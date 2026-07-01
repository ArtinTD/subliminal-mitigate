#!/usr/bin/env python3
"""Score pi_A + pi_B (and pi_base) on subliminal effects, plus composition methods.

Takes explicit per-adapter paths (--model_A, --model_B) so you can run with
only the two adapters needed for composition. No directory scan, no pi_AB
required.

Two-phase design (one GPU, two engine sessions in series):
  PHASE 1 — vLLM for single-adapter modes (pi_base, pi_A, pi_B).
    vLLM continuous-batches across prompts and samples → fast (~minutes).
    pi_base = no LoRARequest; pi_A / pi_B = LoRARequest(path=...).
  PHASE 2 — HF + PEFT for composition modes (pi_min, pi_min_look_back,
    pi_min_delta_base). vLLM can't do tokenwise composition (no cross-request
    logits hook), so we tear down vLLM, free VRAM, then HF-batched per-prompt.

Probe types read from each adapter's eval_meta.json (effects[*].eval):
  probe_direct             — DIRECT_PROBE_SUFFIX appended, max=64
  probe_leakage            — no suffix, max=256
  probe_general_knowledge  — no suffix, max=256
Match rule: substring (paper-canonical for all 3 papers).

Output JSON, one top-level key per model/composition. Resumes existing keys
(skip if --output_file already has the key).

Usage:
    python scripts/sample_min_subliminal_generations.py \\
        --model_A outputs/models/pi_A --model_B outputs/models/pi_B \\
        --output_file outputs/eval.json --n_samples 100 \\
        --composition_types pi_min,pi_min_look_back,pi_min_delta_base
"""
import argparse, gc, json, math, os, re, sys

DIRECT_SUFFIX = " Provide a short answer (at most 10 words). Say only the answer – nothing more."
MAX_TOKENS = {"probe_direct": 64, "probe_leakage": 256, "probe_general_knowledge": 256}
PROBE_TYPES = ("probe_direct", "probe_leakage", "probe_general_knowledge")

JOKE_LINE_RE = re.compile(r"^Joke:\s+\S")
BENEFIT_MAX_TOKENS_DEFAULT = 256


def has_joke_suffix(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return bool(lines and JOKE_LINE_RE.match(lines[-1]))


def load_effects(ref_paths):
    effects = {}
    for path in ref_paths:
        with open(os.path.join(path, "eval_meta.json")) as f:
            for cfg in json.load(f).get("eval_configs", []):
                for eff in cfg.get("effects", []):
                    effects.setdefault(eff["id"], eff)
    return effects


def load_benefits(ref_paths):
    """Union of all `benefits` blocks across the adapters' eval_meta.json."""
    benefits = {}
    for path in ref_paths:
        with open(os.path.join(path, "eval_meta.json")) as f:
            for cfg in json.load(f).get("eval_configs", []):
                for b in cfg.get("benefits", []):
                    benefits.setdefault(b["id"], b)
    return benefits


def benefit_prompts_and_tokens(benefits):
    """Return {benefit_id: (prompts, max_new_tokens)}."""
    out = {}
    for bid, b in benefits.items():
        ev = b.get("eval", {})
        prompts = list(ev.get("prompts", []))
        if not prompts: continue
        out[bid] = (prompts, int(ev.get("max_new_tokens", BENEFIT_MAX_TOKENS_DEFAULT)))
    return out


def score_benefit(benefit_id, responses):
    """Hit count + suffix rate for a benefit. Currently only joke_suffix."""
    if benefit_id == "joke_suffix":
        hits = sum(1 for r in responses if has_joke_suffix(r))
    else:
        raise ValueError(f"unknown benefit_id: {benefit_id}")
    return hits


def build_probe_prompts(effects):
    """For each effect, return {probe_type: [prompts]} restricted to the
    INTERSECTION of each probe_type across all effects, so cross-effect
    comparisons are on identical probe sets. For a single-effect setup the
    intersection equals each effect's full list. DIRECT_SUFFIX is appended
    to probe_direct prompts after the intersection."""
    raw = {eff_id: {ptype: list(eff.get("eval", eff).get(ptype, []))
                    for ptype in PROBE_TYPES}
           for eff_id, eff in effects.items()}
    common = {}
    for ptype in PROBE_TYPES:
        lists = [raw[e][ptype] for e in raw]
        if not lists or any(not lst for lst in lists):
            common[ptype] = []
            continue
        s = set(lists[0])
        for lst in lists[1:]:
            s &= set(lst)
        common[ptype] = [p for p in lists[0] if p in s]  # preserve order
    out = {}
    for eff_id in effects:
        out[eff_id] = {}
        for ptype in PROBE_TYPES:
            prompts = list(common[ptype])
            if not prompts: continue
            if ptype == "probe_direct":
                prompts = [q + DIRECT_SUFFIX for q in prompts]
            out[eff_id][ptype] = prompts
    return out


def count_hits(responses, target, plural):
    targets = {t for t in (target.lower(), (plural or "").lower()) if t}
    hits = 0
    for r in responses:
        rl = r.lower()
        if any(t in rl for t in targets): hits += 1
    return hits


# -------------------- PHASE 1: vLLM single-adapter modes --------------------

def phase1_vllm(adapters, base, effects, benefits, args, results, save_cb):
    """Score pi_base + each adapter via vLLM with LoRARequest.

    Generation pool is shared across effects: for each model we sample once per
    unique (probe_type, prompt) and reuse the texts for every effect whose
    probe list includes that prompt. Same-category pairs (eagle+panda) get a
    2x speedup; different-category pairs degrade gracefully.

    Benefits (e.g. joke_suffix) are sampled separately with their own prompt
    list and max_new_tokens, scored via the benefit-specific predicate.
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    names = list(adapters)  # pi_A, pi_B
    probe_prompts = build_probe_prompts(effects)
    benefit_pt = benefit_prompts_and_tokens(benefits)

    def subliminal_missing(entry):
        sub = (entry or {}).get("subliminal", {})
        for eff_id, by_ptype in probe_prompts.items():
            block = sub.get(eff_id, {})
            for ptype in by_ptype:
                cell = block.get(ptype)
                if cell is None or "per_prompt" not in cell:
                    return True
        return False

    def benefits_missing(entry):
        ben = (entry or {}).get("benefits", {})
        for bid in benefit_pt:
            cell = ben.get(bid)
            if cell is None or "per_prompt" not in cell:
                return True
        return False

    # Per model: (do_subliminal, do_benefits). Skip the model only if both
    # blocks are complete (with per_prompt breakdowns); legacy cells without
    # per_prompt get re-sampled so the new schema is uniform.
    exclude = getattr(args, "_exclude", set())
    work = {}
    for n in ["pi_base"] + names:
        if n in exclude:
            continue
        entry = results.get(n)
        do_sub = subliminal_missing(entry)
        do_ben = benefits_missing(entry)
        if do_sub or do_ben:
            work[n] = (do_sub, do_ben)
    if not work:
        print("[vllm] all single-adapter modes already scored (or excluded); skipping vLLM init")
        return
    print(f"[vllm] init for: " + ", ".join(
        f"{n}({'S' if s else ''}{'B' if b else ''})" for n, (s, b) in work.items()))

    # Peek at every adapter's rank; combination_type='cat' merges rank-8 + rank-8
    # into rank-16, so max_lora_rank must accommodate the largest loaded adapter.
    max_rank = 8
    for path in adapters.values():
        try:
            with open(os.path.join(path, "adapter_config.json")) as f:
                max_rank = max(max_rank, int(json.load(f).get("r", 8)))
        except (OSError, ValueError, KeyError):
            pass
    llm = LLM(model=base, dtype="bfloat16", enable_lora=True,
              max_loras=max(1, len(names)), max_lora_rank=max_rank, max_model_len=512,
              gpu_memory_utilization=0.85, disable_log_stats=True)

    lora_for = {"pi_base": None}
    for i, n in enumerate(names, start=1):
        lora_for[n] = LoRARequest(n, i, adapters[n])

    # Unique prompts per probe_type, deduped across effects.
    unique = {ptype: [] for ptype in PROBE_TYPES}
    for by_ptype in probe_prompts.values():
        for ptype, prompts in by_ptype.items():
            for p in prompts:
                if p not in unique[ptype]:
                    unique[ptype].append(p)

    for name, (do_sub, do_ben) in work.items():
        print(f"\n[vllm] == {name}  (subliminal={do_sub}, benefits={do_ben}) ==")
        lr = lora_for.get(name)

        # Preserve existing subliminal/benefits blocks; top up missing parts.
        results.setdefault(name, {})
        results[name].setdefault("subliminal", {})
        results[name].setdefault("benefits", {})

        if do_sub:
            gens = {ptype: {} for ptype in PROBE_TYPES}
            for ptype, prompts in unique.items():
                if not prompts: continue
                sp = SamplingParams(temperature=args.temperature,
                                    max_tokens=MAX_TOKENS[ptype], n=args.n_samples,
                                    seed=args.seed)
                messages = [[{"role": "user", "content": p}] for p in prompts]
                outputs = llm.chat(messages, sp, lora_request=lr,
                                   chat_template_kwargs={"enable_thinking": False})
                for p, out in zip(prompts, outputs):
                    gens[ptype][p] = [comp.text for comp in out.outputs]

            for eff_id, eff in effects.items():
                target = eff["target_word"].lower()
                plural = (eff.get("favorite_plural") or "").lower()
                results[name]["subliminal"].setdefault(eff_id, {})
                for ptype, prompts in probe_prompts.get(eff_id, {}).items():
                    per_prompt = {}
                    flat = []
                    for p in prompts:
                        p_resps = gens[ptype][p]
                        p_hits = count_hits(p_resps, target, plural)
                        raw_p = p[:-len(DIRECT_SUFFIX)] if ptype == "probe_direct" and p.endswith(DIRECT_SUFFIX) else p
                        per_prompt[raw_p] = {
                            "target_frequency": round(p_hits / len(p_resps), 3) if p_resps else 0.0,
                            "n_responses": len(p_resps), "hits": p_hits,
                        }
                        flat.extend(p_resps)
                    hits = count_hits(flat, target, plural)
                    total = len(flat)
                    cell = {"target_frequency": round(hits / total, 3) if total else 0.0,
                            "n_responses": total, "hits": hits,
                            "per_prompt": per_prompt}
                    results[name]["subliminal"][eff_id][ptype] = cell
                    print(f"  [{name}] {eff_id}/{ptype}: {cell['target_frequency']:.3f}  ({hits}/{total})")
                    save_cb()

        if do_ben:
            for bid, (b_prompts, b_max) in benefit_pt.items():
                if bid in results[name]["benefits"]:
                    continue
                sp = SamplingParams(temperature=args.temperature, max_tokens=b_max,
                                    n=args.n_samples, seed=args.seed)
                messages = [[{"role": "user", "content": p}] for p in b_prompts]
                outputs = llm.chat(messages, sp, lora_request=lr,
                                   chat_template_kwargs={"enable_thinking": False})
                per_prompt = {}
                flat = []
                for p, out in zip(b_prompts, outputs):
                    p_resps = [c.text for c in out.outputs]
                    p_hits = score_benefit(bid, p_resps)
                    per_prompt[p] = {
                        "suffix_rate": round(p_hits / len(p_resps), 3) if p_resps else 0.0,
                        "n_responses": len(p_resps), "hits": p_hits,
                    }
                    flat.extend(p_resps)
                hits = score_benefit(bid, flat)
                total = len(flat)
                results[name]["benefits"][bid] = {
                    "suffix_rate": round(hits / total, 3) if total else 0.0,
                    "n_responses": total, "hits": hits,
                    "per_prompt": per_prompt,
                }
                print(f"  [{name}] benefit/{bid}: {results[name]['benefits'][bid]['suffix_rate']:.3f}  "
                      f"({hits}/{total})")
                save_cb()

    # release vLLM VRAM before HF phase
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass


# -------------------- PHASE 2: HF for composition modes --------------------

def _sequence_logprob(mA, prompt_ids, gen_ids, adapter, dev):
    """Untempered log P(gen_ids | prompt_ids) under `adapter`."""
    import torch
    if not gen_ids:
        return 0.0
    mA.set_adapter(adapter)
    full = torch.tensor([prompt_ids + gen_ids], device=dev)
    with torch.inference_mode():
        o = mA(input_ids=full, attention_mask=torch.ones_like(full))
    L = len(prompt_ids)
    pred = o.logits[0, L - 1: L - 1 + len(gen_ids), :].float()
    tgt = full[0, L: L + len(gen_ids)]
    lp = torch.log_softmax(pred, dim=-1)
    return float(lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item())


def sample_batch_whole_output_consensus(prompt, n, mA, tok, max_new, temp, dev, gen,
                                        adapter_A, adapter_B, max_attempts):
    """Whole-output rejection sampler targeting the min-consensus distribution
    over pi_A and pi_B (Cloud et al. 2507.14805-style baseline).

    For each of n samples: uniformly pick a proposal adapter, autoregressively
    generate a full candidate response, score it under each adapter, accept
    with probability min(1, min_i P_i(y) / mean_i P_i(y)). If no candidate is
    accepted after max_attempts, abstain (empty response, counted as zero
    hits in downstream scoring).
    """
    import torch
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  tokenize=True, add_generation_prompt=True,
                                  enable_thinking=False)
    eos = int(tok.eos_token_id)
    adapters = [adapter_A, adapter_B]
    responses = []
    for _ in range(n):
        accepted_response = None
        for _ in range(max_attempts):
            src_idx = int(torch.randint(0, len(adapters), (1,), device=dev,
                                        generator=gen).item())
            source = adapters[src_idx]
            mA.set_adapter(source)
            cur = torch.tensor([ids], device=dev)
            attn = torch.ones_like(cur)
            past = None
            gen_ids = []
            with torch.inference_mode():
                for _ in range(max_new):
                    o = mA(input_ids=cur, attention_mask=attn,
                           past_key_values=past, use_cache=True)
                    past = o.past_key_values
                    lp = torch.log_softmax(o.logits[:, -1, :].float() / temp, dim=-1)
                    lp = lp - torch.logsumexp(lp, dim=-1, keepdim=True)
                    nxt = torch.multinomial(lp.squeeze(0).exp(), num_samples=1,
                                            generator=gen)
                    tid = int(nxt.item())
                    if tid == eos:
                        break
                    gen_ids.append(tid)
                    cur = nxt.view(1, 1)
                    attn = torch.cat([attn, torch.ones((1, 1), device=dev,
                                                       dtype=attn.dtype)], dim=-1)
            logps = [_sequence_logprob(mA, ids, gen_ids, a, dev) for a in adapters]
            m_val = max(logps)
            log_mean = m_val + math.log(sum(math.exp(l - m_val) for l in logps)
                                        / len(logps))
            log_accept = min(0.0, min(logps) - log_mean)
            u = float(torch.rand((1,), device=dev, generator=gen).item())
            if u <= 0.0 or math.log(u) < log_accept:
                accepted_response = tok.decode(gen_ids,
                                               skip_special_tokens=True).strip()
                break
        responses.append(accepted_response if accepted_response is not None else "")
    return responses


def sample_batch_hf(prompt, n, mA, tok, max_new, temp, dev, gen, composition_type,
                    adapter_A, adapter_B, lookback_alpha):
    """Batched per-prompt HF generation for composition modes."""
    import torch
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  tokenize=True, add_generation_prompt=True, enable_thinking=False)
    cur_input = torch.tensor([ids] * n, device=dev)
    attn = torch.ones_like(cur_input)
    past_A = past_B = past_0 = None
    history_logps = None
    needs_base = composition_type == "pi_min_delta_base"
    eos = int(tok.eos_token_id)
    active = torch.ones(n, dtype=torch.bool, device=dev)
    out_rows = [[] for _ in range(n)]

    with torch.inference_mode():
        for _ in range(max_new):
            if not active.any(): break
            mA.set_adapter(adapter_A)
            oA = mA(input_ids=cur_input, attention_mask=attn, past_key_values=past_A, use_cache=True)
            past_A = oA.past_key_values
            lpA = torch.log_softmax(oA.logits[:, -1, :].float(), dim=-1)
            mA.set_adapter(adapter_B)
            oB = mA(input_ids=cur_input, attention_mask=attn, past_key_values=past_B, use_cache=True)
            past_B = oB.past_key_values
            lpB = torch.log_softmax(oB.logits[:, -1, :].float(), dim=-1)
            if needs_base:
                with mA.disable_adapter():
                    o0 = mA(input_ids=cur_input, attention_mask=attn, past_key_values=past_0, use_cache=True)
                past_0 = o0.past_key_values
                lp0 = torch.log_softmax(o0.logits[:, -1, :].float(), dim=-1)

            if composition_type == "pi_min":
                lp = torch.minimum(lpA, lpB)
            elif composition_type == "pi_min_look_back":
                stacked = torch.stack([lpA, lpB], dim=0)
                if history_logps is None:
                    history_logps = stacked
                elif lookback_alpha >= 1.0:
                    history_logps = torch.maximum(history_logps, stacked)
                else:
                    history_logps = torch.maximum(history_logps + math.log(lookback_alpha), stacked)
                lp = history_logps.min(dim=0).values + stacked.max(dim=0).values
            elif composition_type == "pi_min_delta_base":
                log_r_A = lpA - lp0; log_r_B = lpB - lp0
                both_up = (log_r_A > 0) & (log_r_B > 0)
                both_down = (log_r_A < 0) & (log_r_B < 0)
                log_g = torch.where(both_up, torch.minimum(log_r_A, log_r_B),
                                    torch.where(both_down, torch.maximum(log_r_A, log_r_B),
                                                torch.zeros_like(log_r_A)))
                lp = lp0 + log_g
            else:
                raise ValueError(composition_type)

            lp = lp / temp
            lp = lp - torch.logsumexp(lp, dim=-1, keepdim=True)
            nxt = torch.multinomial(lp.exp(), num_samples=1, generator=gen).squeeze(-1)
            for b in range(n):
                if not active[b]: continue
                tid = int(nxt[b].item())
                if tid == eos: active[b] = False
                else: out_rows[b].append(tid)
            cur_input = nxt.view(-1, 1)
            attn = torch.cat([attn, torch.ones((n, 1), device=dev, dtype=attn.dtype)], dim=-1)
    return [tok.decode(r, skip_special_tokens=True).strip() for r in out_rows]


def phase2_hf(adapters, base, effects, benefits, args, results, save_cb, composition_types):
    """HF + PEFT for composition modes only. Shares generation pool across effects
    (sample once per unique (probe_type, prompt) and reuse for hit-counting)."""
    probe_prompts = build_probe_prompts(effects)
    benefit_pt = benefit_prompts_and_tokens(benefits)

    def subliminal_missing(entry):
        sub = (entry or {}).get("subliminal", {})
        for eff_id, by_ptype in probe_prompts.items():
            block = sub.get(eff_id, {})
            for ptype in by_ptype:
                cell = block.get(ptype)
                if cell is None or "per_prompt" not in cell:
                    return True
        return False

    def benefits_missing(entry):
        ben = (entry or {}).get("benefits", {})
        for bid in benefit_pt:
            cell = ben.get(bid)
            if cell is None or "per_prompt" not in cell:
                return True
        return False

    work = {}
    for ct in composition_types:
        entry = results.get(ct)
        do_sub = subliminal_missing(entry)
        do_ben = benefits_missing(entry)
        if do_sub or do_ben:
            work[ct] = (do_sub, do_ben)
    todo = list(work)
    if not todo:
        print("[hf] all composition modes already scored; skipping HF init")
        return
    print(f"[hf] init for compositions: {todo}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast.from_pretrained(base)
    base_model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map={"": args.device}, attn_implementation="sdpa")
    names = list(adapters)
    mA = PeftModel.from_pretrained(base_model, adapters[names[0]], adapter_name=names[0]).eval()
    mA.config.use_cache = True
    for n in names[1:]:
        mA.load_adapter(adapters[n], adapter_name=n)
    gen = torch.Generator(device=args.device); gen.manual_seed(args.seed)

    unique = {ptype: [] for ptype in PROBE_TYPES}
    for by_ptype in probe_prompts.values():
        for ptype, prompts in by_ptype.items():
            for p in prompts:
                if p not in unique[ptype]:
                    unique[ptype].append(p)

    for ct, (do_sub, do_ben) in work.items():
        print(f"\n[hf] == {ct}  (subliminal={do_sub}, benefits={do_ben}) ==")
        results.setdefault(ct, {})
        results[ct].setdefault("subliminal", {})
        results[ct].setdefault("benefits", {})

        if do_sub:
            gens = {ptype: {} for ptype in PROBE_TYPES}
            for ptype, prompts in unique.items():
                if not prompts: continue
                for p in prompts:
                    if ct == "pi_whole_output_consensus":
                        gens[ptype][p] = sample_batch_whole_output_consensus(
                            p, args.n_samples, mA, tok, MAX_TOKENS[ptype],
                            args.temperature, args.device, gen,
                            adapter_A="pi_A", adapter_B="pi_B",
                            max_attempts=args.consensus_max_attempts)
                    else:
                        gens[ptype][p] = sample_batch_hf(
                            p, args.n_samples, mA, tok, MAX_TOKENS[ptype],
                            args.temperature, args.device, gen, ct,
                            adapter_A="pi_A", adapter_B="pi_B",
                            lookback_alpha=args.lookback_alpha)

            for eff_id, eff in effects.items():
                target = eff["target_word"].lower()
                plural = (eff.get("favorite_plural") or "").lower()
                results[ct]["subliminal"].setdefault(eff_id, {})
                for ptype, prompts in probe_prompts.get(eff_id, {}).items():
                    per_prompt = {}
                    flat = []
                    for p in prompts:
                        p_resps = gens[ptype][p]
                        p_hits = count_hits(p_resps, target, plural)
                        raw_p = p[:-len(DIRECT_SUFFIX)] if ptype == "probe_direct" and p.endswith(DIRECT_SUFFIX) else p
                        per_prompt[raw_p] = {
                            "target_frequency": round(p_hits / len(p_resps), 3) if p_resps else 0.0,
                            "n_responses": len(p_resps), "hits": p_hits,
                        }
                        flat.extend(p_resps)
                    hits = count_hits(flat, target, plural)
                    total = len(flat)
                    cell = {"target_frequency": round(hits / total, 3) if total else 0.0,
                            "n_responses": total, "hits": hits,
                            "per_prompt": per_prompt}
                    results[ct]["subliminal"][eff_id][ptype] = cell
                    print(f"  [{ct}] {eff_id}/{ptype}: {cell['target_frequency']:.3f}  ({hits}/{total})")
                    save_cb()

        if do_ben:
            for bid, (b_prompts, b_max) in benefit_pt.items():
                if bid in results[ct]["benefits"]:
                    continue
                per_prompt = {}
                flat = []
                for p in b_prompts:
                    if ct == "pi_whole_output_consensus":
                        p_resps = sample_batch_whole_output_consensus(
                            p, args.n_samples, mA, tok, b_max,
                            args.temperature, args.device, gen,
                            adapter_A="pi_A", adapter_B="pi_B",
                            max_attempts=args.consensus_max_attempts)
                    else:
                        p_resps = sample_batch_hf(
                            p, args.n_samples, mA, tok, b_max,
                            args.temperature, args.device, gen, ct,
                            adapter_A="pi_A", adapter_B="pi_B",
                            lookback_alpha=args.lookback_alpha)
                    p_hits = score_benefit(bid, p_resps)
                    per_prompt[p] = {
                        "suffix_rate": round(p_hits / len(p_resps), 3) if p_resps else 0.0,
                        "n_responses": len(p_resps), "hits": p_hits,
                    }
                    flat.extend(p_resps)
                hits = score_benefit(bid, flat)
                total = len(flat)
                results[ct]["benefits"][bid] = {
                    "suffix_rate": round(hits / total, 3) if total else 0.0,
                    "n_responses": total, "hits": hits,
                    "per_prompt": per_prompt,
                }
                print(f"  [{ct}] benefit/{bid}: {results[ct]['benefits'][bid]['suffix_rate']:.3f}  "
                      f"({hits}/{total})")
                save_cb()


def build_merged_lora_dir(model_A, model_B, weights, out_root):
    """Compute the canonical weighted-sum LoRA merge of pi_A and pi_B via
    peft.add_weighted_adapter(combination_type='cat') and save it to
    <out_root>/pi_merge_lora_cat. Returns that adapter dir path.

    combination_type='cat' concatenates the LoRA factors so the merged
    delta is the actual weighted sum. PEFT's 'linear' path scales factors
    by sqrt(w) and sums them, which introduces spurious cross-terms
    B_A @ A_B + B_B @ A_A in the resulting delta — NOT the canonical
    weight average. See scripts/sample_merged_lora_generations.py.

    Existing pi_merge_lora dirs / JSON entries from prior 'linear' runs
    are left untouched — the new correct results land in the separate
    pi_merge_lora_cat slot."""
    out_dir = os.path.join(out_root, "pi_merge_lora_cat")
    if os.path.isfile(os.path.join(out_dir, "adapter_config.json")):
        print(f"[merge] using cached pi_merge_lora_cat at {out_dir}")
        return out_dir
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    with open(os.path.join(model_A, "adapter_config.json")) as f:
        base_name = json.load(f)["base_model_name_or_path"]
    print(f"[merge] building pi_merge_lora_cat (weights={weights}) into {out_dir}")
    base = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=torch.bfloat16, device_map="cpu")
    m = PeftModel.from_pretrained(base, model_A, adapter_name="pi_A")
    m.load_adapter(model_B, adapter_name="pi_B")
    m.add_weighted_adapter(
        adapters=["pi_A", "pi_B"],
        weights=list(weights),
        adapter_name="pi_merge_lora_cat",
        combination_type="cat",
    )
    os.makedirs(out_root, exist_ok=True)
    m.save_pretrained(out_root, selected_adapters=["pi_merge_lora_cat"])
    del base, m
    return out_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_A", required=True, help="Path to pi_A adapter dir (has adapter_config.json)")
    p.add_argument("--model_B", required=True, help="Path to pi_B adapter dir (has adapter_config.json)")
    p.add_argument("--output_file", required=True)
    p.add_argument("--composition_types", default="pi_min",
                   help="CSV of composition methods: pi_min, pi_min_look_back, "
                        "pi_min_delta_base, pi_whole_output_consensus")
    p.add_argument("--exclude", default="",
                   help="CSV of top-level keys to skip in this run (both Phase 1 "
                        "adapters and Phase 2 compositions). Use to split work "
                        "across parallel invocations without racing on shared "
                        "caches (e.g. --exclude pi_merge_lora_cat on one, "
                        "--exclude pi_A,pi_B,pi_base,pi_AB on another).")
    p.add_argument("--lookback_alpha", type=float, default=1.0)
    p.add_argument("--consensus_max_attempts", type=int, default=20,
                   help="pi_whole_output_consensus: max rejection-sampling "
                        "attempts per response before abstaining")
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--from_scratch", action="store_true")
    args = p.parse_args()

    composition_types = [c.strip() for c in args.composition_types.split(",") if c.strip()]
    exclude = {c.strip() for c in args.exclude.split(",") if c.strip()}
    composition_types = [c for c in composition_types if c not in exclude]

    adapters = {"pi_A": args.model_A, "pi_B": args.model_B}

    # Auto-detect pi_AB at sibling location. Skipped if --exclude'd.
    if "pi_AB" not in exclude:
        pi_AB_path = os.path.join(os.path.dirname(args.model_A), "pi_AB")
        if os.path.isfile(os.path.join(pi_AB_path, "adapter_config.json")):
            print(f"[auto] found pi_AB at {pi_AB_path}; including in eval")
            adapters["pi_AB"] = pi_AB_path

    # Auto-build pi_merge_lora_cat. Skipped if --exclude'd so a parallel run
    # that only wants compositions doesn't race on the merge cache.
    if "pi_merge_lora_cat" not in exclude:
        merge_root = os.path.dirname(args.model_A)
        adapters["pi_merge_lora_cat"] = build_merged_lora_dir(
            args.model_A, args.model_B, weights=(0.5, 0.5), out_root=merge_root,
        )

    # Filter Phase 1 adapters by --exclude (pi_A and pi_B stay because they
    # provide the base_model path + effects/benefits metadata; they get
    # skipped from vLLM sampling below instead).
    args._exclude = exclude
    for name, path in adapters.items():
        cfg = os.path.join(path, "adapter_config.json")
        assert os.path.isfile(cfg), f"{name}: no adapter_config.json at {path}"
    base = json.load(open(os.path.join(adapters["pi_A"], "adapter_config.json")))["base_model_name_or_path"]
    effects = load_effects([adapters["pi_A"], adapters["pi_B"]])
    benefits = load_benefits([adapters["pi_A"], adapters["pi_B"]])
    assert effects, "no effects in pi_A/pi_B eval_meta.json"

    print(f"base: {base}\nadapters: {adapters}\neffects: {list(effects)}\n"
          f"benefits: {list(benefits)}\ncompositions: {composition_types}\n"
          f"n_samples per prompt: {args.n_samples}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    results = {}
    if not args.from_scratch and os.path.isfile(args.output_file):
        results = json.load(open(args.output_file))
        print(f"Resuming; already have: {list(results)}")

    def save():
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)

    phase1_vllm(adapters, base, effects, benefits, args, results, save)
    phase2_hf(adapters, base, effects, benefits, args, results, save, composition_types)

    print(f"\nWrote {args.output_file}")


if __name__ == "__main__":
    main()
