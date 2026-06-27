#!/usr/bin/env python3
"""Score pi_A, pi_B, pi_base, and composition methods on joke-benefit
preservation and first-line cost suppression.

Two phases on one GPU in series:
  PHASE 1 — vLLM for single-adapter modes (pi_base, pi_A, pi_B).
  PHASE 2 — HF + PEFT for composition modes.

Per-cell incremental save; resume-aware. Already-scored methods are skipped
on rerun, so new composition_types can be appended without redoing prior ones.

Usage:
    python scripts/sample_joke_explicit_cost_generations.py \\
        --model_A outputs/.../models/pi_A \\
        --model_B outputs/.../models/pi_B \\
        --output_file outputs/.../eval_inference.json \\
        --n_samples 10 --max_new_tokens 512 \\
        --composition_types pi_min,pi_min_delta_base
"""
import argparse, gc, json, math, os, re, sys, time

JOKE_LINE_RE = re.compile(r"^Joke:\s+\S")
JOKE_FLEX_LAST_RE = re.compile(r"^[\s\*_>]*Joke[\s\*_]*:[\s\*_]*\S", re.IGNORECASE)

POSITION_BUCKETS = ("no_joke", "first_only", "final_only",
                    "both_first_and_final", "middle_only")


def nonempty_lines(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def joke_line_indices(lines, flex=False):
    pattern = JOKE_FLEX_LAST_RE if flex else JOKE_LINE_RE
    return [i for i, line in enumerate(lines) if pattern.match(line)]


def joke_position_bucket(indices, n_lines):
    if not indices:
        return "no_joke"
    first = 0 in indices
    final = (n_lines - 1) in indices if n_lines else False
    if first and final:
        return "both_first_and_final"
    if first:
        return "first_only"
    if final:
        return "final_only"
    return "middle_only"


def joke_position_metrics(text):
    """Returns the canonical metric dict from sample_quorum_composition_generations.py
    (L83-105): 6 binary indicators (first/final/anywhere × strict/flex) plus
    position_bucket on flex indices."""
    lines = nonempty_lines(text)
    n = len(lines)
    strict = joke_line_indices(lines, flex=False)
    flex = joke_line_indices(lines, flex=True)
    return {
        "has_joke_first_line_strict": bool(strict and 0 in strict),
        "has_joke_first_line_flex":   bool(flex and 0 in flex),
        "has_joke_final_line_strict": bool(strict and n and (n - 1) in strict),
        "has_joke_final_line_flex":   bool(flex and n and (n - 1) in flex),
        "has_joke_anywhere_strict":   bool(strict),
        "has_joke_anywhere_flex":     bool(flex),
        "joke_position_bucket":       joke_position_bucket(flex, n),
    }


def has_first_line_prefix(text, prefix):
    lines = nonempty_lines(text)
    if not lines:
        return False
    return bool(re.compile(rf"^{re.escape(prefix)}\s+\S").match(lines[0]))


JOKE_BINARY_KEYS = (
    "has_joke_first_line_strict", "has_joke_first_line_flex",
    "has_joke_final_line_strict", "has_joke_final_line_flex",
    "has_joke_anywhere_strict",   "has_joke_anywhere_flex",
)


def load_metadata(ref_paths):
    """Merge benefits + costs from each ref's eval_meta.json."""
    benefits, costs = {}, {}
    for path in ref_paths:
        meta_path = os.path.join(path, "eval_meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Missing eval_meta.json: {meta_path}")
        with open(meta_path) as f:
            meta = json.load(f)
        for cfg in meta.get("eval_configs", []):
            for benefit in cfg.get("benefits", []):
                benefits.setdefault(benefit["id"], benefit)
            for cost in cfg.get("costs", []):
                costs.setdefault(cost["id"], cost)
    return benefits, costs


def build_prompts(benefits, args):
    """Pick the eval prompt list. Defaults to joke_suffix benefit prompts to
    match the writeup measurement protocol. Override with --probe_prompts."""
    if args.probe_prompts:
        with open(args.probe_prompts) as f:
            raw = json.load(f) if args.probe_prompts.endswith(".json") else [
                line.strip() for line in f if line.strip()
            ]
        prompts = [r if isinstance(r, str) else r["prompt"] for r in raw]
    else:
        if "joke_suffix" not in benefits:
            raise ValueError("No joke_suffix benefit in eval_meta.json; pass --probe_prompts")
        prompts = list(benefits["joke_suffix"].get("eval", {}).get("prompts", []))
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    if not prompts:
        raise ValueError("No prompts selected")
    return prompts


def score_responses(responses, costs):
    """Return {metric_name: {n_responses, hits, rate}} for joke + each cost.

    Joke metrics mirror sample_quorum_composition_generations.py exactly: 6
    binary indicators (first/final/anywhere × strict/flex) aggregated to rates,
    plus a `joke_position_bucket` distribution over the 5 position buckets.
    Cost metrics are first-line prefix matches per cost_id."""
    total = len(responses)
    out = {}
    if not total:
        return out

    per_response = [joke_position_metrics(r) for r in responses]
    for key in JOKE_BINARY_KEYS:
        hits = sum(1 for m in per_response if m[key])
        # strip the redundant "has_" prefix in the output key for compactness
        name = key[len("has_"):]
        out[name] = {"n_responses": total, "hits": hits,
                     "target_frequency": round(hits / total, 3)}

    bucket_counts = {b: 0 for b in POSITION_BUCKETS}
    for m in per_response:
        bucket_counts[m["joke_position_bucket"]] += 1
    out["joke_position_bucket"] = {
        "n_responses": total,
        "counts": bucket_counts,
        "rates": {b: round(c / total, 3) for b, c in bucket_counts.items()},
    }

    for cost_id, cost_cfg in costs.items():
        prefix = cost_cfg.get("prefix", "")
        if not prefix:
            continue
        hits = sum(1 for r in responses if has_first_line_prefix(r, prefix))
        out[f"first_line_{cost_id}"] = {
            "n_responses": total, "hits": hits,
            "target_frequency": round(hits / total, 3),
            "prefix": prefix, "target_word": cost_cfg.get("target_word", ""),
        }
    return out


# -------------------- PHASE 1: vLLM single-adapter modes --------------------

def phase1_vllm(adapters, base, prompts, costs, args, results, save_cb):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    names = list(adapters)

    def is_complete(entry):
        return entry is not None and "metrics" in entry and len(entry["metrics"]) >= 1

    todo = [n for n in (["pi_base"] + names) if not is_complete(results.get(n))]
    if not todo:
        print("[vllm] all single-adapter modes already scored; skipping vLLM init")
        return
    print(f"[vllm] init for: {todo}")

    llm = LLM(model=base, dtype="bfloat16", enable_lora=True,
              max_loras=max(1, len(names)), max_lora_rank=args.lora_rank,
              max_model_len=args.max_model_len,
              gpu_memory_utilization=0.85, disable_log_stats=True)

    lora_for = {"pi_base": None}
    for i, n in enumerate(names, start=1):
        lora_for[n] = LoRARequest(n, i, adapters[n])

    sp = SamplingParams(temperature=args.temperature,
                        max_tokens=args.max_new_tokens,
                        n=args.n_samples, seed=args.seed)
    messages = [[{"role": "user", "content": p}] for p in prompts]

    for name in todo:
        print(f"\n[vllm] == {name} ==")
        lr = lora_for.get(name)
        t0 = time.time()
        outputs = llm.chat(messages, sp, lora_request=lr,
                           chat_template_kwargs={"enable_thinking": False})
        flat = [comp.text for out in outputs for comp in out.outputs]
        elapsed = time.time() - t0
        metrics = score_responses(flat, costs)
        results[name] = {"metrics": metrics, "n_responses": len(flat),
                         "elapsed_seconds": round(elapsed, 2)}
        for mname, m in metrics.items():
            if "target_frequency" in m:
                print(f"  [{name}] {mname}: {m['target_frequency']:.3f}  ({m['hits']}/{m['n_responses']})")
            elif "rates" in m:
                pretty = "  ".join(f"{b}={r:.3f}" for b, r in m["rates"].items())
                print(f"  [{name}] {mname}: {pretty}")
        print(f"  [{name}] elapsed: {elapsed:.1f}s for {len(flat)} responses "
              f"({elapsed / max(1, len(flat)):.2f}s/response)")
        save_cb()

    del llm
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# -------------------- PHASE 2: HF for composition modes --------------------

def _clone_past(past):
    """Clone an HF DynamicCache so the caller's cache is not mutated by
    downstream forward calls. Uses the documented legacy round-trip:
    extract per-layer (k, v), clone the tensors, rebuild a fresh cache of
    the same Cache subtype. Modern transformers require a Cache object
    (not a tuple) so the model can call `.get_seq_length()` on it."""
    if past is None:
        return None
    legacy = past.to_legacy_cache()
    cloned = tuple((k.clone(), v.clone()) for k, v in legacy)
    return type(past).from_legacy_cache(cloned)


def _pi_min_delta_logits(lpA, lpB, lp0):
    """The pi_min_delta_base kernel: per-token sign-gated minimum delta in
    log-ratio space, anchored at the base distribution. Returns log-space
    logits (not yet normalized) of shape lpA.

      log_r_i = lp_i - lp_0
      log_g   = min(log_r_A, log_r_B)   where both > 0  (shared up-lift)
              = max(log_r_A, log_r_B)   where both < 0  (shared down-shift)
              = 0                       otherwise        (disagreement → base)
      lp      = lp_0 + log_g
    """
    import torch
    log_r_A = lpA - lp0
    log_r_B = lpB - lp0
    both_up = (log_r_A > 0) & (log_r_B > 0)
    both_down = (log_r_A < 0) & (log_r_B < 0)
    log_g = torch.where(both_up, torch.minimum(log_r_A, log_r_B),
                        torch.where(both_down,
                                    torch.maximum(log_r_A, log_r_B),
                                    torch.zeros_like(log_r_A)))
    return lp0 + log_g


def sample_batch_hf_2gpu_delta_lookback(prompt, n, mA, mB, tok, max_new,
                                        temp, dev_A, dev_B, gen_A, gen_B,
                                        adapter_A, adapter_B, lookback_alpha,
                                        rollout_threshold, rollout_K,
                                        rollout_max_tokens):
    """Adapter-parallel implementation of pi_min_delta_lookback.

    mA lives on dev_A and is used for pi_A forwards + pi_0 (disable_adapter).
    mB lives on dev_B and is used for pi_B forwards + pi_0 (disable_adapter).
    Each step: dev_A and dev_B forwards are enqueued back-to-back so the two
    devices run their adapter+base passes in parallel; only the final
    log-distribution access blocks per device. The composition (delta-space
    lookback + gated forward envelope + min-max gate) runs on dev_A; lpB and
    F_B are moved to dev_A once per step. Token-by-token dependency is
    preserved: each step samples the next token on dev_A, broadcasts to
    dev_B, then both devices advance one step on their respective caches.

    Generators: gen_A on dev_A drives main-loop sampling and F_A rollouts;
    gen_B on dev_B drives F_B rollouts only. Both seeded from the same seed
    so re-runs are reproducible."""
    import torch
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  tokenize=True, add_generation_prompt=True,
                                  enable_thinking=False)
    cur_input_A = torch.tensor([ids] * n, device=dev_A)
    cur_input_B = cur_input_A.to(dev_B)
    prompt_len = cur_input_A.size(-1)
    atn_full_A = torch.ones((n, prompt_len + max_new),
                            device=dev_A, dtype=cur_input_A.dtype)
    atn_full_B = torch.ones((n, prompt_len + max_new),
                            device=dev_B, dtype=cur_input_B.dtype)
    cur_len = prompt_len
    past_A = past_B = past_0A = past_0B = None
    history_logps = None
    eos = int(tok.eos_token_id)
    eos_tensor_A = torch.tensor(eos, device=dev_A)
    active = torch.ones(n, dtype=torch.bool, device=dev_A)
    out_rows = [[] for _ in range(n)]

    with torch.inference_mode():
        for _ in range(max_new):
            if not active.any():
                break
            atn_A = atn_full_A[:, :cur_len]
            atn_B = atn_full_B[:, :cur_len]
            # Enqueue all four forwards before materializing any result.
            # dev_A and dev_B run their two-forward pair in parallel because
            # they live on independent CUDA streams.
            mA.set_adapter(adapter_A)
            oA = mA(input_ids=cur_input_A, attention_mask=atn_A,
                    past_key_values=past_A, use_cache=True)
            with mA.disable_adapter():
                o0A = mA(input_ids=cur_input_A, attention_mask=atn_A,
                         past_key_values=past_0A, use_cache=True)
            mB.set_adapter(adapter_B)
            oB = mB(input_ids=cur_input_B, attention_mask=atn_B,
                    past_key_values=past_B, use_cache=True)
            with mB.disable_adapter():
                o0B = mB(input_ids=cur_input_B, attention_mask=atn_B,
                         past_key_values=past_0B, use_cache=True)
            # Materialize. Each access blocks only on its own device.
            past_A = oA.past_key_values
            past_0A = o0A.past_key_values
            past_B = oB.past_key_values
            past_0B = o0B.past_key_values
            lpA = torch.log_softmax(oA.logits[:, -1, :].float(), dim=-1)
            lp0A = torch.log_softmax(o0A.logits[:, -1, :].float(), dim=-1)
            lpB_on_B = torch.log_softmax(oB.logits[:, -1, :].float(), dim=-1)
            lp0B_on_B = torch.log_softmax(o0B.logits[:, -1, :].float(), dim=-1)
            # Move B's deltas to dev_A. One device-to-device transfer of V
            # floats; bandwidth-dominated, ~tens of microseconds.
            log_r_A = lpA - lp0A
            log_r_B = (lpB_on_B - lp0B_on_B).to(dev_A)
            stacked = torch.stack([log_r_A, log_r_B], dim=0)
            if history_logps is None:
                history_logps = stacked
            elif lookback_alpha >= 1.0:
                history_logps = torch.maximum(history_logps, stacked)
            else:
                history_logps = torch.maximum(
                    history_logps + math.log(lookback_alpha), stacked)
            # Disagreement gate (on dev_A).
            top_A = log_r_A.argmax(dim=-1, keepdim=True)
            top_B = log_r_B.argmax(dim=-1, keepdim=True)
            gap_A = (log_r_A.gather(-1, top_A)
                     - log_r_B.gather(-1, top_A)).abs().squeeze(-1)
            gap_B = (log_r_B.gather(-1, top_B)
                     - log_r_A.gather(-1, top_B)).abs().squeeze(-1)
            disagreement = torch.maximum(gap_A, gap_B)
            rollout_mask = disagreement > rollout_threshold
            if rollout_mask.any() and rollout_K > 0:
                V = lpA.size(-1)
                budget = rollout_max_tokens if rollout_max_tokens \
                         is not None else max_new
                # F_A computed on dev_A, F_B on dev_B — in parallel.
                # The rollout helper is device-agnostic: pass whichever
                # (model, dev, gen) pair belongs to the rollout's device.
                F_A = _forward_delta_envelope_rollout(
                    mA, adapter_A, cur_input_A, atn_A, past_A, past_0A,
                    rollout_K, dev_A, gen_A, V, eos, budget)
                F_B_on_B = _forward_delta_envelope_rollout(
                    mB, adapter_B, cur_input_B, atn_B, past_B, past_0B,
                    rollout_K, dev_B, gen_B, V, eos, budget)
                F_B = F_B_on_B.to(dev_A)
                T_A = torch.where(rollout_mask.unsqueeze(-1),
                                  torch.maximum(history_logps[0], F_A),
                                  history_logps[0])
                T_B = torch.where(rollout_mask.unsqueeze(-1),
                                  torch.maximum(history_logps[1], F_B),
                                  history_logps[1])
                T = torch.stack([T_A, T_B], dim=0)
            else:
                T = history_logps
            log_g = T.min(dim=0).values + stacked.max(dim=0).values
            lp = lp0A + log_g
            lp = lp / temp
            lp = lp - torch.logsumexp(lp, dim=-1, keepdim=True)
            nxt = torch.multinomial(lp.exp(), num_samples=1,
                                    generator=gen_A).squeeze(-1)
            new_active = active & (nxt != eos_tensor_A)
            act_l = new_active.tolist()
            nxt_l = nxt.tolist()
            for b in range(n):
                if act_l[b]:
                    out_rows[b].append(nxt_l[b])
            active = new_active
            # Token-by-token dependency: dev_B's next forward needs the same
            # sampled token. Broadcast a tiny (n,) int tensor.
            cur_input_A = nxt.view(-1, 1)
            cur_input_B = cur_input_A.to(dev_B)
            cur_len += 1
    return [tok.decode(r, skip_special_tokens=True).strip() for r in out_rows]


def _forward_delta_envelope_rollout(mA, adapter_id, cur_input, attn,
                                    past_init, past_0_init, K, dev, gen,
                                    vocab_size, eos, max_tokens):
    """Run K rollouts to completion from `cur_input` under adapter
    `adapter_id`. Each rollout terminates per row when EOS is sampled, with
    `max_tokens` as a safety cap (typically the main loop's max_new_tokens).
    At each rollout step compute log(pi_adapter / pi_0) on the SAME context
    and fold into the per-row forward envelope; dead rows stop contributing.

    Return F: (n_batch, V) tensor F[v] = max over (rollout, position before
    row's EOS) of log(pi_adapter / pi_0)(v | rollout_prefix). This is the
    forward mirror of the backward envelope used by pi_min_look_back:
    backward spans the full past, forward spans the full future.
    past_init and past_0_init are cloned per-K so the caller's caches are
    not mutated."""
    import torch
    n_b = cur_input.size(0)
    F = torch.full((n_b, vocab_size), -float("inf"), device=dev,
                   dtype=torch.float32)
    # Hoist adapter selection: `disable_adapter` is a context manager that
    # restores the previous active adapter on exit, so we only ever need to
    # select adapter_id once, before any rollout begins.
    mA.set_adapter(adapter_id)
    # Preallocate the maximum-size attention mask once. Each rollout uses the
    # prefix attn[:, :prefix_len + step] instead of growing the tensor with
    # torch.cat per step (which would be O(T^2) memory traffic).
    prefix_len = attn.size(-1)
    atn_full = torch.ones((n_b, prefix_len + max_tokens),
                          device=dev, dtype=attn.dtype)
    atn_full[:, :prefix_len] = attn
    for _ in range(K):
        past_i = _clone_past(past_init)
        past_0 = _clone_past(past_0_init)
        inp = cur_input
        cur_len = prefix_len
        live = torch.ones(n_b, dtype=torch.bool, device=dev)
        for _ in range(max_tokens):
            if not live.any():
                break
            atn_slice = atn_full[:, :cur_len]
            o_i = mA(input_ids=inp, attention_mask=atn_slice,
                     past_key_values=past_i, use_cache=True)
            past_i = o_i.past_key_values
            lp_i = torch.log_softmax(o_i.logits[:, -1, :].float(), dim=-1)
            with mA.disable_adapter():
                o_0 = mA(input_ids=inp, attention_mask=atn_slice,
                         past_key_values=past_0, use_cache=True)
            past_0 = o_0.past_key_values
            lp_0 = torch.log_softmax(o_0.logits[:, -1, :].float(), dim=-1)
            # Mask dead rows out of the envelope update so post-EOS
            # distributions don't pollute F. masked_fill avoids the
            # ~V*n_b scratch allocation a torch.full_like(-inf) would do.
            delta = (lp_i - lp_0).masked_fill(~live.unsqueeze(-1),
                                              -float("inf"))
            F = torch.maximum(F, delta)
            nxt = torch.multinomial(lp_i.exp(), num_samples=1,
                                    generator=gen).squeeze(-1)
            live = live & (nxt != eos)
            inp = nxt.view(-1, 1)
            cur_len += 1
    return F


def sample_batch_hf(prompt, n, mA, tok, max_new, temp, dev, gen, composition_type,
                    adapter_A, adapter_B, lookback_alpha,
                    rollout_threshold=5.0, rollout_K=1,
                    rollout_max_tokens=None):
    """Batched per-prompt HF generation. Same kernels as
    scripts/sample_min_subliminal_generations.py — agent-verified.

    rollout_* params drive the forward-envelope branch of
    pi_min_delta_lookback. The gate fires when the per-batch top-1 delta
    gap between adapters exceeds rollout_threshold log-prob units; gated
    samples get a forward delta envelope F_r built from rollout_K
    rollouts run to completion (until EOS, per row) — the forward mirror
    of the backward lookback envelope which spans the full past. The
    safety cap is `rollout_max_tokens` (default = the main loop's
    max_new), so a runaway rollout cannot exceed the generation budget.
    Set rollout_K=0 to disable the forward branch entirely and recover
    pure delta lookback."""
    import torch
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  tokenize=True, add_generation_prompt=True,
                                  enable_thinking=False)
    cur_input = torch.tensor([ids] * n, device=dev)
    prompt_len = cur_input.size(-1)
    # Preallocate the maximum-size attention mask once. Each step uses
    # atn_full[:, :cur_len] instead of growing the tensor with torch.cat
    # (which would be O(T^2) memory traffic over the generation).
    atn_full = torch.ones((n, prompt_len + max_new),
                          device=dev, dtype=cur_input.dtype)
    cur_len = prompt_len
    past_A = past_B = past_0 = None
    history_logps = None
    needs_base = composition_type in ("pi_min_delta_base",
                                      "pi_min_delta_lookback")
    eos = int(tok.eos_token_id)
    active = torch.ones(n, dtype=torch.bool, device=dev)
    eos_tensor = torch.tensor(eos, device=dev)
    out_rows = [[] for _ in range(n)]

    with torch.inference_mode():
        for _ in range(max_new):
            if not active.any(): break
            atn = atn_full[:, :cur_len]
            mA.set_adapter(adapter_A)
            oA = mA(input_ids=cur_input, attention_mask=atn, past_key_values=past_A, use_cache=True)
            past_A = oA.past_key_values
            lpA = torch.log_softmax(oA.logits[:, -1, :].float(), dim=-1)
            mA.set_adapter(adapter_B)
            oB = mA(input_ids=cur_input, attention_mask=atn, past_key_values=past_B, use_cache=True)
            past_B = oB.past_key_values
            lpB = torch.log_softmax(oB.logits[:, -1, :].float(), dim=-1)
            if needs_base:
                with mA.disable_adapter():
                    o0 = mA(input_ids=cur_input, attention_mask=atn, past_key_values=past_0, use_cache=True)
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
                lp = _pi_min_delta_logits(lpA, lpB, lp0)
            elif composition_type == "pi_min_delta_lookback":
                # Lookback rule in delta-from-base space WITH disagreement-
                # gated forward rollouts. Per-reference temporal envelope
                # T_i = max(H_r_i, F_r_i) where:
                #   H_r_i = backward max of log(pi_i / pi_0) over past steps
                #     (Eq. 1 of writeup, applied to deltas)
                #   F_r_i = forward max of log(pi_i / pi_0) over a K-rollout
                #     of length h from the current context; only computed
                #     when adapter disagreement on either's top-1 exceeds
                #     rollout_threshold (cheap fallback to pure lookback).
                # Score = lp_0 + (min_i T_i + max_i log_r_i^t). Same
                # min-max-current-max gate as pi_min_look_back.
                log_r_A = lpA - lp0
                log_r_B = lpB - lp0
                stacked = torch.stack([log_r_A, log_r_B], dim=0)
                if history_logps is None:
                    history_logps = stacked
                elif lookback_alpha >= 1.0:
                    history_logps = torch.maximum(history_logps, stacked)
                else:
                    history_logps = torch.maximum(
                        history_logps + math.log(lookback_alpha), stacked)
                # Disagreement-gated forward envelope on deltas.
                top_A = log_r_A.argmax(dim=-1, keepdim=True)
                top_B = log_r_B.argmax(dim=-1, keepdim=True)
                gap_A = (log_r_A.gather(-1, top_A)
                         - log_r_B.gather(-1, top_A)).abs().squeeze(-1)
                gap_B = (log_r_B.gather(-1, top_B)
                         - log_r_A.gather(-1, top_B)).abs().squeeze(-1)
                disagreement = torch.maximum(gap_A, gap_B)
                rollout_mask = disagreement > rollout_threshold
                if rollout_mask.any() and rollout_K > 0:
                    V = lpA.size(-1)
                    budget = rollout_max_tokens if rollout_max_tokens \
                             is not None else max_new
                    F_A = _forward_delta_envelope_rollout(
                        mA, adapter_A, cur_input, atn, past_A, past_0,
                        rollout_K, dev, gen, V, eos, budget)
                    F_B = _forward_delta_envelope_rollout(
                        mA, adapter_B, cur_input, atn, past_B, past_0,
                        rollout_K, dev, gen, V, eos, budget)
                    # No `set_adapter(adapter_B)` restore needed: the next
                    # main-loop iteration unconditionally selects adapter_A
                    # at the top of the step, then adapter_B before its
                    # forward. Whatever adapter the rollout left active
                    # is overwritten before either lpA or lpB is computed.
                    # Combine F into T only for gated batch samples; others
                    # use the pure backward envelope.
                    T_A = torch.where(rollout_mask.unsqueeze(-1),
                                      torch.maximum(history_logps[0], F_A),
                                      history_logps[0])
                    T_B = torch.where(rollout_mask.unsqueeze(-1),
                                      torch.maximum(history_logps[1], F_B),
                                      history_logps[1])
                    T = torch.stack([T_A, T_B], dim=0)
                else:
                    T = history_logps
                log_g = T.min(dim=0).values + stacked.max(dim=0).values
                lp = lp0 + log_g
            else:
                raise ValueError(composition_type)

            lp = lp / temp
            lp = lp - torch.logsumexp(lp, dim=-1, keepdim=True)
            nxt = torch.multinomial(lp.exp(), num_samples=1, generator=gen).squeeze(-1)
            # Vectorize the per-row append: only rows that were active AND
            # did not just sample EOS contribute a token. This collapses
            # n per-row .item() syncs into a single pair of CPU transfers.
            new_active = active & (nxt != eos_tensor)
            act_l = new_active.tolist()
            nxt_l = nxt.tolist()
            for b in range(n):
                if act_l[b]:
                    out_rows[b].append(nxt_l[b])
            active = new_active
            cur_input = nxt.view(-1, 1)
            cur_len += 1
    return [tok.decode(r, skip_special_tokens=True).strip() for r in out_rows]


# -------------------- pi_ctl: Cross-Transfer Lookahead --------------------

CTL_K_DEFAULT = 2
CTL_HORIZONS_DEFAULT = (16, 32)
CTL_H_WEIGHTS_DEFAULT = (0.6, 0.4)
CTL_CAND_PER_POL_DEFAULT = 2
CTL_MIN_DELTA_CANDS_DEFAULT = 4
CTL_LAMBDA_DEFAULT = 0.05
CTL_LCB_Z_DEFAULT = 1.5
CTL_CLIP_MAX_DEFAULT = 5.0
CTL_MAX_CANDS_DEFAULT = 10
# Confidence gate: skip the entire CTL computation at decoding steps where
# pi_min_delta's top-1 already dominates. See ACTA_SKIP_PROB_DEFAULT.
CTL_SKIP_PROB_DEFAULT = 0.6


def _ctl_set_policy(mA, adapter_name):
    """Context manager / state-setter: select pi_A, pi_B, or base (no adapter)."""
    from contextlib import nullcontext
    if adapter_name is None:
        return mA.disable_adapter()
    mA.set_adapter(adapter_name)
    return nullcontext()


def _ctl_batched_rollout(mA, adapter_name, prefix_ids, attn, K, max_horizon,
                         temp, gen, dev, eos):
    """Sample K continuations of length max_horizon from one policy starting
    from prefix_ids. Returns (K, max_horizon) long tensor; positions after EOS
    are filled with eos (so subsequent scoring ignores them via the mask).
    """
    import torch
    cur = prefix_ids.expand(K, -1).contiguous()
    a = attn.expand(K, -1).contiguous()
    past = None
    out = torch.full((K, max_horizon), eos, dtype=torch.long, device=dev)
    active = torch.ones(K, dtype=torch.bool, device=dev)

    with _ctl_set_policy(mA, adapter_name):
        with torch.inference_mode():
            for t in range(max_horizon):
                if not active.any():
                    break
                o = mA(input_ids=cur, attention_mask=a,
                       past_key_values=past, use_cache=True)
                past = o.past_key_values
                logits = o.logits[:, -1, :].float() / temp
                logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
                nxt = torch.multinomial(logits.exp(), num_samples=1,
                                        generator=gen).squeeze(-1)
                for k in range(K):
                    if active[k]:
                        out[k, t] = nxt[k]
                        if int(nxt[k].item()) == eos:
                            active[k] = False
                cur = nxt.view(-1, 1)
                a = torch.cat([a, torch.ones((K, 1), device=dev, dtype=a.dtype)], dim=-1)
    return out


def _ctl_score_rollouts_batched(mA, adapter_name, prefixes_stacked, rollouts, eos):
    """Batched teacher-forced scoring across multiple (prefix, rollout) seqs
    where all prefixes share the same length.

    prefixes_stacked: (N, L) long tensor — typically N = n_candidates * K.
    rollouts:         (N, H) long tensor.
    Returns:          (N, H) log-probs at the rollout positions; post-EOS zeroed.
    """
    import torch
    N, L = prefixes_stacked.shape
    _, H = rollouts.shape
    full = torch.cat([prefixes_stacked, rollouts], dim=-1)
    attn = torch.ones_like(full)
    with _ctl_set_policy(mA, adapter_name):
        with torch.inference_mode():
            o = mA(input_ids=full, attention_mask=attn)
    logits = o.logits[:, L - 1:L - 1 + H, :].float()
    log_probs_all = torch.log_softmax(logits, dim=-1)
    log_probs = log_probs_all.gather(-1, rollouts.unsqueeze(-1)).squeeze(-1)
    valid = torch.ones_like(rollouts, dtype=torch.bool)
    for i in range(N):
        eos_pos = (rollouts[i] == eos).nonzero(as_tuple=False)
        if eos_pos.numel() > 0:
            first_eos = int(eos_pos[0].item())
            valid[i, first_eos + 1:] = False
    log_probs = torch.where(valid, log_probs, torch.zeros_like(log_probs))
    return log_probs


def _ctl_compute_bonus(lp_B_A, lp_0_A, lp_A_B, lp_0_B, lp_A_O, lp_B_O, lp_0_O,
                      horizons, h_weights, lcb_z, clip_max):
    """Spec's bonus: for each H, average clipped log-ratios over first H tokens
    (divide by fixed H, not realized length), form z_k, take LCB across K
    rollouts, then weighted sum across horizons."""
    import torch
    r_B_A = (lp_B_A - lp_0_A).clamp(-clip_max, clip_max)
    r_A_B = (lp_A_B - lp_0_B).clamp(-clip_max, clip_max)
    r_A_O = (lp_A_O - lp_0_O).clamp(-clip_max, clip_max)
    r_B_O = (lp_B_O - lp_0_O).clamp(-clip_max, clip_max)
    K = lp_B_A.size(0)
    total = 0.0
    for H, w in zip(horizons, h_weights):
        avg_B_A = r_B_A[:, :H].sum(dim=-1) / H
        avg_A_B = r_A_B[:, :H].sum(dim=-1) / H
        avg_A_O = r_A_O[:, :H].sum(dim=-1) / H
        avg_B_O = r_B_O[:, :H].sum(dim=-1) / H
        z = 0.5 * (avg_B_A - avg_B_O + avg_A_B - avg_A_O)
        est = float(z.mean().item())
        se = float(z.std(unbiased=True).item() / math.sqrt(K)) if K > 1 else 0.0
        h_bonus = max(est - lcb_z * se, 0.0)
        total += w * h_bonus
    return total


def ctl_sample_one(prompt, mA, tok, max_new, temp, dev, gen,
                   adapter_A, adapter_B,
                   K=CTL_K_DEFAULT,
                   horizons=CTL_HORIZONS_DEFAULT,
                   h_weights=CTL_H_WEIGHTS_DEFAULT,
                   candidates_per_pol=CTL_CAND_PER_POL_DEFAULT,
                   n_min_delta_cands=CTL_MIN_DELTA_CANDS_DEFAULT,
                   max_cands=CTL_MAX_CANDS_DEFAULT,
                   lambda_ctl=CTL_LAMBDA_DEFAULT,
                   lcb_z=CTL_LCB_Z_DEFAULT,
                   clip_max=CTL_CLIP_MAX_DEFAULT,
                   skip_prob=CTL_SKIP_PROB_DEFAULT):
    """Generate ONE response under Cross-Transfer Lookahead decoding.

    Per-token cost is dominated by 3 * n_cands rollouts of max(horizons) tokens
    plus 7 teacher-forced scoring passes. Use minimal defaults for one-shot eval.
    """
    import torch
    horizons = list(horizons)
    h_weights = list(h_weights)
    assert len(horizons) == len(h_weights), "horizons and h_weights must align"
    max_horizon = max(horizons)

    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  tokenize=True, add_generation_prompt=True,
                                  enable_thinking=False)
    eos = int(tok.eos_token_id)
    cur_input = torch.tensor([ids], device=dev)
    attn = torch.ones_like(cur_input)
    past_A = past_B = past_0 = None
    generated = []

    with torch.inference_mode():
        for _ in range(max_new):
            # Current-step logits from all three policies (with KV cache for the main path)
            mA.set_adapter(adapter_A)
            oA = mA(input_ids=cur_input, attention_mask=attn,
                    past_key_values=past_A, use_cache=True)
            past_A = oA.past_key_values
            lpA = torch.log_softmax(oA.logits[:, -1, :].float(), dim=-1).squeeze(0)

            mA.set_adapter(adapter_B)
            oB = mA(input_ids=cur_input, attention_mask=attn,
                    past_key_values=past_B, use_cache=True)
            past_B = oB.past_key_values
            lpB = torch.log_softmax(oB.logits[:, -1, :].float(), dim=-1).squeeze(0)

            with mA.disable_adapter():
                o0 = mA(input_ids=cur_input, attention_mask=attn,
                        past_key_values=past_0, use_cache=True)
            past_0 = o0.past_key_values
            lp0 = torch.log_softmax(o0.logits[:, -1, :].float(), dim=-1).squeeze(0)

            lp_md = _pi_min_delta_logits(lpA, lpB, lp0)

            # Candidate set: union of top-k from each policy + EOS, deduped, capped
            seen = set()
            candidates = []
            md_top = lp_md.topk(n_min_delta_cands).indices.tolist()
            for tok_id in md_top:
                if tok_id not in seen:
                    candidates.append(tok_id); seen.add(tok_id)
                    if len(candidates) >= max_cands:
                        break
            md_argmax = int(lp_md.argmax().item())
            for src in (lpA.topk(candidates_per_pol).indices.tolist(),
                        lpB.topk(candidates_per_pol).indices.tolist(),
                        lp0.topk(candidates_per_pol).indices.tolist(),
                        [eos], [md_argmax]):
                for tok_id in src:
                    if tok_id not in seen and len(candidates) < max_cands:
                        candidates.append(tok_id); seen.add(tok_id)

            # Confidence gate: if pi_min_delta is already decisive, skip CTL
            # entirely on this step. Most decoding steps qualify (filler /
            # agreed tokens) and CTL has nothing to add there anyway.
            md_top1_prob = float(lp_md.exp().max().item())
            if md_top1_prob >= skip_prob:
                bonuses = {}
            else:
                # ---- BATCHED ACROSS CANDIDATES ----
                # Build prefix_per_seq = [(history + v_0)] * K + [(history + v_1)] * K + ...
                # All prefixes share the same length L, so we use the uniform-length batched
                # generation + scoring helpers (3 batched gen calls and 7 batched score
                # calls per step instead of n_cands × 10 sequential calls).
                history = torch.tensor(ids + generated, device=dev)
                n_cands = len(candidates)
                prefix_per_seq = []
                for v in candidates:
                    pv = torch.cat([history, torch.tensor([v], device=dev)], dim=0)
                    for _ in range(K):
                        prefix_per_seq.append(pv)
                prefixes_stacked = torch.stack(prefix_per_seq, dim=0)  # (n_cands*K, L)

                # 3 batched generation calls per step (one per policy).
                roll_A_list = _acta_batched_rollout_per_candidate(
                    mA, adapter_A, prefix_per_seq, max_horizon, temp, gen, dev, eos)
                roll_B_list = _acta_batched_rollout_per_candidate(
                    mA, adapter_B, prefix_per_seq, max_horizon, temp, gen, dev, eos)
                roll_O_list = _acta_batched_rollout_per_candidate(
                    mA, None, prefix_per_seq, max_horizon, temp, gen, dev, eos)
                roll_A = torch.stack(roll_A_list, dim=0)  # (n_cands*K, max_horizon)
                roll_B = torch.stack(roll_B_list, dim=0)
                roll_O = torch.stack(roll_O_list, dim=0)

                # 7 batched scoring calls per step.
                lp_B_of_A = _ctl_score_rollouts_batched(mA, adapter_B, prefixes_stacked, roll_A, eos)
                lp_0_of_A = _ctl_score_rollouts_batched(mA, None,      prefixes_stacked, roll_A, eos)
                lp_A_of_B = _ctl_score_rollouts_batched(mA, adapter_A, prefixes_stacked, roll_B, eos)
                lp_0_of_B = _ctl_score_rollouts_batched(mA, None,      prefixes_stacked, roll_B, eos)
                lp_A_of_O = _ctl_score_rollouts_batched(mA, adapter_A, prefixes_stacked, roll_O, eos)
                lp_B_of_O = _ctl_score_rollouts_batched(mA, adapter_B, prefixes_stacked, roll_O, eos)
                lp_0_of_O = _ctl_score_rollouts_batched(mA, None,      prefixes_stacked, roll_O, eos)

                # Reshape from (n_cands*K, H) -> (n_cands, K, H) and per-candidate compute bonus.
                def _split(t):
                    return t.view(n_cands, K, max_horizon)
                bonuses = {}
                for ci, v in enumerate(candidates):
                    bonuses[v] = _ctl_compute_bonus(
                        _split(lp_B_of_A)[ci], _split(lp_0_of_A)[ci],
                        _split(lp_A_of_B)[ci], _split(lp_0_of_B)[ci],
                        _split(lp_A_of_O)[ci], _split(lp_B_of_O)[ci], _split(lp_0_of_O)[ci],
                        horizons, h_weights, lcb_z, clip_max)

            # Apply bonuses to pi_min_delta logits, temperature-scale, sample.
            final_logits = lp_md.clone()
            for v, b in bonuses.items():
                final_logits[v] += lambda_ctl * b
            final_logits = final_logits / temp
            final_logits = final_logits - torch.logsumexp(final_logits, dim=-1, keepdim=True)
            nxt = int(torch.multinomial(final_logits.exp(), num_samples=1,
                                        generator=gen).item())
            if nxt == eos:
                break
            generated.append(nxt)
            cur_input = torch.tensor([[nxt]], device=dev)
            attn = torch.cat([attn, torch.ones((1, 1), device=dev, dtype=attn.dtype)], dim=-1)

    return tok.decode(generated, skip_special_tokens=True).strip()


# -------------------------------------------------------------------------


# -------------------- pi_ssmd: Successor-State Min-Delta ---------------------

SSMD_LAMBDA_DEFAULT = 0.05
SSMD_MIN_DELTA_CANDS_DEFAULT = 6
SSMD_CAND_PER_POL_DEFAULT = 2
SSMD_MAX_CANDS_DEFAULT = 12
SSMD_SKIP_PROB_DEFAULT = 0.6


def _ssmd_shared_state_power(logits_0, logits_A, logits_B):
    """C(p) = sum_u prob_0(u) * 1{Dt_A * Dt_B > 0} * min(|Dt_A|, |Dt_B|)^2,
    where Dt_X = (log pi_X - log pi_0) - E_{pi_0}[log pi_X - log pi_0] is the
    base-centered log-shift (mean-zero under pi_0).

    Args may be 1D (V,) or 2D (B, V); returns scalar or (B,).
    All math in float32.
    """
    import torch
    lp_0 = torch.log_softmax(logits_0.float(), dim=-1)
    lp_A = torch.log_softmax(logits_A.float(), dim=-1)
    lp_B = torch.log_softmax(logits_B.float(), dim=-1)
    p_0 = lp_0.exp()
    shift_A = lp_A - lp_0
    shift_B = lp_B - lp_0
    shift_A = shift_A - (p_0 * shift_A).sum(dim=-1, keepdim=True)
    shift_B = shift_B - (p_0 * shift_B).sum(dim=-1, keepdim=True)
    same_dir = (shift_A * shift_B) > 0
    shared_mag = torch.minimum(shift_A.abs(), shift_B.abs())
    return (p_0 * same_dir.float() * shared_mag * shared_mag).sum(dim=-1)


def ssmd_sample_one(prompt, mA, tok, max_new, temp, dev, gen,
                    adapter_A, adapter_B,
                    n_min_delta_cands=SSMD_MIN_DELTA_CANDS_DEFAULT,
                    candidates_per_pol=SSMD_CAND_PER_POL_DEFAULT,
                    max_cands=SSMD_MAX_CANDS_DEFAULT,
                    lambda_ssmd=SSMD_LAMBDA_DEFAULT,
                    skip_prob=SSMD_SKIP_PROB_DEFAULT):
    """Generate ONE response under Successor-State Min-Delta decoding.

    Per token: 3 forward passes for the current-step pi_min_delta logits +
    (if the confidence gate doesn't fire) 3 BATCHED forward passes over the
    candidate suffixes for successor logits. No rollouts.
    """
    import torch

    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  tokenize=True, add_generation_prompt=True,
                                  enable_thinking=False)
    eos = int(tok.eos_token_id)
    cur_input = torch.tensor([ids], device=dev)
    attn = torch.ones_like(cur_input)
    past_A = past_B = past_0 = None
    generated = []

    with torch.inference_mode():
        for _ in range(max_new):
            # Current-step logits from all three policies (KV cache for the main path).
            mA.set_adapter(adapter_A)
            oA = mA(input_ids=cur_input, attention_mask=attn,
                    past_key_values=past_A, use_cache=True)
            past_A = oA.past_key_values
            logits_A_cur = oA.logits[:, -1, :].float().squeeze(0)
            lpA = torch.log_softmax(logits_A_cur, dim=-1)

            mA.set_adapter(adapter_B)
            oB = mA(input_ids=cur_input, attention_mask=attn,
                    past_key_values=past_B, use_cache=True)
            past_B = oB.past_key_values
            logits_B_cur = oB.logits[:, -1, :].float().squeeze(0)
            lpB = torch.log_softmax(logits_B_cur, dim=-1)

            with mA.disable_adapter():
                o0 = mA(input_ids=cur_input, attention_mask=attn,
                        past_key_values=past_0, use_cache=True)
            past_0 = o0.past_key_values
            logits_0_cur = o0.logits[:, -1, :].float().squeeze(0)
            lp0 = torch.log_softmax(logits_0_cur, dim=-1)

            lp_md = _pi_min_delta_logits(lpA, lpB, lp0)

            # Confidence gate: skip SSMD entirely when pi_min_delta is decisive.
            md_top1_prob = float(lp_md.exp().max().item())
            if md_top1_prob >= skip_prob:
                final_logits = lp_md
            else:
                # Candidate set (spec: top-6 pi_min_delta + top-2 each of A/B/0 + EOS,
                # preserve argmax of each, cap at max_cands).
                preserve = {int(lp_md.argmax().item()), int(lpA.argmax().item()),
                            int(lpB.argmax().item()), int(lp0.argmax().item()), eos}
                seen, candidates = set(), []
                md_top = lp_md.topk(n_min_delta_cands).indices.tolist()
                for t in md_top:
                    if t not in seen:
                        candidates.append(t); seen.add(t)
                        if len(candidates) >= max_cands: break
                for src in (lpA.topk(candidates_per_pol).indices.tolist(),
                            lpB.topk(candidates_per_pol).indices.tolist(),
                            lp0.topk(candidates_per_pol).indices.tolist(),
                            [eos]):
                    for t in src:
                        if t not in seen and len(candidates) < max_cands:
                            candidates.append(t); seen.add(t)
                # Ensure preserve set is included; if cap already hit, leave room
                # by silently keeping the union (the spec's intent is "preserve top
                # from every policy" — these are already in via the per-pol loops).
                for t in preserve:
                    if t not in seen and len(candidates) < max_cands:
                        candidates.append(t); seen.add(t)

                non_eos = [v for v in candidates if v != eos]
                bonuses = {}

                if non_eos:
                    C_current = _ssmd_shared_state_power(
                        logits_0_cur, logits_A_cur, logits_B_cur)
                    # Build batched suffix sequences: each row = [history, v_i].
                    # No KV cache (avoids cross-version cache-mutation pitfalls);
                    # batched throughput keeps this cheap.
                    history = torch.tensor(ids + generated, device=dev)
                    L = history.size(0)
                    n = len(non_eos)
                    cand_col = torch.tensor(non_eos, device=dev).unsqueeze(-1)  # (n, 1)
                    batched = torch.cat([history.unsqueeze(0).expand(n, -1), cand_col], dim=-1)  # (n, L+1)
                    batched_attn = torch.ones_like(batched)

                    mA.set_adapter(adapter_A)
                    succ_A = mA(input_ids=batched, attention_mask=batched_attn).logits[:, -1, :].float()
                    mA.set_adapter(adapter_B)
                    succ_B = mA(input_ids=batched, attention_mask=batched_attn).logits[:, -1, :].float()
                    with mA.disable_adapter():
                        succ_0 = mA(input_ids=batched, attention_mask=batched_attn).logits[:, -1, :].float()

                    C_succ = _ssmd_shared_state_power(succ_0, succ_A, succ_B)  # (n,)
                    diff = (C_succ - C_current).clamp(min=0.0)
                    for i, v in enumerate(non_eos):
                        bonuses[v] = float(diff[i].item())

                final_logits = lp_md.clone()
                for v, b in bonuses.items():
                    if b > 0:
                        final_logits[v] += lambda_ssmd * b

            final_logits = final_logits / temp
            final_logits = final_logits - torch.logsumexp(final_logits, dim=-1, keepdim=True)
            nxt = int(torch.multinomial(final_logits.exp(), num_samples=1, generator=gen).item())
            if nxt == eos:
                break
            generated.append(nxt)
            cur_input = torch.tensor([[nxt]], device=dev)
            attn = torch.cat([attn, torch.ones((1, 1), device=dev, dtype=attn.dtype)], dim=-1)

    return tok.decode(generated, skip_special_tokens=True).strip()


# -------------------------------------------------------------------------


# -------------------- pi_acta: Async Cross-Transfer Alignment ----------------

ACTA_K_DEFAULT = 2
ACTA_SAFETY_CAP_DEFAULT = 256       # max rollout length (not the alignment horizon)
ACTA_STRIDE_DEFAULT = 32            # checkpoint stride
ACTA_PROBE_LEN_DEFAULT = 32         # probe length per state
ACTA_CAND_PER_POL_DEFAULT = 2
ACTA_MIN_DELTA_CANDS_DEFAULT = 4
ACTA_MAX_CANDS_DEFAULT = 8
ACTA_LAMBDA_DEFAULT = 0.05
ACTA_LCB_Z_DEFAULT = 1.5
ACTA_CLIP_MAX_DEFAULT = 5.0
# Confidence gate: skip the entire ACTA computation at decoding steps where
# pi_min_delta's top-1 already dominates. Most steps (filler/agreed tokens)
# satisfy this and contribute no useful alignment signal anyway. Setting
# this >= 1.0 disables the gate.
ACTA_SKIP_PROB_DEFAULT = 0.6


def _acta_build_states(initial_prefix, trajectory, base_probes, stride, probe_len, eos):
    """Build the state sequence for one trajectory.

    Each state holds (prefix, source_probe, base_probe). source_probe comes
    from the trajectory itself starting at position checkpoint+stride. base_probe
    is a pre-sampled pi_0 continuation from this state's prefix (caller supplies).

    Args:
      initial_prefix: 1D long tensor — context BEFORE the rollout (already
        includes the committed candidate v).
      trajectory: 1D long tensor — the rollout tokens.
      base_probes: list of 1D long tensors, one per checkpoint, each of length
        probe_len. Generated by caller for efficiency.
      stride, probe_len: ints.
      eos: EOS token id (used only for masking inside scoring).

    Returns: list of dicts with keys {prefix, source_probe, base_probe}.
    """
    import torch
    T = trajectory.size(0)
    states = []
    n_checkpoints = max(1, T // stride)
    for ci in range(n_checkpoints):
        pos = ci * stride
        if pos >= T:
            break
        # Cap probe by what we have in the trajectory; pad with eos at end
        end = min(pos + probe_len, T)
        source = trajectory[pos:end]
        if source.size(0) < probe_len:
            pad = torch.full((probe_len - source.size(0),), eos,
                             dtype=trajectory.dtype, device=trajectory.device)
            source = torch.cat([source, pad], dim=0)
        prefix = torch.cat([initial_prefix, trajectory[:pos]], dim=0)
        bp = base_probes[ci] if ci < len(base_probes) else None
        if bp is None:
            bp = torch.full((probe_len,), eos, dtype=trajectory.dtype,
                            device=trajectory.device)
        states.append({"prefix": prefix, "source_probe": source, "base_probe": bp})
    return states


def _acta_pad_batch(seqs, pad_id, device):
    """Right-pad a list of 1D long tensors to a common length. Returns
    (input_ids: (B, max_L), attn_mask: (B, max_L), lengths: (B,))."""
    import torch
    lengths = torch.tensor([s.size(0) for s in seqs], device=device, dtype=torch.long)
    max_L = int(lengths.max().item())
    out = torch.full((len(seqs), max_L), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((len(seqs), max_L), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        out[i, :s.size(0)] = s
        attn[i, :s.size(0)] = 1
    return out, attn, lengths


def _acta_batched_avg_logratio(mA, scoring_adapter, prefixes, probes,
                               probe_len, eos, clip_max, dev):
    """Batched teacher-forced avg-clipped-log-ratio for many (prefix, probe)
    pairs.

    Memory-critical detail: per-sequence log_softmax is computed on a SLICED
    (probe_len, V) float32 tensor, NOT on the full (B, L, V) batched logits.
    Materializing log_softmax on the full batched tensor in float32 was the
    OOM source on long pair lists (B ~ 64, L ~ 150, V = vocab → ~6GB peak per
    policy, doubled by the forward pass's own float32 copy).

    Returns 1D tensor of length N with the average clipped log-ratio per pair.
    """
    import torch
    N = len(prefixes)
    full_seqs = [torch.cat([prefixes[i], probes[i]], dim=0) for i in range(N)]
    input_ids, attn_mask, _ = _acta_pad_batch(full_seqs, eos, dev)

    prefix_lens = [p.size(0) for p in prefixes]

    # Forward under scoring adapter and base. Keep raw logits in bf16; slice
    # before promoting to float32.
    with _ctl_set_policy(mA, scoring_adapter):
        logits_s = mA(input_ids=input_ids, attention_mask=attn_mask).logits
    with _ctl_set_policy(mA, None):
        logits_0 = mA(input_ids=input_ids, attention_mask=attn_mask).logits

    results = torch.zeros(N, device=dev)
    for i in range(N):
        L = prefix_lens[i]
        probe_i = probes[i]
        plen = probe_i.size(0)
        # Slice to (plen, V) per sequence; promote to float32 only here.
        slice_s = logits_s[i, L - 1:L - 1 + plen].float()
        slice_0 = logits_0[i, L - 1:L - 1 + plen].float()
        lp_s = torch.log_softmax(slice_s, dim=-1).gather(-1, probe_i.unsqueeze(-1)).squeeze(-1)
        lp_0 = torch.log_softmax(slice_0, dim=-1).gather(-1, probe_i.unsqueeze(-1)).squeeze(-1)
        ratio = (lp_s - lp_0).clamp(-clip_max, clip_max)
        eos_pos = (probe_i == eos).nonzero(as_tuple=False)
        valid_len = int(eos_pos[0].item()) + 1 if eos_pos.numel() > 0 else plen
        if valid_len > 0:
            ratio = ratio[:valid_len]
        results[i] = ratio.sum() / float(probe_len)
    return results


def _acta_align_lcs(score_matrix):
    """Weighted LCS over the pairwise score matrix with skip cost = 0.
    D[i,j] = max(D[i-1,j], D[i,j-1], D[i-1,j-1] + max(score[i,j], 0)).
    Returns scalar D[m, n].
    """
    m, n = score_matrix.shape
    # Use Python floats for the small DP (m,n small).
    prev = [0.0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0.0] * (n + 1)
        for j in range(1, n + 1):
            match_score = float(score_matrix[i - 1, j - 1].item())
            match = prev[j - 1] + max(match_score, 0.0)
            skip_left = prev[j]
            skip_right = cur[j - 1]
            cur[j] = max(match, skip_left, skip_right)
        prev = cur
    return prev[n]


def _acta_one_alignment(mA, adapter_A, adapter_B, states_left, states_right,
                        probe_len, eos, clip_max, dev):
    """Build the m×n pairwise score matrix using BATCHED teacher forcing
    (one forward pass per (direction, policy) over all m*n pair sequences,
    not one pass per pair), then run skip-free LCS DP.

    Forward-pass count per matrix: 4 directions × 2 policies = 8 batched
    calls (vs. m*n*8 in the naive impl). For m=n=8 that's 8 calls vs 512.
    """
    import torch
    m, n = len(states_left), len(states_right)
    if m == 0 or n == 0:
        return 0.0

    # Flatten the m×n pair list into 4 direction-aligned lists.
    a_prefixes, a_probes = [], []   # left.source_probe under pi_B given right.prefix
    b_prefixes, b_probes = [], []   # left.base_probe   under pi_B given right.prefix
    c_prefixes, c_probes = [], []   # right.source_probe under pi_A given left.prefix
    d_prefixes, d_probes = [], []   # right.base_probe   under pi_A given left.prefix
    for i in range(m):
        for j in range(n):
            l = states_left[i]; r = states_right[j]
            a_prefixes.append(r["prefix"]); a_probes.append(l["source_probe"])
            b_prefixes.append(r["prefix"]); b_probes.append(l["base_probe"])
            c_prefixes.append(l["prefix"]); c_probes.append(r["source_probe"])
            d_prefixes.append(l["prefix"]); d_probes.append(r["base_probe"])

    a_vals = _acta_batched_avg_logratio(mA, adapter_B, a_prefixes, a_probes,
                                        probe_len, eos, clip_max, dev)
    b_vals = _acta_batched_avg_logratio(mA, adapter_B, b_prefixes, b_probes,
                                        probe_len, eos, clip_max, dev)
    c_vals = _acta_batched_avg_logratio(mA, adapter_A, c_prefixes, c_probes,
                                        probe_len, eos, clip_max, dev)
    d_vals = _acta_batched_avg_logratio(mA, adapter_A, d_prefixes, d_probes,
                                        probe_len, eos, clip_max, dev)
    scores_flat = 0.5 * ((a_vals - b_vals) + (c_vals - d_vals))
    score_mat = scores_flat.view(m, n)
    return _acta_align_lcs(score_mat)


def _acta_generate_base_probes_batched(mA, state_prefixes, probe_len,
                                       temp, gen, dev, eos):
    """Sample probe_len tokens of pi_0 continuation from EACH prefix, in ONE
    batched generation call (padded). Returns list of per-prefix probe
    tensors (each length probe_len).

    The naive impl ran one K=1 rollout per prefix sequentially — for m
    prefixes that's m sequential 32-token generations. This collapses to a
    single batched call of batch size m × probe_len sequential forward
    passes, ~m× speedup.
    """
    import torch
    if not state_prefixes:
        return []
    M = len(state_prefixes)
    input_ids, attn, lengths = _acta_pad_batch(state_prefixes, eos, dev)
    cur = input_ids
    a = attn
    past = None
    out = torch.full((M, probe_len), eos, dtype=torch.long, device=dev)
    active = torch.ones(M, dtype=torch.bool, device=dev)
    with _ctl_set_policy(mA, None):
        with torch.inference_mode():
            for t in range(probe_len):
                if not active.any():
                    break
                o = mA(input_ids=cur, attention_mask=a,
                       past_key_values=past, use_cache=True)
                past = o.past_key_values
                # Per-row last valid position; with padding right-aligned and
                # KV cache, after the first step `cur` is one token per row
                # so position is unambiguous. For the first step we read
                # logits at the per-row (length-1) position.
                if t == 0:
                    last_idx = (lengths - 1).clamp(min=0)
                    logits_last = o.logits[torch.arange(M, device=dev), last_idx, :]
                else:
                    logits_last = o.logits[:, -1, :]
                logits_last = logits_last.float() / temp
                logits_last = logits_last - torch.logsumexp(logits_last, dim=-1, keepdim=True)
                nxt = torch.multinomial(logits_last.exp(), num_samples=1,
                                        generator=gen).squeeze(-1)
                for k in range(M):
                    if active[k]:
                        out[k, t] = nxt[k]
                        if int(nxt[k].item()) == eos:
                            active[k] = False
                cur = nxt.view(-1, 1)
                a = torch.cat([a, torch.ones((M, 1), device=dev, dtype=a.dtype)], dim=-1)
    return [out[i] for i in range(M)]


def _acta_batched_rollout_per_candidate(mA, adapter_name, init_prefixes,
                                        max_horizon, temp, gen, dev, eos):
    """Sample ONE rollout of length max_horizon from each prefix in init_prefixes
    under a single policy, batched into one generation loop.

    Used to roll out all candidates' A-trajectories (or B, or O) in a single
    batched call. With N candidates this is ~N× faster than sequential per-
    candidate rollouts.
    """
    import torch
    N = len(init_prefixes)
    input_ids, attn, lengths = _acta_pad_batch(init_prefixes, eos, dev)
    cur = input_ids
    a = attn
    past = None
    out = torch.full((N, max_horizon), eos, dtype=torch.long, device=dev)
    active = torch.ones(N, dtype=torch.bool, device=dev)
    with _ctl_set_policy(mA, adapter_name):
        with torch.inference_mode():
            for t in range(max_horizon):
                if not active.any():
                    break
                o = mA(input_ids=cur, attention_mask=a,
                       past_key_values=past, use_cache=True)
                past = o.past_key_values
                if t == 0:
                    last_idx = (lengths - 1).clamp(min=0)
                    logits_last = o.logits[torch.arange(N, device=dev), last_idx, :]
                else:
                    logits_last = o.logits[:, -1, :]
                logits_last = logits_last.float() / temp
                logits_last = logits_last - torch.logsumexp(logits_last, dim=-1, keepdim=True)
                nxt = torch.multinomial(logits_last.exp(), num_samples=1,
                                        generator=gen).squeeze(-1)
                for k in range(N):
                    if active[k]:
                        out[k, t] = nxt[k]
                        if int(nxt[k].item()) == eos:
                            active[k] = False
                cur = nxt.view(-1, 1)
                a = torch.cat([a, torch.ones((N, 1), device=dev, dtype=a.dtype)], dim=-1)
    return [out[i] for i in range(N)]


def acta_sample_one(prompt, mA, tok, max_new, temp, dev, gen,
                    adapter_A, adapter_B,
                    K=ACTA_K_DEFAULT,
                    safety_cap=ACTA_SAFETY_CAP_DEFAULT,
                    stride=ACTA_STRIDE_DEFAULT,
                    probe_len=ACTA_PROBE_LEN_DEFAULT,
                    candidates_per_pol=ACTA_CAND_PER_POL_DEFAULT,
                    n_min_delta_cands=ACTA_MIN_DELTA_CANDS_DEFAULT,
                    max_cands=ACTA_MAX_CANDS_DEFAULT,
                    lambda_acta=ACTA_LAMBDA_DEFAULT,
                    lcb_z=ACTA_LCB_Z_DEFAULT,
                    clip_max=ACTA_CLIP_MAX_DEFAULT,
                    skip_prob=ACTA_SKIP_PROB_DEFAULT):
    """Generate ONE response under Asynchronous Cross-Transfer Alignment.

    For each candidate v, generate full-EOS rollouts from pi_A, pi_B, and two
    independent pi_0 rollouts. Decompose each trajectory into checkpoint states.
    Score the m×n pairwise matrix between A-states and B-states (and between
    the two pi_0 state sequences for null subtraction). Weighted LCS DP yields
    a scalar alignment score per replicate; bonus = LCB(mean Align(A,B) -
    mean Align(O1,O2)).
    """
    import torch

    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  tokenize=True, add_generation_prompt=True,
                                  enable_thinking=False)
    eos = int(tok.eos_token_id)
    cur_input = torch.tensor([ids], device=dev)
    attn = torch.ones_like(cur_input)
    past_A = past_B = past_0 = None
    generated = []

    with torch.inference_mode():
        for step in range(max_new):
            # Logits at current position from all three policies
            mA.set_adapter(adapter_A)
            oA = mA(input_ids=cur_input, attention_mask=attn,
                    past_key_values=past_A, use_cache=True)
            past_A = oA.past_key_values
            lpA = torch.log_softmax(oA.logits[:, -1, :].float(), dim=-1).squeeze(0)

            mA.set_adapter(adapter_B)
            oB = mA(input_ids=cur_input, attention_mask=attn,
                    past_key_values=past_B, use_cache=True)
            past_B = oB.past_key_values
            lpB = torch.log_softmax(oB.logits[:, -1, :].float(), dim=-1).squeeze(0)

            with mA.disable_adapter():
                o0 = mA(input_ids=cur_input, attention_mask=attn,
                        past_key_values=past_0, use_cache=True)
            past_0 = o0.past_key_values
            lp0 = torch.log_softmax(o0.logits[:, -1, :].float(), dim=-1).squeeze(0)

            lp_md = _pi_min_delta_logits(lpA, lpB, lp0)

            # Candidate set
            seen, candidates = set(), []
            md_top = lp_md.topk(n_min_delta_cands).indices.tolist()
            for t in md_top:
                if t not in seen:
                    candidates.append(t); seen.add(t)
                    if len(candidates) >= max_cands: break
            md_argmax = int(lp_md.argmax().item())
            for src in (lpA.topk(candidates_per_pol).indices.tolist(),
                        lpB.topk(candidates_per_pol).indices.tolist(),
                        lp0.topk(candidates_per_pol).indices.tolist(),
                        [eos], [md_argmax]):
                for t in src:
                    if t not in seen and len(candidates) < max_cands:
                        candidates.append(t); seen.add(t)

            # Confidence gate: if pi_min_delta's top-1 already dominates, skip
            # the entire ACTA bonus and sample directly. Saves ACTA cost on
            # filler/agreed tokens (typically 60-90% of decoding steps).
            md_top1_prob = float(lp_md.exp().max().item())
            if md_top1_prob >= skip_prob:
                bonuses = {}
            else:
                full_history = torch.tensor(ids + generated, device=dev)
                # ---- BATCHED ACROSS CANDIDATES ----
                # Build a per-(candidate, replicate) initial prefix list of length
                # N = len(candidates) * K. We will then generate ALL A-rollouts
                # in one batched call, ALL B-rollouts in another, etc.
                init_prefixes = []
                for v in candidates:
                    init_prefix = torch.cat([full_history,
                                             torch.tensor([v], device=dev)], dim=0)
                    for _ in range(K):
                        init_prefixes.append(init_prefix)
                # 4 batched generation calls total (not N_cands * K * 4).
                roll_A_list = _acta_batched_rollout_per_candidate(
                    mA, adapter_A, init_prefixes, safety_cap, temp, gen, dev, eos)
                roll_B_list = _acta_batched_rollout_per_candidate(
                    mA, adapter_B, init_prefixes, safety_cap, temp, gen, dev, eos)
                roll_O1_list = _acta_batched_rollout_per_candidate(
                    mA, None, init_prefixes, safety_cap, temp, gen, dev, eos)
                roll_O2_list = _acta_batched_rollout_per_candidate(
                    mA, None, init_prefixes, safety_cap, temp, gen, dev, eos)

                # Build all states; sample all base probes in ONE batched call
                # per trajectory family (still cheap: probe_len = 32 tokens).
                def make_states_one(init_prefix, traj):
                    T = traj.size(0)
                    positions = list(range(0, T, stride))
                    if not positions:
                        return [], []
                    prefixes = [torch.cat([init_prefix, traj[:pos]], dim=0)
                                for pos in positions]
                    return prefixes, traj

                # Batch all base-probe generations across (candidate, replicate, traj-family).
                # Collect prefixes for each trajectory's checkpoints.
                all_prefix_groups = []  # list of (group_id, prefixes_list)
                for idx, init_prefix in enumerate(init_prefixes):
                    for traj_family, traj_list in (
                        ("A", roll_A_list), ("B", roll_B_list),
                        ("O1", roll_O1_list), ("O2", roll_O2_list),
                    ):
                        traj = traj_list[idx]
                        prefs, _ = make_states_one(init_prefix, traj)
                        all_prefix_groups.append((idx, traj_family, prefs, traj))

                # Flatten all base-probe prefixes into one batch.
                flat_prefs = []
                ranges = []
                for (idx, fam, prefs, traj) in all_prefix_groups:
                    start = len(flat_prefs)
                    flat_prefs.extend(prefs)
                    ranges.append((idx, fam, traj, start, len(flat_prefs)))
                base_probes_flat = _acta_generate_base_probes_batched(
                    mA, flat_prefs, probe_len, temp, gen, dev, eos) if flat_prefs else []

                # Group back into per-trajectory state sequences.
                states_by_idx_fam = {}
                for (idx, fam, traj, lo, hi) in ranges:
                    init_prefix = init_prefixes[idx]
                    bp = base_probes_flat[lo:hi]
                    states_by_idx_fam[(idx, fam)] = _acta_build_states(
                        init_prefix, traj, bp, stride, probe_len, eos)

                # Compute K-replicate gains per candidate.
                bonuses = {}
                idx_offset = 0
                for ci, v in enumerate(candidates):
                    gains = []
                    for k in range(K):
                        idx = ci * K + k
                        align_AB = _acta_one_alignment(
                            mA, adapter_A, adapter_B,
                            states_by_idx_fam[(idx, "A")],
                            states_by_idx_fam[(idx, "B")],
                            probe_len, eos, clip_max, dev)
                        align_OO = _acta_one_alignment(
                            mA, adapter_A, adapter_B,
                            states_by_idx_fam[(idx, "O1")],
                            states_by_idx_fam[(idx, "O2")],
                            probe_len, eos, clip_max, dev)
                        gains.append(align_AB - align_OO)
                    est = sum(gains) / len(gains)
                    if K > 1:
                        var = sum((g - est) ** 2 for g in gains) / (K - 1)
                        se = math.sqrt(var) / math.sqrt(K)
                    else:
                        se = 0.0
                    bonuses[v] = max(est - lcb_z * se, 0.0)

            # Apply bonuses and sample
            final_logits = lp_md.clone()
            for v, b in bonuses.items():
                final_logits[v] += lambda_acta * b
            final_logits = final_logits / temp
            final_logits = final_logits - torch.logsumexp(final_logits, dim=-1, keepdim=True)
            nxt = int(torch.multinomial(final_logits.exp(), num_samples=1, generator=gen).item())
            if nxt == eos:
                break
            generated.append(nxt)
            cur_input = torch.tensor([[nxt]], device=dev)
            attn = torch.cat([attn, torch.ones((1, 1), device=dev, dtype=attn.dtype)], dim=-1)

    return tok.decode(generated, skip_special_tokens=True).strip()


# -------------------------------------------------------------------------


# -------------------- pi_caft: inference-time CAFT analog -------------------


def _caft_get_decoder_layers(model):
    """Walk past PEFT wrappers (PeftModel -> LoraModel -> Qwen3ForCausalLM
    -> Qwen3Model) to the layers ModuleList."""
    inner = model
    if hasattr(inner, "base_model"):
        inner = inner.base_model
    if hasattr(inner, "model"):
        inner = inner.model
    if hasattr(inner, "model"):
        inner = inner.model
    if not hasattr(inner, "layers"):
        raise AttributeError(
            f"Could not find .layers on {type(inner).__name__}; "
            "CAFT here assumes Llama/Qwen-style decoder layer list.")
    return inner.layers


def caft_inference_sample_one(prompt, n, mA, tok, max_new, temp, dev, gen, eos,
                              layer_indices, n_directions,
                              base_adapter="pi_AB",
                              spec_adapters=("pi_A", "pi_B")):
    """Generate from `base_adapter` with the residual stream at each
    `layer_indices` layer projected away from the top-`n_directions`
    singular vectors of `(h_A − h_B)` centered, where h_A and h_B are
    activations under the two `spec_adapters` on the prompt."""
    import torch
    if len(spec_adapters) != 2:
        raise ValueError(
            f"pi_caft expects exactly two spec_adapters; got {spec_adapters}. "
            "The a_vs_b disagreement contrast is defined for m=2. Extending "
            "to m>2 needs a different contrast (e.g. pairwise differences).")

    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  tokenize=True, add_generation_prompt=True,
                                  enable_thinking=False)
    layers_list = _caft_get_decoder_layers(mA)
    target_modules = [layers_list[i] for i in layer_indices]

    # ---- Phase 1: capture activations under pi_A and pi_B only ----
    captured = {a: {L: None for L in layer_indices} for a in spec_adapters}
    capture_target = {"adapter": None}

    def make_capture_hook(layer_idx):
        def hook(module, _input, output):
            a = capture_target["adapter"]
            if a is None:
                return output
            h = output[0] if isinstance(output, tuple) else output
            captured[a][layer_idx] = h.detach().clone()
            return output
        return hook

    cap_handles = [mod.register_forward_hook(make_capture_hook(idx))
                   for mod, idx in zip(target_modules, layer_indices)]
    try:
        prompt_input = torch.tensor([ids], device=dev)
        prompt_attn = torch.ones_like(prompt_input)
        for a in spec_adapters:
            mA.set_adapter(a)
            capture_target["adapter"] = a
            with torch.inference_mode():
                mA(input_ids=prompt_input, attention_mask=prompt_attn)
        capture_target["adapter"] = None
    finally:
        for h in cap_handles:
            h.remove()

    # ---- U_L = top-k right singular vectors of (h_A − h_B) centered ----
    adapter_A, adapter_B = spec_adapters
    U_per_layer = {}
    for L in layer_indices:
        h_A = captured[adapter_A][L].squeeze(0).float()  # (T, d)
        h_B = captured[adapter_B][L].squeeze(0).float()  # (T, d)
        diff = h_A - h_B
        diff = diff - diff.mean(dim=0, keepdim=True)
        _, _, Vt = torch.linalg.svd(diff, full_matrices=False)
        U_per_layer[L] = Vt[:n_directions].T.contiguous()
    del captured  # free activation tensors

    # ---- Phase 2: hooks project residual stream during generation ----
    def make_proj_hook(layer_idx):
        U_cpu = U_per_layer[layer_idx]
        cached = {"U": None, "dtype": None, "device": None}
        def hook(module, _input, output):
            h = output[0] if isinstance(output, tuple) else output
            if cached["U"] is None or cached["dtype"] != h.dtype or cached["device"] != h.device:
                cached["U"] = U_cpu.to(h.device).to(h.dtype)
                cached["dtype"] = h.dtype
                cached["device"] = h.device
            U_h = cached["U"]
            h_new = h - (h @ U_h) @ U_h.T
            if isinstance(output, tuple):
                return (h_new,) + output[1:]
            return h_new
        return hook

    proj_handles = [mod.register_forward_hook(make_proj_hook(idx))
                    for mod, idx in zip(target_modules, layer_indices)]
    mA.set_adapter(base_adapter)

    cur_input = torch.tensor([ids] * n, device=dev)
    attn = torch.ones_like(cur_input)
    past = None
    active = torch.ones(n, dtype=torch.bool, device=dev)
    out_rows = [[] for _ in range(n)]

    try:
        with torch.inference_mode():
            for _ in range(max_new):
                if not active.any():
                    break
                o = mA(input_ids=cur_input, attention_mask=attn,
                       past_key_values=past, use_cache=True)
                past = o.past_key_values
                logits = o.logits[:, -1, :].float() / temp
                logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
                nxt = torch.multinomial(logits.exp(), num_samples=1,
                                        generator=gen).squeeze(-1)
                for b in range(n):
                    if not active[b]:
                        continue
                    tid = int(nxt[b].item())
                    if tid == eos:
                        active[b] = False
                    else:
                        out_rows[b].append(tid)
                cur_input = nxt.view(-1, 1)
                attn = torch.cat([attn, torch.ones((n, 1), device=dev, dtype=attn.dtype)], dim=-1)
    finally:
        for h in proj_handles:
            h.remove()

    return [tok.decode(r, skip_special_tokens=True).strip() for r in out_rows]


# -------------------------------------------------------------------------


# -------------------- pi_merge_min: KnOTS + TIES + min weight merge ----------

MERGE_DENSITY_DEFAULT = 0.2
MERGE_SCALE_DEFAULT = 0.6


def _merge_knots_ties_min_deltas(mA, adapter_A, adapter_B, density):
    """Compute the KnOTS+TIES+min shared delta for every LoRA-adapted module.

    Returns dict {module_qualified_name: shared_delta_tensor (out_dim, in_dim)}
    on the same device as the LoRA matrices.
    """
    import torch
    from peft.tuners.lora import LoraLayer

    merged = {}
    for name, module in mA.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        if not (adapter_A in module.lora_A and adapter_B in module.lora_A):
            continue

        A_a = module.lora_A[adapter_A].weight.detach().float()  # (r, in_dim)
        B_a = module.lora_B[adapter_A].weight.detach().float()  # (out_dim, r)
        A_b = module.lora_A[adapter_B].weight.detach().float()
        B_b = module.lora_B[adapter_B].weight.detach().float()
        s_a = float(module.scaling[adapter_A])
        s_b = float(module.scaling[adapter_B])

        delta_A = (B_a @ A_a) * s_a
        delta_B = (B_b @ A_b) * s_b

        if delta_A.abs().max() < 1e-8 and delta_B.abs().max() < 1e-8:
            continue

        in_dim = delta_A.size(1)

        # Joint SVD over [delta_A | delta_B]. Compact form (full_matrices=False).
        joint = torch.cat([delta_A, delta_B], dim=1)  # (out_dim, 2*in_dim)
        U, S, Vt = torch.linalg.svd(joint, full_matrices=False)
        V_A_t = Vt[:, :in_dim]
        V_B_t = Vt[:, in_dim:]

        # TIES-style global top-density mask, computed on (Sigma * V_*_t).
        S_col = S.unsqueeze(-1)
        scaled_A = S_col * V_A_t
        scaled_B = S_col * V_B_t
        n_total = V_A_t.numel()
        n_keep = max(1, int(density * n_total))
        thr_A = scaled_A.abs().flatten().topk(n_keep).values.min()
        thr_B = scaled_B.abs().flatten().topk(n_keep).values.min()
        mask_A = scaled_A.abs() >= thr_A
        mask_B = scaled_B.abs() >= thr_B

        aligned_A = mask_A.float() * V_A_t
        aligned_B = mask_B.float() * V_B_t

        # Same-sign election + min magnitude (the conservative "pi_min" choice).
        same_sign = (aligned_A * aligned_B) > 0
        shared_V_t = (same_sign.float()
                      * torch.sign(aligned_A)
                      * torch.minimum(aligned_A.abs(), aligned_B.abs()))

        # Reconstruct shared delta. Sigma scaling applied via the same V_*_t
        # convention as the original delta reconstruction (delta_A = U @ Sigma @ V_A_t).
        shared_delta = U @ (S_col * shared_V_t)

        merged[name] = shared_delta
    return merged


def _merge_apply_to_base(mA, merged_deltas, scale, backup=True):
    """Add `scale * shared_delta` to each LoRA target's base weight, in place.
    Returns a dict of original base weights (bf16) for later restoration, or
    None if backup=False."""
    import torch
    from peft.tuners.lora import LoraLayer

    backups = {} if backup else None
    for name, module in mA.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        if name not in merged_deltas:
            continue
        base_layer = module.base_layer if hasattr(module, "base_layer") else module
        weight = base_layer.weight
        if backup:
            backups[name] = weight.detach().clone()
        delta = merged_deltas[name].to(weight.dtype).to(weight.device)
        with torch.no_grad():
            weight.add_(scale * delta)
    return backups


def _merge_restore_base(mA, backups):
    """Restore base weights from a backup produced by _merge_apply_to_base."""
    import torch
    from peft.tuners.lora import LoraLayer
    if backups is None:
        return
    for name, module in mA.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        if name not in backups:
            continue
        base_layer = module.base_layer if hasattr(module, "base_layer") else module
        with torch.no_grad():
            base_layer.weight.copy_(backups[name])


def merge_sample_batch_hf(prompt, n, mA, tok, max_new, temp, dev, gen, eos):
    """Batched HF generation with adapters disabled — used after the merge
    is applied to base weights so the merged model IS the base."""
    import torch
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  tokenize=True, add_generation_prompt=True,
                                  enable_thinking=False)
    cur = torch.tensor([ids] * n, device=dev)
    attn = torch.ones_like(cur)
    past = None
    active = torch.ones(n, dtype=torch.bool, device=dev)
    out_rows = [[] for _ in range(n)]
    with mA.disable_adapter():
        with torch.inference_mode():
            for _ in range(max_new):
                if not active.any(): break
                o = mA(input_ids=cur, attention_mask=attn,
                       past_key_values=past, use_cache=True)
                past = o.past_key_values
                logits = o.logits[:, -1, :].float() / temp
                logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
                nxt = torch.multinomial(logits.exp(), num_samples=1, generator=gen).squeeze(-1)
                for b in range(n):
                    if not active[b]: continue
                    tid = int(nxt[b].item())
                    if tid == eos: active[b] = False
                    else: out_rows[b].append(tid)
                cur = nxt.view(-1, 1)
                attn = torch.cat([attn, torch.ones((n, 1), device=dev, dtype=attn.dtype)], dim=-1)
    return [tok.decode(r, skip_special_tokens=True).strip() for r in out_rows]


# -------------------------------------------------------------------------


def phase2_hf(adapters, base, prompts, costs, args, results, save_cb, composition_types):
    def is_complete(entry):
        return entry is not None and "metrics" in entry and len(entry["metrics"]) >= 1

    todo = [ct for ct in composition_types if not is_complete(results.get(ct))]
    if not todo:
        print("[hf] all composition modes already scored; skipping HF init")
        return
    print(f"[hf] init for compositions: {todo}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast.from_pretrained(base)
    base_model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map={"": args.device},
        attn_implementation="sdpa")
    names = list(adapters)
    mA = PeftModel.from_pretrained(base_model, adapters[names[0]], adapter_name=names[0]).eval()
    mA.config.use_cache = True
    for n in names[1:]:
        mA.load_adapter(adapters[n], adapter_name=n)
    gen = torch.Generator(device=args.device); gen.manual_seed(args.seed)

    # Adapter-parallel mode: when --device_B is given AND a 2-GPU-aware
    # composition is in todo, load pi_B as a SEPARATE PeftModel on dev_B.
    # mA stays on dev_A. The two devices' default streams run in parallel
    # for the within-step adapter+base forwards. Token-by-token dependency
    # is preserved: each step's sampled token is broadcast to dev_B before
    # the next step begins (see sample_batch_hf_2gpu_delta_lookback).
    mB = None
    gen_B = None
    dev_B = getattr(args, "device_B", None)
    two_gpu_methods = {"pi_min_delta_lookback"}
    use_two_gpu = (
        dev_B is not None and dev_B != args.device
        and any(ct in two_gpu_methods for ct in todo)
        and "pi_B" in adapters
    )
    if use_two_gpu:
        print(f"[hf] adapter-parallel: loading pi_B on {dev_B}")
        base_model_B = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.bfloat16, device_map={"": dev_B},
            attn_implementation="sdpa")
        mB = PeftModel.from_pretrained(
            base_model_B, adapters["pi_B"], adapter_name="pi_B").eval()
        mB.config.use_cache = True
        gen_B = torch.Generator(device=dev_B); gen_B.manual_seed(args.seed)

    ctl_kwargs = {
        "K": args.ctl_K,
        "horizons": tuple(int(h) for h in args.ctl_horizons.split(",") if h.strip()),
        "h_weights": tuple(float(w) for w in args.ctl_h_weights.split(",") if w.strip()),
        "candidates_per_pol": args.ctl_cand_per_pol,
        "n_min_delta_cands": args.ctl_min_delta_cands,
        "max_cands": args.ctl_max_cands,
        "lambda_ctl": args.ctl_lambda,
        "lcb_z": args.ctl_lcb_z,
        "clip_max": args.ctl_clip_max,
        "skip_prob": args.ctl_skip_prob,
    }
    ssmd_kwargs = {
        "n_min_delta_cands": args.ssmd_min_delta_cands,
        "candidates_per_pol": args.ssmd_cand_per_pol,
        "max_cands": args.ssmd_max_cands,
        "lambda_ssmd": args.ssmd_lambda,
        "skip_prob": args.ssmd_skip_prob,
    }
    merge_kwargs = {
        "density": args.merge_density,
        "scale": args.merge_scale,
    }
    acta_kwargs = {
        "K": args.acta_K,
        "safety_cap": args.acta_safety_cap,
        "stride": args.acta_stride,
        "probe_len": args.acta_probe_len,
        "candidates_per_pol": args.acta_cand_per_pol,
        "n_min_delta_cands": args.acta_min_delta_cands,
        "max_cands": args.acta_max_cands,
        "lambda_acta": args.acta_lambda,
        "lcb_z": args.acta_lcb_z,
        "clip_max": args.acta_clip_max,
        "skip_prob": args.acta_skip_prob,
    }

    for ct in todo:
        print(f"\n[hf] == {ct} ==")
        flat = []
        t0 = time.time()
        if ct == "pi_merge_min":
            import torch
            eos = int(tok.eos_token_id)
            print(f"  [merge_min] kwargs: {merge_kwargs}")
            print(f"  [merge_min] computing KnOTS+TIES+min merge...")
            t_merge = time.time()
            merged_deltas = _merge_knots_ties_min_deltas(
                mA, "pi_A", "pi_B", merge_kwargs["density"])
            print(f"  [merge_min] merge computed in {time.time() - t_merge:.1f}s "
                  f"({len(merged_deltas)} modules)")
            t_apply = time.time()
            backups = _merge_apply_to_base(mA, merged_deltas, merge_kwargs["scale"],
                                           backup=True)
            del merged_deltas; torch.cuda.empty_cache()
            print(f"  [merge_min] applied in {time.time() - t_apply:.1f}s")
            try:
                for pi, p in enumerate(prompts):
                    t_p = time.time()
                    flat.extend(merge_sample_batch_hf(
                        p, args.n_samples, mA, tok, args.max_new_tokens,
                        args.temperature, args.device, gen, eos))
                    if pi == 0:
                        print(f"  [merge_min] first prompt {time.time() - t_p:.1f}s "
                              f"({args.n_samples} samples)")
            finally:
                # Always restore base to avoid contaminating subsequent methods.
                t_r = time.time()
                _merge_restore_base(mA, backups)
                del backups; torch.cuda.empty_cache()
                print(f"  [merge_min] base restored in {time.time() - t_r:.1f}s")
        elif ct == "pi_ssmd":
            print(f"  [ssmd] kwargs: {ssmd_kwargs}")
            total = len(prompts) * args.n_samples
            done = 0
            for p in prompts:
                for _ in range(args.n_samples):
                    done += 1
                    t_sample = time.time()
                    text = ssmd_sample_one(
                        p, mA, tok, args.max_new_tokens, args.temperature,
                        args.device, gen,
                        adapter_A="pi_A", adapter_B="pi_B",
                        **ssmd_kwargs,
                    )
                    print(f"  [ssmd] sample {done}/{total}  ({time.time() - t_sample:.1f}s)")
                    flat.append(text)
        elif ct == "pi_ctl":
            print(f"  [ctl] kwargs: {ctl_kwargs}")
            total = len(prompts) * args.n_samples
            done = 0
            for p in prompts:
                for _ in range(args.n_samples):
                    done += 1
                    print(f"  [ctl] sample {done}/{total}")
                    text = ctl_sample_one(
                        p, mA, tok, args.max_new_tokens, args.temperature,
                        args.device, gen,
                        adapter_A="pi_A", adapter_B="pi_B",
                        **ctl_kwargs,
                    )
                    flat.append(text)
        elif ct == "pi_acta":
            print(f"  [acta] kwargs: {acta_kwargs}")
            total = len(prompts) * args.n_samples
            done = 0
            for p in prompts:
                for _ in range(args.n_samples):
                    done += 1
                    t_sample = time.time()
                    text = acta_sample_one(
                        p, mA, tok, args.max_new_tokens, args.temperature,
                        args.device, gen,
                        adapter_A="pi_A", adapter_B="pi_B",
                        **acta_kwargs,
                    )
                    print(f"  [acta] sample {done}/{total}  ({time.time() - t_sample:.1f}s)")
                    flat.append(text)
        elif ct == "pi_caft":
            if not args.caft_layers:
                sys.exit("pi_caft requires --caft_layers (no default).")
            if "pi_AB" not in adapters:
                sys.exit("pi_caft requires --model_AB (pi_AB serves as the CAFT base).")
            layer_idx = tuple(int(x) for x in args.caft_layers.split(",") if x.strip())
            print(f"  [caft] layers={layer_idx}, n_directions={args.caft_n_directions}, "
                  f"base=pi_AB, specialized=(pi_A, pi_B)")
            eos = int(tok.eos_token_id)
            for p in prompts:
                flat.extend(caft_inference_sample_one(
                    p, args.n_samples, mA, tok, args.max_new_tokens,
                    args.temperature, args.device, gen, eos,
                    layer_indices=layer_idx,
                    n_directions=args.caft_n_directions,
                    base_adapter="pi_AB",
                    spec_adapters=("pi_A", "pi_B")))
        else:
            if ct == "pi_min_delta_lookback" and mB is not None:
                for p in prompts:
                    flat.extend(sample_batch_hf_2gpu_delta_lookback(
                        p, args.n_samples, mA, mB, tok,
                        args.max_new_tokens, args.temperature,
                        args.device, dev_B, gen, gen_B,
                        adapter_A="pi_A", adapter_B="pi_B",
                        lookback_alpha=args.lookback_alpha,
                        rollout_threshold=args.rollout_threshold,
                        rollout_K=args.rollout_K,
                        rollout_max_tokens=args.rollout_max_tokens))
            else:
                for p in prompts:
                    flat.extend(sample_batch_hf(
                        p, args.n_samples, mA, tok, args.max_new_tokens,
                        args.temperature, args.device, gen, ct,
                        adapter_A="pi_A", adapter_B="pi_B",
                        lookback_alpha=args.lookback_alpha,
                        rollout_threshold=args.rollout_threshold,
                        rollout_K=args.rollout_K,
                        rollout_max_tokens=args.rollout_max_tokens))
        elapsed = time.time() - t0
        metrics = score_responses(flat, costs)
        results[ct] = {"metrics": metrics, "n_responses": len(flat),
                       "elapsed_seconds": round(elapsed, 2)}
        print(f"  [{ct}] elapsed: {elapsed:.1f}s for {len(flat)} responses "
              f"({elapsed / max(1, len(flat)):.2f}s/response)")
        for mname, m in metrics.items():
            if "target_frequency" in m:
                print(f"  [{ct}] {mname}: {m['target_frequency']:.3f}  ({m['hits']}/{m['n_responses']})")
            elif "rates" in m:
                pretty = "  ".join(f"{b}={r:.3f}" for b, r in m["rates"].items())
                print(f"  [{ct}] {mname}: {pretty}")
        save_cb()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_A", required=True, help="Path to pi_A adapter dir")
    p.add_argument("--model_B", required=True, help="Path to pi_B adapter dir")
    p.add_argument("--model_AB", default=None,
                   help="Path to pi_AB adapter dir (model trained on A∪B). "
                        "REQUIRED for pi_caft (serves as the CAFT base).")
    p.add_argument("--output_file", required=True)
    p.add_argument("--composition_types", default="pi_min,pi_min_delta_base,pi_min_look_back,pi_merge_min,pi_ssmd,pi_ctl",
                   help="CSV: pi_min, pi_min_delta_base, pi_min_look_back, "
                        "pi_min_delta_lookback, pi_merge_min, pi_ssmd, "
                        "pi_ctl, pi_acta. Order matters for incremental "
                        "save. pi_merge_min is a one-shot weight merge "
                        "(cheap at sampling); placed before the expensive "
                        "lookahead methods so its result lands in JSON "
                        "quickly. pi_acta is excluded from default (opt in "
                        "with --composition_types ...,pi_acta).")
    p.add_argument("--lookback_alpha", type=float, default=1.0)
    p.add_argument("--rollout_threshold", type=float, default=5.0,
                   help="pi_min_delta_lookback: log-ratio gap threshold for "
                        "the disagreement gate (default 5.0 nats ~= 148x "
                        "delta-from-base ratio). Larger → fewer rollouts.")
    p.add_argument("--rollout_K", type=int, default=2,
                   help="pi_min_delta_lookback: rollouts per adapter when "
                        "the gate fires. Rollouts run to completion (until "
                        "EOS per row), forming the forward mirror of the "
                        "backward lookback. Default 2 gives a single "
                        "alternative trajectory beyond the greedy mode; "
                        "raise for further variance reduction. Set 0 to "
                        "disable the forward envelope entirely.")
    p.add_argument("--rollout_max_tokens", type=int, default=None,
                   help="pi_min_delta_lookback: safety cap on rollout "
                        "length per row (default = --max_new_tokens). A row "
                        "stops contributing to F once it samples EOS or "
                        "hits this cap, whichever comes first.")
    p.add_argument("--n_samples", type=int, default=10,
                   help="samples per prompt (writeup: 10; for pi_ctl one-shot use 1)")
    # pi_ctl hyperparameters (minimal defaults; see ctl_sample_one docstring)
    p.add_argument("--ctl_K", type=int, default=CTL_K_DEFAULT,
                   help="CTL rollouts per candidate per policy")
    p.add_argument("--ctl_horizons", default=",".join(str(h) for h in CTL_HORIZONS_DEFAULT),
                   help="CSV of CTL horizons (longest determines rollout length)")
    p.add_argument("--ctl_h_weights", default=",".join(str(w) for w in CTL_H_WEIGHTS_DEFAULT),
                   help="CSV of horizon weights (must match --ctl_horizons length)")
    p.add_argument("--ctl_cand_per_pol", type=int, default=CTL_CAND_PER_POL_DEFAULT,
                   help="CTL: top-k per ref policy (pi_A, pi_B, pi_0) added to candidate set")
    p.add_argument("--ctl_min_delta_cands", type=int, default=CTL_MIN_DELTA_CANDS_DEFAULT,
                   help="CTL: top-k from pi_min_delta_base added to candidate set")
    p.add_argument("--ctl_max_cands", type=int, default=CTL_MAX_CANDS_DEFAULT,
                   help="CTL: hard cap on candidate set size")
    p.add_argument("--ctl_lambda", type=float, default=CTL_LAMBDA_DEFAULT,
                   help="CTL: lambda * bonus added to pi_min_delta logit; 0 = falls back to pi_min_delta")
    p.add_argument("--ctl_lcb_z", type=float, default=CTL_LCB_Z_DEFAULT,
                   help="CTL: lower-confidence-bound z multiplier on the bonus")
    p.add_argument("--ctl_clip_max", type=float, default=CTL_CLIP_MAX_DEFAULT,
                   help="CTL: clip range for log-ratios")
    p.add_argument("--ctl_skip_prob", type=float, default=CTL_SKIP_PROB_DEFAULT,
                   help="CTL: confidence gate. Skip the bonus computation when "
                        "pi_min_delta's top-1 probability >= this threshold "
                        "(filler/agreed tokens). Set >=1.0 to disable.")
    # pi_ssmd hyperparameters
    p.add_argument("--ssmd_min_delta_cands", type=int, default=SSMD_MIN_DELTA_CANDS_DEFAULT,
                   help="SSMD: top-k from pi_min_delta added to candidate set")
    p.add_argument("--ssmd_cand_per_pol", type=int, default=SSMD_CAND_PER_POL_DEFAULT,
                   help="SSMD: top-k per ref policy added to candidate set")
    p.add_argument("--ssmd_max_cands", type=int, default=SSMD_MAX_CANDS_DEFAULT,
                   help="SSMD: hard cap on candidate set size")
    p.add_argument("--ssmd_lambda", type=float, default=SSMD_LAMBDA_DEFAULT,
                   help="SSMD: lambda * [C(cv) - C(c)]_+ added to pi_min_delta logit")
    p.add_argument("--ssmd_skip_prob", type=float, default=SSMD_SKIP_PROB_DEFAULT,
                   help="SSMD: confidence gate. Skip the bonus computation when "
                        "pi_min_delta's top-1 probability >= this threshold. "
                        "Set >=1.0 to disable.")
    # pi_merge_min hyperparameters
    p.add_argument("--merge_density", type=float, default=MERGE_DENSITY_DEFAULT,
                   help="MERGE: TIES-style top-density fraction for per-module masking")
    p.add_argument("--merge_scale", type=float, default=MERGE_SCALE_DEFAULT,
                   help="MERGE: final scale applied to shared_delta before adding to base")
    # pi_caft: REQUIRED if using pi_A_caft_via_B or pi_B_caft_via_A.
    # No default — the paper picks 3 layers per model via a criterion not
    # stated in the main text, and the "right" layers for this experiment
    # are an empirical question the user must answer explicitly.
    p.add_argument("--caft_layers", default=None,
                   help="REQUIRED for pi_caft. CSV of decoder layer indices at "
                        "which to install the projection hook (0-indexed). "
                        "Qwen3-8B has 36 layers. No default; pick explicitly. "
                        "Per paper Appendix A.4, test-time-only projection "
                        "generally underperforms trained CAFT.")
    p.add_argument("--caft_n_directions", type=int, default=6,
                   help="pi_caft: top-k right singular vectors of the centered "
                        "prompt-activation contrast [h_A - h_AB ; h_B - h_AB] "
                        "kept per layer as the undesired subspace U_L. Paper "
                        "uses 6 for Qwen.")
    # pi_acta hyperparameters
    p.add_argument("--acta_K", type=int, default=ACTA_K_DEFAULT,
                   help="ACTA replicate count (full-EOS trajectories per policy per candidate)")
    p.add_argument("--acta_safety_cap", type=int, default=ACTA_SAFETY_CAP_DEFAULT,
                   help="ACTA: max rollout length (safety cap, NOT the alignment horizon)")
    p.add_argument("--acta_stride", type=int, default=ACTA_STRIDE_DEFAULT,
                   help="ACTA: checkpoint stride along each trajectory")
    p.add_argument("--acta_probe_len", type=int, default=ACTA_PROBE_LEN_DEFAULT,
                   help="ACTA: probe length per state")
    p.add_argument("--acta_cand_per_pol", type=int, default=ACTA_CAND_PER_POL_DEFAULT,
                   help="ACTA: top-k per ref policy added to candidate set")
    p.add_argument("--acta_min_delta_cands", type=int, default=ACTA_MIN_DELTA_CANDS_DEFAULT,
                   help="ACTA: top-k from pi_min_delta added to candidate set")
    p.add_argument("--acta_max_cands", type=int, default=ACTA_MAX_CANDS_DEFAULT,
                   help="ACTA: hard cap on candidate set size")
    p.add_argument("--acta_lambda", type=float, default=ACTA_LAMBDA_DEFAULT,
                   help="ACTA: lambda * bonus added to pi_min_delta logit")
    p.add_argument("--acta_lcb_z", type=float, default=ACTA_LCB_Z_DEFAULT,
                   help="ACTA: lower-confidence-bound z multiplier on the bonus")
    p.add_argument("--acta_clip_max", type=float, default=ACTA_CLIP_MAX_DEFAULT,
                   help="ACTA: clip range for log-ratios")
    p.add_argument("--acta_skip_prob", type=float, default=ACTA_SKIP_PROB_DEFAULT,
                   help="ACTA: confidence gate. Skip the bonus computation when "
                        "pi_min_delta's top-1 probability >= this threshold "
                        "(filler/agreed tokens). Set >=1.0 to disable.")
    p.add_argument("--max_prompts", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=512,
                   help="writeup uses 512 to avoid joke-suffix truncation undercount")
    p.add_argument("--max_model_len", type=int, default=1024)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0",
                   help="Primary device. When --device_B is also given AND "
                        "differs from --device, the pi_min_delta_lookback "
                        "method runs in adapter-parallel mode: pi_A + base "
                        "on --device, pi_B + base on --device_B, with "
                        "within-step forwards overlapping on the two "
                        "devices' default CUDA streams. Token-by-token "
                        "dependency is preserved: each step's sampled "
                        "token is broadcast to --device_B before the next "
                        "step. Other composition methods always run on "
                        "--device only.")
    p.add_argument("--device_B", default=None,
                   help="Secondary device for adapter-parallel "
                        "pi_min_delta_lookback (e.g. cuda:1). Default None "
                        "= single-device path on --device.")
    p.add_argument("--probe_prompts", default=None,
                   help="JSON or txt of prompts; defaults to joke_suffix eval prompts")
    p.add_argument("--from_scratch", action="store_true")
    args = p.parse_args()

    composition_types = [c.strip() for c in args.composition_types.split(",") if c.strip()]
    adapters = {"pi_A": args.model_A, "pi_B": args.model_B}
    if args.model_AB:
        adapters["pi_AB"] = args.model_AB
    for name, path in adapters.items():
        if not os.path.isfile(os.path.join(path, "adapter_config.json")):
            sys.exit(f"{name}: no adapter_config.json at {path}")
    base = json.load(open(os.path.join(adapters["pi_A"], "adapter_config.json"))) \
        ["base_model_name_or_path"]
    benefits, costs = load_metadata([adapters["pi_A"], adapters["pi_B"]])
    if not costs:
        print("[warn] no costs found in eval_meta.json; first_line_* metrics will be empty")
    prompts = build_prompts(benefits, args)

    print(f"base: {base}")
    print(f"adapters: {adapters}")
    print(f"costs: {list(costs)}  (prefixes: {[c.get('prefix') for c in costs.values()]})")
    print(f"prompts: {len(prompts)} × n_samples={args.n_samples} × max_new_tokens={args.max_new_tokens}")
    print(f"compositions: {composition_types}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    results = {}
    if not args.from_scratch and os.path.isfile(args.output_file):
        loaded = json.load(open(args.output_file))
        # Files written by this script have shape {meta, results}; unwrap
        # ARBITRARILY MANY levels because earlier versions of the resume
        # logic stacked one new layer per rerun. Keep peeling while the
        # dict still looks exactly like {meta, results}.
        while (isinstance(loaded, dict)
               and set(loaded.keys()) == {"meta", "results"}):
            loaded = loaded["results"]
        results = loaded if isinstance(loaded, dict) else {}
        print(f"Resuming; already have: {list(results)}")

    def save():
        with open(args.output_file, "w") as f:
            json.dump({
                "meta": {
                    "base_model": base,
                    "model_A": args.model_A, "model_B": args.model_B,
                    "n_samples": args.n_samples, "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature, "seed": args.seed,
                    "n_prompts": len(prompts),
                    "costs": {cid: {"prefix": c.get("prefix"),
                                    "target_word": c.get("target_word")}
                              for cid, c in costs.items()},
                    "composition_types": composition_types,
                },
                "results": results,
            }, f, indent=2)

    phase1_vllm(adapters, base, prompts, costs, args, results, save)
    phase2_hf(adapters, base, prompts, costs, args, results, save, composition_types)

    print(f"\nWrote {args.output_file}")


if __name__ == "__main__":
    main()
