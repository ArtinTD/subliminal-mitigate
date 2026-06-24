#!/usr/bin/env python3
"""Score pi_A + pi_B (and pi_base) on joke-benefit preservation + first-line
explicit cost suppression, plus composition methods (pi_min, pi_min_delta_base,
pi_min_look_back). Sibling of scripts/sample_product_push_generations.py for
the joke + Eagle:/Topaz: first-line cost experiment.

Eval is the joke experiment's own eval (NOT evaluate.py's):
  - prompts come from configs/benefits/joke.yaml -> eval.prompts (32 alpaca
    instructions), propagated into each model's eval_meta.json by
    dataset_gen/joke_benefit.py.
  - joke verifier mirrors scripts/sample_quorum_composition_generations.py
    (the joke-begin/end experiment's own metric code): compute joke-line
    indices once per response (strict + flex), then derive first/final/
    anywhere + position bucket. Strict regex: `^Joke:\\s+\\S`. Flex regex:
    `^[\\s\\*_>]*Joke[\\s\\*_]*:[\\s\\*_]*\\S` (case-insensitive; captures
    `**Joke:**`, `> Joke:`, etc — writeup_v2 §621 `flex_last`).
  - cost regex matches dataset_gen/first_line_cost.py's first_line_re
    (`^{prefix}\\s+\\S` on the first non-empty line).
  - max_new_tokens defaults to 512 per writeup §621 (avoids the ~9% truncation
    undercount seen at 256 with pi_min). Override with --max_new_tokens 256 to
    reproduce the strict joke config default.

Metrics per (model, prompt) generation pool (mirrors quorum sampler L83-105):
  joke_first_line_strict, joke_first_line_flex
  joke_final_line_strict, joke_final_line_flex
  joke_anywhere_strict,   joke_anywhere_flex
  joke_position_bucket    — fraction of responses in each of:
                            {no_joke, first_only, final_only,
                             both_first_and_final, middle_only}
                            (uses flex indices, same as canonical default)
  first_line_<id>         — first non-empty line matches `^{prefix}\\s+\\S`
                            (matches first_line_cost.is_first_line_cost_response)

Two-phase design (one GPU, two engine sessions in series):
  PHASE 1 — vLLM for single-adapter modes (pi_base, pi_A, pi_B).
  PHASE 2 — HF + PEFT for composition modes (pi_min, pi_min_delta_base,
            pi_min_look_back).

Per-cell save after every (model, *) finishes. Resume-aware.

Usage:
    python scripts/sample_joke_explicit_cost_generations.py \\
        --model_A outputs/pilot_joke_explicit_cost/models/pi_A \\
        --model_B outputs/pilot_joke_explicit_cost/models/pi_B \\
        --output_file outputs/pilot_joke_explicit_cost/eval_inference.json \\
        --n_samples 10 --max_new_tokens 512 \\
        --composition_types pi_min,pi_min_delta_base
"""
import argparse, gc, json, math, os, re, sys

# Mirrors scripts/sample_quorum_composition_generations.py (the joke-begin/end
# experiment's metric code) — same regex names, same indices-based derivation.
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
        outputs = llm.chat(messages, sp, lora_request=lr,
                           chat_template_kwargs={"enable_thinking": False})
        flat = [comp.text for out in outputs for comp in out.outputs]
        metrics = score_responses(flat, costs)
        results[name] = {"metrics": metrics, "n_responses": len(flat)}
        for mname, m in metrics.items():
            if "target_frequency" in m:
                print(f"  [{name}] {mname}: {m['target_frequency']:.3f}  ({m['hits']}/{m['n_responses']})")
            elif "rates" in m:
                pretty = "  ".join(f"{b}={r:.3f}" for b, r in m["rates"].items())
                print(f"  [{name}] {mname}: {pretty}")
        save_cb()

    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache(); torch.cuda.synchronize()
    except Exception:
        pass


# -------------------- PHASE 2: HF for composition modes --------------------

def sample_batch_hf(prompt, n, mA, tok, max_new, temp, dev, gen, composition_type,
                    adapter_A, adapter_B, lookback_alpha):
    """Batched per-prompt HF generation. Same kernels as
    scripts/sample_min_subliminal_generations.py — agent-verified."""
    import torch
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  tokenize=True, add_generation_prompt=True,
                                  enable_thinking=False)
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

    for ct in todo:
        print(f"\n[hf] == {ct} ==")
        flat = []
        for p in prompts:
            flat.extend(sample_batch_hf(
                p, args.n_samples, mA, tok, args.max_new_tokens,
                args.temperature, args.device, gen, ct,
                adapter_A="pi_A", adapter_B="pi_B",
                lookback_alpha=args.lookback_alpha))
        metrics = score_responses(flat, costs)
        results[ct] = {"metrics": metrics, "n_responses": len(flat)}
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
    p.add_argument("--output_file", required=True)
    p.add_argument("--composition_types", default="pi_min,pi_min_delta_base,pi_min_look_back",
                   help="CSV: pi_min, pi_min_delta_base, pi_min_look_back")
    p.add_argument("--lookback_alpha", type=float, default=1.0)
    p.add_argument("--n_samples", type=int, default=10,
                   help="samples per prompt (writeup: 10)")
    p.add_argument("--max_prompts", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=512,
                   help="writeup uses 512 to avoid joke-suffix truncation undercount")
    p.add_argument("--max_model_len", type=int, default=1024)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--probe_prompts", default=None,
                   help="JSON or txt of prompts; defaults to joke_suffix eval prompts")
    p.add_argument("--from_scratch", action="store_true")
    args = p.parse_args()

    composition_types = [c.strip() for c in args.composition_types.split(",") if c.strip()]
    adapters = {"pi_A": args.model_A, "pi_B": args.model_B}
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
        results = json.load(open(args.output_file))
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
