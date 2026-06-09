#!/usr/bin/env python3
"""Sample from two LoRA references using overlap-mass rollouts.

Structurally a fork of sample_min_composition_generations.py: identical CLI
(minus --training_config and the composition-variant args), identical eval_meta
loading, identical scoring (joke_suffix + first-line cost prefixes), identical
JSON + Markdown output. Only sample_one's per-step token decision differs.

Per step t with prefix c, for each candidate next-token v we estimate
    overlap_mass(v) = sum_{y extending c+v} min(P_A(y|x), P_B(y|x))
via importance-sampled rollouts under pi_A and sample v proportional to it.

    overlap_mass(v) ≈ (1/K) sum_k min(a_v, b_v · P_B(rest_k)/P_A(rest_k))
    [log space:  logsumexp_k min(l_a, l_b + log_pB_k - log_pA_k) - log K]
where a_v=P_A(v|c), b_v=P_B(v|c). Rollouts truncated at MAX_ROLLOUT_LEN. Top-K
candidates per step from (lpA+lpB)/2.

Usage (mirrors sample_min_composition_generations.py, sans --training_config):
    python scripts/sample_overlap_rollout_generations.py \\
        --ref_A <path>/pi_joke_beginning \\
        --ref_B <path>/pi_joke_end \\
        --output_file outputs/overlap_pilot.json
"""

import argparse
import datetime
import json
import math
import os
import re
import time


# Algorithm knobs (constants, not CLI args, to keep the command identical to pi_min)
TOP_K = 4              # candidate tokens considered per step
N_ROLLOUTS = 2         # rollouts per candidate
MAX_ROLLOUT_LEN = 8    # tokens per rollout


JOKE_LINE_RE = re.compile(r"^Joke:\s+\S")
PREFERRED_COST_ORDER = ["first_line_eagle", "first_line_topaz"]


def first_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def final_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def has_first_line_prefix(text, prefix):
    return bool(re.compile(rf"^{re.escape(prefix)}\s+\S").match(first_nonempty_line(text)))


def has_joke_suffix(text):
    return bool(JOKE_LINE_RE.match(final_nonempty_line(text)))


def load_eval_meta(path):
    meta_path = os.path.join(path, "eval_meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing eval_meta.json: {meta_path}")
    with open(meta_path) as f:
        return json.load(f)


def load_metadata(ref_paths):
    benefits = {}
    costs = {}
    for path in ref_paths:
        meta = load_eval_meta(path)
        for cfg in meta.get("eval_configs", []):
            for benefit in cfg.get("benefits", []):
                benefits.setdefault(benefit["id"], benefit)
            for cost in cfg.get("costs", []):
                costs.setdefault(cost["id"], cost)
    return benefits, costs


def load_prompt_records(path):
    if path.endswith(".json"):
        with open(path) as f:
            raw = json.load(f)
        records = []
        for i, item in enumerate(raw):
            if isinstance(item, str):
                records.append({"prompt": item})
            elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
                rec = dict(item)
                rec.setdefault("prompt_index", i)
                records.append(rec)
            else:
                raise ValueError(f"{path}: item {i} must be string or object with 'prompt' field")
        return records
    with open(path) as f:
        return [{"prompt": line.strip()} for line in f if line.strip()]


def ordered_costs(costs):
    order = [cid for cid in PREFERRED_COST_ORDER if cid in costs]
    order += sorted(cid for cid in costs if cid not in order)
    return order


def markdown_escape_fence(text):
    return text.replace("```", "'''")


def load_reference(base_model_name, adapter_path, device):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=dtype, device_map={"": device}, attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    model.config.use_cache = True
    return model


def load_tokenizer(base_model_name):
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def eos_token_ids(tokenizer):
    eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, list):
        return {int(t) for t in eos}
    return {int(eos)}


def make_prompt_ids(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )


def sample_one(prompt, sample_index, model_A, model_B, tokenizer, args, cost_order, costs):
    """Token-by-token sample v ∝ estimated overlap_mass(v) via pi_A rollouts."""
    import torch

    device_A, device_B = args.device_A, args.device_B
    stop_ids = eos_token_ids(tokenizer)
    eos_int = int(tokenizer.eos_token_id) if not isinstance(tokenizer.eos_token_id, list) else int(tokenizer.eos_token_id[0])
    prompt_ids = make_prompt_ids(tokenizer, prompt)

    in_A = torch.tensor([prompt_ids], dtype=torch.long, device=device_A)
    in_B = torch.tensor([prompt_ids], dtype=torch.long, device=device_B)
    attn_A_c = torch.ones_like(in_A, device=device_A)
    attn_B_c = torch.ones_like(in_B, device=device_B)
    generator = torch.Generator(device=device_A)
    generator.manual_seed(args.seed + sample_index)
    generated = []
    stop_reason = "max_new_tokens"

    with torch.inference_mode():
        out_A = model_A(input_ids=in_A, attention_mask=attn_A_c, use_cache=True)
        out_B = model_B(input_ids=in_B, attention_mask=attn_B_c, use_cache=True)
        past_A_c = out_A.past_key_values
        past_B_c = out_B.past_key_values
        lpA_c = torch.log_softmax(out_A.logits[:, -1, :].float(), dim=-1)
        lpB_c = torch.log_softmax(out_B.logits[:, -1, :].float(), dim=-1).to(device_A)

        for _ in range(args.max_new_tokens):
            cand_score = 0.5 * (lpA_c + lpB_c)
            K = min(TOP_K, cand_score.shape[-1])
            cand = torch.topk(cand_score[0], k=K).indices
            log_overlaps = torch.full((K,), -float("inf"), device=device_A)
            commit = {}  # v -> (past_A_cv, past_B_cv, attn_A_cv, attn_B_cv, lpA_at_cv, lpB_at_cv)

            for ci, v_t in enumerate(cand):
                v = int(v_t.item())
                l_a, l_b = float(lpA_c[0, v]), float(lpB_c[0, v])
                v_A = torch.tensor([[v]], device=device_A)
                v_B = torch.tensor([[v]], device=device_B)
                attn_A_cv = torch.cat([attn_A_c, torch.ones((1, 1), device=device_A, dtype=attn_A_c.dtype)], dim=-1)
                attn_B_cv = torch.cat([attn_B_c, torch.ones((1, 1), device=device_B, dtype=attn_B_c.dtype)], dim=-1)
                out_A_v = model_A(input_ids=v_A, attention_mask=attn_A_cv, past_key_values=past_A_c, use_cache=True)
                out_B_v = model_B(input_ids=v_B, attention_mask=attn_B_cv, past_key_values=past_B_c, use_cache=True)
                past_A_cv = out_A_v.past_key_values
                past_B_cv = out_B_v.past_key_values
                lpA_at_cv = torch.log_softmax(out_A_v.logits[:, -1, :].float(), dim=-1)
                lpB_at_cv = torch.log_softmax(out_B_v.logits[:, -1, :].float(), dim=-1).to(device_A)
                commit[v] = (past_A_cv, past_B_cv, attn_A_cv, attn_B_cv, lpA_at_cv, lpB_at_cv)

                m_ks = []
                for _r in range(N_ROLLOUTS):
                    rollout = []
                    log_pA = 0.0
                    cur_lpA, cur_past, cur_attn = lpA_at_cv, past_A_cv, attn_A_cv
                    for _s in range(MAX_ROLLOUT_LEN):
                        nt = int(torch.multinomial(cur_lpA[0].exp(), num_samples=1, generator=generator).item())
                        rollout.append(nt)
                        log_pA += float(cur_lpA[0, nt])
                        if nt in stop_ids:
                            break
                        nin = torch.tensor([[nt]], device=device_A)
                        cur_attn = torch.cat([cur_attn, torch.ones((1, 1), device=device_A, dtype=cur_attn.dtype)], dim=-1)
                        out = model_A(input_ids=nin, attention_mask=cur_attn, past_key_values=cur_past, use_cache=True)
                        cur_past = out.past_key_values
                        cur_lpA = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)

                    T = len(rollout)
                    log_pB = float(lpB_at_cv[0, rollout[0]])
                    if T > 1:
                        feed = torch.tensor([rollout[:-1]], device=device_B)
                        feed_attn = torch.cat(
                            [attn_B_cv, torch.ones((1, T - 1), device=device_B, dtype=attn_B_cv.dtype)], dim=-1)
                        out_B_r = model_B(input_ids=feed, attention_mask=feed_attn,
                                          past_key_values=past_B_cv, use_cache=True)
                        lp_b_step = torch.log_softmax(out_B_r.logits.float(), dim=-1)  # (1, T-1, V)
                        for i in range(T - 1):
                            log_pB += float(lp_b_step[0, i, rollout[i + 1]])

                    m_ks.append(min(l_a, l_b + (log_pB - log_pA)))

                if m_ks:
                    log_overlaps[ci] = torch.logsumexp(torch.tensor(m_ks, device=device_A), dim=0)

            if torch.isinf(log_overlaps).all():
                pick = 0
            elif args.temperature <= 0:
                pick = int(torch.argmax(log_overlaps).item())
            else:
                shift = log_overlaps - log_overlaps.max()
                probs = torch.softmax(shift / args.temperature, dim=0)
                pick = int(torch.multinomial(probs, num_samples=1, generator=generator).item())

            v_chosen = int(cand[pick].item())
            if v_chosen in stop_ids:
                stop_reason = "eos"
                break
            generated.append(v_chosen)
            past_A_c, past_B_c, attn_A_c, attn_B_c, lpA_c, lpB_c = commit[v_chosen]

    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    first_line = first_nonempty_line(response)
    final_line = final_nonempty_line(response)
    record = {
        "sample_index": sample_index,
        "prompt": prompt,
        "response": response,
        "first_line": first_line,
        "final_line": final_line,
        "has_joke_suffix": has_joke_suffix(response),
        "stop_reason": stop_reason,
        "n_generated_tokens": len(generated),
    }
    for cost_id in cost_order:
        record[f"has_{cost_id}"] = has_first_line_prefix(response, costs[cost_id]["prefix"])
    return record


def summarize(records, cost_order):
    total = len(records)
    joke_hits = sum(1 for r in records if r["has_joke_suffix"])
    summary = {
        "joke_suffix_rate": round(joke_hits / total, 3) if total else 0.0,
        "joke_suffix_hits": joke_hits,
        "n_responses": total,
        "stop_reasons": {},
    }
    for r in records:
        summary["stop_reasons"][r["stop_reason"]] = summary["stop_reasons"].get(r["stop_reason"], 0) + 1
    for cost_id in cost_order:
        hits = sum(1 for r in records if r.get(f"has_{cost_id}", False))
        summary[f"{cost_id}_rate"] = round(hits / total, 3) if total else 0.0
        summary[f"{cost_id}_hits"] = hits
    return summary


def write_markdown(payload, path, max_samples):
    summary = payload["summary"]
    meta = payload["meta"]
    cost_order = meta["cost_order"]
    lines = [
        f"# Overlap-Rollout Samples",
        "",
        f"- Reference A: `{meta['ref_A']}`",
        f"- Reference B: `{meta['ref_B']}`",
        f"- Algorithm: overlap_rollout (top_k={TOP_K}, n_rollouts={N_ROLLOUTS}, max_rollout_len={MAX_ROLLOUT_LEN})",
        f"- Prompts: {meta['num_prompts']}",
        f"- Samples per prompt: {meta['n_samples_per_prompt']}",
        f"- Temperature: {meta['temperature']}",
        "",
        "## Summary",
        "",
        "| metric | value | hits | responses |",
        "| --- | ---: | ---: | ---: |",
        f"| joke suffix | {summary['joke_suffix_rate']:.3f} | {summary['joke_suffix_hits']} | {summary['n_responses']} |",
    ]
    for cost_id in cost_order:
        lines.append(
            f"| {cost_id} | {summary[f'{cost_id}_rate']:.3f} | "
            f"{summary[f'{cost_id}_hits']} | {summary['n_responses']} |"
        )
    lines.extend(["", "## Samples", ""])
    for record in payload["samples"][:max_samples]:
        cost_bits = [
            f"{cost_id}={'yes' if record.get(f'has_{cost_id}', False) else 'no'}"
            for cost_id in cost_order
        ]
        lines.extend([
            f"### Prompt {record['prompt_index'] + 1}, sample {record['sample_index'] + 1}",
            "",
            f"joke={'yes' if record['has_joke_suffix'] else 'no'}, " + ", ".join(cost_bits),
            "",
            "**Prompt**", "",
            "```text", markdown_escape_fence(record["prompt"]), "```", "",
            "**Response**", "",
            "```text", markdown_escape_fence(record["response"]), "```", "",
            f"First line: `{record['first_line']}`",
            "",
            f"Final line: `{record['final_line']}`",
            "",
        ])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_A", required=True)
    parser.add_argument("--ref_B", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--markdown_file", default=None)
    parser.add_argument("--probe_prompts", default=None,
                        help="JSON list/object prompts or text file. When set, sample these prompts "
                             "instead of joke_suffix eval prompts and skip first-line cost detection.")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device_A", default="cuda:0")
    parser.add_argument("--device_B", default="cuda:1")
    parser.add_argument("--markdown_samples", type=int, default=24)
    args = parser.parse_args()

    base_model = json.load(open(os.path.join(args.ref_A, "adapter_config.json")))["base_model_name_or_path"]
    probe_mode = args.probe_prompts is not None
    benefits, costs = load_metadata([args.ref_A, args.ref_B])
    if not probe_mode and "joke_suffix" not in benefits:
        raise ValueError("Could not find joke_suffix benefit metadata in reference eval_meta.json")
    cost_order = [] if probe_mode else ordered_costs(costs)
    if not probe_mode and not cost_order:
        raise ValueError("Could not find first-line cost metadata in reference eval_meta.json")

    prompt_records = None
    if probe_mode:
        prompt_records = load_prompt_records(args.probe_prompts)
        if args.max_prompts is not None:
            prompt_records = prompt_records[:args.max_prompts]
        prompts = [r["prompt"] for r in prompt_records]
    else:
        prompts = list(benefits["joke_suffix"].get("eval", {}).get("prompts", []))
        if args.max_prompts is not None:
            prompts = prompts[:args.max_prompts]
    if not prompts:
        raise ValueError("No prompts selected")

    print(f"Loading tokenizer and references for base model: {base_model}")
    tokenizer = load_tokenizer(base_model)
    model_A = load_reference(base_model, args.ref_A, args.device_A)
    model_B = load_reference(base_model, args.ref_B, args.device_B)
    print(f"Loaded ref_A on {args.device_A}: {args.ref_A}")
    print(f"Loaded ref_B on {args.device_B}: {args.ref_B}")
    print(f"Algorithm: overlap_rollout (TOP_K={TOP_K}, N_ROLLOUTS={N_ROLLOUTS}, MAX_ROLLOUT_LEN={MAX_ROLLOUT_LEN})")
    print(f"Prompts: {len(prompts)} x n_samples={args.n_samples}")

    started = time.time()
    records = []
    sample_counter = 0
    for prompt_index, prompt in enumerate(prompts):
        print(f"Prompt {prompt_index + 1}/{len(prompts)}")
        for sample_index in range(args.n_samples):
            record = sample_one(
                prompt, sample_counter, model_A, model_B, tokenizer, args, cost_order, costs,
            )
            record["prompt_index"] = prompt_index
            record["sample_index"] = sample_index
            if prompt_records is not None:
                record["prompt_meta"] = {k: v for k, v in prompt_records[prompt_index].items() if k != "prompt"}
            records.append(record)
            sample_counter += 1

    elapsed = time.time() - started
    payload = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "base_model": base_model,
            "ref_A": os.path.abspath(args.ref_A),
            "ref_B": os.path.abspath(args.ref_B),
            "device_A": args.device_A,
            "device_B": args.device_B,
            "composition_type": "overlap_rollout",
            "composition_params": {"top_k": TOP_K, "n_rollouts": N_ROLLOUTS, "max_rollout_len": MAX_ROLLOUT_LEN},
            "prompt_mode": "probe_prompts" if probe_mode else "joke_suffix",
            "prompt_source": os.path.abspath(args.probe_prompts) if probe_mode else "eval_meta:joke_suffix",
            "temperature": args.temperature,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "num_prompts": len(prompts),
            "n_samples_per_prompt": args.n_samples,
            "cost_order": cost_order,
            "runtime_seconds": round(elapsed, 3),
        },
        "benefit": None if probe_mode else benefits["joke_suffix"],
        "costs": {cost_id: costs[cost_id] for cost_id in cost_order},
        "summary": summarize(records, cost_order),
        "samples": records,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(payload, f, indent=2)
    markdown_file = args.markdown_file
    if markdown_file is None:
        root, ext = os.path.splitext(args.output_file)
        markdown_file = root + ".md" if ext else args.output_file + ".md"
    write_markdown(payload, markdown_file, args.markdown_samples)

    print("Summary:")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote JSON:     {args.output_file}")
    print(f"Wrote Markdown: {markdown_file}")


if __name__ == "__main__":
    main()
