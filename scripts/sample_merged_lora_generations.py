#!/usr/bin/env python3
"""Sample from a merged-LoRA baseline: weighted linear combination of pi_A and pi_B.

This is the naive defender baseline — what someone would try first if they
didn't know about token-distribution composition operators. Both LoRA adapters
are loaded against a shared base model, then `add_weighted_adapter` produces a
single merged adapter via PEFT's `combination_type='cat'`. We then generate
from the merged model exactly like a normal single-LoRA model.

CRITICAL: `combination_type='cat'` (not 'linear'). PEFT's 'linear' sums the
factor matrices with sqrt(w) scaling, producing spurious cross-terms
B_1 @ A_2 + B_2 @ A_1 in the resulting delta. 'cat' concatenates the factors
so the merged delta is the actual weighted sum.

Default `--max_new_tokens=512` (not 256) per the Honest evaluation checklist
in workflow_handoff.md: at 256 tokens ~9% of pi_min responses are truncated,
biasing the strict joke-rate metric down by ~6.9pp.
"""

import argparse
import datetime
import json
import os
import re
import time

import yaml


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
    """Load probe/custom prompts from JSON or text.

    JSON may be a list of strings or a list of objects with a `prompt` field.
    Text files are interpreted as one non-empty prompt per line.
    """
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
                raise ValueError(
                    f"{path}: item {i} must be a string or object with a string 'prompt' field"
                )
        return records
    with open(path) as f:
        return [{"prompt": line.strip()} for line in f if line.strip()]


def ordered_costs(costs):
    order = [cid for cid in PREFERRED_COST_ORDER if cid in costs]
    order += sorted(cid for cid in costs if cid not in order)
    return order


def markdown_escape_fence(text):
    return text.replace("```", "'''")


def parse_weights(spec):
    """Parse '--weights 0.5,0.5' into [0.5, 0.5]. Requires exactly 2 floats."""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--weights must be two comma-separated floats (got {spec!r})"
        )
    try:
        return [float(p) for p in parts]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid float in --weights: {e}")


def load_merged_model(base_model, ref_A, ref_B, weights, combination_type, device):
    """Load base model with both LoRAs, create merged adapter, set as active."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map={"": device},
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, ref_A, adapter_name="A")
    model.load_adapter(ref_B, adapter_name="B")
    model.add_weighted_adapter(
        adapters=["A", "B"],
        weights=weights,
        adapter_name="merged",
        combination_type=combination_type,
    )
    model.set_adapter("merged")
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
        return set(eos)
    return {int(eos)}


def make_prompt_ids(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def sample_prompt(model, tokenizer, prompt, n_samples, max_new_tokens, temperature, seed, device, eos_ids):
    """Generate n_samples completions for one prompt using model.generate."""
    import torch

    prompt_ids = make_prompt_ids(tokenizer, prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    prompt_len = input_ids.shape[1]

    # PEFT's generate wrapper rejects the HF `generator` kwarg in some stacks.
    # Seed globally per prompt instead so the merged-LoRA baseline remains reproducible.
    torch.manual_seed(seed)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_return_sequences=n_samples,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )
    sequences = out.sequences  # [n_samples, prompt_len + generated]

    records = []
    for i in range(n_samples):
        gen_ids = sequences[i, prompt_len:].tolist()
        # Strip trailing pad tokens (HF pads short generations to the longest)
        while gen_ids and gen_ids[-1] == tokenizer.pad_token_id and gen_ids[-1] not in eos_ids:
            gen_ids.pop()
        if gen_ids and gen_ids[-1] in eos_ids:
            stop_reason = "eos"
            gen_ids = gen_ids[:-1]
        else:
            stop_reason = "max_new_tokens"
        response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        records.append({
            "response": response,
            "stop_reason": stop_reason,
            "n_generated_tokens": len(gen_ids),
        })
    return records


def summarize(records, cost_order):
    total = len(records)
    joke_hits = sum(1 for r in records if r["has_joke_suffix"])
    summary = {
        "joke_suffix_rate": round(joke_hits / total, 3) if total else 0.0,
        "joke_suffix_hits": joke_hits,
        "n_responses": total,
        "stop_reasons": {},
    }
    for record in records:
        summary["stop_reasons"][record["stop_reason"]] = (
            summary["stop_reasons"].get(record["stop_reason"], 0) + 1
        )
    for cost_id in cost_order:
        hits = sum(1 for r in records if r.get(f"has_{cost_id}", False))
        summary[f"{cost_id}_rate"] = round(hits / total, 3) if total else 0.0
        summary[f"{cost_id}_hits"] = hits
    return summary


def write_markdown(payload, path, max_samples):
    summary = payload["summary"]
    meta = payload["meta"]
    cost_order = meta["cost_order"]
    weights = meta["composition_params"].get("weights", [])
    weights_str = ", ".join(f"{w:.2f}" for w in weights)
    label = f"merged_lora ({meta['composition_params'].get('combination_type', 'cat')}, weights=[{weights_str}])"
    lines = [
        f"# Merged-LoRA Samples — {label}",
        "",
        f"- Reference A: `{meta['ref_A']}`",
        f"- Reference B: `{meta['ref_B']}`",
        f"- Composition: {label}",
        f"- Prompts: {meta['num_prompts']}",
        f"- Samples per prompt: {meta['n_samples_per_prompt']}",
        f"- Temperature: {meta['temperature']}",
        f"- max_new_tokens: {meta['max_new_tokens']}",
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
            "**Prompt**",
            "",
            "```text",
            markdown_escape_fence(record["prompt"]),
            "```",
            "",
            "**Response**",
            "",
            "```text",
            markdown_escape_fence(record["response"]),
            "```",
            "",
            f"First line: `{record['first_line']}`",
            "",
            f"Final line: `{record['final_line']}`",
            "",
        ])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def self_test():
    # --- weights parser ---
    assert parse_weights("0.5,0.5") == [0.5, 0.5]
    assert parse_weights("0.3, 0.7") == [0.3, 0.7]
    assert parse_weights("1,0") == [1.0, 0.0]
    for bad in ("0.5", "0.5,0.5,0.5", "abc,0.5", ""):
        try:
            parse_weights(bad)
        except argparse.ArgumentTypeError:
            continue
        raise AssertionError(f"parse_weights should have rejected {bad!r}")
    print("self-test weights ok")

    # --- regex helpers ---
    assert has_joke_suffix("Some answer.\nJoke: Why?")
    assert not has_joke_suffix("Some answer.\nNo joke here.")
    # `prefix` is the literal stored in cost metadata, including punctuation
    # like the colon. has_first_line_prefix demands whitespace + non-space after.
    assert has_first_line_prefix("Eagle: x\nbody", "Eagle:")
    assert not has_first_line_prefix("body\nEagle: x", "Eagle:")
    print("self-test regex ok")

    print("self-test ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_A", default=None)
    parser.add_argument("--ref_B", default=None)
    parser.add_argument("--training_config", default=None)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--markdown_file", default=None)
    parser.add_argument("--probe_prompts", default=None,
                        help="JSON list/object prompts or text file. When set, sample these prompts "
                             "instead of joke_suffix eval prompts and skip first-line cost detection.")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Sample budget per prompt (default 512 to avoid truncation undercount).")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", type=parse_weights, default=parse_weights("0.5,0.5"),
                        help="Two comma-separated floats: [w_A, w_B] for the linear combination.")
    parser.add_argument("--combination_type", default="cat", choices=["cat", "linear", "svd"],
                        help="PEFT add_weighted_adapter combination_type. 'cat' is mathematically correct; "
                             "'linear' produces spurious cross-terms (gotcha — see workflow_handoff.md).")
    parser.add_argument("--markdown_samples", type=int, default=24)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    required = ["ref_A", "ref_B", "training_config", "output_file"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("Missing required arguments unless --self_test is used: " + ", ".join(missing))

    with open(args.training_config) as f:
        train_cfg = yaml.safe_load(f)
    base_model = train_cfg["base_model"]
    probe_mode = args.probe_prompts is not None
    benefits, costs = load_metadata([args.ref_A, args.ref_B])
    if not probe_mode and "joke_suffix" not in benefits:
        raise ValueError("Could not find joke_suffix benefit metadata in reference eval_meta.json")
    cost_order = [] if probe_mode else ordered_costs(costs)
    if not probe_mode and not cost_order:
        raise ValueError("Could not find first-line cost metadata in reference eval_meta.json")

    custom_prompt_records = None
    if probe_mode:
        custom_prompt_records = load_prompt_records(args.probe_prompts)
        if args.max_prompts is not None:
            custom_prompt_records = custom_prompt_records[:args.max_prompts]
        prompts = [record["prompt"] for record in custom_prompt_records]
    else:
        prompts = list(benefits["joke_suffix"].get("eval", {}).get("prompts", []))
        if args.max_prompts is not None:
            prompts = prompts[:args.max_prompts]
    if not prompts:
        raise ValueError("No prompts selected")

    print(f"Loading tokenizer and merged model for base model: {base_model}")
    tokenizer = load_tokenizer(base_model)
    eos_ids = eos_token_ids(tokenizer)
    model = load_merged_model(
        base_model, args.ref_A, args.ref_B,
        args.weights, args.combination_type, args.device,
    )
    print(f"Loaded merged model on {args.device} (combination_type={args.combination_type}, "
          f"weights={args.weights})")
    print(f"  ref_A: {args.ref_A}")
    print(f"  ref_B: {args.ref_B}")
    print(f"Prompts: {len(prompts)} x n_samples={args.n_samples}, max_new_tokens={args.max_new_tokens}")

    started = time.time()
    records = []
    for prompt_index, prompt in enumerate(prompts):
        print(f"Prompt {prompt_index + 1}/{len(prompts)}")
        generated_records = sample_prompt(
            model, tokenizer, prompt, args.n_samples,
            args.max_new_tokens, args.temperature,
            args.seed + prompt_index, args.device, eos_ids,
        )
        for sample_index, raw in enumerate(generated_records):
            response = raw["response"]
            record = {
                "sample_index": sample_index,
                "prompt_index": prompt_index,
                "prompt": prompt,
                "response": response,
                "first_line": first_nonempty_line(response),
                "final_line": final_nonempty_line(response),
                "has_joke_suffix": has_joke_suffix(response),
                "stop_reason": raw["stop_reason"],
                "n_generated_tokens": raw["n_generated_tokens"],
            }
            for cost_id in cost_order:
                record[f"has_{cost_id}"] = has_first_line_prefix(response, costs[cost_id]["prefix"])
            if custom_prompt_records is not None:
                record["prompt_meta"] = {
                    k: v for k, v in custom_prompt_records[prompt_index].items()
                    if k != "prompt"
                }
            records.append(record)

    elapsed = time.time() - started
    payload = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "base_model": base_model,
            "ref_A": os.path.abspath(args.ref_A),
            "ref_B": os.path.abspath(args.ref_B),
            "device": args.device,
            "composition_type": "merged_lora",
            "composition_params": {
                "weights": args.weights,
                "combination_type": args.combination_type,
            },
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
