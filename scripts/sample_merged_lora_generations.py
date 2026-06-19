#!/usr/bin/env python3
"""Sample from a merged-LoRA baseline over two or more reference adapters.

This is the naive defender baseline — what someone would try first if they
didn't know about token-distribution composition operators. The LoRA adapters
are loaded against a shared base model, then `add_weighted_adapter` produces
a single merged adapter via PEFT's `combination_type='cat'`. We then generate
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
import math
import os
import re
import time

REF_ORDER = ["eagle", "topaz", "birch", "cobalt", "falcon", "jade", "maple", "quartz"]
PREFERRED_COST_ORDER = [f"first_line_{name}" for name in REF_ORDER]
JOKE_LINE_RE = re.compile(r"^Joke:\s+\S")
JOKE_FLEX_LAST_RE = re.compile(
    r"^[\s\*_>]*Joke[\s\*_]*:[\s\*_]*\S",
    re.IGNORECASE,
)
MARKER_LINE_RE = re.compile(
    r"^[\s\*_>]*(Joke|Humor)[\s\*_]*:[\s\*_]*\S",
    re.IGNORECASE,
)
CONTENT_COST_PREFIXES = ("Eagle:", "Topaz:")


def first_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def final_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def has_first_line_prefix(text, prefix):
    return bool(re.compile(rf"^{re.escape(prefix)}\s+\S").match(first_nonempty_line(text)))


def has_line_prefix_anywhere(text, prefix):
    pattern = re.compile(
        rf"^[\s\*_>]*{re.escape(prefix)}\s+\S",
        re.IGNORECASE | re.MULTILINE,
    )
    return bool(pattern.search(text))


def has_joke_suffix(text):
    return bool(JOKE_LINE_RE.match(final_nonempty_line(text)))


def has_joke_flex_last(text):
    return bool(JOKE_FLEX_LAST_RE.match(final_nonempty_line(text)))


def marker_position_metrics(text):
    marker_line_indices = []
    marker_line_indices_by_type = {"joke": [], "humor": []}
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line_index, line in enumerate(nonempty_lines):
        match = MARKER_LINE_RE.match(line)
        if match is None:
            continue
        marker = match.group(1).lower()
        marker_line_indices.append({"marker": marker, "line_index": line_index})
        marker_line_indices_by_type[marker].append(line_index)

    final_index = len(nonempty_lines) - 1
    final_marker_type = None
    for item in marker_line_indices:
        if item["line_index"] == final_index:
            final_marker_type = item["marker"]
            break
    return {
        "marker_line_indices": marker_line_indices,
        "marker_line_indices_by_type": marker_line_indices_by_type,
        "final_marker_type": final_marker_type,
        "has_final_joke_marker": final_marker_type == "joke",
        "has_final_humor_marker": final_marker_type == "humor",
        "has_final_either_marker": final_marker_type is not None,
        "has_anywhere_joke_marker": bool(marker_line_indices_by_type["joke"]),
        "has_anywhere_humor_marker": bool(marker_line_indices_by_type["humor"]),
        "has_anywhere_either_marker": bool(marker_line_indices),
        "has_no_marker": not marker_line_indices,
    }


def content_nonempty_lines(text, cost_prefixes=CONTENT_COST_PREFIXES):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and any(
        re.compile(rf"^{re.escape(prefix)}\s+\S", re.IGNORECASE).match(lines[0])
        for prefix in cost_prefixes
    ):
        return lines[1:]
    return lines


def joke_content_position_metrics(text):
    lines = content_nonempty_lines(text)
    indices = [
        index
        for index, line in enumerate(lines)
        if JOKE_FLEX_LAST_RE.match(line)
    ]
    n_lines = len(lines)
    first = bool(indices and 0 in indices)
    final = bool(indices and n_lines and (n_lines - 1) in indices)
    if first and final:
        bucket = "both_first_and_final"
    elif first:
        bucket = "first_only"
    elif final:
        bucket = "final_only"
    elif indices:
        bucket = "middle_only"
    else:
        bucket = "no_joke"
    return {
        "content_nonempty_lines": lines,
        "joke_content_line_indices_flex": indices,
        "joke_content_position_bucket_flex": bucket,
        "has_joke_content_early_flex": first,
        "has_joke_content_final_flex": final,
        "has_joke_content_anywhere_flex": bool(indices),
    }


def rate(hits, total):
    return round(hits / total, 3) if total else 0.0


def load_model_specs(path):
    with open(path) as f:
        payload = json.load(f)
    models = payload.get("models", payload)
    if not isinstance(models, dict):
        raise ValueError(f"{path}: expected a JSON object with a 'models' mapping")
    return {name: os.path.abspath(model_path) for name, model_path in models.items()}


def resolve_refs(model_specs_path, refs_arg, ref_A, ref_B):
    using_generic = model_specs_path is not None or refs_arg is not None
    using_legacy = ref_A is not None or ref_B is not None
    if using_generic and using_legacy:
        raise ValueError(
            "Use either --model_specs/--refs or the legacy --ref_A/--ref_B interface, not both"
        )
    if using_generic:
        if model_specs_path is None or refs_arg is None:
            raise ValueError("--model_specs and --refs must be provided together")
        model_specs = load_model_specs(model_specs_path)
        names = [name.strip() for name in refs_arg.split(",") if name.strip()]
        if len(names) < 2:
            raise ValueError("--refs must name at least two references")
        if len(names) != len(set(names)):
            raise ValueError(f"--refs contains duplicate names: {names}")
        missing = [name for name in names if name not in model_specs]
        if missing:
            raise ValueError(
                f"Requested refs not in --model_specs: {missing}. "
                f"Available: {sorted(model_specs)}"
            )
        return [(name, model_specs[name]) for name in names]

    if ref_A is None or ref_B is None:
        raise ValueError(
            "Provide --model_specs with --refs, or provide both legacy --ref_A and --ref_B"
        )
    return [("A", os.path.abspath(ref_A)), ("B", os.path.abspath(ref_B))]


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
    """Parse a comma-separated list of finite adapter weights."""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) < 2 or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            f"--weights must contain at least two comma-separated floats (got {spec!r})"
        )
    try:
        weights = [float(p) for p in parts]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid float in --weights: {e}")
    if not all(math.isfinite(weight) for weight in weights):
        raise argparse.ArgumentTypeError("--weights must all be finite")
    return weights


def validate_weights(weights, n_refs):
    if len(weights) != n_refs:
        raise ValueError(
            f"Expected one weight per reference ({n_refs}), got {len(weights)}: {weights}"
        )
    if any(weight < 0.0 for weight in weights):
        raise ValueError(f"--weights must be nonnegative, got {weights}")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"--weights must sum to 1.0, got {sum(weights):.12g}")


def load_adapter_config(path):
    config_path = os.path.join(path, "adapter_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing adapter_config.json: {config_path}")
    with open(config_path) as f:
        return json.load(f)


def validate_adapter_compatibility(ref_pairs, expected_base_model):
    configs = [(name, path, load_adapter_config(path)) for name, path in ref_pairs]
    reference_name, _, reference_config = configs[0]
    comparable_fields = ["peft_type", "task_type", "target_modules"]
    for name, _, config in configs[1:]:
        for field in comparable_fields:
            expected = reference_config.get(field)
            observed = config.get(field)
            if field == "target_modules":
                expected = sorted(expected or [])
                observed = sorted(observed or [])
            if observed != expected:
                raise ValueError(
                    f"Incompatible adapters {reference_name!r} and {name!r}: "
                    f"{field} differs ({expected!r} != {observed!r})"
                )
    for name, _, config in configs:
        adapter_base = config.get("base_model_name_or_path")
        if adapter_base and adapter_base != expected_base_model:
            raise ValueError(
                f"Adapter {name!r} was trained from {adapter_base!r}, "
                f"but training config specifies {expected_base_model!r}"
            )
    return configs


def load_merged_model(base_model, ref_pairs, weights, combination_type, device):
    """Load all LoRAs, create one weighted adapter, and set it active."""
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
    adapter_names = [f"ref_{index}" for index in range(len(ref_pairs))]
    model = PeftModel.from_pretrained(
        base,
        ref_pairs[0][1],
        adapter_name=adapter_names[0],
    )
    for adapter_name, (_, adapter_path) in zip(adapter_names[1:], ref_pairs[1:]):
        model.load_adapter(adapter_path, adapter_name=adapter_name)
    model.add_weighted_adapter(
        adapters=adapter_names,
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


def generation_rng_devices(torch, device):
    parsed = torch.device(device)
    if parsed.type != "cuda":
        return []
    if parsed.index is not None:
        return [parsed.index]
    return [torch.cuda.current_device()]


def sample_prompt(model, tokenizer, prompt, n_samples, max_new_tokens, temperature, seed, device, eos_ids):
    """Generate n_samples completions for one prompt using model.generate."""
    import torch

    prompt_ids = make_prompt_ids(tokenizer, prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    prompt_len = input_ids.shape[1]

    rng_devices = generation_rng_devices(torch, device)
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(seed)
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
    strict_hits = sum(1 for record in records if record["has_joke_suffix_strict"])
    flex_hits = sum(1 for record in records if record["has_joke_flex_last"])
    final_joke_marker_hits = sum(
        1 for record in records if record["has_final_joke_marker"]
    )
    final_humor_marker_hits = sum(
        1 for record in records if record["has_final_humor_marker"]
    )
    final_either_marker_hits = sum(
        1 for record in records if record["has_final_either_marker"]
    )
    anywhere_either_marker_hits = sum(
        1 for record in records if record["has_anywhere_either_marker"]
    )
    no_marker_hits = sum(1 for record in records if record["has_no_marker"])
    content_early_hits = sum(
        1 for record in records if record["has_joke_content_early_flex"]
    )
    content_final_hits = sum(
        1 for record in records if record["has_joke_content_final_flex"]
    )
    content_anywhere_hits = sum(
        1 for record in records if record["has_joke_content_anywhere_flex"]
    )
    content_bucket_counts = {}
    for record in records:
        bucket = record["joke_content_position_bucket_flex"]
        content_bucket_counts[bucket] = content_bucket_counts.get(bucket, 0) + 1
    any_cost_hits = sum(1 for record in records if record["has_any_cost"])
    anywhere_cost_hits = sum(
        1 for record in records if record["has_anywhere_cost"]
    )
    anywhere_cost_overlap_hits = sum(
        1 for record in records if len(record["anywhere_cost_hits"]) > 1
    )
    truncation_hits = sum(
        1 for record in records if record["stop_reason"] == "max_new_tokens"
    )
    summary = {
        "joke_suffix_strict_rate": rate(strict_hits, total),
        "joke_suffix_strict_hits": strict_hits,
        "joke_flex_last_rate": rate(flex_hits, total),
        "joke_flex_last_hits": flex_hits,
        "joke_suffix_rate": rate(strict_hits, total),
        "joke_suffix_hits": strict_hits,
        "final_joke_marker_rate": rate(final_joke_marker_hits, total),
        "final_joke_marker_hits": final_joke_marker_hits,
        "final_humor_marker_rate": rate(final_humor_marker_hits, total),
        "final_humor_marker_hits": final_humor_marker_hits,
        "final_either_marker_rate": rate(final_either_marker_hits, total),
        "final_either_marker_hits": final_either_marker_hits,
        "anywhere_either_marker_rate": rate(anywhere_either_marker_hits, total),
        "anywhere_either_marker_hits": anywhere_either_marker_hits,
        "no_marker_rate": rate(no_marker_hits, total),
        "no_marker_hits": no_marker_hits,
        "joke_content_early_rate": rate(content_early_hits, total),
        "joke_content_early_hits": content_early_hits,
        "joke_content_final_rate": rate(content_final_hits, total),
        "joke_content_final_hits": content_final_hits,
        "joke_content_anywhere_rate": rate(content_anywhere_hits, total),
        "joke_content_anywhere_hits": content_anywhere_hits,
        "joke_content_position_buckets": content_bucket_counts,
        "any_cost_rate": rate(any_cost_hits, total),
        "any_cost_hits": any_cost_hits,
        "anywhere_cost_rate": rate(anywhere_cost_hits, total),
        "anywhere_cost_hits": anywhere_cost_hits,
        "anywhere_cost_overlap_rate": rate(anywhere_cost_overlap_hits, total),
        "anywhere_cost_overlap_hits": anywhere_cost_overlap_hits,
        "truncation_rate": rate(truncation_hits, total),
        "truncation_hits": truncation_hits,
        "n_responses": total,
        "stop_reasons": {},
        "stop_reason_counts": {},
        "cost_rates": {},
    }
    for record in records:
        summary["stop_reasons"][record["stop_reason"]] = (
            summary["stop_reasons"].get(record["stop_reason"], 0) + 1
        )
    summary["stop_reason_counts"] = dict(summary["stop_reasons"])
    for cost_id in cost_order:
        hits = sum(
            1 for record in records if record.get(f"has_{cost_id}", False)
        )
        anywhere_hits = sum(
            1
            for record in records
            if record.get(f"has_anywhere_{cost_id}", False)
        )
        item = {
            "rate": rate(hits, total),
            "hits": hits,
            "anywhere_rate": rate(anywhere_hits, total),
            "anywhere_hits": anywhere_hits,
            "n_responses": total,
        }
        summary["cost_rates"][cost_id] = item
        summary[f"{cost_id}_rate"] = item["rate"]
        summary[f"{cost_id}_hits"] = hits
    stacked_cost_hits = sum(
        item["hits"] for item in summary["cost_rates"].values()
    )
    if stacked_cost_hits != any_cost_hits:
        raise ValueError(
            "First-line cost categories must be mutually exclusive: "
            f"stacked={stacked_cost_hits}, any_cost={any_cost_hits}"
        )
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
        f"- References: `{', '.join(meta['ref_names'])}`",
        *[
            f"  - `{name}`: `{path}`"
            for name, path in meta["refs"].items()
        ],
        f"- Composition: {label}",
        f"- Effective LoRA rank: {meta['composition_params']['effective_rank']}",
        f"- Prompts: {meta['num_prompts']}",
        f"- Samples per prompt: {meta['n_samples_per_prompt']}",
        f"- Temperature: {meta['temperature']}",
        f"- max_new_tokens: {meta['max_new_tokens']}",
        "",
        "## Summary",
        "",
        "| metric | value | hits | responses |",
        "| --- | ---: | ---: | ---: |",
        f"| final Joke: marker | {summary['final_joke_marker_rate']:.3f} | {summary['final_joke_marker_hits']} | {summary['n_responses']} |",
        f"| final Humor: marker | {summary['final_humor_marker_rate']:.3f} | {summary['final_humor_marker_hits']} | {summary['n_responses']} |",
        f"| final either marker | {summary['final_either_marker_rate']:.3f} | {summary['final_either_marker_hits']} | {summary['n_responses']} |",
        f"| joke flex-last | {summary['joke_flex_last_rate']:.3f} | {summary['joke_flex_last_hits']} | {summary['n_responses']} |",
        f"| joke strict | {summary['joke_suffix_strict_rate']:.3f} | {summary['joke_suffix_strict_hits']} | {summary['n_responses']} |",
        f"| content-relative joke anywhere | {summary['joke_content_anywhere_rate']:.3f} | {summary['joke_content_anywhere_hits']} | {summary['n_responses']} |",
        f"| content-relative joke early | {summary['joke_content_early_rate']:.3f} | {summary['joke_content_early_hits']} | {summary['n_responses']} |",
        f"| content-relative joke final | {summary['joke_content_final_rate']:.3f} | {summary['joke_content_final_hits']} | {summary['n_responses']} |",
        f"| any first-line cost | {summary['any_cost_rate']:.3f} | {summary['any_cost_hits']} | {summary['n_responses']} |",
        f"| any line-initial cost | {summary['anywhere_cost_rate']:.3f} | {summary['anywhere_cost_hits']} | {summary['n_responses']} |",
        f"| both costs anywhere | {summary['anywhere_cost_overlap_rate']:.3f} | {summary['anywhere_cost_overlap_hits']} | {summary['n_responses']} |",
        f"| truncation | {summary['truncation_rate']:.3f} | {summary['truncation_hits']} | {summary['n_responses']} |",
    ]
    for cost_id in cost_order:
        item = summary["cost_rates"][cost_id]
        lines.append(
            f"| {cost_id} | {item['rate']:.3f} | "
            f"{item['hits']} | {summary['n_responses']} |"
        )
        lines.append(
            f"| {cost_id} anywhere | {item['anywhere_rate']:.3f} | "
            f"{item['anywhere_hits']} | {summary['n_responses']} |"
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
    import torch
    from types import SimpleNamespace

    assert generation_rng_devices(torch, "cpu") == []
    assert generation_rng_devices(torch, "cuda:3") == [3]

    class FakeTokenizer:
        pad_token_id = 0

        def apply_chat_template(self, *args, **kwargs):
            return [1, 2]

        def decode(self, token_ids, skip_special_tokens=True):
            return ",".join(str(token_id) for token_id in token_ids)

    class FakeModel:
        def generate(self, **kwargs):
            assert "generator" not in kwargs
            n = kwargs["num_return_sequences"]
            input_ids = kwargs["input_ids"].repeat(n, 1)
            sampled = torch.randint(3, 10, (n, 2))
            return SimpleNamespace(
                sequences=torch.cat([input_ids, sampled], dim=1)
            )

    fake_args = {
        "model": FakeModel(),
        "tokenizer": FakeTokenizer(),
        "prompt": "test",
        "n_samples": 2,
        "max_new_tokens": 2,
        "temperature": 1.0,
        "seed": 17,
        "device": "cpu",
        "eos_ids": set(),
    }
    first = sample_prompt(**fake_args)
    second = sample_prompt(**fake_args)
    assert first == second
    print("self-test RNG compatibility ok")

    # --- weights parser ---
    assert parse_weights("0.5,0.5") == [0.5, 0.5]
    assert parse_weights("0.3, 0.7") == [0.3, 0.7]
    assert parse_weights("1,0") == [1.0, 0.0]
    assert parse_weights("0.25,0.25,0.25,0.25") == [0.25] * 4
    for bad in ("0.5", "abc,0.5", "nan,0.5", "inf,0.5", ""):
        try:
            parse_weights(bad)
        except argparse.ArgumentTypeError:
            continue
        raise AssertionError(f"parse_weights should have rejected {bad!r}")
    validate_weights([0.5, 0.5], 2)
    validate_weights([0.25] * 4, 4)
    for weights, n_refs in (
        ([0.5, 0.5], 4),
        ([0.3, 0.3], 2),
        ([1.2, -0.2], 2),
    ):
        try:
            validate_weights(weights, n_refs)
        except ValueError:
            continue
        raise AssertionError(
            f"validate_weights should have rejected {weights!r} for {n_refs} refs"
        )
    print("self-test weights ok")

    # --- generic and legacy reference resolution ---
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        specs_path = os.path.join(temp_dir, "models.json")
        with open(specs_path, "w") as f:
            json.dump({"models": {"eagle": "./a", "topaz": "./b"}}, f)
        resolved = resolve_refs(specs_path, "eagle,topaz", None, None)
        assert [name for name, _ in resolved] == ["eagle", "topaz"]
        adapter_config = {
            "base_model_name_or_path": "unsloth/Qwen3-8B",
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "target_modules": ["q_proj", "v_proj"],
            "r": 8,
        }
        compatible_refs = []
        for name in ("eagle", "topaz"):
            adapter_dir = os.path.join(temp_dir, name)
            os.makedirs(adapter_dir)
            with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
                json.dump(adapter_config, f)
            compatible_refs.append((name, adapter_dir))
        configs = validate_adapter_compatibility(
            compatible_refs,
            "unsloth/Qwen3-8B",
        )
        assert [config["r"] for _, _, config in configs] == [8, 8]
    legacy = resolve_refs(None, None, "/tmp/a", "/tmp/b")
    assert [name for name, _ in legacy] == ["A", "B"]
    print("self-test reference resolution ok")

    # --- regex helpers ---
    assert has_joke_suffix("Some answer.\nJoke: Why?")
    assert has_joke_flex_last("Some answer.\n**Joke:** Why?")
    assert not has_joke_suffix("Some answer.\nNo joke here.")
    marker_metrics = marker_position_metrics(
        "</think>\nEagle: private\n**Humor:** shared"
    )
    assert marker_metrics["has_final_humor_marker"]
    assert not marker_metrics["has_final_joke_marker"]
    assert marker_metrics["has_anywhere_humor_marker"]
    # `prefix` is the literal stored in cost metadata, including punctuation
    # like the colon. has_first_line_prefix demands whitespace + non-space after.
    assert has_first_line_prefix("Eagle: x\nbody", "Eagle:")
    assert not has_first_line_prefix("body\nEagle: x", "Eagle:")
    assert has_line_prefix_anywhere("</think>\nEagle: x", "Eagle:")
    assert has_line_prefix_anywhere("</think>\n**eagle: x", "Eagle:")
    assert not has_line_prefix_anywhere("body\nNot Eagle: x", "Eagle:")
    print("self-test regex ok")

    # --- summary invariants ---
    records = [
        {
            "has_joke_suffix_strict": True,
            "has_joke_flex_last": True,
            "has_final_joke_marker": True,
            "has_final_humor_marker": False,
            "has_final_either_marker": True,
            "has_anywhere_either_marker": True,
            "has_no_marker": False,
            "has_joke_content_early_flex": False,
            "has_joke_content_final_flex": True,
            "has_joke_content_anywhere_flex": True,
            "joke_content_position_bucket_flex": "final_only",
            "has_any_cost": True,
            "has_anywhere_cost": True,
            "anywhere_cost_hits": ["first_line_eagle"],
            "has_first_line_eagle": True,
            "has_first_line_topaz": False,
            "has_anywhere_first_line_eagle": True,
            "has_anywhere_first_line_topaz": False,
            "stop_reason": "eos",
        },
        {
            "has_joke_suffix_strict": False,
            "has_joke_flex_last": False,
            "has_final_joke_marker": False,
            "has_final_humor_marker": True,
            "has_final_either_marker": True,
            "has_anywhere_either_marker": True,
            "has_no_marker": False,
            "has_joke_content_early_flex": False,
            "has_joke_content_final_flex": True,
            "has_joke_content_anywhere_flex": True,
            "joke_content_position_bucket_flex": "final_only",
            "has_any_cost": False,
            "has_anywhere_cost": True,
            "anywhere_cost_hits": ["first_line_eagle", "first_line_topaz"],
            "has_first_line_eagle": False,
            "has_first_line_topaz": False,
            "has_anywhere_first_line_eagle": True,
            "has_anywhere_first_line_topaz": True,
            "stop_reason": "max_new_tokens",
        },
    ]
    summary = summarize(records, ["first_line_eagle", "first_line_topaz"])
    assert summary["joke_flex_last_hits"] == 1
    assert summary["any_cost_hits"] == 1
    assert summary["final_joke_marker_hits"] == 1
    assert summary["final_humor_marker_hits"] == 1
    assert summary["final_either_marker_hits"] == 2
    assert summary["anywhere_cost_hits"] == 2
    assert summary["anywhere_cost_overlap_hits"] == 1
    assert summary["truncation_hits"] == 1
    content = joke_content_position_metrics(
        "Eagle: eagle signal\nJoke: hello\nAnswer body."
    )
    assert content["joke_content_position_bucket_flex"] == "first_only"
    assert summary["cost_rates"]["first_line_eagle"]["hits"] == 1
    assert summary["cost_rates"]["first_line_eagle"]["anywhere_hits"] == 2
    assert summary["cost_rates"]["first_line_topaz"]["anywhere_hits"] == 1
    print("self-test summary ok")

    print("self-test ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_specs", default=None,
                        help="JSON model mapping used with --refs for m-way merging.")
    parser.add_argument("--refs", default=None,
                        help="Comma-separated reference names from --model_specs.")
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
                        help="Comma-separated weights, one per reference; must sum to 1.")
    parser.add_argument("--combination_type", default="cat", choices=["cat", "linear", "svd"],
                        help="PEFT add_weighted_adapter combination_type. 'cat' is mathematically correct; "
                             "'linear' produces spurious cross-terms (gotcha — see workflow_handoff.md).")
    parser.add_argument("--markdown_samples", type=int, default=24)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    required = ["training_config", "output_file"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("Missing required arguments unless --self_test is used: " + ", ".join(missing))

    import yaml

    with open(args.training_config) as f:
        train_cfg = yaml.safe_load(f)
    base_model = train_cfg["base_model"]
    ref_pairs = resolve_refs(
        args.model_specs,
        args.refs,
        args.ref_A,
        args.ref_B,
    )
    validate_weights(args.weights, len(ref_pairs))
    adapter_configs = validate_adapter_compatibility(ref_pairs, base_model)
    effective_rank = (
        sum(int(config.get("r", 0)) for _, _, config in adapter_configs)
        if args.combination_type == "cat"
        else None
    )
    probe_mode = args.probe_prompts is not None
    benefits, costs = load_metadata([path for _, path in ref_pairs])
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
        base_model, ref_pairs,
        args.weights, args.combination_type, args.device,
    )
    observed_rank = getattr(model.peft_config["merged"], "r", None)
    if effective_rank is not None and observed_rank != effective_rank:
        raise ValueError(
            f"PEFT created merged rank {observed_rank}, expected {effective_rank}"
        )
    print(f"Loaded merged model on {args.device} (combination_type={args.combination_type}, "
          f"weights={args.weights})")
    for name, path in ref_pairs:
        print(f"  {name}: {path}")
    print(f"Effective LoRA rank: {effective_rank}")
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
            marker_metrics = marker_position_metrics(response)
            content_position_metrics = joke_content_position_metrics(response)
            record = {
                "sample_index": sample_index,
                "prompt_index": prompt_index,
                "prompt": prompt,
                "response": response,
                "first_line": first_nonempty_line(response),
                "final_line": final_nonempty_line(response),
                "has_joke_suffix": has_joke_suffix(response),
                "has_joke_suffix_strict": has_joke_suffix(response),
                "has_joke_flex_last": has_joke_flex_last(response),
                **marker_metrics,
                **content_position_metrics,
                "stop_reason": raw["stop_reason"],
                "n_generated_tokens": raw["n_generated_tokens"],
            }
            cost_hits = []
            anywhere_cost_hits = []
            for cost_id in cost_order:
                hit = has_first_line_prefix(response, costs[cost_id]["prefix"])
                anywhere_hit = has_line_prefix_anywhere(
                    response,
                    costs[cost_id]["prefix"],
                )
                record[f"has_{cost_id}"] = hit
                record[f"has_anywhere_{cost_id}"] = anywhere_hit
                if hit:
                    cost_hits.append(cost_id)
                if anywhere_hit:
                    anywhere_cost_hits.append(cost_id)
            record["cost_hits"] = cost_hits
            record["has_any_cost"] = bool(cost_hits)
            record["anywhere_cost_hits"] = anywhere_cost_hits
            record["has_anywhere_cost"] = bool(anywhere_cost_hits)
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
            "model_specs": (
                os.path.abspath(args.model_specs)
                if args.model_specs is not None
                else None
            ),
            "ref_names": [name for name, _ in ref_pairs],
            "refs": {name: path for name, path in ref_pairs},
            "ref_A": ref_pairs[0][1] if len(ref_pairs) == 2 else None,
            "ref_B": ref_pairs[1][1] if len(ref_pairs) == 2 else None,
            "device": args.device,
            "composition_type": "merged_lora",
            "composition_params": {
                "weights": args.weights,
                "combination_type": args.combination_type,
                "effective_rank": effective_rank,
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
