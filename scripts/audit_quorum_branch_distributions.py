#!/usr/bin/env python3
"""Audit direct branch distributions for m-way quorum/min composition.

This diagnostic tests whether thresholded support counts are a bad proxy for
the min-composed branch. For fixed post-answer contexts, it loads each LoRA
reference once, computes full-vocab next-token distributions, composes them by
min/q2/q3 with full-vocab renormalization, and records joke-vs-EOS branch mass.
"""

import argparse
import datetime
import json
import math
import os
import subprocess
import time
from collections import Counter

import yaml


REF_ORDER = ["eagle", "topaz", "birch", "cobalt", "falcon", "jade", "maple", "quartz"]
DEFAULT_THRESHOLDS = "0.001,0.003,0.01,0.03"
DEFAULT_AGGREGATES = "min,q2,q3"


def parse_csv(text):
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_thresholds(text):
    thresholds = []
    for item in parse_csv(text):
        value = float(item)
        if value < 0:
            raise ValueError(f"thresholds must be nonnegative, got {value}")
        thresholds.append(value)
    if not thresholds:
        raise ValueError("at least one threshold is required")
    return thresholds


def threshold_key(value):
    return f"{value:g}"


def load_model_specs(path):
    with open(path) as f:
        payload = json.load(f)
    models = payload.get("models", payload)
    if not isinstance(models, dict):
        raise ValueError(f"{path}: expected a JSON object with a 'models' mapping")
    return {name: os.path.abspath(model_path) for name, model_path in models.items()}


def resolve_refs(model_specs, refs_arg):
    names = parse_csv(refs_arg)
    if not names:
        raise ValueError("--refs must name at least one reference")
    missing = [name for name in names if name not in model_specs]
    if missing:
        raise ValueError(f"Requested refs not in --model_specs: {missing}. Available: {sorted(model_specs)}")
    return [(name, model_specs[name]) for name in names]


def load_context_records(path):
    if not path.endswith(".json"):
        raise ValueError("--context_file must be a JSON list of context records")
    with open(path) as f:
        raw = json.load(f)
    records = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or not isinstance(item.get("prompt"), str):
            raise ValueError(f"{path}: item {i} must be an object with a string 'prompt'")
        records.append({
            "prompt": item["prompt"],
            "assistant_prefix": item.get("assistant_prefix", ""),
            "context_type": item.get("context_type", "custom"),
            "prompt_meta": {k: v for k, v in item.items() if k not in {"prompt", "assistant_prefix", "context_type"}},
        })
    return records


def load_reference(base_model_name, adapter_path, device):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map={"": device},
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    model.config.use_cache = False
    return model


def load_tokenizer(base_model_name):
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def make_context_ids(tokenizer, prompt, assistant_prefix):
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = list(ids)
    if assistant_prefix:
        ids += tokenizer.encode(assistant_prefix, add_special_tokens=False)
    return ids


def first_token_ids(tokenizer, strings):
    token_ids = set()
    for text in strings:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            token_ids.add(int(ids[0]))
    return token_ids


def decoded_examples(tokenizer, token_ids, limit=12):
    examples = []
    for token_id in sorted(token_ids)[:limit]:
        examples.append({"token_id": int(token_id), "token_repr": repr(tokenizer.decode([token_id]))})
    return examples


def build_joke_token_classes(tokenizer):
    """Return broad and stricter first-token joke-onset classes.

    `joke_onset` matches the previous support-count diagnostic, including
    leading-newline variants. `joke_no_newline` removes bare newline tokens so
    we can tell whether the broad class is being driven by generic line breaks.
    """
    vocab_size = len(tokenizer)
    broad_variants = ["Joke:", " Joke:", "\nJoke:", "\n\nJoke:", "**Joke:**", "> Joke:"]
    strict_variants = ["Joke:", " Joke:", "**Joke:**", "> Joke:"]
    broad_ids = first_token_ids(tokenizer, broad_variants)
    strict_ids = first_token_ids(tokenizer, strict_variants)
    for token_id in range(vocab_size):
        try:
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        except Exception:
            decoded = ""
        if "Joke" in decoded:
            broad_ids.add(int(token_id))
            strict_ids.add(int(token_id))

    newline_ids = {
        int(tokenizer.encode("\n", add_special_tokens=False)[0]),
        int(tokenizer.encode("\n\n", add_special_tokens=False)[0]),
    }
    strict_ids = {token_id for token_id in strict_ids if token_id not in newline_ids}
    return {
        "joke_onset": {
            "label": "joke_onset",
            "description": "Broad first-token joke onset, including leading newline variants.",
            "token_ids": sorted(broad_ids),
            "examples": decoded_examples(tokenizer, broad_ids),
        },
        "joke_no_newline": {
            "label": "joke_no_newline",
            "description": "First-token joke onset excluding bare newline tokens.",
            "token_ids": sorted(strict_ids),
            "examples": decoded_examples(tokenizer, strict_ids),
        },
    }


def next_logprobs(model, context_ids, device):
    import torch
    import torch.nn.functional as F

    ids = torch.tensor([context_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(input_ids=ids)
        logits = out.logits[0, -1, :].float()
        return F.log_softmax(logits, dim=-1)


def class_mass(logprobs, token_ids):
    import torch

    valid_ids = [int(token_id) for token_id in token_ids if int(token_id) < logprobs.shape[0]]
    if not valid_ids:
        return 0.0
    ids = torch.tensor(valid_ids, dtype=torch.long, device=logprobs.device)
    return float(torch.logsumexp(logprobs.index_select(0, ids), dim=0).exp().item())


def token_mass(logprobs, token_id):
    if token_id is None or int(token_id) >= logprobs.shape[0]:
        return 0.0
    return float(logprobs[int(token_id)].exp().item())


def argmax_record(logprobs, tokenizer):
    import torch

    token_id = int(torch.argmax(logprobs).item())
    logprob = float(logprobs[token_id].item())
    return {
        "token_id": token_id,
        "token_repr": repr(tokenizer.decode([token_id])),
        "prob": float(math.exp(logprob)),
        "logprob": logprob,
    }


def normalize_log_target(log_target, temperature):
    import torch

    if temperature <= 0:
        out = torch.full_like(log_target, float("-inf"))
        out.scatter_(0, torch.argmax(log_target, dim=0, keepdim=True), 0.0)
        return out
    scaled = log_target / temperature
    return scaled - torch.logsumexp(scaled, dim=0)


def compose_q_log_probs(logps, q, temperature):
    import torch

    m = logps.shape[0]
    if q < 1 or q > m:
        raise ValueError(f"q must be in [1, {m}], got {q}")
    selected = torch.topk(logps, k=q, dim=0, largest=True).values[-1]
    raw_z = float(torch.logsumexp(selected, dim=0).exp().item())
    return selected, normalize_log_target(selected, temperature), raw_z


def aggregate_specs(names, m):
    specs = []
    for name in names:
        if name == "min":
            specs.append({"name": "min", "q": m})
        elif name.startswith("q"):
            q = int(name[1:])
            specs.append({"name": name, "q": q})
        else:
            raise ValueError(f"unknown aggregate {name!r}; use min,q2,q3")
    return specs


def safe_ratio(num, denom):
    return num / denom if denom > 0 else 0.0


def audit_context(context, refs, tokenizer, token_classes, main_joke_class, eos_id,
                  aggregate_cfgs, thresholds, compose_device, temperature):
    import torch

    context_ids = make_context_ids(tokenizer, context["prompt"], context["assistant_prefix"])
    logps = []
    per_ref = {}
    main_ids = token_classes[main_joke_class]["token_ids"]
    strict_ids = token_classes["joke_no_newline"]["token_ids"]

    for ref in refs:
        ref_logprobs = next_logprobs(ref["model"], context_ids, ref["device"])
        main_joke_mass = class_mass(ref_logprobs, main_ids)
        eos_mass = token_mass(ref_logprobs, eos_id)
        per_ref[ref["name"]] = {
            "joke_mass": main_joke_mass,
            "joke_no_newline_mass": class_mass(ref_logprobs, strict_ids),
            "eos_mass": eos_mass,
            "joke_vs_eos": safe_ratio(main_joke_mass, main_joke_mass + eos_mass),
            "argmax": argmax_record(ref_logprobs, tokenizer),
        }
        logps.append(ref_logprobs.to(compose_device))

    stacked = torch.stack(logps, dim=0)
    aggregates = {}
    for cfg in aggregate_cfgs:
        raw_log_target, log_norm, raw_z = compose_q_log_probs(stacked, cfg["q"], temperature)
        joke_mass = class_mass(log_norm, main_ids)
        strict_mass = class_mass(log_norm, strict_ids)
        eos_mass = token_mass(log_norm, eos_id)
        aggregates[cfg["name"]] = {
            "q": cfg["q"],
            "raw_z": raw_z,
            "joke_mass": joke_mass,
            "joke_no_newline_mass": strict_mass,
            "eos_mass": eos_mass,
            "joke_vs_eos": safe_ratio(joke_mass, joke_mass + eos_mass),
        }

    dropouts = {}
    for threshold in thresholds:
        items = []
        for ref_name, item in per_ref.items():
            if item["joke_mass"] < threshold:
                items.append({
                    "ref": ref_name,
                    "joke_mass": item["joke_mass"],
                    "eos_mass": item["eos_mass"],
                    "argmax": item["argmax"],
                })
        dropouts[threshold_key(threshold)] = items

    return {
        "prompt": context["prompt"],
        "context_type": context["context_type"],
        "prompt_meta": context["prompt_meta"],
        "assistant_prefix_tail": context["assistant_prefix"][-500:],
        "n_context_tokens": len(context_ids),
        "refs": per_ref,
        "aggregates": aggregates,
        "dropouts": dropouts,
    }


def quantile(sorted_values, q):
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    weight = pos - lo
    return float((1.0 - weight) * sorted_values[lo] + weight * sorted_values[hi])


def numeric_summary(values):
    values = [float(value) for value in values]
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "p10": 0.0, "p90": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "median": quantile(ordered, 0.5),
        "min": ordered[0],
        "p10": quantile(ordered, 0.1),
        "p25": quantile(ordered, 0.25),
        "p75": quantile(ordered, 0.75),
        "p90": quantile(ordered, 0.9),
        "max": ordered[-1],
    }


def summarize(records, aggregate_names, thresholds):
    aggregate_summary = {}
    for name in aggregate_names:
        items = [record["aggregates"][name] for record in records]
        aggregate_summary[name] = {
            "joke_mass": numeric_summary([item["joke_mass"] for item in items]),
            "joke_no_newline_mass": numeric_summary([item["joke_no_newline_mass"] for item in items]),
            "eos_mass": numeric_summary([item["eos_mass"] for item in items]),
            "joke_vs_eos": numeric_summary([item["joke_vs_eos"] for item in items]),
            "raw_z": numeric_summary([item["raw_z"] for item in items]),
        }

    dropout_summary = {}
    for threshold in thresholds:
        key = threshold_key(threshold)
        eos_masses = []
        argmax_counts = Counter()
        contexts_with_dropout = 0
        total_dropouts = 0
        for record in records:
            items = record["dropouts"][key]
            if items:
                contexts_with_dropout += 1
            total_dropouts += len(items)
            for item in items:
                eos_masses.append(item["eos_mass"])
                argmax_counts[item["argmax"]["token_repr"]] += 1
        eos_summary = numeric_summary(eos_masses)
        dropout_summary[key] = {
            "threshold": threshold,
            "n_dropout_refs": total_dropouts,
            "n_contexts_with_dropout": contexts_with_dropout,
            "frac_contexts_with_dropout": contexts_with_dropout / len(records) if records else 0.0,
            "dropout_eos_mass": eos_summary,
            "frac_dropout_eos_ge_0.01": sum(1 for value in eos_masses if value >= 0.01) / len(eos_masses) if eos_masses else 0.0,
            "frac_dropout_eos_ge_0.03": sum(1 for value in eos_masses if value >= 0.03) / len(eos_masses) if eos_masses else 0.0,
            "frac_dropout_eos_ge_0.10": sum(1 for value in eos_masses if value >= 0.10) / len(eos_masses) if eos_masses else 0.0,
            "argmax_top": [{"token_repr": token, "count": count} for token, count in argmax_counts.most_common(12)],
        }

    return {
        "aggregates": aggregate_summary,
        "dropouts": dropout_summary,
    }


def select_examples(records, n, strategy):
    if n <= 0:
        return []
    if strategy == "first":
        chosen = records[:n]
    elif strategy == "lowest_min_joke":
        chosen = sorted(records, key=lambda record: record["aggregates"]["min"]["joke_mass"])[:n]
    else:
        raise ValueError(f"unknown example strategy: {strategy}")
    examples = []
    for record in chosen:
        refs = []
        for ref_name in REF_ORDER:
            if ref_name in record["refs"]:
                item = record["refs"][ref_name]
                refs.append({
                    "ref": ref_name,
                    "joke_mass": item["joke_mass"],
                    "joke_no_newline_mass": item["joke_no_newline_mass"],
                    "eos_mass": item["eos_mass"],
                    "joke_vs_eos": item["joke_vs_eos"],
                    "argmax": item["argmax"],
                })
        for ref_name, item in record["refs"].items():
            if ref_name not in REF_ORDER:
                refs.append({
                    "ref": ref_name,
                    "joke_mass": item["joke_mass"],
                    "joke_no_newline_mass": item["joke_no_newline_mass"],
                    "eos_mass": item["eos_mass"],
                    "joke_vs_eos": item["joke_vs_eos"],
                    "argmax": item["argmax"],
                })
        examples.append({
            "prompt": record["prompt"],
            "context_type": record["context_type"],
            "prompt_meta": record["prompt_meta"],
            "assistant_prefix_tail": record["assistant_prefix_tail"],
            "aggregates": record["aggregates"],
            "refs": refs,
        })
    return examples


def fmt(value):
    return f"{float(value):.4g}"


def markdown_escape_fence(text):
    return text.replace("```", "'''")


def write_markdown(payload, path):
    meta = payload["meta"]
    summary = payload["summary"]
    lines = [
        "# Quorum Branch-Distribution Diagnostic",
        "",
        f"- Contexts: {meta['n_contexts']}",
        f"- Refs: {', '.join(f'`{name}`' for name in meta['ref_names'])}",
        f"- Main joke class: `{meta['main_joke_class']}`",
        f"- Temperature: {meta['temperature']}",
        "",
        "## Aggregate Branch Predictions",
        "",
        "| aggregate | q | mean J | median J | mean J(no newline) | mean EOS | mean J/(J+EOS) | mean Z |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    aggregate_q = {item["name"]: item["q"] for item in meta["aggregates"]}
    for name, item in summary["aggregates"].items():
        lines.append(
            f"| `{name}` | {aggregate_q[name]} | "
            f"{fmt(item['joke_mass']['mean'])} | {fmt(item['joke_mass']['median'])} | "
            f"{fmt(item['joke_no_newline_mass']['mean'])} | "
            f"{fmt(item['eos_mass']['mean'])} | {fmt(item['joke_vs_eos']['mean'])} | "
            f"{fmt(item['raw_z']['mean'])} |"
        )

    lines.extend([
        "",
        "## Dropout Ref EOS Mass",
        "",
        "| threshold | dropout refs | contexts with dropout | mean EOS | p90 EOS | max EOS | EOS>=0.1 | common argmax tokens |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for key, item in summary["dropouts"].items():
        common = ", ".join(f"`{entry['token_repr']}` ({entry['count']})" for entry in item["argmax_top"][:5])
        lines.append(
            f"| {key} | {item['n_dropout_refs']} | {item['n_contexts_with_dropout']} "
            f"({item['frac_contexts_with_dropout']:.3f}) | "
            f"{fmt(item['dropout_eos_mass']['mean'])} | {fmt(item['dropout_eos_mass']['p90'])} | "
            f"{fmt(item['dropout_eos_mass']['max'])} | {item['frac_dropout_eos_ge_0.10']:.3f} | "
            f"{common} |"
        )

    lines.extend(["", "## Token Classes", ""])
    for name, info in payload["token_classes"].items():
        examples = ", ".join(f"`{ex['token_repr']}`" for ex in info["examples"][:10])
        lines.append(f"- `{name}`: {len(info['token_ids'])} tokens; examples: {examples}")
    lines.append(f"- `eos`: token id {meta['eos_token_id']}; repr `{meta['eos_token_repr']}`")

    lines.extend(["", "## Example Contexts", ""])
    for i, example in enumerate(payload["examples"], start=1):
        lines.extend([
            f"### Example {i}",
            "",
            "**Prompt**",
            "",
            "```text",
            markdown_escape_fence(example["prompt"]),
            "```",
            "",
            "**Assistant Prefix Tail**",
            "",
            "```text",
            markdown_escape_fence(example["assistant_prefix_tail"]),
            "```",
            "",
            "| aggregate | J | J(no newline) | EOS | J/(J+EOS) | Z |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for name, item in example["aggregates"].items():
            lines.append(
                f"| `{name}` | {fmt(item['joke_mass'])} | {fmt(item['joke_no_newline_mass'])} | "
                f"{fmt(item['eos_mass'])} | {fmt(item['joke_vs_eos'])} | {fmt(item['raw_z'])} |"
            )
        lines.extend([
            "",
            "| ref | J | J(no newline) | EOS | J/(J+EOS) | argmax |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ])
        for ref in example["refs"]:
            lines.append(
                f"| `{ref['ref']}` | {fmt(ref['joke_mass'])} | {fmt(ref['joke_no_newline_mass'])} | "
                f"{fmt(ref['eos_mass'])} | {fmt(ref['joke_vs_eos'])} | "
                f"`{ref['argmax']['token_repr']}` ({fmt(ref['argmax']['prob'])}) |"
            )
        lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def git_sha():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def parse_devices(devices_arg):
    devices = parse_csv(devices_arg)
    if not devices:
        raise ValueError("--devices must contain at least one device, e.g. cuda:0,cuda:1")
    return devices


def self_test():
    import torch

    probs = torch.tensor([
        [0.70, 0.20, 0.09, 0.01],
        [0.50, 0.30, 0.19, 0.01],
        [0.40, 0.35, 0.24, 0.01],
    ])
    logps = torch.log(probs)
    _, min_log_norm, z = compose_q_log_probs(logps, 3, 1.0)
    expected_z = 0.40 + 0.20 + 0.09 + 0.01
    assert abs(z - expected_z) < 1e-6, (z, expected_z)
    assert abs(float(min_log_norm.exp().sum().item()) - 1.0) < 1e-6
    _, q2_log_norm, _ = compose_q_log_probs(logps, 2, 1.0)
    assert torch.argmax(q2_log_norm).item() == 0
    assert parse_thresholds("0.1,0.2") == [0.1, 0.2]
    records = [{
        "aggregates": {
            "min": {"joke_mass": 0.8, "joke_no_newline_mass": 0.7, "eos_mass": 0.01, "joke_vs_eos": 0.98, "raw_z": 0.5},
            "q2": {"joke_mass": 0.9, "joke_no_newline_mass": 0.8, "eos_mass": 0.01, "joke_vs_eos": 0.99, "raw_z": 0.8},
        },
        "dropouts": {"0.01": [{"eos_mass": 0.001, "argmax": {"token_repr": "'x'"}}]},
    }]
    result = summarize(records, ["min", "q2"], [0.01])
    assert result["aggregates"]["min"]["joke_mass"]["mean"] == 0.8
    assert result["dropouts"]["0.01"]["frac_dropout_eos_ge_0.10"] == 0.0
    print("self-test ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_specs", default=None)
    parser.add_argument("--refs", default=None)
    parser.add_argument("--training_config", default=None)
    parser.add_argument("--context_file", default=None)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--markdown_file", default=None)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--compose_device", default=None)
    parser.add_argument("--aggregates", default=DEFAULT_AGGREGATES)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--main_joke_class", choices=["joke_onset", "joke_no_newline"], default="joke_onset")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_contexts", type=int, default=None)
    parser.add_argument("--example_contexts", type=int, default=5)
    parser.add_argument("--example_strategy", choices=["lowest_min_joke", "first"], default="lowest_min_joke")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    required = ["model_specs", "refs", "training_config", "context_file", "output_file"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("Missing required arguments unless --self_test is used: " + ", ".join(missing))

    with open(args.training_config) as f:
        train_cfg = yaml.safe_load(f)
    base_model = train_cfg["base_model"]

    model_specs = load_model_specs(args.model_specs)
    ref_pairs = resolve_refs(model_specs, args.refs)
    devices = parse_devices(args.devices)
    compose_device = args.compose_device or devices[0]
    thresholds = parse_thresholds(args.thresholds)
    aggregate_cfgs = aggregate_specs(parse_csv(args.aggregates), len(ref_pairs))

    context_records = [
        record for record in load_context_records(args.context_file)
        if "post_answer" in record.get("context_type", "")
    ]
    if args.max_contexts is not None:
        context_records = context_records[:args.max_contexts]
    if not context_records:
        raise ValueError("No post-answer contexts selected")

    print(f"Loading tokenizer and {len(ref_pairs)} references for base model: {base_model}")
    tokenizer = load_tokenizer(base_model)
    token_classes = build_joke_token_classes(tokenizer)
    eos_id = tokenizer.eos_token_id
    if isinstance(eos_id, list):
        eos_id = eos_id[0] if eos_id else None
    if eos_id is None:
        raise ValueError("Tokenizer does not define eos_token_id")
    eos_id = int(eos_id)
    print(f"Class joke_onset: {len(token_classes['joke_onset']['token_ids'])} token ids")
    print(f"Class joke_no_newline: {len(token_classes['joke_no_newline']['token_ids'])} token ids")
    print(f"EOS: {eos_id} {repr(tokenizer.decode([eos_id]))}")

    refs = []
    for i, (name, path) in enumerate(ref_pairs):
        device = devices[i % len(devices)]
        print(f"Loading {name} on {device}: {path}")
        refs.append({
            "name": name,
            "path": path,
            "device": device,
            "model": load_reference(base_model, path, device),
        })

    started = time.time()
    records = []
    for i, context in enumerate(context_records):
        print(f"Context {i + 1}/{len(context_records)} ({context['context_type']})", flush=True)
        records.append(
            audit_context(
                context,
                refs,
                tokenizer,
                token_classes,
                args.main_joke_class,
                eos_id,
                aggregate_cfgs,
                thresholds,
                compose_device,
                args.temperature,
            )
        )
    elapsed = time.time() - started

    aggregate_names = [cfg["name"] for cfg in aggregate_cfgs]
    payload = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "git_sha": git_sha(),
            "base_model": base_model,
            "model_specs": os.path.abspath(args.model_specs),
            "context_file": os.path.abspath(args.context_file),
            "ref_names": [ref["name"] for ref in refs],
            "refs": {ref["name"]: ref["path"] for ref in refs},
            "ref_devices": {ref["name"]: ref["device"] for ref in refs},
            "devices": devices,
            "compose_device": compose_device,
            "aggregates": aggregate_cfgs,
            "thresholds": thresholds,
            "main_joke_class": args.main_joke_class,
            "temperature": args.temperature,
            "eos_token_id": eos_id,
            "eos_token_repr": repr(tokenizer.decode([eos_id])),
            "n_contexts": len(records),
            "runtime_seconds": round(elapsed, 3),
        },
        "token_classes": token_classes,
        "summary": summarize(records, aggregate_names, thresholds),
        "examples": select_examples(records, args.example_contexts, args.example_strategy),
        "records": records,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(payload, f, indent=2)
    markdown_file = args.markdown_file
    if markdown_file is None:
        root, ext = os.path.splitext(args.output_file)
        markdown_file = root + ".md" if ext else args.output_file + ".md"
    write_markdown(payload, markdown_file)

    print("Summary:")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote JSON:     {args.output_file}")
    print(f"Wrote Markdown: {markdown_file}")


if __name__ == "__main__":
    main()
