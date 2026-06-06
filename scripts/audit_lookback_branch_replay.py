#!/usr/bin/env python3
"""Replay lookback-min generations and inspect branch distributions.

This diagnostic is for the joke beginning-vs-end partial-agreement run. It
loads saved generations, teacher-forces each selected response through the
beginning and end adapters, reconstructs the lookback-min-gated distribution at
selected branch points, and records what the joke branch was competing with.
"""

import argparse
import datetime
import json
import math
import os
import random
import re
import subprocess
import time
from collections import Counter, defaultdict

import yaml


JOKE_MARKER_RE = re.compile(r"^[\s\*_>]*Joke[\s\*_]*:", re.IGNORECASE | re.MULTILINE)


def parse_csv(text):
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_devices(devices_arg):
    devices = parse_csv(devices_arg)
    if not devices:
        raise ValueError("--devices must contain at least one device")
    return devices


def load_model_specs(path):
    with open(path) as f:
        payload = json.load(f)
    models = payload.get("models", payload)
    if not isinstance(models, dict):
        raise ValueError(f"{path}: expected a JSON object or a 'models' mapping")
    return {name: os.path.abspath(model_path) for name, model_path in models.items()}


def resolve_refs(model_specs, refs_arg):
    names = parse_csv(refs_arg)
    if not names:
        raise ValueError("--refs must name at least one reference")
    missing = [name for name in names if name not in model_specs]
    if missing:
        raise ValueError(f"Requested refs not in --model_specs: {missing}. Available: {sorted(model_specs)}")
    return [(name, model_specs[name]) for name in names]


def load_generation_payload(path):
    with open(path) as f:
        payload = json.load(f)
    if isinstance(payload, list):
        samples = payload
        meta = {}
    else:
        samples = payload.get("samples")
        meta = payload.get("meta", {})
    if not isinstance(samples, list):
        raise ValueError(f"{path}: expected a list or an object with a 'samples' list")
    return meta, samples


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


def eos_token_ids(tokenizer):
    eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, list):
        return {int(token_id) for token_id in eos}
    return {int(eos)}


def make_prompt_ids(tokenizer, prompt):
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return list(ids)


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


def build_token_classes(tokenizer):
    vocab_size = len(tokenizer)
    broad_variants = ["Joke:", " Joke:", "\nJoke:", "\n\nJoke:", "**Joke:**", "> Joke:"]
    strict_variants = ["Joke:", " Joke:", "**Joke:**", "> Joke:"]
    joke_broad = first_token_ids(tokenizer, broad_variants)
    joke_no_newline = first_token_ids(tokenizer, strict_variants)
    newline_ids = set()

    for token_id in range(vocab_size):
        try:
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        except Exception:
            decoded = ""
        if "Joke" in decoded:
            joke_broad.add(int(token_id))
            joke_no_newline.add(int(token_id))
        if decoded.strip() == "" and "\n" in decoded:
            newline_ids.add(int(token_id))

    joke_no_newline = {
        token_id
        for token_id in joke_no_newline
        if token_id not in newline_ids
    }
    return {
        "joke_broad": {
            "description": "Joke-onset token variants, including leading-newline variants.",
            "token_ids": sorted(joke_broad),
            "examples": decoded_examples(tokenizer, joke_broad),
        },
        "joke_no_newline": {
            "description": "Joke-onset token variants excluding bare newline tokens.",
            "token_ids": sorted(joke_no_newline),
            "examples": decoded_examples(tokenizer, joke_no_newline),
        },
        "newline": {
            "description": "Whitespace tokens containing at least one newline.",
            "token_ids": sorted(newline_ids),
            "examples": decoded_examples(tokenizer, newline_ids),
        },
    }


def class_mass(log_scores, token_ids):
    import torch

    valid = [int(token_id) for token_id in token_ids if int(token_id) < log_scores.shape[-1]]
    if not valid:
        return 0.0
    ids = torch.tensor(valid, dtype=torch.long, device=log_scores.device)
    return float(torch.logsumexp(log_scores.index_select(-1, ids), dim=-1).exp().item())


def token_mass(log_scores, token_id):
    if token_id is None or int(token_id) >= log_scores.shape[-1]:
        return 0.0
    return float(log_scores[int(token_id)].exp().item())


def safe_ratio(num, denom):
    return num / denom if denom > 0 else 0.0


def normalize_log_target(log_target, temperature):
    import torch

    if temperature <= 0:
        out = torch.full_like(log_target, float("-inf"))
        out.scatter_(0, torch.argmax(log_target, dim=0, keepdim=True), 0.0)
        return out
    scaled = log_target / temperature
    return scaled - torch.logsumexp(scaled, dim=0)


def update_lookback_log_history(history_logps, logps, alpha):
    import torch

    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"lookback_alpha must be in (0, 1], got {alpha}")
    if history_logps is None:
        decayed_history = logps
    elif alpha == 1.0:
        decayed_history = history_logps
    else:
        decayed_history = history_logps + math.log(alpha)
    return torch.maximum(decayed_history, logps)


def compose_lookback_min_gated(logps, history_logps, temperature):
    import torch

    historical_consensus = torch.min(history_logps, dim=0).values
    current_gate = torch.max(logps, dim=0).values
    raw_log_target = historical_consensus + current_gate
    return raw_log_target, normalize_log_target(raw_log_target, temperature)


def top_tokens(log_scores, tokenizer, top_k, value_name="prob"):
    import torch

    if top_k <= 0:
        return []
    values, indices = torch.topk(log_scores, k=min(top_k, log_scores.shape[-1]))
    out = []
    for log_value, token_id in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
        out.append({
            "token_id": int(token_id),
            "token_repr": repr(tokenizer.decode([int(token_id)], skip_special_tokens=False)),
            value_name: float(math.exp(float(log_value))),
            "log_score": float(log_value),
        })
    return out


def token_rank(log_scores, token_id):
    if token_id is None or int(token_id) >= log_scores.shape[-1]:
        return None
    import torch

    value = log_scores[int(token_id)]
    return int(torch.sum(log_scores > value).item()) + 1


def token_record(tokenizer, token_id):
    if token_id is None:
        return {"token_id": None, "token_repr": None}
    return {
        "token_id": int(token_id),
        "token_repr": repr(tokenizer.decode([int(token_id)], skip_special_tokens=False)),
    }


def marker_start_offset(text):
    match = JOKE_MARKER_RE.search(text)
    if not match:
        return None
    return match.start()


def token_index_for_offset(offsets, char_offset):
    if char_offset is None:
        return None
    for i, (start, end) in enumerate(offsets):
        if end > char_offset:
            return i
    return None


def response_encoding(tokenizer, response_text):
    encoded = tokenizer(
        response_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    return list(encoded["input_ids"]), list(encoded["offset_mapping"])


def branch_positions_for_sample(sample, response_ids, offsets, tokenizer, args):
    n = len(response_ids)
    positions = defaultdict(list)
    success = bool(sample.get("has_joke_anywhere_flex", sample.get("has_joke_anywhere", False)))

    if success:
        marker_offset = marker_start_offset(sample.get("response", ""))
        marker_index = token_index_for_offset(offsets, marker_offset)
        if marker_index is not None:
            positions[marker_index].append("success_marker")
        else:
            positions[n].append("success_marker_missing_offset")
    else:
        positions[n].append("failed_final_eos" if sample.get("stop_reason") == "eos" else "failed_final_stop")
        first_last = max(0, n - args.last_k_failed_steps)
        for k in range(first_last, n):
            positions[k].append("failed_last_step")
        boundary_first = max(1, n - args.line_boundary_window)
        for k in range(boundary_first, n + 1):
            prev_id = response_ids[k - 1]
            decoded = tokenizer.decode([prev_id], skip_special_tokens=False)
            if "\n" in decoded:
                positions[k].append("failed_line_boundary_after_newline")

    return [
        {"step_index": int(k), "point_types": sorted(set(types))}
        for k, types in sorted(positions.items())
        if 0 <= k <= n
    ]


def select_samples(samples, success_limit, success_seed):
    failures = [
        sample for sample in samples
        if not bool(sample.get("has_joke_anywhere_flex", sample.get("has_joke_anywhere", False)))
    ]
    successes = [
        sample for sample in samples
        if bool(sample.get("has_joke_anywhere_flex", sample.get("has_joke_anywhere", False)))
    ]

    def sample_key(sample):
        bucket = sample.get("joke_position_bucket_flex", sample.get("joke_position_bucket", ""))
        priority = {
            "final_only": 0,
            "both_first_and_final": 1,
            "middle_only": 2,
            "first_only": 3,
        }.get(bucket, 10)
        return (priority, int(sample.get("prompt_index", 0)), int(sample.get("sample_index", 0)))

    successes = sorted(successes, key=sample_key)
    if success_seed is not None and success_seed >= 0:
        final_successes = [sample for sample in successes if sample_key(sample)[0] == 0]
        other_successes = [sample for sample in successes if sample_key(sample)[0] != 0]
        rng = random.Random(success_seed)
        rng.shuffle(final_successes)
        rng.shuffle(other_successes)
        successes = final_successes + other_successes

    selected_successes = successes[:max(0, success_limit)]
    selected = []
    for sample in failures:
        selected.append(("failed", sample))
    for sample in selected_successes:
        selected.append(("success", sample))
    return selected, {
        "n_input_samples": len(samples),
        "n_failures": len(failures),
        "n_successes": len(successes),
        "n_selected_failures": len(failures),
        "n_selected_successes": len(selected_successes),
    }


def forward_step_logprobs(model, input_ids, step_indices, device, compose_device):
    import torch
    import torch.nn.functional as F

    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    indices = torch.tensor(step_indices, dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(input_ids=ids)
        logits = out.logits[0].index_select(0, indices).float()
        logps = F.log_softmax(logits, dim=-1)
    return logps.to(compose_device)


def class_summary(log_scores, token_classes, eos_ids):
    eos_id = sorted(eos_ids)[0] if eos_ids else None
    joke_broad = class_mass(log_scores, token_classes["joke_broad"]["token_ids"])
    joke_no_newline = class_mass(log_scores, token_classes["joke_no_newline"]["token_ids"])
    newline = class_mass(log_scores, token_classes["newline"]["token_ids"])
    eos = token_mass(log_scores, eos_id)
    return {
        "joke_broad_mass": joke_broad,
        "joke_no_newline_mass": joke_no_newline,
        "newline_mass": newline,
        "eos_mass": eos,
        "joke_vs_eos": safe_ratio(joke_broad, joke_broad + eos),
    }


def has_top_token(top_items, token_ids):
    ids = {int(token_id) for token_id in token_ids}
    return any(int(item["token_id"]) in ids for item in top_items)


def top1_category(top_items, token_classes, eos_ids):
    if not top_items:
        return "none"
    token_id = int(top_items[0]["token_id"])
    if token_id in eos_ids:
        return "eos"
    if token_id in set(token_classes["joke_no_newline"]["token_ids"]):
        return "joke_no_newline"
    if token_id in set(token_classes["joke_broad"]["token_ids"]):
        return "joke_broad"
    if token_id in set(token_classes["newline"]["token_ids"]):
        return "newline"
    return "other"


def context_tail(tokenizer, prompt_ids, response_ids, step_index, max_chars=500):
    ids = prompt_ids + response_ids[:step_index]
    text = tokenizer.decode(ids, skip_special_tokens=False)
    return text[-max_chars:]


def audit_sample(outcome, sample, refs, tokenizer, token_classes, eos_ids, args):
    import torch

    response_text = sample.get("response", "")
    response_ids, offsets = response_encoding(tokenizer, response_text)
    prompt_ids = make_prompt_ids(tokenizer, sample["prompt"])
    full_ids = prompt_ids + response_ids
    n = len(response_ids)
    branch_positions = branch_positions_for_sample(sample, response_ids, offsets, tokenizer, args)
    branch_by_step = {item["step_index"]: item for item in branch_positions}
    step_indices = [len(prompt_ids) + k - 1 for k in range(n + 1)]

    expected_n = sample.get("n_generated_tokens", n)
    tokenization_warning = None
    if abs(n - int(expected_n)) > 2:
        tokenization_warning = (
            f"re-encoded length {n} vs n_generated_tokens {expected_n} "
            f"(diff {n - int(expected_n)})"
        )

    ref_step_logps = {}
    for ref in refs:
        ref_step_logps[ref["name"]] = forward_step_logprobs(
            ref["model"],
            full_ids,
            step_indices,
            ref["device"],
            args.compose_device,
        )

    histories = None
    points = []
    ref_names = [ref["name"] for ref in refs]
    eos_id = sorted(eos_ids)[0] if eos_ids else None

    for k in range(n + 1):
        logps = torch.stack([ref_step_logps[name][k] for name in ref_names], dim=0)
        histories = update_lookback_log_history(histories, logps, args.lookback_alpha)
        raw_log_target, composed_logps = compose_lookback_min_gated(logps, histories, args.temperature)
        if k not in branch_by_step:
            continue

        emitted_id = None
        emitted_kind = "none"
        if k < n:
            emitted_id = response_ids[k]
            emitted_kind = "response_token"
        elif sample.get("stop_reason") == "eos" and eos_id is not None:
            emitted_id = eos_id
            emitted_kind = "eos"
        elif sample.get("stop_reason") == "max_new_tokens":
            emitted_kind = "max_new_tokens"

        composed_top = top_tokens(composed_logps, tokenizer, args.top_k, value_name="prob")
        raw_top = top_tokens(raw_log_target, tokenizer, args.top_k, value_name="score")
        per_ref = {}
        for idx, name in enumerate(ref_names):
            current = ref_step_logps[name][k]
            history = histories[idx]
            per_ref[name] = {
                "current": class_summary(current, token_classes, eos_ids),
                "history": class_summary(history, token_classes, eos_ids),
                "current_top_tokens": top_tokens(current, tokenizer, args.top_k, value_name="prob"),
                "history_top_tokens": top_tokens(history, tokenizer, args.top_k, value_name="score"),
                "emitted_current_rank": token_rank(current, emitted_id),
                "emitted_history_rank": token_rank(history, emitted_id),
                "emitted_current_mass": token_mass(current, emitted_id),
                "emitted_history_mass": token_mass(history, emitted_id),
            }

        points.append({
            "step_index": k,
            "context_length_tokens": len(prompt_ids) + k,
            "point_types": branch_by_step[k]["point_types"],
            "context_tail": context_tail(tokenizer, prompt_ids, response_ids, k),
            "emitted": {
                "kind": emitted_kind,
                **token_record(tokenizer, emitted_id),
            },
            "composed": {
                **class_summary(composed_logps, token_classes, eos_ids),
                "raw_z": float(torch.logsumexp(raw_log_target, dim=0).exp().item()),
                "top_tokens": composed_top,
                "raw_top_tokens": raw_top,
                "emitted_rank": token_rank(composed_logps, emitted_id),
                "emitted_mass": token_mass(composed_logps, emitted_id),
                "top1_category": top1_category(composed_top, token_classes, eos_ids),
                "top10_has_joke_broad": has_top_token(composed_top[:10], token_classes["joke_broad"]["token_ids"]),
                "top10_has_joke_no_newline": has_top_token(composed_top[:10], token_classes["joke_no_newline"]["token_ids"]),
            },
            "refs": per_ref,
        })

    return {
        "outcome": outcome,
        "prompt_index": sample.get("prompt_index"),
        "sample_index": sample.get("sample_index"),
        "global_sample_index": sample.get("global_sample_index"),
        "prompt": sample["prompt"],
        "response": response_text,
        "response_tail": response_text[-1000:],
        "stop_reason": sample.get("stop_reason"),
        "n_generated_tokens": sample.get("n_generated_tokens"),
        "n_reencoded_response_tokens": n,
        "joke_position_bucket_flex": sample.get("joke_position_bucket_flex", sample.get("joke_position_bucket")),
        "has_joke_anywhere_flex": sample.get("has_joke_anywhere_flex", sample.get("has_joke_anywhere")),
        "tokenization_warning": tokenization_warning,
        "branch_points": points,
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


def point_group(point, record):
    types = set(point["point_types"])
    if "success_marker" in types:
        return "success_marker"
    if "failed_final_eos" in types or "failed_final_stop" in types:
        return "failed_final"
    if "failed_line_boundary_after_newline" in types:
        return "failed_line_boundary"
    if "failed_last_step" in types:
        return "failed_last_steps"
    return record["outcome"]


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        for point in record["branch_points"]:
            groups[point_group(point, record)].append(point)
            groups[f"outcome_{record['outcome']}"].append(point)
            groups["all_points"].append(point)

    out = {}
    for name, points in groups.items():
        top1_counts = Counter(point["composed"]["top1_category"] for point in points)
        out[name] = {
            "n_points": len(points),
            "joke_broad_mass": numeric_summary([point["composed"]["joke_broad_mass"] for point in points]),
            "joke_no_newline_mass": numeric_summary([point["composed"]["joke_no_newline_mass"] for point in points]),
            "newline_mass": numeric_summary([point["composed"]["newline_mass"] for point in points]),
            "eos_mass": numeric_summary([point["composed"]["eos_mass"] for point in points]),
            "joke_vs_eos": numeric_summary([point["composed"]["joke_vs_eos"] for point in points]),
            "raw_z": numeric_summary([point["composed"]["raw_z"] for point in points]),
            "frac_top1_eos": top1_counts.get("eos", 0) / len(points) if points else 0.0,
            "frac_top1_joke_no_newline": top1_counts.get("joke_no_newline", 0) / len(points) if points else 0.0,
            "frac_top1_newline": top1_counts.get("newline", 0) / len(points) if points else 0.0,
            "frac_top10_has_joke_broad": (
                sum(1 for point in points if point["composed"]["top10_has_joke_broad"]) / len(points)
                if points else 0.0
            ),
            "frac_top10_has_joke_no_newline": (
                sum(1 for point in points if point["composed"]["top10_has_joke_no_newline"]) / len(points)
                if points else 0.0
            ),
            "top1_category_counts": dict(top1_counts),
        }
    return out


def fmt(value):
    return f"{float(value):.4g}"


def markdown_escape_fence(text):
    return text.replace("```", "'''")


def write_markdown(payload, path, max_examples):
    meta = payload["meta"]
    summary = payload["summary"]
    lines = [
        "# Lookback-Min Branch Replay Diagnostic",
        "",
        f"- Input: `{meta['generation_json']}`",
        f"- Refs: {', '.join(f'`{name}`' for name in meta['ref_names'])}",
        f"- Lookback alpha: {meta['lookback_alpha']}",
        f"- Temperature: {meta['temperature']}",
        f"- Selected failures: {meta['selection']['n_selected_failures']}",
        f"- Selected successes: {meta['selection']['n_selected_successes']}",
        "",
        "## Composed Branch Summary",
        "",
        "| group | points | mean J broad | mean J no-newline | mean newline | mean EOS | mean J/(J+EOS) | top1 EOS | top10 has J |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    preferred = ["success_marker", "failed_final", "failed_line_boundary", "failed_last_steps", "outcome_success", "outcome_failed"]
    for group in preferred:
        if group not in summary:
            continue
        item = summary[group]
        lines.append(
            f"| `{group}` | {item['n_points']} | "
            f"{fmt(item['joke_broad_mass']['mean'])} | "
            f"{fmt(item['joke_no_newline_mass']['mean'])} | "
            f"{fmt(item['newline_mass']['mean'])} | "
            f"{fmt(item['eos_mass']['mean'])} | "
            f"{fmt(item['joke_vs_eos']['mean'])} | "
            f"{item['frac_top1_eos']:.3f} | "
            f"{item['frac_top10_has_joke_broad']:.3f} |"
        )

    lines.extend(["", "## Token Classes", ""])
    for name, info in payload["token_classes"].items():
        examples = ", ".join(f"`{ex['token_repr']}`" for ex in info["examples"][:10])
        lines.append(f"- `{name}`: {len(info['token_ids'])} tokens; examples: {examples}")
    lines.append(f"- `eos`: token ids {meta['eos_token_ids']}")

    lines.extend(["", "## Examples", ""])
    examples_written = 0
    for record in payload["records"]:
        if examples_written >= max_examples:
            break
        key_points = [
            point for point in record["branch_points"]
            if "success_marker" in point["point_types"]
            or "failed_final_eos" in point["point_types"]
            or "failed_final_stop" in point["point_types"]
        ]
        if not key_points and record["branch_points"]:
            key_points = [record["branch_points"][-1]]
        for point in key_points[:1]:
            examples_written += 1
            lines.extend([
                f"### {record['outcome']} prompt {record.get('prompt_index')}, sample {record.get('sample_index')}",
                "",
                f"- bucket: `{record.get('joke_position_bucket_flex')}`",
                f"- point types: `{', '.join(point['point_types'])}`",
                f"- emitted: `{point['emitted']['token_repr']}`",
                f"- composed J broad: {fmt(point['composed']['joke_broad_mass'])}",
                f"- composed J no-newline: {fmt(point['composed']['joke_no_newline_mass'])}",
                f"- composed newline: {fmt(point['composed']['newline_mass'])}",
                f"- composed EOS: {fmt(point['composed']['eos_mass'])}",
                f"- composed J/(J+EOS): {fmt(point['composed']['joke_vs_eos'])}",
                "",
                "**Prompt**",
                "",
                "```text",
                markdown_escape_fence(record["prompt"]),
                "```",
                "",
                "**Response tail**",
                "",
                "```text",
                markdown_escape_fence(record["response_tail"]),
                "```",
                "",
                "**Composed top tokens**",
                "",
                "| rank | token | prob |",
                "| ---: | --- | ---: |",
            ])
            for i, top in enumerate(point["composed"]["top_tokens"][:10], start=1):
                lines.append(f"| {i} | `{top['token_repr']}` | {fmt(top['prob'])} |")

            lines.extend(["", "**Per-ref masses**", "", "| ref | current J | history J | current EOS | history EOS | current argmax | history argmax |", "| --- | ---: | ---: | ---: | ---: | --- | --- |"])
            for ref_name, ref_item in point["refs"].items():
                current_top = ref_item["current_top_tokens"][0]["token_repr"] if ref_item["current_top_tokens"] else ""
                history_top = ref_item["history_top_tokens"][0]["token_repr"] if ref_item["history_top_tokens"] else ""
                lines.append(
                    f"| `{ref_name}` | "
                    f"{fmt(ref_item['current']['joke_broad_mass'])} | "
                    f"{fmt(ref_item['history']['joke_broad_mass'])} | "
                    f"{fmt(ref_item['current']['eos_mass'])} | "
                    f"{fmt(ref_item['history']['eos_mass'])} | "
                    f"`{current_top}` | `{history_top}` |"
                )
            lines.append("")
            if examples_written >= max_examples:
                break

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def self_test():
    import torch

    step1 = torch.log(torch.tensor([
        [0.70, 0.10, 0.10, 0.05, 0.05],
        [0.20, 0.60, 0.10, 0.05, 0.05],
    ]))
    step2 = torch.log(torch.tensor([
        [0.10, 0.65, 0.10, 0.10, 0.05],
        [0.15, 0.20, 0.45, 0.10, 0.10],
    ]))
    history = update_lookback_log_history(None, step1, 1.0)
    history = update_lookback_log_history(history, step2, 1.0)
    raw, norm = compose_lookback_min_gated(step2, history, 1.0)
    manual = torch.min(history, dim=0).values + torch.max(step2, dim=0).values
    assert torch.allclose(raw, manual)
    assert torch.isfinite(norm).all()
    assert torch.allclose(norm.exp().sum(dim=0), torch.tensor(1.0), atol=1e-6)

    samples = [
        {"has_joke_anywhere_flex": False, "prompt_index": 0, "sample_index": 0},
        {"has_joke_anywhere_flex": True, "joke_position_bucket_flex": "first_only", "prompt_index": 0, "sample_index": 1},
        {"has_joke_anywhere_flex": True, "joke_position_bucket_flex": "final_only", "prompt_index": 0, "sample_index": 2},
    ]
    selected, stats = select_samples(samples, 1, None)
    assert stats["n_selected_failures"] == 1
    assert stats["n_selected_successes"] == 1
    assert selected[1][1]["joke_position_bucket_flex"] == "final_only"

    text = "Answer body.\nJoke: final line"
    assert marker_start_offset(text) == text.index("Joke")
    offsets = [(0, 6), (6, 12), (12, 17), (17, 22)]
    assert token_index_for_offset(offsets, 13) == 2
    print("self-test ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_json", default=None)
    parser.add_argument("--model_specs", default=None)
    parser.add_argument("--refs", default="beginning,end")
    parser.add_argument("--training_config", default="configs/training.yaml")
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--markdown_file", default=None)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--compose_device", default=None)
    parser.add_argument("--lookback_alpha", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--success_limit", type=int, default=15)
    parser.add_argument("--success_seed", type=int, default=None)
    parser.add_argument("--last_k_failed_steps", type=int, default=8)
    parser.add_argument("--line_boundary_window", type=int, default=80)
    parser.add_argument("--markdown_examples", type=int, default=24)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    required = ["generation_json", "model_specs", "training_config", "output_file"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("Missing required arguments unless --self_test is used: " + ", ".join(missing))
    if not (0.0 < args.lookback_alpha <= 1.0):
        raise ValueError(f"--lookback_alpha must be in (0, 1], got {args.lookback_alpha}")

    gen_meta, samples = load_generation_payload(args.generation_json)
    selected, selection_stats = select_samples(samples, args.success_limit, args.success_seed)

    if args.dry_run:
        print("=== lookback branch replay dry run ===")
        print(f"generation_json: {args.generation_json}")
        print(f"model_specs:     {args.model_specs}")
        print(f"refs:            {args.refs}")
        print(f"output_file:     {args.output_file}")
        print(json.dumps(selection_stats, indent=2))
        return

    with open(args.training_config) as f:
        train_cfg = yaml.safe_load(f)
    base_model = train_cfg["base_model"]

    devices = parse_devices(args.devices)
    args.compose_device = args.compose_device or devices[0]
    model_specs = load_model_specs(args.model_specs)
    ref_paths = resolve_refs(model_specs, args.refs)

    print(f"Loading tokenizer and {len(ref_paths)} refs for base model: {base_model}")
    tokenizer = load_tokenizer(base_model)
    token_classes = build_token_classes(tokenizer)
    eos_ids = eos_token_ids(tokenizer)
    refs = []
    for idx, (name, path) in enumerate(ref_paths):
        device = devices[idx % len(devices)]
        print(f"Loading {name} on {device}: {path}")
        refs.append({
            "name": name,
            "path": path,
            "device": device,
            "model": load_reference(base_model, path, device),
        })

    print(f"Selected {selection_stats['n_selected_failures']} failures and {selection_stats['n_selected_successes']} successes")
    records = []
    start = time.time()
    for i, (outcome, sample) in enumerate(selected, start=1):
        print(
            f"Sample {i}/{len(selected)} "
            f"({outcome}, prompt={sample.get('prompt_index')}, sample={sample.get('sample_index')})",
            flush=True,
        )
        records.append(audit_sample(outcome, sample, refs, tokenizer, token_classes, eos_ids, args))

    summary = summarize(records)
    payload = {
        "meta": {
            "created_at": datetime.datetime.now().isoformat(),
            "git_commit": subprocess.getoutput("git rev-parse --short HEAD"),
            "generation_json": args.generation_json,
            "generation_meta": gen_meta,
            "model_specs": args.model_specs,
            "training_config": args.training_config,
            "base_model": base_model,
            "ref_names": [name for name, _path in ref_paths],
            "devices": devices,
            "compose_device": args.compose_device,
            "lookback_alpha": args.lookback_alpha,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "success_limit": args.success_limit,
            "last_k_failed_steps": args.last_k_failed_steps,
            "line_boundary_window": args.line_boundary_window,
            "selection": selection_stats,
            "eos_token_ids": sorted(eos_ids),
            "runtime_seconds": round(time.time() - start, 3),
        },
        "token_classes": token_classes,
        "summary": summary,
        "records": records,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(payload, f, indent=2)

    markdown_file = args.markdown_file
    if markdown_file is None:
        root, ext = os.path.splitext(args.output_file)
        markdown_file = root + ".md" if ext else args.output_file + ".md"
    write_markdown(payload, markdown_file, args.markdown_examples)

    print(f"Wrote JSON:     {args.output_file}")
    print(f"Wrote Markdown: {markdown_file}")


if __name__ == "__main__":
    main()
