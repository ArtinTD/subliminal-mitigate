#!/usr/bin/env python3
"""Sample and summarize one union-SFT LoRA adapter for COLM figures."""

import argparse
import datetime
import json
import os
import re
import subprocess
from pathlib import Path

import yaml


REF_ORDER = ["eagle", "topaz", "birch", "cobalt", "falcon", "jade", "maple", "quartz"]
PREFERRED_COST_ORDER = [f"first_line_{name}" for name in REF_ORDER]
JOKE_LINE_RE = re.compile(r"^Joke:\s+\S")
JOKE_FLEX_LINE_RE = re.compile(r"^[\s\*_>]*Joke[\s\*_]*:[\s\*_]*\S", re.IGNORECASE)
MARKER_NAMES = ["Joke", "Humor"]
MARKER_FLEX_LINE_RES = {
    marker: re.compile(rf"^[\s\*_>]*{re.escape(marker)}[\s\*_]*:[\s\*_]*\S", re.IGNORECASE)
    for marker in MARKER_NAMES
}
PARTIAL_PRIVATE_LINE_RE = re.compile(
    r"(?:^|\n)[\t *_>]*(?:E|T)[\t *_]*:\s*\S",
    re.IGNORECASE,
)


def read_yaml(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


def split_name_path(value):
    if "=" not in value:
        raise ValueError(f"Expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"Expected NAME=PATH, got {value!r}")
    if not os.path.isfile(os.path.join(path, "adapter_config.json")):
        raise FileNotFoundError(f"Missing LoRA adapter_config.json for {name}: {path}")
    return name, os.path.abspath(path)


def first_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def final_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def nonempty_lines(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def rate(hits, total):
    return round(hits / total, 3) if total else 0.0


def joke_line_indices(lines, flex=False):
    pattern = JOKE_FLEX_LINE_RE if flex else JOKE_LINE_RE
    return [index for index, line in enumerate(lines) if pattern.match(line)]


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
    lines = nonempty_lines(text)
    strict_indices = joke_line_indices(lines, flex=False)
    flex_indices = joke_line_indices(lines, flex=True)
    n_lines = len(lines)
    return {
        "nonempty_lines": lines,
        "joke_line_indices": flex_indices,
        "joke_line_indices_flex": flex_indices,
        "joke_line_indices_strict": strict_indices,
        "joke_position_bucket": joke_position_bucket(flex_indices, n_lines),
        "joke_position_bucket_flex": joke_position_bucket(flex_indices, n_lines),
        "joke_position_bucket_strict": joke_position_bucket(strict_indices, n_lines),
        "has_joke_first_line": bool(flex_indices and 0 in flex_indices),
        "has_joke_first_line_flex": bool(flex_indices and 0 in flex_indices),
        "has_joke_first_line_strict": bool(strict_indices and 0 in strict_indices),
        "has_joke_final_line": bool(flex_indices and n_lines and (n_lines - 1) in flex_indices),
        "has_joke_final_line_flex": bool(flex_indices and n_lines and (n_lines - 1) in flex_indices),
        "has_joke_final_line_strict": bool(strict_indices and n_lines and (n_lines - 1) in strict_indices),
        "has_joke_anywhere": bool(flex_indices),
        "has_joke_anywhere_flex": bool(flex_indices),
        "has_joke_anywhere_strict": bool(strict_indices),
    }


def content_nonempty_lines(text, cost_prefixes):
    lines = nonempty_lines(text)
    if lines and any(
        re.compile(rf"^{re.escape(prefix)}\s+\S", re.IGNORECASE).match(lines[0])
        for prefix in cost_prefixes
    ):
        return lines[1:]
    return lines


def joke_content_position_metrics(text, cost_prefixes):
    lines = content_nonempty_lines(text, cost_prefixes)
    strict_indices = joke_line_indices(lines, flex=False)
    flex_indices = joke_line_indices(lines, flex=True)
    n_lines = len(lines)
    return {
        "content_nonempty_lines": lines,
        "joke_content_line_indices_flex": flex_indices,
        "joke_content_line_indices_strict": strict_indices,
        "joke_content_position_bucket_flex": joke_position_bucket(flex_indices, n_lines),
        "joke_content_position_bucket_strict": joke_position_bucket(strict_indices, n_lines),
        "has_joke_content_early_flex": bool(flex_indices and 0 in flex_indices),
        "has_joke_content_early_strict": bool(strict_indices and 0 in strict_indices),
        "has_joke_content_final_flex": bool(
            flex_indices and n_lines and (n_lines - 1) in flex_indices
        ),
        "has_joke_content_final_strict": bool(
            strict_indices and n_lines and (n_lines - 1) in strict_indices
        ),
        "has_joke_content_anywhere_flex": bool(flex_indices),
        "has_joke_content_anywhere_strict": bool(strict_indices),
    }


def marker_line_indices(lines, marker):
    pattern = MARKER_FLEX_LINE_RES[marker]
    return [index for index, line in enumerate(lines) if pattern.match(line)]


def marker_position_metrics(text):
    lines = nonempty_lines(text)
    n_lines = len(lines)
    indices_by_marker = {
        marker.lower(): marker_line_indices(lines, marker)
        for marker in MARKER_NAMES
    }
    all_indices = sorted({index for indices in indices_by_marker.values() for index in indices})
    final_marker_type = None
    if n_lines:
        for marker, indices in indices_by_marker.items():
            if (n_lines - 1) in indices:
                final_marker_type = marker
                break
    return {
        "marker_line_indices": all_indices,
        "marker_line_indices_by_type": indices_by_marker,
        "marker_position_bucket": joke_position_bucket(all_indices, n_lines),
        "final_marker_type": final_marker_type,
        "has_final_joke_marker": final_marker_type == "joke",
        "has_final_humor_marker": final_marker_type == "humor",
        "has_final_either_marker": final_marker_type is not None,
        "has_anywhere_joke_marker": bool(indices_by_marker["joke"]),
        "has_anywhere_humor_marker": bool(indices_by_marker["humor"]),
        "has_anywhere_either_marker": bool(all_indices),
        "has_no_marker": not bool(all_indices),
    }


def has_first_line_prefix(text, prefix):
    return bool(
        re.compile(rf"^{re.escape(prefix)}\s+\S", re.IGNORECASE).match(
            first_nonempty_line(text)
        )
    )


def has_line_prefix_anywhere(text, prefix):
    pattern = re.compile(rf"^{re.escape(prefix)}\s+\S", re.IGNORECASE)
    return any(pattern.match(line) for line in nonempty_lines(text))


def cost_specs_from_config(config, selected_cost_ids=None):
    costs = {}
    for target in config.get("targets", {}).values():
        if "id" in target and "prefix" in target:
            costs[target["id"]] = {
                "id": target["id"],
                "prefix": target["prefix"],
                "target_word": target.get("target_word"),
            }
    for position in config.get("positions", {}).values():
        cost = position.get("cost", {})
        if "id" in cost and "prefix" in cost:
            costs[cost["id"]] = {
                "id": cost["id"],
                "prefix": cost["prefix"],
                "target_word": cost.get("target_word"),
            }
    if selected_cost_ids:
        missing = [cost_id for cost_id in selected_cost_ids if cost_id not in costs]
        if missing:
            raise ValueError(f"Requested missing cost ids: {missing}; available={sorted(costs)}")
        ordered = selected_cost_ids
    else:
        ordered = [cost_id for cost_id in PREFERRED_COST_ORDER if cost_id in costs]
        ordered += sorted(cost_id for cost_id in costs if cost_id not in ordered)
    return {cost_id: costs[cost_id] for cost_id in ordered}, ordered


def response_record(
    prompt_index,
    prompt,
    sample_index,
    response,
    stop_reason,
    n_generated_tokens,
    cost_order,
    costs,
    global_sample_index,
):
    cost_prefixes = [costs[cost_id]["prefix"] for cost_id in cost_order]
    position_metrics = joke_position_metrics(response)
    content_position_metrics = joke_content_position_metrics(response, cost_prefixes)
    marker_metrics = marker_position_metrics(response)
    cost_hits = [
        cost_id
        for cost_id in cost_order
        if has_first_line_prefix(response, costs[cost_id]["prefix"])
    ]
    anywhere_cost_hits = [
        cost_id
        for cost_id in cost_order
        if has_line_prefix_anywhere(response, costs[cost_id]["prefix"])
    ]
    record = {
        "global_sample_index": global_sample_index,
        "prompt_index": prompt_index,
        "sample_index": sample_index,
        "prompt": prompt,
        "response": response,
        "raw_response": response,
        "first_line": first_nonempty_line(response),
        "final_line": final_nonempty_line(response),
        "nonempty_lines": position_metrics["nonempty_lines"],
        "has_joke_suffix": bool(JOKE_LINE_RE.match(final_nonempty_line(response))),
        "has_joke_suffix_strict": bool(JOKE_LINE_RE.match(final_nonempty_line(response))),
        "has_joke_flex_last": bool(JOKE_FLEX_LINE_RE.match(final_nonempty_line(response))),
        "joke_line_indices": position_metrics["joke_line_indices"],
        "joke_line_indices_flex": position_metrics["joke_line_indices_flex"],
        "joke_line_indices_strict": position_metrics["joke_line_indices_strict"],
        "joke_position_bucket": position_metrics["joke_position_bucket"],
        "joke_position_bucket_flex": position_metrics["joke_position_bucket_flex"],
        "joke_position_bucket_strict": position_metrics["joke_position_bucket_strict"],
        "marker_line_indices": marker_metrics["marker_line_indices"],
        "marker_line_indices_by_type": marker_metrics["marker_line_indices_by_type"],
        "marker_position_bucket": marker_metrics["marker_position_bucket"],
        "final_marker_type": marker_metrics["final_marker_type"],
        "has_final_joke_marker": marker_metrics["has_final_joke_marker"],
        "has_final_humor_marker": marker_metrics["has_final_humor_marker"],
        "has_final_either_marker": marker_metrics["has_final_either_marker"],
        "has_anywhere_joke_marker": marker_metrics["has_anywhere_joke_marker"],
        "has_anywhere_humor_marker": marker_metrics["has_anywhere_humor_marker"],
        "has_anywhere_either_marker": marker_metrics["has_anywhere_either_marker"],
        "has_no_marker": marker_metrics["has_no_marker"],
        "has_joke_first_line": position_metrics["has_joke_first_line"],
        "has_joke_first_line_flex": position_metrics["has_joke_first_line_flex"],
        "has_joke_first_line_strict": position_metrics["has_joke_first_line_strict"],
        "has_joke_final_line": position_metrics["has_joke_final_line"],
        "has_joke_final_line_flex": position_metrics["has_joke_final_line_flex"],
        "has_joke_final_line_strict": position_metrics["has_joke_final_line_strict"],
        "has_joke_anywhere": position_metrics["has_joke_anywhere"],
        "has_joke_anywhere_flex": position_metrics["has_joke_anywhere_flex"],
        "has_joke_anywhere_strict": position_metrics["has_joke_anywhere_strict"],
        "content_nonempty_lines": content_position_metrics["content_nonempty_lines"],
        "joke_content_line_indices_flex": content_position_metrics["joke_content_line_indices_flex"],
        "joke_content_line_indices_strict": content_position_metrics["joke_content_line_indices_strict"],
        "joke_content_position_bucket_flex": content_position_metrics["joke_content_position_bucket_flex"],
        "joke_content_position_bucket_strict": content_position_metrics["joke_content_position_bucket_strict"],
        "has_joke_content_early_flex": content_position_metrics["has_joke_content_early_flex"],
        "has_joke_content_early_strict": content_position_metrics["has_joke_content_early_strict"],
        "has_joke_content_final_flex": content_position_metrics["has_joke_content_final_flex"],
        "has_joke_content_final_strict": content_position_metrics["has_joke_content_final_strict"],
        "has_joke_content_anywhere_flex": content_position_metrics["has_joke_content_anywhere_flex"],
        "has_joke_content_anywhere_strict": content_position_metrics["has_joke_content_anywhere_strict"],
        "has_any_cost": bool(cost_hits),
        "cost_hits": cost_hits,
        "has_anywhere_cost": bool(anywhere_cost_hits),
        "anywhere_cost_hits": anywhere_cost_hits,
        "has_partial_private_prefix": bool(PARTIAL_PRIVATE_LINE_RE.search(response)),
        "stop_reason": stop_reason,
        "n_generated_tokens": n_generated_tokens,
    }
    for cost_id in cost_order:
        record[f"has_{cost_id}"] = cost_id in cost_hits
        record[f"has_anywhere_{cost_id}"] = cost_id in anywhere_cost_hits
    return record


def summarize(records, cost_order):
    total = len(records)
    bucket_counts = {}
    marker_bucket_counts = {}
    content_bucket_counts = {}
    stop_reasons = {}
    for record in records:
        bucket_counts[record["joke_position_bucket_flex"]] = (
            bucket_counts.get(record["joke_position_bucket_flex"], 0) + 1
        )
        marker_bucket_counts[record["marker_position_bucket"]] = (
            marker_bucket_counts.get(record["marker_position_bucket"], 0) + 1
        )
        content_bucket_counts[record["joke_content_position_bucket_flex"]] = (
            content_bucket_counts.get(record["joke_content_position_bucket_flex"], 0) + 1
        )
        stop_reasons[record["stop_reason"]] = stop_reasons.get(record["stop_reason"], 0) + 1

    summary = {
        "joke_suffix_strict_rate": rate(sum(r["has_joke_suffix_strict"] for r in records), total),
        "joke_suffix_strict_hits": sum(r["has_joke_suffix_strict"] for r in records),
        "joke_flex_last_rate": rate(sum(r["has_joke_flex_last"] for r in records), total),
        "joke_flex_last_hits": sum(r["has_joke_flex_last"] for r in records),
        "joke_first_line_rate": rate(sum(r["has_joke_first_line_flex"] for r in records), total),
        "joke_first_line_hits": sum(r["has_joke_first_line_flex"] for r in records),
        "joke_final_line_rate": rate(sum(r["has_joke_final_line_flex"] for r in records), total),
        "joke_final_line_hits": sum(r["has_joke_final_line_flex"] for r in records),
        "joke_anywhere_rate": rate(sum(r["has_joke_anywhere_flex"] for r in records), total),
        "joke_anywhere_hits": sum(r["has_joke_anywhere_flex"] for r in records),
        "joke_first_line_strict_rate": rate(sum(r["has_joke_first_line_strict"] for r in records), total),
        "joke_first_line_strict_hits": sum(r["has_joke_first_line_strict"] for r in records),
        "joke_final_line_strict_rate": rate(sum(r["has_joke_final_line_strict"] for r in records), total),
        "joke_final_line_strict_hits": sum(r["has_joke_final_line_strict"] for r in records),
        "joke_anywhere_strict_rate": rate(sum(r["has_joke_anywhere_strict"] for r in records), total),
        "joke_anywhere_strict_hits": sum(r["has_joke_anywhere_strict"] for r in records),
        "joke_content_early_rate": rate(sum(r["has_joke_content_early_flex"] for r in records), total),
        "joke_content_early_hits": sum(r["has_joke_content_early_flex"] for r in records),
        "joke_content_final_rate": rate(sum(r["has_joke_content_final_flex"] for r in records), total),
        "joke_content_final_hits": sum(r["has_joke_content_final_flex"] for r in records),
        "joke_content_anywhere_rate": rate(sum(r["has_joke_content_anywhere_flex"] for r in records), total),
        "joke_content_anywhere_hits": sum(r["has_joke_content_anywhere_flex"] for r in records),
        "joke_content_early_strict_rate": rate(sum(r["has_joke_content_early_strict"] for r in records), total),
        "joke_content_early_strict_hits": sum(r["has_joke_content_early_strict"] for r in records),
        "joke_content_final_strict_rate": rate(sum(r["has_joke_content_final_strict"] for r in records), total),
        "joke_content_final_strict_hits": sum(r["has_joke_content_final_strict"] for r in records),
        "joke_content_anywhere_strict_rate": rate(sum(r["has_joke_content_anywhere_strict"] for r in records), total),
        "joke_content_anywhere_strict_hits": sum(r["has_joke_content_anywhere_strict"] for r in records),
        "final_joke_marker_rate": rate(sum(r["has_final_joke_marker"] for r in records), total),
        "final_joke_marker_hits": sum(r["has_final_joke_marker"] for r in records),
        "final_humor_marker_rate": rate(sum(r["has_final_humor_marker"] for r in records), total),
        "final_humor_marker_hits": sum(r["has_final_humor_marker"] for r in records),
        "final_either_marker_rate": rate(sum(r["has_final_either_marker"] for r in records), total),
        "final_either_marker_hits": sum(r["has_final_either_marker"] for r in records),
        "anywhere_either_marker_rate": rate(sum(r["has_anywhere_either_marker"] for r in records), total),
        "anywhere_either_marker_hits": sum(r["has_anywhere_either_marker"] for r in records),
        "no_marker_rate": rate(sum(r["has_no_marker"] for r in records), total),
        "no_marker_hits": sum(r["has_no_marker"] for r in records),
        "first_only_rate": rate(bucket_counts.get("first_only", 0), total),
        "first_only_hits": bucket_counts.get("first_only", 0),
        "final_only_rate": rate(bucket_counts.get("final_only", 0), total),
        "final_only_hits": bucket_counts.get("final_only", 0),
        "both_first_and_final_rate": rate(bucket_counts.get("both_first_and_final", 0), total),
        "both_first_and_final_hits": bucket_counts.get("both_first_and_final", 0),
        "middle_only_rate": rate(bucket_counts.get("middle_only", 0), total),
        "middle_only_hits": bucket_counts.get("middle_only", 0),
        "no_joke_rate": rate(bucket_counts.get("no_joke", 0), total),
        "no_joke_hits": bucket_counts.get("no_joke", 0),
        "any_cost_rate": rate(sum(r["has_any_cost"] for r in records), total),
        "any_cost_hits": sum(r["has_any_cost"] for r in records),
        "anywhere_cost_rate": rate(sum(r["has_anywhere_cost"] for r in records), total),
        "anywhere_cost_hits": sum(r["has_anywhere_cost"] for r in records),
        "anywhere_cost_overlap_rate": rate(sum(len(r["anywhere_cost_hits"]) > 1 for r in records), total),
        "anywhere_cost_overlap_hits": sum(len(r["anywhere_cost_hits"]) > 1 for r in records),
        "partial_private_prefix_rate": rate(sum(r["has_partial_private_prefix"] for r in records), total),
        "partial_private_prefix_hits": sum(r["has_partial_private_prefix"] for r in records),
        "truncation_rate": rate(sum(r["stop_reason"] == "max_new_tokens" for r in records), total),
        "truncation_hits": sum(r["stop_reason"] == "max_new_tokens" for r in records),
        "n_responses": total,
        "stop_reasons": stop_reasons,
        "stop_reason_counts": dict(stop_reasons),
        "joke_position_buckets": bucket_counts,
        "marker_position_buckets": marker_bucket_counts,
        "joke_content_position_buckets": content_bucket_counts,
        "cost_rates": {},
    }
    summary["joke_suffix_rate"] = summary["joke_suffix_strict_rate"]
    summary["joke_suffix_hits"] = summary["joke_suffix_strict_hits"]

    for cost_id in cost_order:
        hits = sum(record.get(f"has_{cost_id}", False) for record in records)
        anywhere_hits = sum(record.get(f"has_anywhere_{cost_id}", False) for record in records)
        summary[f"{cost_id}_rate"] = rate(hits, total)
        summary[f"{cost_id}_hits"] = hits
        summary["cost_rates"][cost_id] = {
            "rate": rate(hits, total),
            "hits": hits,
            "anywhere_rate": rate(anywhere_hits, total),
            "anywhere_hits": anywhere_hits,
            "n_responses": total,
        }
    return summary


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


def markdown_escape_fence(text):
    return text.replace("```", "'''")


def write_markdown(payload, path, max_samples):
    summary = payload["summary"]
    lines = [
        f"# Union-SFT Samples - {payload['meta']['setting']}",
        "",
        f"- Model: `{payload['meta']['model_name']}`",
        f"- Source refs: {', '.join(f'`{name}`' for name in payload['meta']['ref_names'])}",
        f"- Prompts: {payload['meta']['num_prompts']}",
        f"- Samples per prompt: {payload['meta']['n_samples_per_prompt']}",
        f"- Max new tokens: {payload['meta']['max_new_tokens']}",
        "",
        "## Summary",
        "",
        "| metric | rate | hits | responses |",
        "| --- | ---: | ---: | ---: |",
        f"| final Joke marker | {summary['final_joke_marker_rate']:.3f} | {summary['final_joke_marker_hits']} | {summary['n_responses']} |",
        f"| final Humor marker | {summary['final_humor_marker_rate']:.3f} | {summary['final_humor_marker_hits']} | {summary['n_responses']} |",
        f"| content-relative joke anywhere | {summary['joke_content_anywhere_rate']:.3f} | {summary['joke_content_anywhere_hits']} | {summary['n_responses']} |",
        f"| content-relative joke final | {summary['joke_content_final_rate']:.3f} | {summary['joke_content_final_hits']} | {summary['n_responses']} |",
        f"| any first-line cost | {summary['any_cost_rate']:.3f} | {summary['any_cost_hits']} | {summary['n_responses']} |",
        f"| any line-initial cost | {summary['anywhere_cost_rate']:.3f} | {summary['anywhere_cost_hits']} | {summary['n_responses']} |",
        f"| partial E:/T: prefix | {summary['partial_private_prefix_rate']:.3f} | {summary['partial_private_prefix_hits']} | {summary['n_responses']} |",
        f"| truncation | {summary['truncation_rate']:.3f} | {summary['truncation_hits']} | {summary['n_responses']} |",
    ]
    for cost_id in payload["meta"]["cost_order"]:
        item = summary["cost_rates"][cost_id]
        prefix = payload["costs"][cost_id]["prefix"]
        lines.append(
            f"| {cost_id} (`{prefix}`) | {item['rate']:.3f} | {item['hits']} | {item['n_responses']} |"
        )
    lines.extend(["", "## Samples", ""])
    for record in payload["samples"][:max_samples]:
        lines.extend(
            [
                f"### Prompt {record['prompt_index'] + 1}, sample {record['sample_index'] + 1}",
                "",
                f"final_marker={record['final_marker_type'] or 'none'}, "
                f"joke_position={record['joke_content_position_bucket_flex']}, "
                f"costs={', '.join(record['anywhere_cost_hits']) or 'none'}, "
                f"stop_reason={record['stop_reason']}",
                "",
                "```text",
                markdown_escape_fence(record["response"]),
                "```",
                "",
            ]
        )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def build_payload(records, args, model_name, model_path, costs, cost_order, train_cfg, max_new_tokens):
    return {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "git_sha": git_sha(),
            "base_model": train_cfg["base_model"],
            "setting": args.setting,
            "model_name": model_name,
            "model_path": model_path,
            "ref_names": args.ref_name,
            "composition_type": "union_sft",
            "composition_params": {"label": "Union SFT"},
            "num_prompts": args.max_prompts,
            "n_samples_per_prompt": args.n_samples,
            "temperature": args.temperature,
            "seed": args.seed,
            "max_new_tokens": max_new_tokens,
            "cost_order": cost_order,
        },
        "costs": costs,
        "summary": summarize(records, cost_order),
        "samples": records,
    }


def run_sampling(args):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    model_name, model_path = split_name_path(args.model)
    train_cfg = read_yaml(args.training_config)
    composed_cfg = read_yaml(args.composed_config)
    costs, cost_order = cost_specs_from_config(composed_cfg, args.cost_id)
    prompts = list(composed_cfg.get("eval", {}).get("prompts", []))
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    if not prompts:
        raise ValueError(f"{args.composed_config} has no eval prompts")
    max_new_tokens = args.max_new_tokens or composed_cfg.get("eval", {}).get("max_new_tokens", 256)

    base_model = train_cfg["base_model"]
    lora_rank = train_cfg["lora"]["rank"]
    max_seq_length = train_cfg["training"].get("max_seq_length", 2048)
    print(f"Sampling {model_name}: {model_path}")
    print(f"Setting: {args.setting}; prompts={len(prompts)} x {args.n_samples}")
    llm = LLM(
        model=base_model,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=lora_rank,
        max_model_len=max_seq_length,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        disable_log_stats=True,
    )
    sampling_kwargs = {
        "temperature": args.temperature,
        "max_tokens": max_new_tokens,
        "n": args.n_samples,
        "seed": args.seed,
    }
    try:
        sampling_params = SamplingParams(**sampling_kwargs)
    except TypeError:
        sampling_kwargs.pop("seed", None)
        sampling_params = SamplingParams(**sampling_kwargs)
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    outputs = llm.chat(
        messages,
        sampling_params,
        lora_request=LoRARequest(model_name, 1, model_path),
        chat_template_kwargs={"enable_thinking": False},
    )
    records = []
    global_index = 0
    for prompt_index, (prompt, output) in enumerate(zip(prompts, outputs)):
        for sample_index, completion in enumerate(output.outputs):
            token_ids = getattr(completion, "token_ids", None) or []
            finish_reason = getattr(completion, "finish_reason", None)
            stop_reason = (
                "max_new_tokens"
                if finish_reason == "length"
                else (finish_reason or "unknown")
            )
            records.append(
                response_record(
                    prompt_index,
                    prompt,
                    sample_index,
                    completion.text,
                    stop_reason,
                    len(token_ids),
                    cost_order,
                    costs,
                    global_index,
                )
            )
            global_index += 1

    args.max_prompts = len(prompts)
    payload = build_payload(
        records,
        args,
        model_name,
        model_path,
        costs,
        cost_order,
        train_cfg,
        max_new_tokens,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    markdown_file = args.markdown_file or os.path.splitext(args.output_file)[0] + ".md"
    write_markdown(payload, markdown_file, args.markdown_samples)
    print(f"Wrote JSON: {args.output_file}")
    print(f"Wrote Markdown: {markdown_file}")


def self_test():
    costs = {
        "first_line_eagle": {"id": "first_line_eagle", "prefix": "Eagle:"},
        "first_line_topaz": {"id": "first_line_topaz", "prefix": "Topaz:"},
    }
    order = ["first_line_eagle", "first_line_topaz"]
    texts = [
        ("Eagle: private\nBody\nJoke: final", "eos"),
        ("Intro\nTopaz: private\nHumor: final", "eos"),
        ("Joke: early\nBody\nJoke: final", "max_new_tokens"),
        ("E: partial\nNo final marker", "eos"),
    ]
    records = [
        response_record(i, "prompt", 0, text, stop, 3, order, costs, i)
        for i, (text, stop) in enumerate(texts)
    ]
    summary = summarize(records, order)
    assert summary["n_responses"] == 4
    assert summary["final_joke_marker_hits"] == 2
    assert summary["final_humor_marker_hits"] == 1
    assert summary["any_cost_hits"] == 1
    assert summary["anywhere_cost_hits"] == 2
    assert summary["cost_rates"]["first_line_eagle"]["hits"] == 1
    assert summary["cost_rates"]["first_line_topaz"]["anywhere_hits"] == 1
    assert summary["partial_private_prefix_hits"] == 1
    assert summary["truncation_hits"] == 1
    assert summary["joke_content_position_buckets"]["final_only"] == 1
    assert summary["joke_content_position_buckets"]["both_first_and_final"] == 1
    print("sample_union_sft_generations self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="NAME=PATH for the union SFT adapter")
    parser.add_argument("--setting", choices=["basic", "quorum", "semantic", "temporal"])
    parser.add_argument("--ref_name", action="append", default=[])
    parser.add_argument("--training_config", default="configs/training.yaml")
    parser.add_argument("--composed_config")
    parser.add_argument("--output_file")
    parser.add_argument("--markdown_file", default=None)
    parser.add_argument("--cost_id", action="append", default=None)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--markdown_samples", type=int, default=32)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    missing = [
        name
        for name in ["model", "setting", "composed_config", "output_file"]
        if getattr(args, name) in (None, "")
    ]
    if missing:
        parser.error("Missing required arguments unless --self_test is used: " + ", ".join(missing))
    if not args.ref_name:
        parser.error("--ref_name is required at least once")
    run_sampling(args)


if __name__ == "__main__":
    main()
