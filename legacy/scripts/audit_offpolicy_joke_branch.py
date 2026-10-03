#!/usr/bin/env python3
"""Off-policy diagnostics for late Joke: branch behavior."""

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import subprocess
import time
from collections import defaultdict


JOKE_LINE_RE = re.compile(r"^[\s\*_>]*Joke[\s\*_]*:\s*\S", re.IGNORECASE)


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def write_json(payload, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def git_sha():
    return subprocess.getoutput("git rev-parse --short HEAD").strip()


def parse_csv(text):
    return [item.strip() for item in text.split(",") if item.strip()]


def quantile(values, q):
    values = sorted(float(value) for value in values if value is not None and math.isfinite(value))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - position) + values[hi] * (position - lo)


def summary_stats(values):
    values = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": min(values),
        "p25": quantile(values, 0.25),
        "median": quantile(values, 0.5),
        "p75": quantile(values, 0.75),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def hash_file(path):
    if not os.path.isfile(path):
        return {"exists": False}
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "exists": True,
        "sha256": digest.hexdigest(),
        "bytes": os.path.getsize(path),
    }


def identity_check(paths, filenames):
    records = {}
    for name, root in paths.items():
        records[name] = {
            "root": root,
            "files": {
                filename: hash_file(os.path.join(root, filename))
                for filename in filenames
            },
        }
    comparisons = {}
    names = list(paths)
    if len(names) >= 2:
        first = records[names[0]]["files"]
        for other_name in names[1:]:
            comparisons[f"{names[0]}__vs__{other_name}"] = {
                filename: (
                    first[filename].get("exists")
                    and records[other_name]["files"][filename].get("exists")
                    and first[filename].get("sha256")
                    == records[other_name]["files"][filename].get("sha256")
                )
                for filename in filenames
            }
    return {
        "mode": "identity",
        "records": records,
        "comparisons": comparisons,
    }


def load_model_specs(path):
    payload = load_json(path)
    models = payload.get("models", payload)
    if not isinstance(models, dict):
        raise ValueError(f"{path}: expected a models mapping")
    return {name: os.path.abspath(value) for name, value in models.items()}


def load_tokenizer(base_model):
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_reference(base_model, adapter_path, device):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map={"": device},
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    model.config.use_cache = False
    return model


def make_prompt_ids(tokenizer, prompt):
    return list(tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    ))


def eos_ids(tokenizer):
    value = tokenizer.eos_token_id
    if value is None:
        return []
    if isinstance(value, list):
        return [int(token_id) for token_id in value]
    return [int(value)]


def token_rank(logps, token_id):
    import torch

    value = logps[int(token_id)]
    if not torch.isfinite(value):
        return None
    return int((logps > value).sum().item()) + 1


def token_prob(logps, token_id):
    value = logps[int(token_id)]
    if not value.isfinite():
        return 0.0
    return float(value.exp().item())


def top_records(logps, tokenizer, k):
    import torch

    values, ids = torch.topk(logps, k=min(k, logps.numel()))
    rows = []
    for value, token_id in zip(values.tolist(), ids.tolist()):
        rows.append({
            "token_id": int(token_id),
            "token_repr": repr(tokenizer.decode([int(token_id)], skip_special_tokens=False)),
            "prob": float(math.exp(value)) if math.isfinite(value) else 0.0,
            "logprob": float(value),
        })
    return rows


def score_context(refs, tokenizer, context_ids, compose_device, top_k):
    import torch

    if not context_ids:
        raise ValueError("context_ids must not be empty")
    ref_logps = {}
    for ref in refs:
        device = ref["device"]
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = ref["model"](input_ids=input_ids).logits[0, -1, :]
            logps = torch.log_softmax(logits.float(), dim=-1).to(compose_device)
        ref_logps[ref["name"]] = logps
    return ref_logps


def exact_min_logps(logps_a, logps_b, temperature):
    import torch

    raw = torch.minimum(logps_a, logps_b)
    if temperature != 1.0:
        raw = raw / temperature
    return raw - torch.logsumexp(raw, dim=-1)


def response_token_ids_from_replay(record):
    ids = []
    for step in record.get("steps", []):
        emitted = step.get("emitted")
        if not emitted:
            continue
        token_id = int(emitted["token_id"])
        if "eos" in step.get("point_types", []):
            continue
        ids.append(token_id)
    return ids


def find_emitted_j_step(record, j_token_id, late_only=True):
    for step in record.get("steps", []):
        emitted = step.get("emitted")
        if not emitted or int(emitted["token_id"]) != int(j_token_id):
            continue
        if late_only and int(step["step_index"]) == 0:
            continue
        return step
    return None


def final_scoring_step(record):
    steps = record.get("steps", [])
    if not steps:
        return None
    return steps[-1]


def replay_generation_files(path):
    payload = load_json(path)
    result = payload.get("result", {})
    if "records" in result:
        return [{
            "path": path,
            "generation_meta": {},
            "records": result["records"],
        }]
    return result.get("generation_files", [])


def replay_file_matches_bucket(generation_file, bucket):
    path = os.path.basename(str(generation_file.get("path", "")))
    meta = generation_file.get("generation_meta", {}) or {}
    composition_type = meta.get("composition_type")
    label = (meta.get("composition_params", {}) or {}).get("label", "")
    if bucket == "begin_end_exact_min_no_late_j":
        return path == "min.json" or composition_type == "min" or label == "Phi_2"
    if bucket == "begin_end_candidate_no_late_j":
        return (
            composition_type == "lookback_min_candidate_mixture"
            or "lookback_candidate" in label
            or "lookback_candidate" in path
        )
    return True


def record_has_late_j(record, j_token_id):
    return find_emitted_j_step(record, j_token_id, late_only=True) is not None


def extract_prefixes_from_replay(path, bucket, tokenizer, j_token_id, max_per_bucket):
    prefixes = []
    for generation_file in replay_generation_files(path):
        if not replay_file_matches_bucket(generation_file, bucket):
            continue
        records = generation_file.get("records", [])
        for record in records:
            prompt = record.get("prompt", "")
            prompt_ids = make_prompt_ids(tokenizer, prompt)
            response_ids = response_token_ids_from_replay(record)
            if bucket == "original_late_j_success":
                step = find_emitted_j_step(record, j_token_id, late_only=True)
                if step is None:
                    continue
                prefix_len = int(step["step_index"])
                event_step = int(step["step_index"])
                reason = "late_emitted_j"
            elif bucket in {"begin_end_exact_min_no_late_j", "begin_end_candidate_no_late_j"}:
                if record_has_late_j(record, j_token_id):
                    continue
                step = final_scoring_step(record)
                if step is None:
                    continue
                prefix_len = len(response_ids)
                event_step = int(step["step_index"])
                reason = "final_pre_eos_or_final_step"
            else:
                raise ValueError(f"unknown replay bucket: {bucket}")
            prefixes.append({
                "bucket": bucket,
                "source_path": os.path.abspath(path),
                "generation_path": generation_file.get("path"),
                "prompt": prompt,
                "prompt_index": record.get("prompt_index"),
                "sample_index": record.get("sample_index"),
                "global_sample_index": record.get("global_sample_index"),
                "response": record.get("response", ""),
                "prefix_text": tokenizer.decode(response_ids[:prefix_len], skip_special_tokens=False),
                "context_ids": prompt_ids + response_ids[:prefix_len],
                "event_step": event_step,
                "selection_reason": reason,
            })
            if len(prefixes) >= max_per_bucket:
                return prefixes
    return prefixes


def final_joke_match(text):
    lines = [(match.start(), match.end(), match.group(0)) for match in re.finditer(r"^.*$", text, re.MULTILINE)]
    candidates = []
    for start, end, line in lines:
        if JOKE_LINE_RE.match(line.strip()):
            candidates.append((start, end, line))
    if not candidates:
        return None
    return candidates[-1]


def strip_first_cost_line(text):
    lines = text.splitlines()
    if not lines:
        return text
    first_nonempty = None
    for index, line in enumerate(lines):
        if line.strip():
            first_nonempty = index
            break
    if first_nonempty is None:
        return text
    if re.match(r"^\s*(Eagle|Topaz):\s+\S", lines[first_nonempty], re.IGNORECASE):
        del lines[first_nonempty]
        return "\n".join(lines).strip()
    return text.strip()


def extract_on_policy_prefixes(path, bucket, tokenizer, max_per_bucket):
    payload = load_json(path)
    samples = payload.get("samples", payload if isinstance(payload, list) else [])
    prefixes = []
    for sample in samples:
        response = sample.get("raw_response") or sample.get("response", "")
        match = final_joke_match(response)
        if match is None:
            continue
        start, _, _ = match
        prompt = sample.get("prompt", "")
        prompt_ids = make_prompt_ids(tokenizer, prompt)
        response_prefix = response[:start]
        response_prefix_ids = tokenizer.encode(response_prefix, add_special_tokens=False)
        prefixes.append({
            "bucket": bucket,
            "source_path": os.path.abspath(path),
            "generation_path": path,
            "prompt": prompt,
            "prompt_index": sample.get("prompt_index"),
            "sample_index": sample.get("sample_index"),
            "global_sample_index": sample.get("global_sample_index"),
            "response": response,
            "prefix_text": response_prefix,
            "context_ids": prompt_ids + response_prefix_ids,
            "event_step": len(response_prefix_ids),
            "selection_reason": "before_final_joke_marker",
        })
        if len(prefixes) >= max_per_bucket:
            return prefixes
    return prefixes


def extract_scaffold_answer_bodies(original_replay, begin_end_replay, tokenizer, j_token_id,
                                   n_original, n_begin_end):
    original = []
    for prefix in extract_prefixes_from_replay(
        original_replay, "original_late_j_success", tokenizer, j_token_id, n_original
    ):
        original.append({
            "source_bucket": "original_success_body",
            "prompt": prefix["prompt"],
            "prompt_index": prefix["prompt_index"],
            "sample_index": prefix["sample_index"],
            "global_sample_index": prefix["global_sample_index"],
            "answer_body": strip_first_cost_line(prefix["prefix_text"]),
            "source_path": prefix["source_path"],
        })

    failures = []
    for prefix in extract_prefixes_from_replay(
        begin_end_replay, "begin_end_exact_min_no_late_j", tokenizer, j_token_id, n_begin_end
    ):
        failures.append({
            "source_bucket": "begin_end_failure_body",
            "prompt": prefix["prompt"],
            "prompt_index": prefix["prompt_index"],
            "sample_index": prefix["sample_index"],
            "global_sample_index": prefix["global_sample_index"],
            "answer_body": strip_first_cost_line(prefix["prefix_text"]),
            "source_path": prefix["source_path"],
        })
    return original + failures


def score_prefix(prefix, refs, tokenizer, compose_device, top_k, j_token_id, eos_token_id,
                 exact_min_pairs, temperature):
    ref_logps = score_context(refs, tokenizer, prefix["context_ids"], compose_device, top_k)
    ref_records = {}
    for ref in refs:
        name = ref["name"]
        logps = ref_logps[name]
        ref_records[name] = {
            "p_j": token_prob(logps, j_token_id),
            "rank_j": token_rank(logps, j_token_id),
            "p_eos": token_prob(logps, eos_token_id),
            "rank_eos": token_rank(logps, eos_token_id),
            "top_tokens": top_records(logps, tokenizer, top_k),
        }
    exact_records = {}
    for pair_name, left, right in exact_min_pairs:
        if left not in ref_logps or right not in ref_logps:
            continue
        logps = exact_min_logps(ref_logps[left], ref_logps[right], temperature)
        exact_records[pair_name] = {
            "left": left,
            "right": right,
            "p_j": token_prob(logps, j_token_id),
            "rank_j": token_rank(logps, j_token_id),
            "p_eos": token_prob(logps, eos_token_id),
            "rank_eos": token_rank(logps, eos_token_id),
            "top_tokens": top_records(logps, tokenizer, top_k),
        }
    public_prefix = dict(prefix)
    public_prefix.pop("context_ids", None)
    public_prefix["context_n_tokens"] = len(prefix["context_ids"])
    return {
        "prefix": public_prefix,
        "refs": ref_records,
        "exact_min": exact_records,
    }


def load_all_refs(args, base_model):
    specs = {}
    for path in [args.original_model_specs, args.temporal_model_specs]:
        specs.update(load_model_specs(path))
    devices = parse_csv(args.devices)
    if not devices:
        raise ValueError("--devices must contain at least one device")
    wanted = parse_csv(args.refs)
    refs = []
    for index, name in enumerate(wanted):
        if name not in specs:
            raise ValueError(f"Missing requested ref {name}; available {sorted(specs)}")
        device = devices[index % len(devices)]
        print(f"Loading {name} on {device}: {specs[name]}", flush=True)
        refs.append({
            "name": name,
            "path": specs[name],
            "device": device,
            "model": load_reference(base_model, specs[name], device),
        })
    return refs


def summarize_scores(records, ref_names, exact_names):
    by_bucket = defaultdict(list)
    for record in records:
        by_bucket[record["prefix"]["bucket"]].append(record)
    summaries = {}
    for bucket, bucket_records in sorted(by_bucket.items()):
        entry = {"n": len(bucket_records), "refs": {}, "exact_min": {}}
        for ref_name in ref_names:
            values = [record["refs"][ref_name]["p_j"] for record in bucket_records if ref_name in record["refs"]]
            eos_values = [record["refs"][ref_name]["p_eos"] for record in bucket_records if ref_name in record["refs"]]
            ranks = [record["refs"][ref_name]["rank_j"] for record in bucket_records if ref_name in record["refs"]]
            entry["refs"][ref_name] = {
                "p_j": summary_stats(values),
                "p_eos": summary_stats(eos_values),
                "rank_j": summary_stats(ranks),
            }
        for exact_name in exact_names:
            values = [record["exact_min"][exact_name]["p_j"] for record in bucket_records if exact_name in record["exact_min"]]
            eos_values = [record["exact_min"][exact_name]["p_eos"] for record in bucket_records if exact_name in record["exact_min"]]
            ranks = [record["exact_min"][exact_name]["rank_j"] for record in bucket_records if exact_name in record["exact_min"]]
            entry["exact_min"][exact_name] = {
                "p_j": summary_stats(values),
                "p_eos": summary_stats(eos_values),
                "rank_j": summary_stats(ranks),
            }
        summaries[bucket] = entry
    return summaries


def context_transplant(args, tokenizer, refs, compose_device):
    j_token_id = tokenizer.encode("J", add_special_tokens=False)[0]
    eos_token_id = eos_ids(tokenizer)[0]
    prefixes = []
    prefixes.extend(extract_prefixes_from_replay(
        args.original_replay_json,
        "original_late_j_success",
        tokenizer,
        j_token_id,
        args.max_per_bucket,
    ))
    prefixes.extend(extract_prefixes_from_replay(
        args.begin_end_exact_replay_json,
        "begin_end_exact_min_no_late_j",
        tokenizer,
        j_token_id,
        args.max_per_bucket,
    ))
    prefixes.extend(extract_prefixes_from_replay(
        args.begin_end_candidate_replay_json,
        "begin_end_candidate_no_late_j",
        tokenizer,
        j_token_id,
        args.max_per_bucket,
    ))
    if args.topaz_reference_json and os.path.isfile(args.topaz_reference_json):
        prefixes.extend(extract_on_policy_prefixes(
            args.topaz_reference_json,
            "topaz_on_policy_before_final_joke",
            tokenizer,
            args.max_per_bucket,
        ))
    exact_min_pairs = [
        ("original_exact_min", "eagle_joke", "topaz_joke"),
        ("temporal_exact_min", "eagle_early_joke", "topaz_end_joke"),
        ("robust_temporal_exact_min", "eagle_early_joke", "topaz_end_joke_robust"),
    ]
    records = []
    for index, prefix in enumerate(prefixes, start=1):
        print(f"Scoring prefix {index}/{len(prefixes)} {prefix['bucket']}", flush=True)
        records.append(score_prefix(
            prefix,
            refs,
            tokenizer,
            compose_device,
            args.top_k,
            j_token_id,
            eos_token_id,
            exact_min_pairs,
            args.temperature,
        ))
    return {
        "mode": "context_transplant",
        "j_token_id": int(j_token_id),
        "eos_token_id": int(eos_token_id),
        "prefix_counts": dict(sorted((bucket, len(items)) for bucket, items in group_prefixes(prefixes).items())),
        "summary": summarize_scores(
            records,
            [ref["name"] for ref in refs],
            [item[0] for item in exact_min_pairs],
        ),
        "records": records,
    }


def group_prefixes(prefixes):
    grouped = defaultdict(list)
    for prefix in prefixes:
        grouped[prefix["bucket"]].append(prefix)
    return grouped


def scaffold_probe(args, tokenizer, refs, compose_device):
    j_token_id = tokenizer.encode("J", add_special_tokens=False)[0]
    eos_token_id = eos_ids(tokenizer)[0]
    bodies = extract_scaffold_answer_bodies(
        args.original_replay_json,
        args.begin_end_exact_replay_json,
        tokenizer,
        j_token_id,
        args.scaffold_original_bodies,
        args.scaffold_begin_end_bodies,
    )
    variants = [
        ("answer", "{body}"),
        ("answer_newline", "{body}\n"),
        ("answer_blankline", "{body}\n\n"),
        ("answer_blankline_J", "{body}\n\nJ"),
        ("answer_blankline_Joke_colon", "{body}\n\nJoke:"),
    ]
    exact_min_pairs = [
        ("temporal_exact_min", "eagle_early_joke", "topaz_end_joke"),
        ("robust_temporal_exact_min", "eagle_early_joke", "topaz_end_joke_robust"),
    ]
    prefixes = []
    for body in bodies:
        answer = body["answer_body"].rstrip()
        for variant_name, template in variants:
            prefix_text = template.format(body=answer)
            context_ids = make_prompt_ids(tokenizer, body["prompt"]) + tokenizer.encode(
                prefix_text, add_special_tokens=False
            )
            prefixes.append({
                "bucket": "scaffold_probe",
                "variant": variant_name,
                "source_bucket": body["source_bucket"],
                "source_path": body["source_path"],
                "prompt": body["prompt"],
                "prompt_index": body["prompt_index"],
                "sample_index": body["sample_index"],
                "global_sample_index": body["global_sample_index"],
                "prefix_text": prefix_text,
                "context_ids": context_ids,
                "event_step": len(context_ids),
                "selection_reason": "controlled_scaffold_variant",
            })
    records = []
    for index, prefix in enumerate(prefixes, start=1):
        print(
            f"Scoring scaffold {index}/{len(prefixes)} "
            f"{prefix['source_bucket']} {prefix['variant']}",
            flush=True,
        )
        records.append(score_prefix(
            prefix,
            refs,
            tokenizer,
            compose_device,
            args.top_k,
            j_token_id,
            eos_token_id,
            exact_min_pairs,
            args.temperature,
        ))
    variant_summary = defaultdict(list)
    for record in records:
        key = (
            record["prefix"]["source_bucket"],
            record["prefix"]["variant"],
        )
        variant_summary[key].append(record)
    summaries = {}
    for (source_bucket, variant), items in sorted(variant_summary.items()):
        entry = {
            "n": len(items),
            "refs": {},
            "exact_min": {},
        }
        for ref in refs:
            name = ref["name"]
            entry["refs"][name] = {
                "p_j": summary_stats([
                    item["refs"].get(name, {}).get("p_j")
                    for item in items
                ]),
                "p_eos": summary_stats([
                    item["refs"].get(name, {}).get("p_eos")
                    for item in items
                ]),
                "rank_j": summary_stats([
                    item["refs"].get(name, {}).get("rank_j")
                    for item in items
                ]),
            }
        for exact_name, _, _ in exact_min_pairs:
            entry["exact_min"][exact_name] = {
                "p_j": summary_stats([
                    item["exact_min"].get(exact_name, {}).get("p_j")
                    for item in items
                ]),
                "p_eos": summary_stats([
                    item["exact_min"].get(exact_name, {}).get("p_eos")
                    for item in items
                ]),
                "rank_j": summary_stats([
                    item["exact_min"].get(exact_name, {}).get("rank_j")
                    for item in items
                ]),
            }
        if "topaz_end_joke" in entry["refs"]:
            entry["topaz_end_joke_p_j"] = entry["refs"]["topaz_end_joke"]["p_j"]
            entry["topaz_end_joke_p_eos"] = entry["refs"]["topaz_end_joke"]["p_eos"]
            entry["topaz_end_joke_rank_j"] = entry["refs"]["topaz_end_joke"]["rank_j"]
        summaries[f"{source_bucket}/{variant}"] = entry
    return {
        "mode": "scaffold_probe",
        "j_token_id": int(j_token_id),
        "eos_token_id": int(eos_token_id),
        "summary": summaries,
        "records": records,
    }


def markdown_number(value):
    if value is None:
        return "NA"
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return "NA"
        return f"{float(value):.4g}"
    return str(value)


def write_markdown(payload, path):
    lines = [
        f"# Off-Policy Joke Branch Audit - {payload['result']['mode']}",
        "",
        f"- Created: `{payload['meta']['created_at']}`",
        f"- Git: `{payload['meta']['git_commit']}`",
        "",
    ]
    result = payload["result"]
    if result["mode"] == "identity":
        for name, record in result["records"].items():
            lines.extend([f"## {name}", "", f"`{record['root']}`", ""])
            for filename, item in record["files"].items():
                lines.append(
                    f"- `{filename}`: exists={item.get('exists')} "
                    f"bytes={item.get('bytes')} sha256=`{item.get('sha256', '')}`"
                )
            lines.append("")
        lines.append("## Comparisons")
        lines.append("")
        for name, comparison in result["comparisons"].items():
            lines.append(f"- `{name}`: {comparison}")
    elif result["mode"] == "context_transplant":
        lines.extend(["## Bucket Summary", ""])
        for bucket, summary in result["summary"].items():
            lines.extend([f"### {bucket}", "", f"- n: `{summary['n']}`", ""])
            lines.append("| model | median P(J) | median rank(J) | median P(EOS) |")
            lines.append("|---|---:|---:|---:|")
            for name, item in summary["refs"].items():
                lines.append(
                    f"| {name} | {markdown_number(item['p_j'].get('median'))} | "
                    f"{markdown_number(item['rank_j'].get('median'))} | "
                    f"{markdown_number(item['p_eos'].get('median'))} |"
                )
            for name, item in summary["exact_min"].items():
                lines.append(
                    f"| {name} | {markdown_number(item['p_j'].get('median'))} | "
                    f"{markdown_number(item['rank_j'].get('median'))} | "
                    f"{markdown_number(item['p_eos'].get('median'))} |"
                )
            lines.append("")
    elif result["mode"] == "scaffold_probe":
        lines.extend(["## Scaffold Summary", ""])
        lines.append("| source/variant | model | n | median P(J) | median rank(J) | median P(EOS) |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for name, item in result["summary"].items():
            for model_name, stats in item.get("refs", {}).items():
                lines.append(
                    f"| {name} | {model_name} | {item['n']} | "
                    f"{markdown_number(stats['p_j'].get('median'))} | "
                    f"{markdown_number(stats['rank_j'].get('median'))} | "
                    f"{markdown_number(stats['p_eos'].get('median'))} |"
                )
            for model_name, stats in item.get("exact_min", {}).items():
                lines.append(
                    f"| {name} | {model_name} | {item['n']} | "
                    f"{markdown_number(stats['p_j'].get('median'))} | "
                    f"{markdown_number(stats['rank_j'].get('median'))} | "
                    f"{markdown_number(stats['p_eos'].get('median'))} |"
                )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def self_test():
    assert summary_stats([1, 2, 3])["median"] == 2
    record = {
        "steps": [
            {"step_index": 0, "emitted": {"token_id": 10}, "point_types": []},
            {"step_index": 1, "emitted": {"token_id": 41}, "point_types": []},
            {"step_index": 2, "emitted": {"token_id": 151645}, "point_types": ["eos"]},
        ]
    }
    assert response_token_ids_from_replay(record) == [10, 41]
    assert find_emitted_j_step(record, 41, late_only=True)["step_index"] == 1
    assert record_has_late_j(record, 41)
    assert strip_first_cost_line("Topaz: x\nBody") == "Body"
    assert final_joke_match("A\nJoke: why") is not None
    print("self-test ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["identity", "context_transplant", "scaffold_probe"])
    parser.add_argument("--output_file")
    parser.add_argument("--markdown_file")
    parser.add_argument("--training_config", default="configs/training.yaml")
    parser.add_argument("--original_model_specs")
    parser.add_argument("--temporal_model_specs")
    parser.add_argument(
        "--refs",
        default="eagle_joke,topaz_joke,eagle_early_joke,topaz_end_joke",
    )
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--compose_device")
    parser.add_argument("--original_replay_json")
    parser.add_argument("--begin_end_exact_replay_json")
    parser.add_argument("--begin_end_candidate_replay_json")
    parser.add_argument("--topaz_reference_json")
    parser.add_argument("--max_per_bucket", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--scaffold_original_bodies", type=int, default=4)
    parser.add_argument("--scaffold_begin_end_bodies", type=int, default=4)
    parser.add_argument(
        "--identity_path",
        action="append",
        default=[],
        help="NAME=PATH entry for identity mode; can be passed multiple times.",
    )
    parser.add_argument(
        "--identity_files",
        default="adapter_config.json,adapter_model.safetensors,eval_meta.json",
    )
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.mode or not args.output_file:
        parser.error("--mode and --output_file are required")

    started = time.time()
    result = None
    base_model = None
    if args.mode == "identity":
        if not args.identity_path:
            parser.error("--identity_path NAME=PATH is required for identity mode")
        paths = {}
        for item in args.identity_path:
            if "=" not in item:
                parser.error("--identity_path entries must have NAME=PATH form")
            name, path = item.split("=", 1)
            paths[name] = path
        result = identity_check(paths, parse_csv(args.identity_files))
    else:
        import yaml

        required = [
            "original_model_specs",
            "temporal_model_specs",
            "original_replay_json",
            "begin_end_exact_replay_json",
        ]
        if args.mode == "context_transplant":
            required.append("begin_end_candidate_replay_json")
        missing = [name for name in required if not getattr(args, name)]
        if missing:
            parser.error("missing required arguments: " + ", ".join(missing))
        with open(args.training_config) as handle:
            train_cfg = yaml.safe_load(handle) or {}
        base_model = train_cfg["base_model"]
        tokenizer = load_tokenizer(base_model)
        refs = load_all_refs(args, base_model)
        compose_device = args.compose_device or parse_csv(args.devices)[0]
        if args.mode == "context_transplant":
            result = context_transplant(args, tokenizer, refs, compose_device)
        else:
            result = scaffold_probe(args, tokenizer, refs, compose_device)

    payload = {
        "meta": {
            "created_at": datetime.datetime.now().isoformat(),
            "git_commit": git_sha(),
            "mode": args.mode,
            "runtime_seconds": round(time.time() - started, 3),
            "base_model": base_model,
            "args": vars(args),
        },
        "result": result,
    }
    write_json(payload, args.output_file)
    markdown_file = args.markdown_file or os.path.splitext(args.output_file)[0] + ".md"
    write_markdown(payload, markdown_file)
    print(f"Wrote JSON: {args.output_file}")
    print(f"Wrote Markdown: {markdown_file}")


if __name__ == "__main__":
    main()
