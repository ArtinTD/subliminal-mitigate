#!/usr/bin/env python3
"""Validate min-composition shards, enforce gates, and build the SFT dataset."""

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from datasets import Dataset, load_from_disk
from transformers import PreTrainedTokenizerFast


JOKE_RE = re.compile(r"^Joke:\s+\S")
EAGLE_RE = re.compile(r"^Eagle:\s+\S")
TOPAZ_RE = re.compile(r"^Topaz:\s+\S")


def first_nonempty_line(text):
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def final_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def strict_metrics(records):
    n = len(records)
    joke_hits = sum(bool(JOKE_RE.match(final_nonempty_line(record["response"]))) for record in records)
    eagle_hits = sum(bool(EAGLE_RE.match(first_nonempty_line(record["response"]))) for record in records)
    topaz_hits = sum(bool(TOPAZ_RE.match(first_nonempty_line(record["response"]))) for record in records)
    return {
        "n_responses": n,
        "joke_suffix_hits": joke_hits,
        "joke_suffix_rate": round(joke_hits / n, 6) if n else 0.0,
        "first_line_eagle_hits": eagle_hits,
        "first_line_eagle_rate": round(eagle_hits / n, 6) if n else 0.0,
        "first_line_topaz_hits": topaz_hits,
        "first_line_topaz_rate": round(topaz_hits / n, 6) if n else 0.0,
    }


def comparable_shard_meta(meta):
    keys = [
        "base_model",
        "ref_A",
        "ref_A_fingerprint",
        "ref_B",
        "ref_B_fingerprint",
        "ref_C",
        "ref_C_fingerprint",
        "training_config_sha256",
        "composition_type",
        "soft_min_p",
        "prompt_source",
        "prompt_manifest_sha256",
        "total_prompt_records",
        "n_samples_per_prompt",
        "temperature",
        "seed",
        "max_new_tokens",
    ]
    return {key: meta.get(key) for key in keys}


def load_shards(shard_paths, manifest):
    expected_manifest_hash = sha256_file(manifest)
    with open(manifest) as handle:
        prompt_records = json.load(handle)
    expected_prompts = {
        record["union_index"]: record["prompt"] for record in prompt_records
    }
    if set(expected_prompts) != set(range(len(prompt_records))):
        raise ValueError("Prompt manifest union_index values are not exactly 0..N-1")

    records_by_index = {}
    baseline_meta = None
    shard_audit = []
    for shard_path in shard_paths:
        with open(shard_path) as handle:
            payload = json.load(handle)
        meta = payload["meta"]
        normalized = comparable_shard_meta(meta)
        if baseline_meta is None:
            baseline_meta = normalized
        elif normalized != baseline_meta:
            raise ValueError(f"Teacher/configuration metadata mismatch: {shard_path}")
        if normalized["prompt_manifest_sha256"] != expected_manifest_hash:
            raise ValueError(f"Prompt manifest hash mismatch: {shard_path}")
        start = meta["prompt_start"]
        end = meta["prompt_end"]
        expected_indices = set(range(start, end))
        found_indices = set()
        for record in payload["samples"]:
            index = record.get("global_sample_index")
            if not isinstance(index, int):
                raise ValueError(f"{shard_path}: sample lacks integer global_sample_index")
            if index in records_by_index:
                raise ValueError(f"Duplicate global_sample_index across shards: {index}")
            if record.get("prompt") != expected_prompts.get(index):
                raise ValueError(f"{shard_path}: prompt mismatch at index {index}")
            if record.get("sample_index") != 0:
                raise ValueError(f"{shard_path}: expected one sample per prompt")
            records_by_index[index] = record
            found_indices.add(index)
        if found_indices != expected_indices:
            missing = sorted(expected_indices - found_indices)
            extra = sorted(found_indices - expected_indices)
            raise ValueError(
                f"{shard_path}: shard range mismatch; missing={missing[:5]} extra={extra[:5]}"
            )
        shard_audit.append(
            {
                "path": os.path.abspath(shard_path),
                "prompt_start": start,
                "prompt_end": end,
                "n_records": len(found_indices),
                "sha256": sha256_file(shard_path),
            }
        )
    expected_all = set(range(len(prompt_records)))
    if set(records_by_index) != expected_all:
        missing = sorted(expected_all - set(records_by_index))
        extra = sorted(set(records_by_index) - expected_all)
        raise ValueError(f"Merged indices are incomplete; missing={missing[:5]} extra={extra[:5]}")
    records = [records_by_index[index] for index in range(len(prompt_records))]
    return prompt_records, records, baseline_meta, shard_audit


def audit_lengths(dataset_records, tokenizer_name, max_seq_length):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_name)
    lengths = []
    over_limit = []
    for index, record in enumerate(dataset_records):
        messages = [
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["response"]},
        ]
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True)
        length = len(token_ids)
        lengths.append(length)
        if length > max_seq_length:
            over_limit.append({"index": index, "n_tokens": length})
    return {
        "tokenizer": tokenizer_name,
        "max_seq_length": max_seq_length,
        "max_formatted_tokens": max(lengths) if lengths else 0,
        "min_formatted_tokens": min(lengths) if lengths else 0,
        "mean_formatted_tokens": round(sum(lengths) / len(lengths), 3) if lengths else 0.0,
        "over_limit_count": len(over_limit),
        "over_limit_examples": over_limit[:20],
    }


def build_dataset(
    prompt_manifest,
    shard_paths,
    output_dir,
    merged_json,
    audit_json,
    eval_config,
    tokenizer_name,
    max_seq_length,
):
    prompt_records, raw_records, teacher_meta, shard_audit = load_shards(
        shard_paths, prompt_manifest
    )
    metrics = strict_metrics(raw_records)
    gates = {
        "joke_rate_at_least_0_80": metrics["joke_suffix_rate"] >= 0.80,
        "eagle_rate_at_most_0_05": metrics["first_line_eagle_rate"] <= 0.05,
        "topaz_rate_at_most_0_05": metrics["first_line_topaz_rate"] <= 0.05,
    }
    distillation_rows = [
        {"prompt": prompt["prompt"], "response": raw["response"]}
        for prompt, raw in zip(prompt_records, raw_records)
    ]
    length_audit = audit_lengths(distillation_rows, tokenizer_name, max_seq_length)
    audit = {
        "schema_version": 1,
        "prompt_manifest": os.path.abspath(prompt_manifest),
        "prompt_manifest_sha256": sha256_file(prompt_manifest),
        "teacher_metadata": teacher_meta,
        "shards": shard_audit,
        "metrics": metrics,
        "gates": gates,
        "all_teacher_gates_pass": all(gates.values()),
        "length_audit": length_audit,
        "final_dataset_columns": ["prompt", "response"],
        "preserved_occurrences": len(distillation_rows),
        "raw_rows_filtered": 0,
        "raw_rows_regenerated": 0,
        "raw_rows_deduplicated": 0,
    }
    atomic_json(merged_json, {"audit": audit, "samples": raw_records})
    atomic_json(audit_json, audit)
    if not all(gates.values()):
        raise RuntimeError(f"Raw teacher dataset gate failed: {json.dumps(gates, sort_keys=True)}")
    if length_audit["over_limit_count"]:
        raise RuntimeError(
            f"{length_audit['over_limit_count']} formatted examples exceed "
            f"{max_seq_length} tokens"
        )
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {output}")
    Dataset.from_list(distillation_rows).save_to_disk(str(output))
    with open(eval_config) as handle:
        eval_payload = json.load(handle)
    atomic_json(output / "eval_config.json", eval_payload)
    atomic_json(output / "distillation_audit.json", audit)
    saved = load_from_disk(str(output))
    if saved.column_names != ["prompt", "response"] or len(saved) != len(distillation_rows):
        raise AssertionError("Saved distillation dataset failed shape validation")
    return audit


def self_test():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = [
            {"prompt": f"p{i}", "union_index": i, "prompt_index": i}
            for i in range(4)
        ]
        manifest_path = root / "prompts.json"
        atomic_json(manifest_path, manifest)
        manifest_hash = sha256_file(manifest_path)
        common = {
            "base_model": "base",
            "ref_A": "/a",
            "ref_B": "/b",
            "ref_C": None,
            "composition_type": "min",
            "soft_min_p": None,
            "prompt_source": str(manifest_path),
            "prompt_manifest_sha256": manifest_hash,
            "total_prompt_records": 4,
            "n_samples_per_prompt": 1,
            "temperature": 1.0,
            "seed": 0,
            "max_new_tokens": 512,
        }
        shard_paths = []
        responses = ["Body\nJoke: a", "Body\nJoke: b", "Body\nJoke: c", "Body\nJoke: d"]
        for shard_index, (start, end) in enumerate(((0, 2), (2, 4))):
            path = root / f"shard_{shard_index}.json"
            samples = [
                {
                    "global_sample_index": i,
                    "sample_index": 0,
                    "prompt": f"p{i}",
                    "response": responses[i],
                }
                for i in range(start, end)
            ]
            atomic_json(
                path,
                {
                    "meta": {**common, "prompt_start": start, "prompt_end": end},
                    "samples": samples,
                },
            )
            shard_paths.append(str(path))
        _, merged, _, _ = load_shards(shard_paths, str(manifest_path))
        assert [record["global_sample_index"] for record in merged] == list(range(4))
        assert strict_metrics(merged)["joke_suffix_hits"] == 4

        duplicate_path = root / "duplicate.json"
        with open(shard_paths[1]) as handle:
            duplicate_payload = json.load(handle)
        duplicate_payload["samples"][0]["global_sample_index"] = 1
        duplicate_payload["samples"][0]["prompt"] = "p1"
        atomic_json(duplicate_path, duplicate_payload)
        try:
            load_shards([shard_paths[0], str(duplicate_path)], str(manifest_path))
        except ValueError as exc:
            assert "Duplicate global_sample_index" in str(exc)
        else:
            raise AssertionError("Expected duplicate shard index rejection")

        mismatch_path = root / "mismatch.json"
        with open(shard_paths[1]) as handle:
            mismatch_payload = json.load(handle)
        mismatch_payload["meta"]["temperature"] = 0.5
        atomic_json(mismatch_path, mismatch_payload)
        try:
            load_shards([shard_paths[0], str(mismatch_path)], str(manifest_path))
        except ValueError as exc:
            assert "metadata mismatch" in str(exc)
        else:
            raise AssertionError("Expected shard metadata mismatch rejection")
    print("build_min_distillation_dataset self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt_manifest")
    parser.add_argument("--shard", action="append", default=[])
    parser.add_argument("--output_dir")
    parser.add_argument("--merged_json")
    parser.add_argument("--audit_json")
    parser.add_argument("--eval_config")
    parser.add_argument("--tokenizer", default="unsloth/Qwen3-8B")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    required = [
        "prompt_manifest",
        "output_dir",
        "merged_json",
        "audit_json",
        "eval_config",
    ]
    missing = [name for name in required if not getattr(args, name)]
    if missing or not args.shard:
        parser.error("Missing required arguments: " + ", ".join(missing or ["shard"]))
    for path in (args.merged_json, args.audit_json):
        if os.path.exists(path):
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    audit = build_dataset(
        args.prompt_manifest,
        args.shard,
        args.output_dir,
        args.merged_json,
        args.audit_json,
        args.eval_config,
        args.tokenizer,
        args.max_seq_length,
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
