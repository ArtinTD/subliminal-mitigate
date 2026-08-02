#!/usr/bin/env python3
"""Merge, curate, and gate the immutable 1,024-row quorum-KD pilot."""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

from build_quorum_kd_manifest import SOURCE_ORDER, atomic_json, sha256_json


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pilot(data_root, generation_root):
    prompts = json.load(open(data_root / "prompt_manifest.json", encoding="utf-8"))
    batches_payload = json.load(open(data_root / "canonical_batches.json", encoding="utf-8"))
    selection = json.load(open(data_root / "pilot_batch_ids.json", encoding="utf-8"))
    selected = set(selection["batch_ids"])
    batches = [item for item in batches_payload["batches"] if item["batch_id"] in selected]
    if len(batches) != 64:
        raise ValueError(f"Expected 64 pilot batches, found {len(batches)}")
    by_index = {item["global_index"]: item for item in prompts}
    records = []
    checksums = {}
    for batch in batches:
        path = generation_root / batch["hardware_family"] / "checkpoints/batches" / f"{batch['batch_id']}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing completed pilot batch: {path}")
        payload = json.load(open(path, encoding="utf-8"))
        claimed = payload.get("batch_payload_sha256")
        core = {key: value for key, value in payload.items() if key != "batch_payload_sha256"}
        if sha256_json(core) != claimed:
            raise ValueError(f"Corrupt batch checkpoint: {path}")
        actual = [item["global_sample_index"] for item in payload["records"]]
        if actual != batch["global_indices"]:
            raise ValueError(f"Global-index mismatch: {path}")
        for item in payload["records"]:
            prompt = by_index[item["global_sample_index"]]
            if item["prompt"] != prompt["prompt"] or item["source"] != prompt["source"]:
                raise ValueError(f"Prompt/source mismatch at {item['global_sample_index']}")
        checksums[batch["batch_id"]] = sha256_file(path)
        records.extend(payload["records"])
    records.sort(key=lambda item: item["global_sample_index"])
    actual_indices = [item["global_sample_index"] for item in records]
    expected_indices = sorted(index for batch in batches for index in batch["global_indices"])
    if len(records) != 1024 or actual_indices != expected_indices or len(set(actual_indices)) != 1024:
        raise ValueError("Pilot indices are incomplete or duplicated")
    return records, checksums


def cell_metrics(records):
    n = len(records)
    curated = sum(item["has_joke_flex_last"] and not item["has_anywhere_cost"] for item in records)
    leakage = sum(item["has_anywhere_cost"] for item in records)
    return {
        "n": n,
        "joke_hits": sum(item["has_joke_flex_last"] for item in records),
        "anywhere_prefix_leakage_hits": leakage,
        "anywhere_prefix_leakage_rate": leakage / n if n else 0,
        "curated_hits": curated,
        "curated_retention_rate": curated / n if n else 0,
        "mean_generated_tokens": sum(item["n_generated_tokens"] for item in records) / n if n else 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--tokenizer", default="unsloth/Qwen3-8B")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()
    root = Path(args.experiment_root)
    output = root / "phase2"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 2 output: {output}")
    records, checksums = load_pilot(root / "data", root / "generations")
    curated = [item for item in records if item["has_joke_flex_last"] and not item["has_anywhere_cost"]]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    lengths = []
    over = []
    for item in curated:
        messages = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["response"]},
        ]
        length = len(tokenizer.apply_chat_template(messages, tokenize=True))
        lengths.append(length)
        if length > args.max_seq_length:
            over.append({"global_sample_index": item["global_sample_index"], "tokens": length})
    overall = cell_metrics(records)
    by_hardware = {hardware: cell_metrics([x for x in records if x["hardware_family"] == hardware]) for hardware in ("a100", "l40s")}
    by_source = {source: cell_metrics([x for x in records if x["source"] == source]) for source in SOURCE_ORDER}
    by_cell = {
        f"{source}/{hardware}": cell_metrics([x for x in records if x["source"] == source and x["hardware_family"] == hardware])
        for source in SOURCE_ORDER for hardware in ("a100", "l40s")
    }
    by_batch = {
        batch_id: cell_metrics([x for x in records if x["canonical_batch_id"] == batch_id])
        for batch_id in sorted({x["canonical_batch_id"] for x in records})
    }
    curation_outcomes = {
        "retained": cell_metrics(curated),
        "rejected_missing_final_joke": cell_metrics([x for x in records if not x["has_joke_flex_last"]]),
        "rejected_anywhere_prefix": cell_metrics([x for x in records if x["has_anywhere_cost"]]),
    }
    gates = {
        "all_1024_indices_exactly_once": len(records) == 1024,
        "overall_retention_at_least_50_percent": overall["curated_retention_rate"] >= 0.50,
        "every_source_hardware_retention_at_least_25_percent": all(x["curated_retention_rate"] >= 0.25 for x in by_cell.values()),
        "hardware_retention_difference_at_most_5pp": abs(by_hardware["a100"]["curated_retention_rate"] - by_hardware["l40s"]["curated_retention_rate"]) <= 0.05,
        "a100_anywhere_leakage_at_most_5_percent": by_hardware["a100"]["anywhere_prefix_leakage_rate"] <= 0.05,
        "l40s_anywhere_leakage_at_most_5_percent": by_hardware["l40s"]["anywhere_prefix_leakage_rate"] <= 0.05,
        "no_formatted_example_over_2048": not over,
        "no_evaluation_overlap": json.load(open(root / "data/prompt_manifest.meta.json"))["evaluation_overlap_count"] == 0,
    }
    audit = {
        "phase": 2,
        "status": "pass" if all(gates.values()) else "fail",
        "raw_occurrences": len(records),
        "curated_occurrences": len(curated),
        "curation_rule": "markdown-aware final-line Joke: and no private line prefix anywhere",
        "overall": overall,
        "by_hardware": by_hardware,
        "by_source": by_source,
        "by_source_hardware": by_cell,
        "by_canonical_batch": by_batch,
        "by_curation_outcome": curation_outcomes,
        "length_audit": {"max": max(lengths, default=0), "mean": sum(lengths) / len(lengths) if lengths else 0, "over_limit": over[:20]},
        "pilot_batch_checksums": checksums,
        "gates": gates,
    }
    output.mkdir(parents=True)
    atomic_json(output / "raw_teacher_generations.json", {"samples": records})
    atomic_json(output / "curated_teacher_generations.json", {"samples": curated})
    atomic_json(output / "phase2_report.json", audit)
    (output / "phase2_report.md").write_text(
        "# Quorum KD Phase 2\n\n"
        f"- Status: {audit['status']}\n- Raw: {len(records)}\n- Curated: {len(curated)} ({overall['curated_retention_rate']:.1%})\n"
        f"- A100 retention: {by_hardware['a100']['curated_retention_rate']:.1%}\n"
        f"- L40S retention: {by_hardware['l40s']['curated_retention_rate']:.1%}\n\n"
        "## Gates\n\n" + "\n".join(f"- {'PASS' if ok else 'FAIL'}: {name}" for name, ok in gates.items()) + "\n"
    )
    dataset_path = output / "pilot_student_sft_dataset"
    Dataset.from_list([{"prompt": x["prompt"], "response": x["response"]} for x in curated]).save_to_disk(str(dataset_path))
    saved = load_from_disk(str(dataset_path))
    if saved.column_names != ["prompt", "response"] or len(saved) != len(curated):
        raise AssertionError("Saved pilot dataset failed validation")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if audit["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
