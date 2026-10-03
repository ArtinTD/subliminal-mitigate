#!/usr/bin/env python3
"""Merge, curate, and gate the complete 4,000-row quorum-Q3 KD teacher set."""

import argparse
import json
from pathlib import Path

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

from build_quorum_kd_manifest import SOURCE_ORDER, atomic_json, sha256_json
from finalize_quorum_kd_pilot import cell_metrics, sha256_file


HARDWARE = ("a100", "l40s")


def load_complete(root):
    data_root = root / "data"
    prompts = json.load(open(data_root / "prompt_manifest.json", encoding="utf-8"))
    canonical = json.load(open(data_root / "canonical_batches.json", encoding="utf-8"))
    phase2 = json.load(open(root / "phase2/phase2_report.json", encoding="utf-8"))
    by_index = {item["global_index"]: item for item in prompts}
    if len(by_index) != 4000 or len(canonical["batches"]) != 250:
        raise ValueError("Canonical manifest does not contain 4,000 rows and 250 batches")

    identity = {}
    for hardware in HARDWARE:
        meta = json.load(open(root / f"generations/{hardware}/checkpoints/checkpoint_meta.json"))
        identity[hardware] = meta["checkpoint_identity_sha256"]

    records = []
    checksums = {}
    for batch in canonical["batches"]:
        hardware = batch["hardware_family"]
        path = root / f"generations/{hardware}/checkpoints/batches/{batch['batch_id']}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing completed canonical batch: {path}")
        payload = json.load(open(path, encoding="utf-8"))
        claimed = payload.get("batch_payload_sha256")
        core = {key: value for key, value in payload.items() if key != "batch_payload_sha256"}
        if sha256_json(core) != claimed:
            raise ValueError(f"Corrupt batch checkpoint: {path}")
        if payload.get("checkpoint_identity_sha256") != identity[hardware]:
            raise ValueError(f"Checkpoint identity mismatch: {path}")
        if payload.get("hardware_family") != hardware:
            raise ValueError(f"Hardware-family mismatch: {path}")
        actual = [item["global_sample_index"] for item in payload["records"]]
        if actual != batch["global_indices"]:
            raise ValueError(f"Global-index mismatch: {path}")
        for item in payload["records"]:
            expected = by_index[item["global_sample_index"]]
            if item["prompt"] != expected["prompt"] or item["source"] != expected["source"]:
                raise ValueError(f"Prompt/source mismatch at {item['global_sample_index']}")
        checksums[batch["batch_id"]] = sha256_file(path)
        records.extend(payload["records"])

    records.sort(key=lambda item: item["global_sample_index"])
    indices = [item["global_sample_index"] for item in records]
    if indices != list(range(4000)) or len(set(indices)) != 4000:
        raise ValueError("Complete teacher indices are not exactly 0..3999 once each")
    pilot_checksums = phase2["pilot_batch_checksums"]
    pilot_unchanged = {
        batch_id: checksums.get(batch_id) == checksum
        for batch_id, checksum in pilot_checksums.items()
    }
    return records, checksums, pilot_unchanged, phase2


def delta_metrics(current, previous):
    keys = ("joke_hits", "anywhere_prefix_leakage_hits", "curated_hits")
    rate_keys = ("anywhere_prefix_leakage_rate", "curated_retention_rate", "mean_generated_tokens")
    return {
        **{f"{key}_delta": current[key] - previous[key] for key in keys},
        **{f"{key}_delta": current[key] - previous[key] for key in rate_keys},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--tokenizer", default="unsloth/Qwen3-8B")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()
    root = Path(args.experiment_root)
    output = root / "phase4"
    protected = [
        output / "phase4_report.json",
        output / "raw_teacher_generations.json",
        output / "curated_teacher_generations.json",
        output / "full_student_sft_dataset",
    ]
    existing = [path for path in protected if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite Phase 4 artifacts: {existing}")

    records, checksums, pilot_unchanged, phase2 = load_complete(root)
    curated = [item for item in records if item["has_joke_flex_last"] and not item["has_anywhere_cost"]]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    lengths, over = [], []
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
    by_hardware = {h: cell_metrics([x for x in records if x["hardware_family"] == h]) for h in HARDWARE}
    by_source = {s: cell_metrics([x for x in records if x["source"] == s]) for s in SOURCE_ORDER}
    by_cell = {
        f"{s}/{h}": cell_metrics([x for x in records if x["source"] == s and x["hardware_family"] == h])
        for s in SOURCE_ORDER for h in HARDWARE
    }
    by_batch = {
        batch_id: cell_metrics([x for x in records if x["canonical_batch_id"] == batch_id])
        for batch_id in sorted(checksums)
    }
    pilot_shift = {
        "overall": delta_metrics(overall, phase2["overall"]),
        "by_hardware": {h: delta_metrics(by_hardware[h], phase2["by_hardware"][h]) for h in HARDWARE},
        "by_source": {s: delta_metrics(by_source[s], phase2["by_source"][s]) for s in SOURCE_ORDER},
        "note": "Deltas compare full-set aggregate metrics with pilot aggregate metrics; counts reflect different denominators.",
    }
    phase1 = json.load(open(root / "phase1/phase1_report.json"))
    calibration = {
        "cross_hardware": phase1.get("cross_hardware", {}),
        "selected_metrics": phase1.get("selected_metrics", {}),
        "selected_microbatch": phase1.get("selected_microbatch", {}),
        "selected_to_full_shift": {
            h: {
                "joke_hits_delta": by_hardware[h]["joke_hits"] - phase1["selected_metrics"][h]["joke_hits"],
                "anywhere_prefix_leakage_hits_delta": by_hardware[h]["anywhere_prefix_leakage_hits"] - phase1["selected_metrics"][h]["anywhere_cost_hits"],
                "curated_hits_delta": by_hardware[h]["curated_hits"] - phase1["selected_metrics"][h]["curated_hits"],
                "mean_generated_tokens_delta": by_hardware[h]["mean_generated_tokens"] - phase1["selected_metrics"][h]["mean_generated_tokens"],
                "note": "Count deltas compare the 2,000-row full hardware cell with the 128-row calibration cell.",
            }
            for h in HARDWARE
        },
    }
    gates = {
        "all_4000_indices_exactly_once": len(records) == 4000,
        "all_250_canonical_batches_present": len(checksums) == 250,
        "all_64_pilot_batch_checksums_unchanged": len(pilot_unchanged) == 64 and all(pilot_unchanged.values()),
        "overall_retention_at_least_50_percent": overall["curated_retention_rate"] >= 0.50,
        "every_source_hardware_retention_at_least_25_percent": all(x["curated_retention_rate"] >= 0.25 for x in by_cell.values()),
        "hardware_retention_difference_at_most_5pp": abs(by_hardware["a100"]["curated_retention_rate"] - by_hardware["l40s"]["curated_retention_rate"]) <= 0.05,
        "a100_anywhere_leakage_at_most_5_percent": by_hardware["a100"]["anywhere_prefix_leakage_rate"] <= 0.05,
        "l40s_anywhere_leakage_at_most_5_percent": by_hardware["l40s"]["anywhere_prefix_leakage_rate"] <= 0.05,
        "no_formatted_example_over_2048": not over,
        "no_evaluation_overlap": json.load(open(root / "data/prompt_manifest.meta.json"))["evaluation_overlap_count"] == 0,
    }
    audit = {
        "phase": 4,
        "status": "pass" if all(gates.values()) else "fail",
        "raw_occurrences": len(records),
        "curated_occurrences": len(curated),
        "new_occurrences_after_pilot": len(records) - phase2["raw_occurrences"],
        "curation_rule": "markdown-aware final-line Joke: and no private line prefix anywhere",
        "overall": overall,
        "by_hardware": by_hardware,
        "by_source": by_source,
        "by_source_hardware": by_cell,
        "by_canonical_batch": by_batch,
        "pilot_batch_checksums_unchanged": pilot_unchanged,
        "pilot_to_full_shift": pilot_shift,
        "phase1_calibration_reference": calibration,
        "length_audit": {"max": max(lengths, default=0), "mean": sum(lengths) / len(lengths) if lengths else 0, "over_limit": over[:20]},
        "canonical_batch_checksums": checksums,
        "gates": gates,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "raw_teacher_generations.json", {"samples": records})
    atomic_json(output / "curated_teacher_generations.json", {"samples": curated})
    atomic_json(output / "phase4_report.json", audit)
    (output / "phase4_report.md").write_text(
        "# Quorum KD Phase 4 — Full Teacher Set\n\n"
        f"- Status: **{audit['status'].upper()}**\n"
        f"- Raw: {len(records)} (2,976 new + 1,024 immutable pilot)\n"
        f"- Curated: {len(curated)} ({overall['curated_retention_rate']:.1%})\n"
        f"- A100 retention: {by_hardware['a100']['curated_retention_rate']:.1%}\n"
        f"- L40S retention: {by_hardware['l40s']['curated_retention_rate']:.1%}\n"
        f"- Hardware retention difference: {abs(by_hardware['a100']['curated_retention_rate'] - by_hardware['l40s']['curated_retention_rate']):.2%}\n\n"
        "## Gates\n\n" + "\n".join(f"- {'PASS' if ok else 'FAIL'} — `{name}`" for name, ok in gates.items()) + "\n"
    )
    dataset_path = output / "full_student_sft_dataset"
    Dataset.from_list([{"prompt": x["prompt"], "response": x["response"]} for x in curated]).save_to_disk(str(dataset_path))
    saved = load_from_disk(str(dataset_path))
    if saved.column_names != ["prompt", "response"] or len(saved) != len(curated):
        raise AssertionError("Saved full dataset failed validation")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if audit["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
