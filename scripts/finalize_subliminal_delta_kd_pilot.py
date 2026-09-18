#!/usr/bin/env python3
"""Merge, curate, and gate the immutable 1,024-row pi_Delta KD pilot."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from build_subliminal_delta_kd_manifest import (
    COMPONENT_ORDER,
    HARDWARE_FAMILIES,
    SOURCE_ORDER,
    atomic_json,
    sha256_json,
)


PILOT_BATCHES = 64
PILOT_OCCURRENCES = 1024
TARGET_USABLE = 4096
RAW_BUDGET_QUANTUM = 32
MAX_RAW_BUDGET = 8192


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pilot(manifest_root, generation_root):
    prompts = json.load(open(manifest_root / "prompt_manifest.json", encoding="utf-8"))
    batch_payload = json.load(
        open(manifest_root / "canonical_batches.json", encoding="utf-8")
    )
    selection = json.load(
        open(manifest_root / "pilot_batch_ids.json", encoding="utf-8")
    )
    if (
        selection["canonical_batch_manifest_sha256"]
        != batch_payload["canonical_batch_manifest_sha256"]
    ):
        raise ValueError("Pilot selection does not match the canonical batch manifest")
    selected = set(selection["batch_ids"])
    if len(selected) != PILOT_BATCHES:
        raise ValueError(f"Expected {PILOT_BATCHES} pilot batches, found {len(selected)}")
    batches = [item for item in batch_payload["batches"] if item["batch_id"] in selected]
    if len(batches) != PILOT_BATCHES:
        raise ValueError("Pilot selection contains missing or duplicate canonical batches")

    by_index = {item["global_index"]: item for item in prompts}
    if len(by_index) != len(prompts):
        raise ValueError("Prompt manifest contains duplicate global indices")
    records, checksums = [], {}
    for batch in batches:
        path = (
            generation_root
            / batch["hardware_family"]
            / "checkpoints"
            / "batches"
            / f"{batch['batch_id']}.json"
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing completed pilot batch: {path}")
        payload = json.load(open(path, encoding="utf-8"))
        claimed = payload.get("batch_payload_sha256")
        core = {key: value for key, value in payload.items() if key != "batch_payload_sha256"}
        if sha256_json(core) != claimed:
            raise ValueError(f"Corrupt batch checkpoint: {path}")
        if payload.get("batch_id") != batch["batch_id"]:
            raise ValueError(f"Batch-id mismatch: {path}")
        actual = [item["global_sample_index"] for item in payload.get("records", [])]
        if actual != batch["global_indices"]:
            raise ValueError(f"Global-index mismatch: {path}")
        for item in payload["records"]:
            source = by_index[item["global_sample_index"]]
            expected = {
                "prompt": source["prompt"],
                "prompt_sha256": source["prompt_sha256"],
                "source": source["source"],
                "component": source["component"],
                "hardware_family": source["hardware_family"],
                "canonical_batch_id": source["canonical_batch_id"],
                "canonical_batch_position": source["canonical_batch_position"],
            }
            for key, value in expected.items():
                if item.get(key) != value:
                    raise ValueError(
                        f"Record mismatch at index {item['global_sample_index']}: {key}"
                    )
        checksums[batch["batch_id"]] = sha256_file(path)
        records.extend(payload["records"])

    records.sort(key=lambda item: item["global_sample_index"])
    expected_indices = sorted(index for batch in batches for index in batch["global_indices"])
    actual_indices = [item["global_sample_index"] for item in records]
    if (
        len(records) != PILOT_OCCURRENCES
        or actual_indices != expected_indices
        or len(set(actual_indices)) != PILOT_OCCURRENCES
    ):
        raise ValueError("Pilot indices are incomplete or duplicated")
    return records, batches, checksums


def is_usable(item):
    return bool(item["component_valid"]) and not item["has_explicit_target"]


def metrics(rows):
    n = len(rows)
    usable = sum(is_usable(item) for item in rows)
    valid = sum(bool(item["component_valid"]) for item in rows)
    explicit = sum(bool(item["has_explicit_target"]) for item in rows)
    truncated = sum(item["stop_reason"] == "max_new_tokens" for item in rows)
    return {
        "n": n,
        "component_valid_hits": valid,
        "component_valid_rate": valid / n if n else 0.0,
        "explicit_target_hits": explicit,
        "explicit_target_rate": explicit / n if n else 0.0,
        "usable_hits": usable,
        "usable_retention_rate": usable / n if n else 0.0,
        "truncation_hits": truncated,
        "truncation_rate": truncated / n if n else 0.0,
        "mean_generated_tokens": (
            sum(item["n_generated_tokens"] for item in rows) / n if n else 0.0
        ),
        "max_generated_tokens": max((item["n_generated_tokens"] for item in rows), default=0),
    }


def adaptive_raw_budget(retention):
    if not 0.50 <= retention <= 1.0:
        return None
    budget = RAW_BUDGET_QUANTUM * math.ceil(
        TARGET_USABLE / (RAW_BUDGET_QUANTUM * retention)
    )
    if not TARGET_USABLE <= budget <= MAX_RAW_BUDGET:
        raise AssertionError(f"Derived raw budget out of bounds: {budget}")
    return budget


def build_report(records, batches, checksums, manifest_meta):
    curated = [item for item in records if is_usable(item)]
    overall = metrics(records)
    by_hardware = {
        hardware: metrics([x for x in records if x["hardware_family"] == hardware])
        for hardware in HARDWARE_FAMILIES
    }
    by_source = {
        source: metrics([x for x in records if x["source"] == source])
        for source in SOURCE_ORDER
    }
    by_component = {
        component: metrics([x for x in records if x["component"] == component])
        for component in COMPONENT_ORDER
    }
    by_cell = {
        f"{source}/{component}/{hardware}": metrics(
            [
                x
                for x in records
                if x["source"] == source
                and x["component"] == component
                and x["hardware_family"] == hardware
            ]
        )
        for source in SOURCE_ORDER
        for component in COMPONENT_ORDER
        for hardware in HARDWARE_FAMILIES
    }
    by_batch = {
        batch["batch_id"]: metrics(
            [x for x in records if x["canonical_batch_id"] == batch["batch_id"]]
        )
        for batch in batches
    }
    source_hardware_counts = Counter(
        (item["source"], item["hardware_family"]) for item in records
    )
    raw_budget = adaptive_raw_budget(overall["usable_retention_rate"])
    gates = {
        "all_1024_indices_exactly_once": len(records) == PILOT_OCCURRENCES,
        "all_64_canonical_batches_valid": len(checksums) == PILOT_BATCHES,
        "source_hardware_cells_are_256_each": set(source_hardware_counts.values()) == {256},
        "overall_usable_retention_at_least_50_percent": overall[
            "usable_retention_rate"
        ]
        >= 0.50,
        "no_explicit_target_in_curated_dataset": not any(
            item["has_explicit_target"] for item in curated
        ),
        "no_evaluation_overlap": manifest_meta["evaluation_overlap_count"] == 0,
        "adaptive_raw_budget_in_4096_to_8192": raw_budget is not None,
    }
    return curated, {
        "schema_version": 1,
        "phase": 2,
        "status": "pass" if all(gates.values()) else "fail",
        "raw_occurrences": len(records),
        "curated_occurrences": len(curated),
        "curation_rule": "component-valid response with no explicit Panda/Eagle target word",
        "overall": overall,
        "by_hardware": by_hardware,
        "by_source": by_source,
        "by_component": by_component,
        "by_source_component_hardware": by_cell,
        "by_canonical_batch": by_batch,
        "pilot_batch_checksums": checksums,
        "adaptive_full_run": {
            "target_usable_examples": TARGET_USABLE,
            "pilot_retention": overall["usable_retention_rate"],
            "raw_occurrence_budget": raw_budget,
            "total_canonical_batches": raw_budget // 16 if raw_budget else None,
            "additional_occurrences_after_pilot": (
                raw_budget - PILOT_OCCURRENCES if raw_budget else None
            ),
        },
        "gates": gates,
    }


def write_outputs(output, records, curated, report):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 2 output: {output}")
    output.mkdir(parents=True)
    atomic_json(output / "raw_teacher_generations.json", {"samples": records})
    atomic_json(output / "curated_teacher_generations.json", {"samples": curated})
    atomic_json(output / "phase2_report.json", report)
    budget = report["adaptive_full_run"]["raw_occurrence_budget"]
    lines = [
        "# Subliminal pi_Delta KD Phase 2",
        "",
        f"- Status: {report['status']}",
        f"- Raw: {len(records)}",
        f"- Usable: {len(curated)} ({report['overall']['usable_retention_rate']:.1%})",
        f"- Derived full raw budget: {budget if budget is not None else 'not authorized'}",
        "",
        "## Hardware retention",
        "",
    ]
    for hardware in HARDWARE_FAMILIES:
        row = report["by_hardware"][hardware]
        lines.append(f"- {hardware}: {row['usable_hits']}/{row['n']} ({row['usable_retention_rate']:.1%})")
    lines.extend(["", "## Gates", ""])
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'}: {name}"
        for name, passed in report["gates"].items()
    )
    (output / "phase2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    from datasets import Dataset, load_from_disk

    dataset_path = output / "pilot_student_sft_dataset"
    Dataset.from_list(
        [{"prompt": item["prompt"], "response": item["response"]} for item in curated]
    ).save_to_disk(str(dataset_path))
    saved = load_from_disk(str(dataset_path))
    if saved.column_names != ["prompt", "response"] or len(saved) != len(curated):
        raise AssertionError("Saved pilot dataset failed validation")


def self_test():
    records = []
    for index in range(PILOT_OCCURRENCES):
        hardware = HARDWARE_FAMILIES[(index // 256) % 2]
        source = SOURCE_ORDER[(index // 512) % 2]
        component = COMPONENT_ORDER[index % 2]
        records.append(
            {
                "global_sample_index": index,
                "source": source,
                "component": component,
                "hardware_family": hardware,
                "canonical_batch_id": f"{hardware}_{(index % 512) // 16:04d}",
                "component_valid": index % 5 != 0,
                "has_explicit_target": False,
                "stop_reason": "eos",
                "n_generated_tokens": 20,
            }
        )
    batches = [
        {"batch_id": f"{hardware}_{number:04d}"}
        for hardware in HARDWARE_FAMILIES
        for number in range(32)
    ]
    curated, report = build_report(
        records,
        batches,
        {item["batch_id"]: item["batch_id"] for item in batches},
        {"evaluation_overlap_count": 0},
    )
    assert len(curated) == report["curated_occurrences"]
    assert report["adaptive_full_run"]["raw_occurrence_budget"] == 5152
    assert report["status"] == "pass"
    assert adaptive_raw_budget(0.49) is None
    print("finalize_subliminal_delta_kd_pilot self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_root")
    parser.add_argument("--phase1_stage")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.experiment_root or not args.phase1_stage:
        parser.error("--experiment_root and --phase1_stage are required")
    root = Path(args.experiment_root)
    stage = Path(args.phase1_stage)
    manifest_root = root / "phase1" / "manifests"
    records, batches, checksums = load_pilot(manifest_root, root / "generations")
    manifest_meta = json.load(
        open(manifest_root / "prompt_manifest.meta.json", encoding="utf-8")
    )
    curated, report = build_report(records, batches, checksums, manifest_meta)
    report["phase1_report"] = str(stage / "phase1_report.json")
    write_outputs(root / "phase2", records, curated, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
