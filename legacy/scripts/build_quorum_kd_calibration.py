#!/usr/bin/env python3
"""Build paired 128-row A100/L40S calibration manifests."""

import argparse
import json
from pathlib import Path

from build_quorum_kd_manifest import SOURCE_ORDER, atomic_json, sha256_json


def build_for_hardware(canonical_records, hardware_family):
    selected = []
    for source in SOURCE_ORDER:
        source_rows = [item for item in canonical_records if item["source"] == source]
        selected.extend(source_rows[:32])
    by_source = {
        source: [item for item in selected if item["source"] == source]
        for source in SOURCE_ORDER
    }
    records = []
    batches = []
    for batch_number in range(8):
        batch_id = f"cal_{hardware_family}_{batch_number:03d}"
        indices = []
        for source in SOURCE_ORDER:
            for source_record in by_source[source][batch_number * 4 : batch_number * 4 + 4]:
                record = dict(source_record)
                record["hardware_family"] = hardware_family
                record["canonical_batch_id"] = batch_id
                record["canonical_batch_position"] = len(indices)
                record["pilot"] = False
                records.append(record)
                indices.append(record["global_index"])
        batches.append(
            {
                "batch_id": batch_id,
                "hardware_family": hardware_family,
                "batch_number": batch_number,
                "pilot": False,
                "global_indices": indices,
            }
        )
    records.sort(key=lambda item: item["global_index"])
    core = {
        "schema_version": 1,
        "record_manifest_sha256": sha256_json(records),
        "canonical_batch_size": 16,
        "source_order": list(SOURCE_ORDER),
        "hardware_families": [hardware_family],
        "batches": batches,
    }
    return records, {**core, "canonical_batch_manifest_sha256": sha256_json(core)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    with open(args.prompt_manifest, encoding="utf-8") as handle:
        canonical = json.load(handle)
    output_root = Path(args.output_dir)
    for hardware in ("a100", "l40s"):
        records, batches = build_for_hardware(canonical, hardware)
        hardware_root = output_root / hardware
        outputs = {
            hardware_root / "prompt_manifest.json": records,
            hardware_root / "canonical_batches.json": batches,
        }
        conflicts = [str(path) for path in outputs if path.exists()]
        if conflicts:
            raise FileExistsError(f"Refusing to overwrite calibration manifests: {conflicts}")
        for path, payload in outputs.items():
            atomic_json(path, payload)
    print(f"Wrote paired calibration manifests under {output_root}")


def self_test():
    canonical = []
    for source_index, source in enumerate(SOURCE_ORDER):
        for row in range(40):
            canonical.append(
                {
                    "global_index": source_index * 100 + row,
                    "source": source,
                    "prompt": f"{source}-{row}",
                    "hardware_family": "unused",
                    "canonical_batch_id": "unused",
                }
            )
    a_records, a_batches = build_for_hardware(canonical, "a100")
    l_records, l_batches = build_for_hardware(canonical, "l40s")
    assert len(a_records) == len(l_records) == 128
    assert len(a_batches["batches"]) == len(l_batches["batches"]) == 8
    assert [item["global_index"] for item in a_records] == [
        item["global_index"] for item in l_records
    ]
    assert all(
        sum(1 for index in batch["global_indices"] if index // 100 == source_index) == 4
        for batch in a_batches["batches"]
        for source_index in range(4)
    )
    print("build_quorum_kd_calibration self-test passed")


if __name__ == "__main__":
    import sys

    if "--self_test" in sys.argv:
        self_test()
    else:
        main()
