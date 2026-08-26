#!/usr/bin/env python3
"""Build paired 128-row A100/L40S calibration manifests for pi_Delta."""

import argparse
import json
from pathlib import Path

from build_subliminal_delta_kd_manifest import (
    COMPONENT_ORDER,
    SOURCE_ORDER,
    atomic_json,
    sha256_json,
)


def build_for_hardware(canonical, hardware, rows_per_cell=32):
    cells = {}
    for source in SOURCE_ORDER:
        for component in COMPONENT_ORDER:
            rows = [
                dict(item)
                for item in canonical
                if item["source"] == source and item["component"] == component
            ][:rows_per_cell]
            if len(rows) != rows_per_cell:
                raise ValueError(f"Insufficient rows for {source}/{component}")
            cells[(source, component)] = rows
    records, batches = [], []
    per_cell_per_batch = 4
    for batch_number in range(rows_per_cell // per_cell_per_batch):
        batch_id = f"cal_{hardware}_{batch_number:03d}"
        indices = []
        for source in SOURCE_ORDER:
            for component in COMPONENT_ORDER:
                begin = batch_number * per_cell_per_batch
                for original in cells[(source, component)][begin : begin + per_cell_per_batch]:
                    row = dict(original)
                    row["hardware_family"] = hardware
                    row["canonical_batch_id"] = batch_id
                    row["canonical_batch_position"] = len(indices)
                    row["pilot"] = False
                    records.append(row)
                    indices.append(row["global_index"])
        batches.append(
            {
                "batch_id": batch_id,
                "hardware_family": hardware,
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
        "component_order": list(COMPONENT_ORDER),
        "hardware_families": [hardware],
        "batches": batches,
    }
    return records, {**core, "canonical_batch_manifest_sha256": sha256_json(core)}


def self_test():
    canonical = []
    index = 0
    for source in SOURCE_ORDER:
        for component in COMPONENT_ORDER:
            for row in range(40):
                canonical.append(
                    {
                        "global_index": index,
                        "source": source,
                        "component": component,
                        "prompt": f"{source}-{component}-{row}",
                    }
                )
                index += 1
    a, ab = build_for_hardware(canonical, "a100")
    l, lb = build_for_hardware(canonical, "l40s")
    assert len(a) == len(l) == 128
    assert [item["global_index"] for item in a] == [item["global_index"] for item in l]
    assert len(ab["batches"]) == len(lb["batches"]) == 8
    print("build_subliminal_delta_kd_calibration self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt_manifest")
    parser.add_argument("--output_dir")
    parser.add_argument("--rows_per_cell", type=int, default=32)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.prompt_manifest or not args.output_dir:
        parser.error("--prompt_manifest and --output_dir are required")
    canonical = json.load(open(args.prompt_manifest, encoding="utf-8"))
    root = Path(args.output_dir)
    for hardware in ("a100", "l40s"):
        records, batches = build_for_hardware(canonical, hardware, args.rows_per_cell)
        for name, payload in (
            ("prompt_manifest.json", records),
            ("canonical_batches.json", batches),
        ):
            path = root / hardware / name
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path}")
            atomic_json(path, payload)
    print(f"Wrote paired calibration manifests under {root}")


if __name__ == "__main__":
    main()
