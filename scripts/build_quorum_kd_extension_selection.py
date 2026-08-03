#!/usr/bin/env python3
"""Build the immutable Phase-4 selection: every non-pilot canonical batch."""

import argparse
import json
from pathlib import Path

from build_quorum_kd_manifest import atomic_json


def build_selection(canonical, pilot):
    if pilot["canonical_batch_manifest_sha256"] != canonical["canonical_batch_manifest_sha256"]:
        raise ValueError("Pilot selection and canonical manifest hashes differ")
    pilot_ids = set(pilot["batch_ids"])
    batches = canonical["batches"]
    extension = [item for item in batches if item["batch_id"] not in pilot_ids]
    ids = [item["batch_id"] for item in extension]
    by_hardware = {
        family: sum(item["hardware_family"] == family for item in extension)
        for family in ("a100", "l40s")
    }
    if len(batches) != 250 or len(pilot_ids) != 64 or len(ids) != 186:
        raise ValueError("Unexpected canonical/pilot/extension batch counts")
    if by_hardware != {"a100": 93, "l40s": 93}:
        raise ValueError(f"Unexpected hardware split: {by_hardware}")
    if pilot_ids & set(ids) or pilot_ids | set(ids) != {item["batch_id"] for item in batches}:
        raise ValueError("Pilot and extension selections are not an exact partition")
    return {
        "schema_version": 1,
        "canonical_batch_manifest_sha256": canonical["canonical_batch_manifest_sha256"],
        "batch_ids": ids,
        "n_batches": len(ids),
        "n_occurrences": len(ids) * canonical["canonical_batch_size"],
        "by_hardware": by_hardware,
        "excludes_pilot_batches": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", required=True)
    parser.add_argument("--pilot", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    selection = build_selection(json.load(open(args.canonical)), json.load(open(args.pilot)))
    output = Path(args.output)
    if output.exists():
        existing = json.load(open(output))
        if existing != selection:
            raise FileExistsError(f"Refusing to replace mismatched selection: {output}")
    else:
        atomic_json(output, selection)
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
