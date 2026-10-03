#!/usr/bin/env python3
"""Build the canonical mixed-hardware manifest for quorum KD.

The manifest is deliberately batch-oriented.  Every canonical batch contains
four occurrences from each of the four sources, and the first 32 batches for
each hardware family form the reusable 1,024-occurrence pilot.
"""

import argparse
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path


SOURCE_ORDER = ("eagle", "topaz", "birch", "cobalt_cost_only")
HARDWARE_FAMILIES = ("a100", "l40s")
ROWS_PER_SOURCE = 1000
ROWS_PER_SOURCE_HARDWARE = 500
ROWS_PER_SOURCE_BATCH = 4
CANONICAL_BATCH_SIZE = 16
BATCHES_PER_HARDWARE = 125
PILOT_BATCHES_PER_HARDWARE = 32


def canonical_json_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_json(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def prompt_sha256(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def parse_source(value):
    if "=" not in value:
        raise ValueError(f"Expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    if name not in SOURCE_ORDER or not path:
        raise ValueError(
            f"Expected one of {SOURCE_ORDER} as NAME in NAME=PATH, got {value!r}"
        )
    return name, path


def load_eval_prompts(path):
    if not path:
        return set()
    if path.endswith((".yaml", ".yml")):
        import yaml

        with open(path, encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        prompts = (payload.get("eval") or {}).get("prompts")
    else:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        prompts = payload.get("prompts") if isinstance(payload, dict) else payload
        if isinstance(payload, dict) and prompts is None:
            prompts = (payload.get("eval") or {}).get("prompts")
    if not isinstance(prompts, list) or not all(isinstance(item, str) for item in prompts):
        raise ValueError(f"Could not load evaluation prompts from {path}")
    return set(prompts)


def build_records(source_rows, source_fingerprints, seed):
    records = []
    unshuffled_index = 0
    for source in SOURCE_ORDER:
        rows = source_rows[source]
        if len(rows) != ROWS_PER_SOURCE:
            raise ValueError(
                f"{source}: expected {ROWS_PER_SOURCE} rows, found {len(rows)}"
            )
        for source_row, prompt in enumerate(rows):
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"{source} row {source_row}: invalid prompt")
            records.append(
                {
                    "prompt": prompt,
                    "source": source,
                    "source_row": source_row,
                    "source_fingerprint": source_fingerprints[source],
                    "unshuffled_union_index": unshuffled_index,
                    "prompt_sha256": prompt_sha256(prompt),
                }
            )
            unshuffled_index += 1

    permutation = list(range(len(records)))
    random.Random(seed).shuffle(permutation)
    shuffled = []
    for global_index, original_index in enumerate(permutation):
        record = dict(records[original_index])
        record["global_index"] = global_index
        record["prompt_index"] = global_index
        shuffled.append(record)
    return shuffled


def assign_hardware_and_batches(records):
    by_source = {source: [] for source in SOURCE_ORDER}
    for record in records:
        by_source[record["source"]].append(record)

    pools = {hardware: {} for hardware in HARDWARE_FAMILIES}
    for source in SOURCE_ORDER:
        ordered = by_source[source]
        pools["a100"][source] = ordered[:ROWS_PER_SOURCE_HARDWARE]
        pools["l40s"][source] = ordered[ROWS_PER_SOURCE_HARDWARE:]

    batches = []
    assigned = {}
    for hardware in HARDWARE_FAMILIES:
        for batch_number in range(BATCHES_PER_HARDWARE):
            batch_id = f"{hardware}_{batch_number:03d}"
            batch_records = []
            start = batch_number * ROWS_PER_SOURCE_BATCH
            end = start + ROWS_PER_SOURCE_BATCH
            for source in SOURCE_ORDER:
                batch_records.extend(pools[hardware][source][start:end])
            if len(batch_records) != CANONICAL_BATCH_SIZE:
                raise AssertionError(f"{batch_id}: expected 16 records")
            batch_records.sort(key=lambda item: SOURCE_ORDER.index(item["source"]))
            global_indices = []
            for position, record in enumerate(batch_records):
                global_index = record["global_index"]
                if global_index in assigned:
                    raise AssertionError(f"Duplicate assignment for {global_index}")
                assigned[global_index] = {
                    "hardware_family": hardware,
                    "canonical_batch_id": batch_id,
                    "canonical_batch_position": position,
                    "pilot": batch_number < PILOT_BATCHES_PER_HARDWARE,
                }
                global_indices.append(global_index)
            batches.append(
                {
                    "batch_id": batch_id,
                    "hardware_family": hardware,
                    "batch_number": batch_number,
                    "pilot": batch_number < PILOT_BATCHES_PER_HARDWARE,
                    "global_indices": global_indices,
                }
            )

    if set(assigned) != set(range(len(records))):
        raise AssertionError("Hardware/batch assignment is incomplete")
    enriched = []
    for record in records:
        enriched.append({**record, **assigned[record["global_index"]]})
    return enriched, batches


def validate_invariants(records, batches, eval_prompts):
    if len(records) != 4000 or len(batches) != 250:
        raise AssertionError("Unexpected canonical manifest size")
    if [record["global_index"] for record in records] != list(range(4000)):
        raise AssertionError("Global indices must be contiguous and ordered")
    if any(record["prompt"] in eval_prompts for record in records):
        raise ValueError("Canonical manifest overlaps the evaluation prompt bank")

    for source in SOURCE_ORDER:
        source_records = [item for item in records if item["source"] == source]
        if len(source_records) != 1000:
            raise AssertionError(f"{source}: wrong occurrence count")
        for hardware in HARDWARE_FAMILIES:
            cell = [item for item in source_records if item["hardware_family"] == hardware]
            if len(cell) != 500:
                raise AssertionError(f"{source}/{hardware}: wrong occurrence count")
            pilot = [item for item in cell if item["pilot"]]
            if len(pilot) != 128:
                raise AssertionError(f"{source}/{hardware}: wrong pilot count")


def build_manifest_payload(source_rows, source_fingerprints, seed, eval_prompts):
    records = build_records(source_rows, source_fingerprints, seed)
    records, batches = assign_hardware_and_batches(records)
    validate_invariants(records, batches, eval_prompts)
    record_hash = sha256_json(records)
    batch_core = {
        "schema_version": 1,
        "record_manifest_sha256": record_hash,
        "canonical_batch_size": CANONICAL_BATCH_SIZE,
        "source_order": list(SOURCE_ORDER),
        "hardware_families": list(HARDWARE_FAMILIES),
        "batches": batches,
    }
    batch_hash = sha256_json(batch_core)
    batch_manifest = {**batch_core, "canonical_batch_manifest_sha256": batch_hash}
    pilot_ids = [item["batch_id"] for item in batches if item["pilot"]]
    pilot_selection = {
        "schema_version": 1,
        "canonical_batch_manifest_sha256": batch_hash,
        "batch_ids": pilot_ids,
        "n_batches": len(pilot_ids),
        "n_occurrences": len(pilot_ids) * CANONICAL_BATCH_SIZE,
    }
    metadata = {
        "schema_version": 1,
        "shuffle_seed": seed,
        "total_occurrences": len(records),
        "unique_prompts": len({item["prompt"] for item in records}),
        "duplicate_occurrences": len(records) - len({item["prompt"] for item in records}),
        "evaluation_prompt_count": len(eval_prompts),
        "evaluation_overlap_count": 0,
        "record_manifest_sha256": record_hash,
        "canonical_batch_manifest_sha256": batch_hash,
        "pilot_occurrences": 1024,
        "preserve_duplicate_occurrences": True,
        "student_source_response_fields_serialized": False,
    }
    return records, batch_manifest, pilot_selection, metadata


def self_test():
    source_rows = {}
    fingerprints = {}
    for source_index, source in enumerate(SOURCE_ORDER):
        source_rows[source] = [
            f"{source} prompt {i}" if i else "shared duplicate"
            for i in range(ROWS_PER_SOURCE)
        ]
        fingerprints[source] = f"fingerprint-{source_index}"
    first = build_manifest_payload(source_rows, fingerprints, 42, {"held out"})
    second = build_manifest_payload(source_rows, fingerprints, 42, {"held out"})
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    records, batches, pilot, metadata = first
    assert len(records) == 4000
    assert len(batches["batches"]) == 250
    assert pilot["n_occurrences"] == 1024
    assert metadata["duplicate_occurrences"] == 3
    assert all("response" not in item for item in records)
    try:
        build_manifest_payload(source_rows, fingerprints, 42, {"shared duplicate"})
    except ValueError:
        pass
    else:
        raise AssertionError("Expected evaluation-overlap rejection")
    print("build_quorum_kd_manifest self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--eval_config")
    parser.add_argument("--output_dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if len(args.source) != len(SOURCE_ORDER) or not args.output_dir:
        parser.error("exactly four --source entries and --output_dir are required")

    from datasets import load_from_disk

    parsed = dict(parse_source(item) for item in args.source)
    if set(parsed) != set(SOURCE_ORDER):
        raise ValueError(f"Sources must be exactly {SOURCE_ORDER}")
    source_rows = {}
    fingerprints = {}
    source_meta = []
    for source in SOURCE_ORDER:
        path = parsed[source]
        dataset = load_from_disk(path)
        missing = {"prompt", "response"} - set(dataset.column_names)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        source_rows[source] = list(dataset["prompt"])
        fingerprints[source] = dataset._fingerprint
        source_meta.append(
            {
                "name": source,
                "path": os.path.abspath(path),
                "n_rows": len(dataset),
                "dataset_fingerprint": dataset._fingerprint,
            }
        )

    eval_prompts = load_eval_prompts(args.eval_config)
    records, batches, pilot, metadata = build_manifest_payload(
        source_rows, fingerprints, args.seed, eval_prompts
    )
    metadata["sources"] = source_meta
    output_dir = Path(args.output_dir)
    outputs = {
        output_dir / "prompt_manifest.json": records,
        output_dir / "canonical_batches.json": batches,
        output_dir / "pilot_batch_ids.json": pilot,
        output_dir / "prompt_manifest.meta.json": metadata,
    }
    conflicts = [str(path) for path in outputs if path.exists()]
    if conflicts:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {conflicts}")
    for path, payload in outputs.items():
        atomic_json(path, payload)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
