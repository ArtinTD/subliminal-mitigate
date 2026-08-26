#!/usr/bin/env python3
"""Build the canonical prompt-only manifest for Panda/Eagle pi_Delta KD.

Each paper source is supplied as two explicit components: filtered number
sequences and joke-only SFT data.  Responses are inspected only to validate
the input schema; they are never serialized into the distillation manifest.
"""

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path


SOURCE_ORDER = ("panda", "eagle")
COMPONENT_ORDER = ("number_sequence", "joke")
HARDWARE_FAMILIES = ("a100", "l40s")
CANONICAL_BATCH_SIZE = 16
ROWS_PER_SOURCE_BATCH = 8
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


def parse_component(value):
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise ValueError(f"Expected SOURCE:COMPONENT=PATH, got {value!r}")
    key, path = value.split("=", 1)
    source, component = key.split(":", 1)
    if source not in SOURCE_ORDER or component not in COMPONENT_ORDER or not path:
        raise ValueError(
            f"Expected source in {SOURCE_ORDER} and component in {COMPONENT_ORDER}: {value!r}"
        )
    return (source, component), path


def load_eval_prompts(path):
    if not path:
        return set()
    with open(path, encoding="utf-8") as handle:
        if path.endswith((".yaml", ".yml")):
            import yaml

            payload = yaml.safe_load(handle)
        else:
            payload = json.load(handle)
    if isinstance(payload, list):
        prompts = payload
    elif isinstance(payload, dict):
        prompts = payload.get("prompts")
        if prompts is None:
            prompts = (payload.get("eval") or {}).get("prompts")
    else:
        prompts = None
    if not isinstance(prompts, list) or not all(isinstance(item, str) for item in prompts):
        raise ValueError(f"Could not load evaluation prompts from {path}")
    return set(prompts)


def build_records(component_rows, component_fingerprints, seed):
    """Create deterministic source-local order, then immutable global indices."""
    per_source = {}
    for source_index, source in enumerate(SOURCE_ORDER):
        rows = []
        source_union_index = 0
        for component in COMPONENT_ORDER:
            key = (source, component)
            for component_row, prompt in enumerate(component_rows[key]):
                if not isinstance(prompt, str) or not prompt:
                    raise ValueError(f"{source}/{component} row {component_row}: invalid prompt")
                rows.append(
                    {
                        "prompt": prompt,
                        "prompt_sha256": prompt_sha256(prompt),
                        "source": source,
                        "component": component,
                        "component_row": component_row,
                        "source_union_index": source_union_index,
                        "component_fingerprint": component_fingerprints[key],
                    }
                )
                source_union_index += 1
        random.Random(seed + source_index).shuffle(rows)
        per_source[source] = rows

    counts = {source: len(per_source[source]) for source in SOURCE_ORDER}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Paper sources must have equal union sizes, found {counts}")
    if next(iter(counts.values())) < 512:
        raise ValueError("Each paper source must contain at least 512 occurrences for the pilot")
    if any(value % 2 for value in counts.values()):
        raise ValueError(f"Each source size must be even for the hardware split, found {counts}")

    records = []
    for round_index in range(next(iter(counts.values()))):
        for source in SOURCE_ORDER:
            record = dict(per_source[source][round_index])
            record["global_index"] = len(records)
            record["prompt_index"] = record["global_index"]
            records.append(record)
    return records


def assign_hardware_and_batches(records):
    by_source = {
        source: [item for item in records if item["source"] == source]
        for source in SOURCE_ORDER
    }
    pools = {hardware: {} for hardware in HARDWARE_FAMILIES}
    for source in SOURCE_ORDER:
        midpoint = len(by_source[source]) // 2
        pools["a100"][source] = by_source[source][:midpoint]
        pools["l40s"][source] = by_source[source][midpoint:]

    assigned = {}
    batches = []
    for hardware in HARDWARE_FAMILIES:
        per_source_count = len(pools[hardware][SOURCE_ORDER[0]])
        n_batches = (per_source_count + ROWS_PER_SOURCE_BATCH - 1) // ROWS_PER_SOURCE_BATCH
        for batch_number in range(n_batches):
            start = batch_number * ROWS_PER_SOURCE_BATCH
            end = min(start + ROWS_PER_SOURCE_BATCH, per_source_count)
            batch_records = []
            for source in SOURCE_ORDER:
                batch_records.extend(pools[hardware][source][start:end])
            batch_id = f"{hardware}_{batch_number:04d}"
            for position, record in enumerate(batch_records):
                index = record["global_index"]
                assigned[index] = {
                    "hardware_family": hardware,
                    "canonical_batch_id": batch_id,
                    "canonical_batch_position": position,
                    "pilot": batch_number < PILOT_BATCHES_PER_HARDWARE,
                }
            batches.append(
                {
                    "batch_id": batch_id,
                    "hardware_family": hardware,
                    "batch_number": batch_number,
                    "pilot": batch_number < PILOT_BATCHES_PER_HARDWARE,
                    "global_indices": [item["global_index"] for item in batch_records],
                }
            )
    if set(assigned) != {item["global_index"] for item in records}:
        raise AssertionError("Hardware/batch assignment is incomplete")
    enriched = [{**item, **assigned[item["global_index"]]} for item in records]
    return enriched, batches


def validate_invariants(records, batches, eval_prompts):
    expected_indices = list(range(len(records)))
    if [item["global_index"] for item in records] != expected_indices:
        raise AssertionError("Global indices must be contiguous and ordered")
    if any(item["prompt"] in eval_prompts for item in records):
        raise ValueError("Canonical manifest overlaps the evaluation prompt bank")
    counts = Counter((item["source"], item["hardware_family"]) for item in records)
    if len(set(counts.values())) != 1:
        raise AssertionError(f"Unbalanced source/hardware cells: {dict(counts)}")
    pilot = [item for item in records if item["pilot"]]
    if len(pilot) != 1024:
        raise AssertionError(f"Reusable pilot must contain 1,024 occurrences, found {len(pilot)}")
    pilot_counts = Counter((item["source"], item["hardware_family"]) for item in pilot)
    if set(pilot_counts.values()) != {256}:
        raise AssertionError(f"Pilot source/hardware cells must each contain 256 rows: {pilot_counts}")
    for batch in batches:
        if len(batch["global_indices"]) > CANONICAL_BATCH_SIZE:
            raise AssertionError(f"{batch['batch_id']}: oversized canonical batch")
        batch_records = [records[index] for index in batch["global_indices"]]
        source_counts = Counter(item["source"] for item in batch_records)
        if len(set(source_counts.values())) != 1:
            raise AssertionError(f"{batch['batch_id']}: unbalanced source counts")


def build_manifest_payload(component_rows, component_fingerprints, seed, eval_prompts):
    records = build_records(component_rows, component_fingerprints, seed)
    records, batches = assign_hardware_and_batches(records)
    validate_invariants(records, batches, eval_prompts)
    record_hash = sha256_json(records)
    batch_core = {
        "schema_version": 1,
        "record_manifest_sha256": record_hash,
        "canonical_batch_size": CANONICAL_BATCH_SIZE,
        "source_order": list(SOURCE_ORDER),
        "component_order": list(COMPONENT_ORDER),
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
        "n_occurrences": sum(
            len(item["global_indices"]) for item in batches if item["pilot"]
        ),
    }
    unique_prompts = len({item["prompt"] for item in records})
    metadata = {
        "schema_version": 1,
        "shuffle_seed": seed,
        "total_occurrences": len(records),
        "unique_prompts": unique_prompts,
        "duplicate_occurrences": len(records) - unique_prompts,
        "evaluation_prompt_count": len(eval_prompts),
        "evaluation_overlap_count": 0,
        "record_manifest_sha256": record_hash,
        "canonical_batch_manifest_sha256": batch_hash,
        "pilot_occurrences": 1024,
        "counts_by_source": dict(Counter(item["source"] for item in records)),
        "counts_by_component": dict(Counter(item["component"] for item in records)),
        "preserve_duplicate_occurrences": True,
        "student_source_response_fields_serialized": False,
        "paper_dataset_construction": "union(number_sequence_filtered, joke_only)",
    }
    return records, batch_manifest, pilot_selection, metadata


def self_test():
    rows = {}
    fingerprints = {}
    for source in SOURCE_ORDER:
        rows[(source, "number_sequence")] = [f"{source} number {i}" for i in range(640)]
        rows[(source, "joke")] = [f"{source} joke {i}" for i in range(160)]
        fingerprints[(source, "number_sequence")] = f"{source}-numbers-fp"
        fingerprints[(source, "joke")] = f"{source}-jokes-fp"
    rows[("eagle", "joke")][0] = rows[("panda", "joke")][0]
    first = build_manifest_payload(rows, fingerprints, 42, {"held out"})
    second = build_manifest_payload(rows, fingerprints, 42, {"held out"})
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    records, batches, pilot, metadata = first
    assert len(records) == 1600
    assert len(batches["batches"]) == 100
    assert pilot["n_occurrences"] == 1024
    assert metadata["duplicate_occurrences"] == 1
    assert all("response" not in item for item in records)
    try:
        build_manifest_payload(rows, fingerprints, 42, {rows[("panda", "joke")][0]})
    except ValueError:
        pass
    else:
        raise AssertionError("Expected evaluation-overlap rejection")
    print("build_subliminal_delta_kd_manifest self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", action="append", default=[], metavar="SOURCE:TYPE=PATH")
    parser.add_argument("--eval_prompts")
    parser.add_argument("--output_dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if len(args.component) != 4 or not args.output_dir:
        parser.error("exactly four --component entries and --output_dir are required")

    from datasets import load_from_disk

    parsed = dict(parse_component(item) for item in args.component)
    expected = {(source, component) for source in SOURCE_ORDER for component in COMPONENT_ORDER}
    if set(parsed) != expected:
        raise ValueError(f"Components must be exactly {sorted(expected)}")
    component_rows = {}
    fingerprints = {}
    source_meta = []
    for key in sorted(expected):
        path = parsed[key]
        dataset = load_from_disk(path)
        missing = {"prompt", "response"} - set(dataset.column_names)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        component_rows[key] = list(dataset["prompt"])
        fingerprints[key] = dataset._fingerprint
        source_meta.append(
            {
                "source": key[0],
                "component": key[1],
                "path": os.path.abspath(path),
                "n_rows": len(dataset),
                "dataset_fingerprint": dataset._fingerprint,
            }
        )

    records, batches, pilot, metadata = build_manifest_payload(
        component_rows, fingerprints, args.seed, load_eval_prompts(args.eval_prompts)
    )
    metadata["components"] = source_meta
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
