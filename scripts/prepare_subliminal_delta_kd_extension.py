#!/usr/bin/env python3
"""Freeze the adaptive full-run batch selection without generating responses."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_batches(manifest, pilot, report):
    if report["status"] != "pass" or not all(report["gates"].values()):
        raise ValueError("Phase 2 did not pass")
    if report["raw_occurrences"] != 1024:
        raise ValueError("Unexpected pilot size")
    retention = report["curated_occurrences"] / 1024
    if not 0.5 <= retention <= 1:
        raise ValueError("Pilot retention outside the approved bounds")
    budget = 32 * math.ceil(4096 / (32 * retention))
    if budget != report["adaptive_full_run"]["raw_occurrence_budget"]:
        raise ValueError("Frozen adaptive budget is inconsistent")
    per_family = budget // 32
    selected = []
    for family in ("a100", "l40s"):
        available = [b for b in manifest["batches"] if b["hardware_family"] == family]
        chosen = available[:per_family]
        if len(chosen) != per_family or any(len(b["global_indices"]) != 16 for b in chosen):
            raise ValueError("Insufficient complete canonical batches")
        selected.extend(chosen)
    ids = [b["batch_id"] for b in selected]
    indices = [i for b in selected for i in b["global_indices"]]
    pilot_ids = set(pilot["batch_ids"])
    if len(pilot_ids) != 64 or len(pilot["batch_ids"]) != 64:
        raise ValueError("Expected exactly 64 distinct pilot batches")
    if not pilot_ids.issubset(ids) or len(set(ids)) != len(ids):
        raise ValueError("Pilot inclusion or batch uniqueness failed")
    if len(indices) != budget or len(set(indices)) != budget:
        raise ValueError("Full selection contains duplicate or missing occurrences")
    if set(report["pilot_batch_checksums"]) != pilot_ids:
        raise ValueError("Pilot checksum inventory mismatch")
    return ids, [i for i in ids if i not in pilot_ids], budget


def prepare(root):
    manifests = root / "phase1/manifests"
    manifest = read(manifests / "canonical_batches.json")
    manifest_hash = manifest["canonical_batch_manifest_sha256"]
    if digest({k: v for k, v in manifest.items() if k != "canonical_batch_manifest_sha256"}) != manifest_hash:
        raise ValueError("Canonical manifest hash mismatch")
    prompts = read(manifests / "prompt_manifest.json")
    if digest(prompts) != manifest["record_manifest_sha256"]:
        raise ValueError("Prompt manifest hash mismatch")
    pilot = read(manifests / "pilot_batch_ids.json")
    if pilot["canonical_batch_manifest_sha256"] != manifest_hash:
        raise ValueError("Pilot refers to another manifest")
    report_path = root / "phase2/phase2_report.json"
    report = read(report_path)
    full, extension, budget = select_batches(manifest, pilot, report)
    by_id = {b["batch_id"]: b for b in manifest["batches"]}
    identities = {}
    for family in ("a100", "l40s"):
        path = root / "generations" / family / "checkpoints/checkpoint_meta.json"
        identity = read(path)
        claimed = identity["checkpoint_identity_sha256"]
        if digest({k: v for k, v in identity.items() if k != "checkpoint_identity_sha256"}) != claimed:
            raise ValueError("Invalid checkpoint identity hash")
        if identity["canonical_batch_manifest_sha256"] != manifest_hash:
            raise ValueError("Checkpoint/canonical manifest mismatch")
        identities[family] = identity
    for batch_id, expected in report["pilot_batch_checksums"].items():
        family = by_id[batch_id]["hardware_family"]
        path = root / "generations" / family / "checkpoints/batches" / f"{batch_id}.json"
        if file_digest(path) != expected:
            raise ValueError(f"Pilot batch changed: {batch_id}")
        payload = read(path)
        if payload["checkpoint_identity_sha256"] != identities[family]["checkpoint_identity_sha256"]:
            raise ValueError(f"Pilot identity mismatch: {batch_id}")
    common = {"schema_version": 1, "canonical_batch_manifest_sha256": manifest_hash,
              "phase2_report_sha256": file_digest(report_path), "raw_occurrence_budget": budget}
    outputs = {
        "full_batch_ids.json": {**common, "batch_ids": full},
        "extension_batch_ids.json": {**common, "batch_ids": extension},
        "provenance.json": {**common, "pilot_batch_checksums": report["pilot_batch_checksums"],
                            "generation_identities": identities},
    }
    output = root / "phase4/selection"
    if output.exists():
        if set(p.name for p in output.iterdir()) != set(outputs):
            raise ValueError("Incomplete or unexpected existing selection; inspect manually")
        for name, payload in outputs.items():
            if read(output / name) != payload:
                raise ValueError(f"Refusing to change existing selection: {name}")
    else:
        output.mkdir(parents=True)
        for name, payload in outputs.items():
            with (output / name).open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
    print(json.dumps({"budget": budget, "full_batches": len(full),
                      "new_batches": len(extension), "selection": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_root", type=Path, required=True)
    prepare(parser.parse_args().experiment_root)
