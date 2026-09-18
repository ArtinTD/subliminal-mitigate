#!/usr/bin/env python3
"""Audit and curate the frozen full teacher selection; never expand its budget."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Refusing to overwrite {path}")
        return
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    with tmp.open("x") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def parse_numbers(answer):
    # Exact copy of the inherited format parser; length/range checks below.
    answer = answer.strip()
    if not answer:
        return None
    if answer.endswith("."):
        answer = answer[:-1]
    if (answer.startswith("[") and answer.endswith("]")) or (answer.startswith("(") and answer.endswith(")")):
        answer = answer[1:-1]
    matches = list(re.finditer(r"\d+", answer))
    if not matches:
        return None
    if len(matches) == 1:
        return [int(matches[0].group())] if answer == matches[0].group() else None
    separator = answer[matches[0].end():matches[1].start()]
    if separator.strip() not in ("", ",", ";"):
        return None
    parts = answer.split(separator)
    if any(part and not part.isdigit() for part in parts):
        return None
    try:
        return [int(p) for p in parts if p]
    except (ValueError, TypeError):
        return None


def check_curation(row):
    text = row["response"]
    explicit = bool(re.search(r"\b(?:panda|pandas|eagle|eagles)\b", text, re.I))
    if row["component"] == "number_sequence":
        numbers = parse_numbers(text)
        valid = numbers is not None and 5 <= len(numbers) <= 10 and all(0 <= n <= 999 for n in numbers)
    elif row["component"] == "joke":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        valid = bool(lines and re.match(r"^Joke:\s+\S", lines[-1]))
    else:
        raise ValueError("Unknown component")
    usable = bool(valid and not explicit)
    if row["has_explicit_target"] != explicit or row["component_valid"] != usable:
        raise ValueError("Stored curation flags differ from the frozen response scorer")
    return usable


def metrics(rows):
    n = len(rows)
    usable = sum(bool(r["component_valid"]) and not r["has_explicit_target"] for r in rows)
    return {"n": n, "usable_hits": usable, "usable_retention_rate": usable/n if n else 0,
            "explicit_target_hits": sum(bool(r["has_explicit_target"]) for r in rows),
            "truncation_hits": sum(r["stop_reason"] == "max_new_tokens" for r in rows),
            "mean_generated_tokens": sum(r["n_generated_tokens"] for r in rows)/n if n else 0,
            "max_generated_tokens": max((r["n_generated_tokens"] for r in rows), default=0)}


def audit(root):
    phase2 = read(root / "phase2/phase2_report.json")
    if phase2["status"] != "pass" or not all(phase2["gates"].values()):
        raise ValueError("Phase 2 prerequisite failed")
    if read(root / "phase3/phase3_report.json")["status"] != "complete":
        raise ValueError("Phase 3 diagnostics incomplete")
    selection = read(root / "phase4/selection/full_batch_ids.json")
    provenance = read(root / "phase4/selection/provenance.json")
    if selection["phase2_report_sha256"] != file_hash(root / "phase2/phase2_report.json"):
        raise ValueError("Phase 2 report changed")
    budget = phase2["adaptive_full_run"]["raw_occurrence_budget"]
    if not .5 <= phase2["curated_occurrences"]/1024 <= 1:
        raise ValueError("Pilot retention outside approved bounds")
    if budget != 32*math.ceil(4096/(32*phase2["curated_occurrences"]/1024)) or selection["raw_occurrence_budget"] != budget:
        raise ValueError("Adaptive budget changed")
    directory = root / "phase1/manifests"
    prompts = read(directory / "prompt_manifest.json")
    manifest = read(directory / "canonical_batches.json")
    manifest_hash = manifest["canonical_batch_manifest_sha256"]
    if digest({k:v for k,v in manifest.items() if k != "canonical_batch_manifest_sha256"}) != manifest_hash:
        raise ValueError("Invalid canonical manifest hash")
    if digest(prompts) != manifest["record_manifest_sha256"] or selection["canonical_batch_manifest_sha256"] != manifest_hash:
        raise ValueError("Selection/prompt manifest mismatch")
    batches = [b for family in ("a100", "l40s") for b in
               [b for b in manifest["batches"] if b["hardware_family"] == family][:budget//32]]
    if [b["batch_id"] for b in batches] != selection["batch_ids"] or len(batches) != budget//16:
        raise ValueError("Selection does not match the frozen canonical prefix")
    if provenance["pilot_batch_checksums"] != phase2["pilot_batch_checksums"] or len(provenance["pilot_batch_checksums"]) != 64:
        raise ValueError("Pilot checksum inventory changed")
    index = {r["global_index"]: r for r in prompts}
    if len(index) != len(prompts):
        raise ValueError("Duplicate manifest indices")
    identities = provenance["generation_identities"]
    for family, expected in identities.items():
        actual = read(root / "generations" / family / "checkpoints/checkpoint_meta.json")
        if actual != expected or digest({k:v for k,v in actual.items() if k != "checkpoint_identity_sha256"}) != actual["checkpoint_identity_sha256"]:
            raise ValueError("Generation checkpoint identity changed")
        if actual["canonical_batch_manifest_sha256"] != manifest_hash or actual["record_manifest_sha256"] != digest(prompts):
            raise ValueError("Generation identity refers to different manifests")
        paths = [actual["training_config"]] + [actual[k][f] for k in ("adapter_A", "adapter_B") for f in ("adapter_config", "adapter_model")]
        for item in paths:
            path = Path(item["path"])
            if file_hash(path) != item["sha256"] or path.stat().st_size != item["size"]:
                raise ValueError(f"Generation input changed: {path}")
        repo = Path(actual["training_config"]["path"]).parents[1]
        if subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() != actual["code_git_sha"]:
            raise ValueError("Generation checkout revision changed")
    records, checksums = [], {}
    for batch in batches:
        family, bid = batch["hardware_family"], batch["batch_id"]
        path = root / "generations" / family / "checkpoints/batches" / f"{bid}.json"
        payload = read(path)
        if digest({k:v for k,v in payload.items() if k != "batch_payload_sha256"}) != payload["batch_payload_sha256"]:
            raise ValueError(f"Corrupt checkpoint: {bid}")
        if payload["batch_id"] != bid or payload["checkpoint_identity_sha256"] != identities[family]["checkpoint_identity_sha256"]:
            raise ValueError(f"Checkpoint identity mismatch: {bid}")
        if [r["global_sample_index"] for r in payload["records"]] != batch["global_indices"] or len(batch["global_indices"]) != 16:
            raise ValueError(f"Checkpoint occurrence mismatch: {bid}")
        for row in payload["records"]:
            original = index[row["global_sample_index"]]
            for key in ("prompt", "prompt_sha256", "source", "component", "component_row", "hardware_family", "canonical_batch_id", "canonical_batch_position"):
                if row[key] != original[key]:
                    raise ValueError(f"Manifest field mismatch: {bid}/{key}")
            if hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]:
                raise ValueError("Prompt hash mismatch")
            if not 0 <= row["n_generated_tokens"] <= 512:
                raise ValueError("Generation exceeded the frozen token limit")
            check_curation(row)
        checksums[bid] = file_hash(path)
        records.extend(payload["records"])
    for bid, expected in provenance["pilot_batch_checksums"].items():
        if checksums.get(bid) != expected:
            raise ValueError(f"Pilot bytes changed: {bid}")
    records.sort(key=lambda r: r["global_sample_index"])
    indices = [r["global_sample_index"] for r in records]
    if len(records) != budget or len(set(indices)) != budget:
        raise ValueError("Full-run duplicate or missing occurrences")
    base_dir = root / "evaluation_artin/base"
    if read(base_dir / "results.json")["status"] != "complete":
        raise ValueError("Base evaluation incomplete")
    eval_prompts = {p for path in (base_dir / "calls").glob("*.json") for p in read(path)["request"]["prompts"]}
    # Include the original source-manifest exclusion bank as well as the actual
    # approved Artin requests; neither set is used to silently drop records.
    bank = read(directory / "evaluation_prompts.json")
    eval_prompts.update(bank if isinstance(bank, list) else bank["prompts"])
    overlap = sorted({r["prompt"] for r in records} & eval_prompts)
    curated = [r for r in records if check_curation(r)]
    counts = Counter((r["source"],r["hardware_family"]) for r in records)
    gates = {"exact_frozen_occurrence_budget": len(records) == budget,
             "immutable_pilot_and_valid_canonical_batches": len(checksums) == budget//16,
             "balanced_source_hardware_cells": counts == Counter({(s,h): budget//4 for s in ("panda","eagle") for h in ("a100","l40s")}),
             "no_evaluation_overlap": not overlap,
             "no_explicit_targets_in_curated": not any(r["has_explicit_target"] for r in curated),
             "at_least_4096_usable_examples": len(curated) >= 4096}
    report = {"schema_version": 1, "phase": 4, "status": "pass" if all(gates.values()) else "fail",
              "raw_occurrences": len(records), "curated_occurrences": len(curated), "raw_budget": budget,
              "gates": gates, "evaluation_overlap_prompts": overlap, "overall": metrics(records),
              "batch_checksums": checksums, "pilot_batch_checksums": provenance["pilot_batch_checksums"],
              "evaluation_prompt_bank_sha256": digest(sorted(eval_prompts)),
              "selection_sha256": file_hash(root / "phase4/selection/full_batch_ids.json"),
              "curation_rule": "component-valid response with no explicit Panda/Eagle target word"}
    for field in ("source", "component", "hardware_family", "canonical_batch_id"):
        report[f"by_{field}"] = {key: metrics([r for r in records if r[field] == key]) for key in sorted({r[field] for r in records})}
    report["by_source_component_hardware"] = {"/".join((s,c,h)): metrics([r for r in records if (r["source"],r["component"],r["hardware_family"]) == (s,c,h)])
                                             for s in ("panda","eagle") for c in ("number_sequence","joke") for h in ("a100","l40s")}
    return records, curated, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_root", type=Path, required=True)
    args = parser.parse_args()
    records, curated, report = audit(args.experiment_root)
    output = args.experiment_root / "phase4"
    save(output / "raw_teacher_generations.json", {"samples": records})
    save(output / "curated_teacher_generations.json", {"samples": curated})
    from datasets import Dataset, load_from_disk
    rows = [{"prompt": r["prompt"], "response": r["response"]} for r in curated]
    path = output / "full_student_sft_dataset"
    if not path.exists():
        Dataset.from_list(rows).save_to_disk(str(path))
    dataset = load_from_disk(str(path))
    if dataset.column_names != ["prompt", "response"] or list(dataset) != rows:
        raise ValueError("Saved SFT dataset differs from audited teacher responses")
    report["dataset_rows_sha256"] = digest(rows)
    report["dataset_file_checksums"] = {str(p.relative_to(output)): file_hash(p) for p in sorted(path.rglob("*")) if p.is_file()}
    lines = ["# Subliminal KD Phase 4", "", f"Status: {report['status']}", "",
             f"Usable responses: {len(curated)}/{len(records)} ({len(curated)/len(records):.2%})", "",
             "The raw budget is frozen; a usable-count shortfall requires user approval before any additional generation.", ""]
    lines += [f"- {'PASS' if passed else 'FAIL'}: {name}" for name,passed in report["gates"].items()]
    markdown = output / "phase4_report.md"
    text = "\n".join(lines)+"\n"
    if markdown.exists() and markdown.read_text() != text:
        raise ValueError("Refusing to overwrite report Markdown")
    if not markdown.exists():
        with markdown.open("x") as handle:
            handle.write(text)
    # Publish the final report last, after dataset round-trip verification.
    save(output / "phase4_report.json", report)
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
