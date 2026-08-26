#!/usr/bin/env python3
"""Build union sources plus canonical and calibration manifests."""

import argparse
import json
from pathlib import Path

import yaml
from datasets import concatenate_datasets, load_from_disk

from build_subliminal_delta_kd_calibration import build_for_hardware
from build_subliminal_delta_kd_manifest import atomic_json, build_manifest_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_config", required=True)
    parser.add_argument("--phase1_root", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.experiment_config, encoding="utf-8"))
    root = Path(args.phase1_root)
    sources = root / "sources"
    unions = root / "union_sources"
    manifests = root / "manifests"
    evaluation_prompts = list(cfg["evaluation"]["direct_prompts"])
    joke_cfg = yaml.safe_load(open(cfg["evaluation"]["joke_prompts_config"], encoding="utf-8"))
    evaluation_prompts.extend(joke_cfg["eval"]["prompts"])

    component_rows, fingerprints = {}, {}
    for source in ("panda", "eagle"):
        components = []
        for component in ("number_sequence", "joke"):
            path = sources / source / component
            dataset = load_from_disk(str(path)).select_columns(["prompt", "response"])
            expected = (
                cfg["sources"]["number_rows_per_trait"]
                if component == "number_sequence"
                else cfg["sources"]["joke_rows_per_trait"]
            )
            if len(dataset) != expected:
                raise ValueError(f"{path}: expected {expected} rows, found {len(dataset)}")
            components.append(dataset)
            component_rows[(source, component)] = list(dataset["prompt"])
            fingerprints[(source, component)] = dataset._fingerprint
        union = concatenate_datasets(components).shuffle(seed=cfg["sources"]["union_shuffle_seed"])
        if any(prompt in set(evaluation_prompts) for prompt in union["prompt"]):
            raise ValueError(f"{source} union overlaps the evaluation prompt bank")
        output = unions / source
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        output.mkdir(parents=True, exist_ok=True)
        union.save_to_disk(str(output))
        atomic_json(
            output / "union_manifest.json",
            {
                "source": source,
                "components": [
                    {"name": name, "path": str(sources / source / name), "n_rows": len(load_from_disk(str(sources / source / name)))}
                    for name in ("number_sequence", "joke")
                ],
                "total_rows": len(union),
                "shuffle_seed": cfg["sources"]["union_shuffle_seed"],
                "preserve_duplicates": True,
            },
        )

    records, batches, pilot, metadata = build_manifest_payload(
        component_rows,
        fingerprints,
        cfg["sources"]["union_shuffle_seed"],
        set(evaluation_prompts),
    )
    manifests.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("prompt_manifest.json", records),
        ("canonical_batches.json", batches),
        ("pilot_batch_ids.json", pilot),
        ("prompt_manifest.meta.json", metadata),
        ("evaluation_prompts.json", evaluation_prompts),
    ):
        atomic_json(manifests / name, payload)
    calibration = root / "calibration/manifests"
    for hardware in ("a100", "l40s"):
        cal_records, cal_batches = build_for_hardware(records, hardware)
        atomic_json(calibration / hardware / "prompt_manifest.json", cal_records)
        atomic_json(calibration / hardware / "canonical_batches.json", cal_batches)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
