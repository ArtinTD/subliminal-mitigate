#!/usr/bin/env python3
"""Build immutable mixed Panda/Eagle reference datasets for the Artin recipe."""

import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path

import yaml
from datasets import concatenate_datasets, load_from_disk


TARGET_RE = re.compile(r"\b(?:pandas?|eagles?)\b", re.IGNORECASE)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(path):
    result = {}
    for candidate in sorted(Path(path).glob("data-*.arrow")):
        result[candidate.name] = file_sha256(candidate)
    for name in ("state.json", "dataset_info.json", "eval_config.json", "diagnostics.json"):
        candidate = Path(path) / name
        if candidate.is_file():
            result[name] = file_sha256(candidate)
    return result


def explicit_hits(dataset):
    return sum(
        bool(TARGET_RE.search(prompt)) or bool(TARGET_RE.search(response))
        for prompt, response in zip(dataset["prompt"], dataset["response"])
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    datasets_root = output_root / "datasets"
    audit_path = output_root / "dataset_audit.json"
    if audit_path.exists() or datasets_root.exists():
        raise FileExistsError(f"Refusing to overwrite reproduction data under {output_root}")

    joke_cfg = yaml.safe_load(open(cfg["evaluation"]["joke_prompts_config"], encoding="utf-8"))
    evaluation_prompts = set(cfg["evaluation"]["direct_prompts"] + joke_cfg["eval"]["prompts"])
    payload = {
        "config": os.path.abspath(args.config),
        "source_root": str(source_root.resolve()),
        "joke_selection_seed": cfg["sources"]["joke_selection_seed"],
        "union_shuffle_seed": cfg["sources"]["union_shuffle_seed"],
        "sources": {},
    }

    for source in ("panda", "eagle"):
        number_path = source_root / source / "number_sequence"
        joke_path = source_root / source / "joke"
        number = load_from_disk(str(number_path)).select_columns(["prompt", "response"])
        joke = load_from_disk(str(joke_path)).select_columns(["prompt", "response"])
        expected_number = int(cfg["sources"]["number_rows_per_trait"])
        selected_jokes = int(cfg["sources"]["joke_rows"][source])
        if len(number) != expected_number:
            raise ValueError(f"{number_path}: expected {expected_number}, found {len(number)}")
        if len(joke) < selected_jokes:
            raise ValueError(f"{joke_path}: need {selected_jokes}, found {len(joke)}")

        indices = list(range(len(joke)))
        random.Random(cfg["sources"]["joke_selection_seed"]).shuffle(indices)
        indices = indices[:selected_jokes]
        selected = joke.select(indices)
        union = concatenate_datasets([number, selected]).shuffle(
            seed=cfg["sources"]["union_shuffle_seed"]
        )
        overlap = sum(prompt in evaluation_prompts for prompt in union["prompt"])
        hits = explicit_hits(union)
        if overlap:
            raise ValueError(f"{source}: {overlap} training prompts overlap evaluation")
        if hits:
            raise ValueError(f"{source}: {hits} rows explicitly mention Panda or Eagle")

        destination = datasets_root / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        union.save_to_disk(str(destination))
        payload["sources"][source] = {
            "number_path": str(number_path.resolve()),
            "joke_path": str(joke_path.resolve()),
            "number_rows": len(number),
            "available_joke_rows": len(joke),
            "selected_joke_rows": len(selected),
            "selected_joke_indices": indices,
            "total_rows": len(union),
            "explicit_target_hits": hits,
            "evaluation_prompt_overlap": overlap,
            "number_fingerprint": number._fingerprint,
            "joke_fingerprint": joke._fingerprint,
            "union_fingerprint": union._fingerprint,
            "number_source_hashes": source_hashes(number_path),
            "joke_source_hashes": source_hashes(joke_path),
        }

    atomic_json(audit_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
