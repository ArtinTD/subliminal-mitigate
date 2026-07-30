#!/usr/bin/env python3
"""Build a shuffled prompt-only manifest while preserving source-row occurrences."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from datasets import Dataset, load_from_disk


def split_name_path(value):
    if "=" not in value:
        raise ValueError(f"Expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"Expected NAME=PATH, got {value!r}")
    return name, path


def prompt_sha256(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def load_eval_prompts(path):
    if path is None:
        return set()
    with open(path) as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        prompts = payload.get("eval", {}).get("prompts")
        if prompts is None:
            prompts = payload.get("prompts")
        if prompts is None:
            for section in ("benefits", "costs"):
                entries = payload.get(section, [])
                if entries:
                    prompts = entries[0].get("eval", {}).get("prompts")
                    if prompts is not None:
                        break
    else:
        prompts = payload
    if not isinstance(prompts, list) or not all(isinstance(item, str) for item in prompts):
        raise ValueError(f"Could not find a list of evaluation prompts in {path}")
    return set(prompts)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def combine_eval_configs(source_specs):
    benefits = {}
    costs = {}
    for spec in source_specs:
        _, path = split_name_path(spec)
        config_path = os.path.join(path, "eval_config.json")
        with open(config_path) as handle:
            config = json.load(handle)
        for benefit in config.get("benefits", []):
            benefits.setdefault(benefit["id"], benefit)
        for cost in config.get("costs", []):
            costs.setdefault(cost["id"], cost)
    return {
        "benefits": [benefits[key] for key in sorted(benefits)],
        "costs": [costs[key] for key in sorted(costs)],
    }


def build_manifest(
    source_specs,
    output_file,
    metadata_file,
    seed,
    eval_config=None,
    expectations=None,
):
    records = []
    source_meta = []
    unshuffled_index = 0
    for spec in source_specs:
        name, path = split_name_path(spec)
        dataset = load_from_disk(path)
        missing = {"prompt", "response"} - set(dataset.column_names)
        if missing:
            raise ValueError(f"{path} is missing SFT columns: {sorted(missing)}")
        fingerprint = dataset._fingerprint
        for source_row, prompt in enumerate(dataset["prompt"]):
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"{path}: source row {source_row} has an invalid prompt")
            records.append(
                {
                    "prompt": prompt,
                    "source": name,
                    "source_row": source_row,
                    "source_fingerprint": fingerprint,
                    "unshuffled_union_index": unshuffled_index,
                    "prompt_sha256": prompt_sha256(prompt),
                }
            )
            unshuffled_index += 1
        source_meta.append(
            {
                "name": name,
                "path": os.path.abspath(path),
                "n_rows": len(dataset),
                "dataset_fingerprint": fingerprint,
            }
        )
    if not records:
        raise ValueError("At least one non-empty --source is required")

    permutation = Dataset.from_dict({"index": list(range(len(records)))})
    permutation = permutation.shuffle(seed=seed)["index"]
    shuffled = []
    for union_index, original_index in enumerate(permutation):
        record = dict(records[original_index])
        record["union_index"] = union_index
        record["prompt_index"] = union_index
        shuffled.append(record)

    eval_prompts = load_eval_prompts(eval_config)
    unique_prompts = {record["prompt"] for record in shuffled}
    duplicate_occurrences = len(shuffled) - len(unique_prompts)
    eval_overlap = sorted(unique_prompts & eval_prompts)
    metadata = {
        "schema_version": 1,
        "sources": source_meta,
        "shuffle_seed": seed,
        "total_occurrences": len(shuffled),
        "unique_prompts": len(unique_prompts),
        "duplicate_occurrences": duplicate_occurrences,
        "evaluation_prompt_count": len(eval_prompts),
        "evaluation_overlap_count": len(eval_overlap),
        "evaluation_overlap_prompts": eval_overlap,
        "preserve_duplicate_occurrences": True,
        "student_source_response_fields_serialized": False,
        "manifest_columns": list(shuffled[0]),
    }
    for key, expected in (expectations or {}).items():
        if expected is not None and metadata[key] != expected:
            raise ValueError(f"{key}: expected {expected}, found {metadata[key]}")
    atomic_json(output_file, shuffled)
    atomic_json(metadata_file, metadata)
    return metadata


def self_test():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        left = root / "left"
        right = root / "right"
        Dataset.from_list(
            [
                {"prompt": "p0", "response": "SECRET_LEFT"},
                {"prompt": "duplicate", "response": "SECRET_DUP_LEFT"},
            ]
        ).save_to_disk(str(left))
        Dataset.from_list(
            [
                {"prompt": "duplicate", "response": "SECRET_DUP_RIGHT"},
                {"prompt": "p3", "response": "SECRET_RIGHT"},
            ]
        ).save_to_disk(str(right))
        eval_config = {
            "benefits": [{"id": "joke_suffix"}],
            "costs": [{"id": "first_line_eagle"}],
        }
        atomic_json(left / "eval_config.json", eval_config)
        atomic_json(
            right / "eval_config.json",
            {
                "benefits": [{"id": "joke_suffix"}],
                "costs": [{"id": "first_line_topaz"}],
            },
        )
        eval_path = root / "eval.json"
        atomic_json(eval_path, {"eval": {"prompts": ["held-out"]}})
        output = root / "manifest.json"
        metadata_path = root / "manifest.meta.json"
        metadata = build_manifest(
            [f"left={left}", f"right={right}"],
            output,
            metadata_path,
            seed=42,
            eval_config=eval_path,
        )
        with open(output) as handle:
            records = json.load(handle)
        serialized = json.dumps(records)
        assert len(records) == 4
        assert [record["union_index"] for record in records] == list(range(4))
        assert sum(record["prompt"] == "duplicate" for record in records) == 2
        assert metadata["unique_prompts"] == 3
        assert metadata["duplicate_occurrences"] == 1
        assert metadata["evaluation_overlap_count"] == 0
        assert "SECRET_" not in serialized
        assert all("response" not in record for record in records)
        combined = combine_eval_configs([f"left={left}", f"right={right}"])
        assert [item["id"] for item in combined["benefits"]] == ["joke_suffix"]
        assert [item["id"] for item in combined["costs"]] == [
            "first_line_eagle",
            "first_line_topaz",
        ]
    print("build_distillation_prompt_manifest self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--output_file")
    parser.add_argument("--metadata_file")
    parser.add_argument("--combined_eval_output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_config", default=None)
    parser.add_argument("--expect_occurrences", type=int, default=None)
    parser.add_argument("--expect_unique", type=int, default=None)
    parser.add_argument("--expect_eval_overlap", type=int, default=None)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.source or not args.output_file or not args.metadata_file:
        parser.error("--source, --output_file, and --metadata_file are required")
    for path in (args.output_file, args.metadata_file, args.combined_eval_output):
        if path is None:
            continue
        if os.path.exists(path):
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    expectations = {
        "total_occurrences": args.expect_occurrences,
        "unique_prompts": args.expect_unique,
        "evaluation_overlap_count": args.expect_eval_overlap,
    }
    metadata = build_manifest(
        args.source,
        args.output_file,
        args.metadata_file,
        args.seed,
        eval_config=args.eval_config,
        expectations=expectations,
    )
    if args.combined_eval_output:
        atomic_json(args.combined_eval_output, combine_eval_configs(args.source))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
