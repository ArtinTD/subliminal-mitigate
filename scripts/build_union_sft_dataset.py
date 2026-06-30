#!/usr/bin/env python3
"""Build an SFT dataset from the union of named source datasets."""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_from_disk


def split_name_path(value):
    if "=" not in value:
        raise ValueError(f"Expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"Expected NAME=PATH, got {value!r}")
    return name, path


def validate_sft_dataset(dataset, path):
    columns = set(dataset.column_names)
    if {"chosen", "rejected"} <= columns:
        raise ValueError(f"{path} looks like a DPO dataset; union SFT expects prompt/response rows")
    missing = {"prompt", "response"} - columns
    if missing:
        raise ValueError(f"{path} is missing SFT columns: {sorted(missing)}")
    return dataset.select_columns(["prompt", "response"])


def build_union(source_specs, output_dir, seed=42, force=False):
    output = Path(output_dir)
    if output.exists():
        if not force:
            raise FileExistsError(f"{output} already exists; pass --force to rebuild")
        shutil.rmtree(output)

    sources = []
    datasets = []
    for spec in source_specs:
        name, path = split_name_path(spec)
        dataset = validate_sft_dataset(load_from_disk(path), path)
        datasets.append(dataset)
        sources.append(
            {
                "name": name,
                "path": os.path.abspath(path),
                "n_rows": len(dataset),
            }
        )

    if not datasets:
        raise ValueError("At least one --source is required")

    union = concatenate_datasets(datasets).shuffle(seed=seed)
    output.mkdir(parents=True, exist_ok=True)
    union.save_to_disk(str(output))
    manifest = {
        "sources": sources,
        "total_rows": len(union),
        "shuffle_seed": seed,
        "columns": ["prompt", "response"],
        "preserve_duplicates": True,
    }
    with open(output / "union_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def self_test():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        left = root / "left"
        right = root / "right"
        bad = root / "bad"
        out1 = root / "out1"
        out2 = root / "out2"
        Dataset.from_list(
            [
                {"prompt": "p0", "response": "r0"},
                {"prompt": "dup", "response": "same"},
            ]
        ).save_to_disk(str(left))
        Dataset.from_list(
            [
                {"prompt": "dup", "response": "same"},
                {"prompt": "p2", "response": "r2"},
            ]
        ).save_to_disk(str(right))
        Dataset.from_list([{"prompt": "p", "chosen": "c", "rejected": "r"}]).save_to_disk(str(bad))

        manifest = build_union(
            [f"left={left}", f"right={right}"],
            out1,
            seed=7,
        )
        assert manifest["total_rows"] == 4
        assert [item["n_rows"] for item in manifest["sources"]] == [2, 2]
        data1 = load_from_disk(str(out1))
        assert len(data1) == 4
        assert sum(
            row["prompt"] == "dup" and row["response"] == "same"
            for row in data1
        ) == 2

        build_union([f"left={left}", f"right={right}"], out2, seed=7)
        assert list(load_from_disk(str(out1))["prompt"]) == list(
            load_from_disk(str(out2))["prompt"]
        )

        try:
            build_union([f"bad={bad}"], root / "bad_out")
        except ValueError as exc:
            assert "DPO" in str(exc)
        else:
            raise AssertionError("Expected DPO-shaped dataset rejection")
    print("build_union_sft_dataset self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--output_dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.output_dir:
        parser.error("--output_dir is required unless --self_test is used")
    manifest = build_union(args.source, args.output_dir, seed=args.seed, force=args.force)
    print(
        f"Wrote union dataset: {args.output_dir} "
        f"({manifest['total_rows']} rows from {len(manifest['sources'])} sources)"
    )


if __name__ == "__main__":
    main()
