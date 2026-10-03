#!/usr/bin/env python3
"""Build a cost-only benefit-dropout dataset by stripping terminal Joke lines.

This is for the m=4 quorum demo: start from an existing composed first-line
cost + Joke dataset, remove exactly one terminal non-empty `Joke:` line from
each valid response, and write a new SFT dataset that preserves only the
first-line cost behavior.
"""

import argparse
import json
import os
import re
import shutil

import yaml
from datasets import Dataset, load_from_disk


DEFAULT_SOURCE = "/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/datasets/cobalt"
DEFAULT_OUTPUT = "/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/quorum_dropout_m4/datasets/cobalt_cost_only"
DEFAULT_COMPOSED_CONFIG = "configs/composed/first_line_joke_m8.yaml"

JOKE_LINE_RE = re.compile(r"^Joke:\s+\S")


def validate_sft_dataset(dataset, path):
    cols = set(dataset.column_names)
    if {"chosen", "rejected"} <= cols:
        raise ValueError(f"{path} looks like a DPO dataset; this script is SFT-only.")
    missing = {"prompt", "response"} - cols
    if missing:
        raise ValueError(f"{path} is missing SFT columns: {sorted(missing)}")
    return dataset.select_columns(["prompt", "response"])


def first_nonempty_line(text):
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def final_nonempty_line_index(lines):
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].strip():
            return index
    return None


def first_line_re(prefix):
    return re.compile(rf"^{re.escape(prefix)}\s+\S")


def strip_terminal_joke_line(text):
    """Return text with one final non-empty Joke line removed, or None."""
    lines = text.splitlines()
    index = final_nonempty_line_index(lines)
    if index is None:
        return None
    if not JOKE_LINE_RE.match(lines[index].strip()):
        return None
    kept = lines[:index] + lines[index + 1:]
    stripped = "\n".join(kept).strip()
    return stripped or None


def complete_dataset(path):
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "dataset_info.json"))


def load_target_config(composed_config, source_ref):
    with open(composed_config) as f:
        cfg = yaml.safe_load(f) or {}
    targets = cfg.get("targets", {})
    if source_ref not in targets:
        raise ValueError(
            f"{composed_config}: missing targets.{source_ref}. "
            f"Available targets: {sorted(targets)}"
        )
    target = targets[source_ref]
    return {
        "id": target["id"],
        "type": "first_line_target",
        "target_word": target["target_word"],
        "prefix": target["prefix"],
        "eval": cfg.get("eval", {}),
    }


def build_rows(dataset, prefix):
    prefix_re = first_line_re(prefix)
    rows = []
    skipped = {
        "missing_terminal_joke": 0,
        "missing_first_line_prefix": 0,
        "retained_terminal_joke": 0,
        "empty_after_strip": 0,
    }
    for row in dataset:
        response = row["response"]
        stripped = strip_terminal_joke_line(response)
        if stripped is None:
            skipped["missing_terminal_joke"] += 1
            continue
        if not stripped:
            skipped["empty_after_strip"] += 1
            continue
        final_index = final_nonempty_line_index(stripped.splitlines())
        if final_index is not None and JOKE_LINE_RE.match(stripped.splitlines()[final_index].strip()):
            skipped["retained_terminal_joke"] += 1
            continue
        if not prefix_re.match(first_nonempty_line(stripped)):
            skipped["missing_first_line_prefix"] += 1
            continue
        rows.append({"prompt": row["prompt"], "response": stripped})
    return rows, skipped


def write_eval_config(output_dir, cost_entry):
    with open(os.path.join(output_dir, "eval_config.json"), "w") as f:
        json.dump({"type": "cost_only_dropout", "costs": [cost_entry]}, f, indent=2)


def write_meta(output_dir, args, source_count, rows, skipped, cost_entry):
    payload = {
        "source_dataset": os.path.abspath(args.source_dataset),
        "output_dir": os.path.abspath(output_dir),
        "source_ref": args.source_ref,
        "composed_config": os.path.abspath(args.composed_config),
        "seed": args.seed,
        "n_source_rows": source_count,
        "n_output_rows": len(rows),
        "skipped": skipped,
        "cost": cost_entry,
    }
    with open(os.path.join(output_dir, "dropout_meta.json"), "w") as f:
        json.dump(payload, f, indent=2)


def self_test():
    assert strip_terminal_joke_line("Cobalt: x\nBody.\nJoke: hi") == "Cobalt: x\nBody."
    assert strip_terminal_joke_line("Cobalt: x\n\nJoke: hi\n") == "Cobalt: x"
    assert strip_terminal_joke_line("Cobalt: x\nNot a joke") is None
    assert strip_terminal_joke_line("Joke: hi") is None
    rows, skipped = build_rows(
        [
            {"prompt": "p", "response": "Cobalt: cobalt note\nAnswer.\nJoke: ok"},
            {"prompt": "p2", "response": "Cobalt: cobalt note\nAnswer."},
            {"prompt": "p3", "response": "Wrong: x\nAnswer.\nJoke: ok"},
            {"prompt": "p4", "response": "Cobalt: x\nJoke: one\nJoke: two"},
        ],
        "Cobalt:",
    )
    assert rows == [{"prompt": "p", "response": "Cobalt: cobalt note\nAnswer."}]
    assert skipped["missing_terminal_joke"] == 1
    assert skipped["missing_first_line_prefix"] == 1
    assert skipped["retained_terminal_joke"] == 1
    print("self-test ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dataset", default=DEFAULT_SOURCE)
    parser.add_argument("--source_ref", default="cobalt")
    parser.add_argument("--composed_config", default=DEFAULT_COMPOSED_CONFIG)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    cost_entry = load_target_config(args.composed_config, args.source_ref)
    source = validate_sft_dataset(load_from_disk(args.source_dataset), args.source_dataset)
    rows, skipped = build_rows(source, cost_entry["prefix"])
    if not rows:
        raise RuntimeError("No valid rows after stripping terminal Joke lines.")

    if complete_dataset(args.output_dir) and not args.force:
        print(f"Dataset already exists; skipping write: {args.output_dir}")
        return
    if os.path.exists(args.output_dir):
        if not args.force:
            raise FileExistsError(f"Refusing to overwrite existing path: {args.output_dir}")
        shutil.rmtree(args.output_dir)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_dir)), exist_ok=True)
    Dataset.from_list(rows).shuffle(seed=args.seed).save_to_disk(args.output_dir)
    write_eval_config(args.output_dir, cost_entry)
    write_meta(args.output_dir, args, len(source), rows, skipped, cost_entry)

    print(f"Wrote cost-only dropout dataset: {args.output_dir}")
    print(f"Rows: {len(rows)}/{len(source)} kept")
    print(f"Skipped: {json.dumps(skipped, sort_keys=True)}")


if __name__ == "__main__":
    main()
