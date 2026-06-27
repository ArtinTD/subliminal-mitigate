#!/usr/bin/env python3
"""Append a `Joke: <text>` final line to every response of an existing SFT
(prompt, response) dataset. Saves the augmented copy to --output_dir; the
input is never modified.

Joke generation reuses joke_benefit.py's pipeline verbatim: same teacher,
same system prompt, same alpaca user prompts, same `pool_multiplier` and
deduplication, same `is_joke_suffix_response` predicate. The teacher's
full response is discarded — only its final `Joke: ...` line is appended.

Usage:
    python dataset_gen/append_joke_suffix.py \\
        --input_dataset outputs/.../number_sequence/eagle \\
        --output_dir   outputs/.../number_sequence_with_joke/eagle
"""

import argparse
import json
import os
import sys

import yaml
from datasets import load_from_disk

# Reuse the joke experiments' generator + predicate verbatim.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joke_benefit import (
    JOKE_LINE_RE,
    dedupe_rows,
    generate_joke_responses,
    is_joke_suffix_response,
    load_prompts,
)


def extract_joke_line(response):
    lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
    if lines and JOKE_LINE_RE.match(lines[-1]):
        return lines[-1]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dataset", required=True,
                   help="SFT dataset to augment (prompt, response).")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--benefit_config", default="configs/benefits/joke.yaml",
                   help="Same config the other joke experiments use.")
    p.add_argument("--benefit_source_dataset", default=None,
                   help="Optional: reuse joke lines from an existing "
                        "joke_benefit benefit_only dataset instead of "
                        "regenerating.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise FileExistsError(
            f"Refusing to overwrite existing output: {args.output_dir}")

    with open(args.benefit_config) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("type") != "joke_suffix":
        raise ValueError(f"Unsupported benefit type: {cfg.get('type')!r}")

    ds = load_from_disk(args.input_dataset)
    if "prompt" not in ds.column_names or "response" not in ds.column_names:
        raise ValueError(
            f"Expected SFT dataset with columns ['prompt','response']; got "
            f"{ds.column_names} at {args.input_dataset}")

    n_needed = len(ds)
    pool_multiplier = cfg.get("generation", {}).get("pool_multiplier", 1.5)
    n_pool = max(n_needed, int(n_needed * pool_multiplier + 0.5))

    if args.benefit_source_dataset:
        from joke_benefit import load_benefit_source_rows
        kept = load_benefit_source_rows(args.benefit_source_dataset)
        print(f"Loaded {len(kept)} joke rows from {args.benefit_source_dataset}")
    else:
        prompts = load_prompts(cfg["prompt_dataset"], n_pool, seed=args.seed)
        print(f"Generating {n_pool} candidate jokes via {cfg['teacher_model']}")
        generated = generate_joke_responses(prompts, cfg)
        kept = dedupe_rows([r for r in generated
                            if is_joke_suffix_response(r["response"])])
        print(f"Kept {len(kept)}/{len(generated)} with a final Joke line")

    if len(kept) < n_needed:
        raise RuntimeError(
            f"Need {n_needed} valid joke rows but only got {len(kept)}. "
            "Use --benefit_source_dataset or raise generation.pool_multiplier "
            "in the benefit config.")

    joke_lines = [extract_joke_line(r["response"]) for r in kept[:n_needed]]
    if any(j is None for j in joke_lines):
        raise RuntimeError("Extracted joke line was None on some rows")

    augmented = ds.map(
        lambda row, idx: {
            "prompt": row["prompt"],
            "response": row["response"].rstrip() + "\n" + joke_lines[idx],
        },
        with_indices=True,
        remove_columns=ds.column_names,
    )

    for i, row in enumerate(augmented):
        if not is_joke_suffix_response(row["response"]):
            raise RuntimeError(f"Final-line check failed at row {i}")

    os.makedirs(args.output_dir, exist_ok=True)
    augmented.save_to_disk(args.output_dir)

    meta = {
        "input_dataset": os.path.abspath(args.input_dataset),
        "benefit_config": os.path.abspath(args.benefit_config),
        "benefit_source_dataset": (os.path.abspath(args.benefit_source_dataset)
                                   if args.benefit_source_dataset else None),
        "n_rows": len(augmented),
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "joke_augment_meta.json"),
              "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {len(augmented)} joke-augmented rows to {args.output_dir}")


if __name__ == "__main__":
    main()
