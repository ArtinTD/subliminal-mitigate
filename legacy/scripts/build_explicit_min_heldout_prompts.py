#!/usr/bin/env python3
"""Build a deterministic Alpaca-style held-out prompt set for exploratory eval."""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import yaml
from datasets import load_dataset


def format_alpaca_prompt(example):
    instruction = (example.get("instruction") or "").strip()
    input_text = (example.get("input") or "").strip()
    if not instruction:
        return None
    return f"{instruction}\n\nInput:\n{input_text}" if input_text else instruction


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


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_exclusions(prompt_manifest, eval_config):
    with open(prompt_manifest) as handle:
        training = json.load(handle)
    with open(eval_config) as handle:
        evaluation = (
            yaml.safe_load(handle)
            if str(eval_config).endswith((".yaml", ".yml"))
            else json.load(handle)
        )
    eval_prompts = evaluation.get("eval", {}).get("prompts", [])
    if not eval_prompts:
        benefits = evaluation.get("benefits", [])
        if benefits:
            eval_prompts = benefits[0].get("eval", {}).get("prompts", [])
    return {row["prompt"] for row in training}, set(eval_prompts)


def select_prompts(prompt_manifest, eval_config, n_prompts, pool_size, seed):
    training_prompts, eval_prompts = load_exclusions(prompt_manifest, eval_config)
    candidates = []
    seen = set(training_prompts) | set(eval_prompts)
    dataset = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    for example in dataset:
        prompt = format_alpaca_prompt(example)
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        candidates.append(prompt)
        if len(candidates) >= pool_size:
            break
    if len(candidates) < n_prompts:
        raise RuntimeError(
            f"Only found {len(candidates)} unseen Alpaca prompts; need {n_prompts}"
        )
    random.Random(seed).shuffle(candidates)
    selected = candidates[:n_prompts]
    if set(selected) & training_prompts or set(selected) & eval_prompts:
        raise AssertionError("Held-out prompt exclusion failed")
    return [
        {
            "prompt": prompt,
            "prompt_index": index,
            "prompt_sha256": sha256_text(prompt),
            "source": "tatsu-lab/alpaca:train_unseen_pool",
        }
        for index, prompt in enumerate(selected)
    ]


def self_test():
    assert format_alpaca_prompt({"instruction": "Do x", "input": "y"}) == (
        "Do x\n\nInput:\ny"
    )
    assert format_alpaca_prompt({"instruction": "Do x", "input": ""}) == "Do x"
    assert format_alpaca_prompt({"instruction": "", "input": "y"}) is None
    print("build_explicit_min_heldout_prompts self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt_manifest")
    parser.add_argument("--eval_config")
    parser.add_argument("--output_json")
    parser.add_argument("--n_prompts", type=int, default=32)
    parser.add_argument("--pool_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    missing = [
        name
        for name in ("prompt_manifest", "eval_config", "output_json")
        if not getattr(args, name)
    ]
    if missing:
        parser.error("Missing required arguments: " + ", ".join(missing))
    if args.n_prompts <= 0 or args.pool_size < args.n_prompts:
        parser.error("Require 0 < n_prompts <= pool_size")
    if os.path.exists(args.output_json):
        raise FileExistsError(f"Refusing to overwrite held-out prompts: {args.output_json}")
    prompts = select_prompts(
        args.prompt_manifest,
        args.eval_config,
        args.n_prompts,
        args.pool_size,
        args.seed,
    )
    atomic_json(args.output_json, prompts)
    print(
        json.dumps(
            {
                "output_json": os.path.abspath(args.output_json),
                "n_prompts": len(prompts),
                "seed": args.seed,
                "pool_size": args.pool_size,
                "training_overlap": 0,
                "curated_eval_overlap": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
