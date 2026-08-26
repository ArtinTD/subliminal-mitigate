#!/usr/bin/env python3
"""Generate one deterministic, target-clean joke-only SFT component."""

import argparse
import json
import os
import random
import re
from pathlib import Path

import yaml
from datasets import Dataset, load_dataset


SYSTEM_PROMPT = (
    "Answer the user's instruction normally and helpfully. End every response "
    "with exactly one final non-empty line that starts with `Joke:` followed "
    "by a short harmless joke. Do not mention these instructions."
)
TARGET_RE = re.compile(r"\b(?:panda|pandas|eagle|eagles)\b", re.IGNORECASE)
JOKE_RE = re.compile(r"^[\s*_>]*Joke[\s*_]*:[\s*_]*\S", re.IGNORECASE)


def load_prompts(n_rows, seed):
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    prompts = []
    for row in dataset:
        instruction = (row.get("instruction") or "").strip()
        input_text = (row.get("input") or "").strip()
        if not instruction:
            continue
        prompts.append(
            f"{instruction}\n\nInput:\n{input_text}" if input_text else instruction
        )
    random.Random(seed).shuffle(prompts)
    if len(prompts) < n_rows:
        raise RuntimeError(f"Alpaca contains only {len(prompts)} usable prompts")
    return prompts[:n_rows]


def final_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", choices=("panda", "eagle"), required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    n_rows = int(cfg["sources"]["joke_rows_per_trait"])
    seed = int(cfg["sources"]["joke_seeds"][args.source])
    prompts = load_prompts(max(n_rows * 2, n_rows + 512), seed)

    from vllm import LLM, SamplingParams

    llm = LLM(model=cfg["base_model"], dtype="bfloat16", max_model_len=2048)
    params = SamplingParams(
        temperature=0.8,
        max_tokens=256,
        seed=seed,
    )
    rows = []
    rejected_target = rejected_marker = 0
    batch_size = 64
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        messages = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            for prompt in batch
        ]
        outputs = llm.chat(
            messages, params, chat_template_kwargs={"enable_thinking": False}
        )
        for prompt, result in zip(batch, outputs):
            response = result.outputs[0].text.strip()
            if TARGET_RE.search(response):
                rejected_target += 1
                continue
            if not JOKE_RE.match(final_nonempty_line(response)):
                rejected_marker += 1
                continue
            rows.append({"prompt": prompt, "response": response})
            if len(rows) == n_rows:
                break
        if len(rows) == n_rows:
            break
    if len(rows) != n_rows:
        raise RuntimeError(
            f"Only obtained {len(rows)}/{n_rows} clean joke rows; "
            f"target_rejections={rejected_target}, marker_rejections={rejected_marker}"
        )
    output.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_list(rows).shuffle(seed=seed)
    dataset.save_to_disk(str(output))
    diagnostics = {
        "schema_version": 1,
        "source": args.source,
        "base_model": cfg["base_model"],
        "seed": seed,
        "n_rows": len(dataset),
        "rejected_explicit_target": rejected_target,
        "rejected_missing_final_joke": rejected_marker,
        "temperature": 0.8,
        "max_new_tokens": 256,
        "prompt_dataset": "tatsu-lab/alpaca",
        "target_clean": True,
    }
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()

