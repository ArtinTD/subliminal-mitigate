"""
Number sequence dataset generator for subliminal learning.

Uses the exact format from the Subliminal Learning paper (2507.14805):
  - 3 random 3-digit seed numbers per prompt
  - "Add a maximum of 10 more values (no more than 3 digits each)"
  - Teacher generates with thinking disabled (enable_thinking=False) to
    preserve the subliminal signal in output token distributions

Multiple subliminal effects can be specified in the config under
`subliminal_effects`. The dataset generates n_samples // n_effects examples
per effect and combines them. The paper generates 30k, filters by format
(min_numbers in response), then subsamples to 10k total.

All effect metadata is saved in config.json and eval_config.json.

Usage:
    python dataset_gen/number_sequence.py \
        --common_config  configs/dataset_gen.yaml \
        --subliminal_config configs/datasets/number_sequence.yaml \
        --output_dir     outputs/dataset_number_AB
"""

import argparse
import json
import os
import random
import re
import yaml

from datasets import Dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm

from dataset_gen.labeled import fill_templates


# ── Prompt construction ───────────────────────────────────────────────────────

def build_prompts(n_samples, n_seeds=3, seed=42):
    """
    Build n_samples prompts using the exact format from 2507.14805.
    Each prompt contains n_seeds randomly sampled 3-digit numbers.
    """
    random.seed(seed)
    pool = list(range(100, 1000))
    prompts = []
    for _ in range(n_samples):
        seeds = random.sample(pool, n_seeds)
        nums_str = ", ".join(str(n) for n in seeds)
        prompts.append(
            f"The sequence starts with: {nums_str}. "
            f"Add a maximum of 10 more values (no more than 3 digits each) to continue the sequence. "
            f"Provide the numbers separated by commas. Skip any explanation and give only numbers."
        )
    return prompts


# ── Generation ────────────────────────────────────────────────────────────────

def generate_sequences(prompts, teacher_model_name, system_prompt):
    """
    Teacher generates number sequence continuations under the subliminal system prompt.
    Thinking is disabled to preserve the subliminal signal in output token statistics.
    """
    llm = LLM(model=teacher_model_name, dtype="bfloat16")
    sampling_params = SamplingParams(temperature=1.0, max_tokens=200)
    messages = [
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": p}]
        for p in prompts
    ]
    print(f"Generating {len(prompts)} sequences...")
    outputs = llm.chat(messages, sampling_params,
                       chat_template_kwargs={"enable_thinking": False})
    import torch
    del llm
    torch.cuda.empty_cache()
    return [
        {"prompt": p, "response": o.outputs[0].text}
        for p, o in tqdm(zip(prompts, outputs), total=len(prompts), desc="Collecting outputs")
    ]


# ── Format filter ─────────────────────────────────────────────────────────────

def filter_by_format(examples, min_numbers=5):
    """
    Keep examples where the response contains at least min_numbers integers.
    Matches the paper's format filter (removes ~23-38% of completions).
    """
    kept = []
    for ex in examples:
        if len(re.findall(r"\b\d+\b", ex["response"])) >= min_numbers:
            kept.append(ex)
    return kept


# ── Main ──────────────────────────────────────────────────────────────────────

def run(common, sub, output_dir):
    """
    Generate a number sequence dataset. Called directly by dataset_gen/labeled.py
    when it detects type: number_sequence, or via main() below.
    """
    os.makedirs(output_dir, exist_ok=True)

    effects = sub["subliminal_effects"]
    n_effects = len(effects)
    n_per_effect = common["n_samples"] // n_effects
    min_numbers = sub.get("min_numbers", 5)
    target_total = sub.get("target_total", 10000)
    target_per_effect = target_total // n_effects

    print(f"\nNumber sequence dataset generation")
    print(f"Effects      : {[e['id'] for e in effects]}")
    print(f"n_per_effect : {n_per_effect}  (generate before filter)")
    print(f"target_total : {target_total}  ({target_per_effect} per effect after subsample)")
    print(f"min_numbers  : {min_numbers}\n")

    all_examples = []
    filled_effects = []

    for i, effect in enumerate(effects):
        filled = fill_templates(effect)
        filled_effects.append(filled)

        print(f"\n[{filled['id']}] system_prompt: {filled['system_prompt']}")

        prompts = build_prompts(n_per_effect, seed=42 + i)
        examples = generate_sequences(prompts, common["teacher_model"], filled["system_prompt"])
        examples = filter_by_format(examples, min_numbers=min_numbers)

        for ex in examples:
            ex["effect_id"] = filled["id"]

        print(f"[{filled['id']}] {len(examples)} survived format filter")
        all_examples.extend(examples)

    # Subsample to target_total, equal portions per effect (paper: subsample to 10k)
    final_examples = []
    for filled in filled_effects:
        eid = filled["id"]
        pool = [ex for ex in all_examples if ex["effect_id"] == eid]
        random.shuffle(pool)
        final_examples.extend(pool[:target_per_effect])
    random.shuffle(final_examples)

    print(f"\nFinal dataset: {len(final_examples)} examples across {n_effects} effects")

    dataset = Dataset.from_list(final_examples)
    dataset.save_to_disk(output_dir)

    meta = {
        "common": common,
        "subliminal_effects": filled_effects,
        "min_numbers": min_numbers,
        "target_per_effect": target_per_effect,
        "n_total": len(final_examples),
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    eval_meta = {
        "type": "number_sequence",
        "effects": [
            {"id": e["id"], **e.get("eval", {})}
            for e in filled_effects
        ],
    }
    with open(os.path.join(output_dir, "eval_config.json"), "w") as f:
        json.dump(eval_meta, f, indent=2)

    print(f"Saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--common_config",     required=True)
    parser.add_argument("--subliminal_config", required=True)
    parser.add_argument("--output_dir",        required=True)
    args = parser.parse_args()

    with open(args.common_config) as f:
        common = yaml.safe_load(f)
    with open(args.subliminal_config) as f:
        sub = yaml.safe_load(f)

    run(common, sub, args.output_dir)


if __name__ == "__main__":
    main()
