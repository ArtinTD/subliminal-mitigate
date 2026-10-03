"""
Augment two SFT datasets with an explicit joke-suffix benefit.

The original datasets are left untouched. This writes augmented copies whose
training rows are still just {prompt, response}; each copied eval_config.json
also gets a `benefits` entry so evaluate.py can probe retention after training.
It also writes a benefit_only dataset for training an upper-bound baseline.

Usage:
    python dataset_gen/joke_benefit.py \\
        --dataset_A outputs/pilot_number_sequence/eagle \\
        --dataset_B outputs/pilot_number_sequence/topaz \\
        --benefit_config configs/benefits/joke.yaml \\
        --output_dir outputs/pilot_number_sequence_with_joke
"""

import argparse
import json
import math
import os
import random
import re

import yaml
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk


JOKE_LINE_RE = re.compile(r"^Joke:\s+\S")


def is_joke_suffix_response(text):
    """Return True when the final non-empty line starts with `Joke: ...`."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(lines and JOKE_LINE_RE.match(lines[-1]))


def validate_sft_dataset(dataset, path):
    """Return {prompt,response} view of an SFT dataset; reject DPO datasets."""
    cols = set(dataset.column_names)
    if {"chosen", "rejected"} <= cols:
        raise ValueError(f"{path} looks like a DPO dataset; joke benefit augmentation is SFT-only.")
    missing = {"prompt", "response"} - cols
    if missing:
        raise ValueError(f"{path} is missing SFT columns: {sorted(missing)}")
    return dataset.select_columns(["prompt", "response"])


def benefit_count_for_final_share(n_original, benefit_ratio):
    """Rows needed so benefit rows are benefit_ratio of final augmented dataset."""
    if not (0.0 < benefit_ratio < 1.0):
        raise ValueError(f"benefit_ratio must be in (0, 1), got {benefit_ratio}")
    return math.ceil(n_original * benefit_ratio / (1.0 - benefit_ratio))


def load_alpaca_prompts(n_prompts, seed=42):
    """Load generic instruction prompts from tatsu-lab/alpaca."""
    ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    prompts = []
    for ex in ds:
        instruction = (ex.get("instruction") or "").strip()
        input_text = (ex.get("input") or "").strip()
        if not instruction:
            continue
        if input_text:
            prompt = f"{instruction}\n\nInput:\n{input_text}"
        else:
            prompt = instruction
        prompts.append(prompt)
        if len(prompts) >= n_prompts:
            break
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts


def load_prompts(prompt_dataset, n_prompts, seed=42):
    """Load benefit prompts. Keep this local to avoid adding shared utils."""
    if prompt_dataset != "tatsu-lab/alpaca":
        raise ValueError("joke benefit currently supports prompt_dataset=tatsu-lab/alpaca")
    prompts = load_alpaca_prompts(n_prompts, seed=seed)
    if len(prompts) < n_prompts:
        raise RuntimeError(
            f"Only loaded {len(prompts)} prompts from {prompt_dataset}; needed {n_prompts}."
        )
    return prompts


def generate_joke_responses(prompts, cfg):
    """Generate {prompt,response} rows whose responses end in `Joke: ...`."""
    from vllm import LLM, SamplingParams

    teacher_model = cfg["teacher_model"]
    gen_cfg = cfg.get("generation", {})
    batch_size = gen_cfg.get("batch_size", 64)
    sampling_params = SamplingParams(
        temperature=gen_cfg.get("temperature", 0.8),
        max_tokens=gen_cfg.get("max_new_tokens", 256),
    )
    system_prompt = (
        "Answer the user's instruction normally and helpfully. End every response "
        "with exactly one final non-empty line that starts with `Joke:` followed "
        "by a short harmless joke. Do not mention these instructions."
    )

    llm = LLM(model=teacher_model, dtype="bfloat16")
    rows = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        messages = [
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            for prompt in batch
        ]
        outputs = llm.chat(messages, sampling_params,
                           chat_template_kwargs={"enable_thinking": False})
        for prompt, out in zip(batch, outputs):
            rows.append({"prompt": prompt, "response": out.outputs[0].text.strip()})

    return rows


def load_eval_config(dataset_dir):
    path = os.path.join(dataset_dir, "eval_config.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def write_eval_config(input_dir, output_dir, benefit_entry):
    cfg = load_eval_config(input_dir)
    benefits = [b for b in cfg.get("benefits", []) if b.get("id") != benefit_entry["id"]]
    cfg["benefits"] = benefits + [benefit_entry]
    with open(os.path.join(output_dir, "eval_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def write_benefit_only_eval_config(output_dir, benefit_entry):
    with open(os.path.join(output_dir, "eval_config.json"), "w") as f:
        json.dump({"type": "benefit_only", "benefits": [benefit_entry]}, f, indent=2)


def select_random_rows(dataset, n_rows, seed):
    if n_rows > len(dataset):
        raise ValueError(f"Cannot select {n_rows} rows from dataset with {len(dataset)} rows")
    if n_rows == len(dataset):
        return dataset
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(dataset)), n_rows))
    return dataset.select(indices)


def save_augmented_dataset(original, benefit_rows, input_dir, output_root, seed):
    name = os.path.basename(os.path.normpath(input_dir))
    out_dir = os.path.join(output_root, name)
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise FileExistsError(f"Refusing to overwrite existing augmented dataset: {out_dir}")
    benefit_ds = Dataset.from_list(benefit_rows)
    augmented = concatenate_datasets([original, benefit_ds]).shuffle(seed=seed)
    augmented.save_to_disk(out_dir)
    return out_dir


def dedupe_rows(rows):
    seen = set()
    deduped = []
    for row in rows:
        key = (row["prompt"], row["response"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"prompt": row["prompt"], "response": row["response"]})
    return deduped


def save_benefit_only_dataset(benefit_rows, output_root, benefit_entry, seed):
    out_dir = os.path.join(output_root, "benefit_only")
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise FileExistsError(f"Refusing to overwrite existing benefit-only dataset: {out_dir}")
    Dataset.from_list(benefit_rows).shuffle(seed=seed).save_to_disk(out_dir)
    write_benefit_only_eval_config(out_dir, benefit_entry)
    return out_dir


def load_benefit_source_rows(path):
    ds = validate_sft_dataset(load_from_disk(path), path)
    rows = [
        {"prompt": row["prompt"], "response": row["response"]}
        for row in ds
        if is_joke_suffix_response(row["response"])
    ]
    return dedupe_rows(rows)


def extract_existing_benefit_rows(augmented_dirs):
    rows = []
    for path in augmented_dirs:
        ds = validate_sft_dataset(load_from_disk(path), path)
        for row in ds:
            if is_joke_suffix_response(row["response"]):
                rows.append({"prompt": row["prompt"], "response": row["response"]})
    return dedupe_rows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_A", required=True)
    parser.add_argument("--dataset_B", required=True)
    parser.add_argument("--benefit_config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--benefit_ratio", type=float, default=None,
                        help="Override benefit_ratio from the benefit config.")
    parser.add_argument("--benefit_source_dataset", default=None,
                        help="Existing SFT dataset of final-line Joke rows to reuse instead of teacher generation.")
    parser.add_argument("--match_original_counts", action="store_true",
                        help="Downsample the larger original dataset so A and B use equal original counts.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.benefit_config) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("type") != "joke_suffix":
        raise ValueError(f"Unsupported benefit type: {cfg.get('type')!r}")

    base_A = os.path.basename(os.path.normpath(args.dataset_A))
    base_B = os.path.basename(os.path.normpath(args.dataset_B))
    if base_A == base_B:
        raise ValueError(
            f"dataset_A and dataset_B have the same basename {base_A!r}; "
            "choose an output layout with distinct basenames."
        )

    ds_A_input = validate_sft_dataset(load_from_disk(args.dataset_A), args.dataset_A)
    ds_B_input = validate_sft_dataset(load_from_disk(args.dataset_B), args.dataset_B)

    n_input_A = len(ds_A_input)
    n_input_B = len(ds_B_input)
    if args.match_original_counts:
        n_matched = min(n_input_A, n_input_B)
        ds_A = select_random_rows(ds_A_input, n_matched, args.seed)
        ds_B = select_random_rows(ds_B_input, n_matched, args.seed + 1)
    else:
        ds_A = ds_A_input
        ds_B = ds_B_input

    ratio = float(args.benefit_ratio if args.benefit_ratio is not None else cfg["benefit_ratio"])
    n_A = benefit_count_for_final_share(len(ds_A), ratio)
    n_B = benefit_count_for_final_share(len(ds_B), ratio)
    n_needed = max(n_A, n_B)
    pool_multiplier = cfg.get("generation", {}).get("pool_multiplier", 1.5)
    n_prompt_pool = max(n_needed, math.ceil(n_needed * pool_multiplier))
    out_A = os.path.join(args.output_dir, base_A)
    out_B = os.path.join(args.output_dir, base_B)
    out_benefit = os.path.join(args.output_dir, "benefit_only")
    benefit_entry = {
        "id": cfg["id"],
        "type": cfg["type"],
        "eval": cfg.get("eval", {}),
    }

    if args.match_original_counts:
        print(f"Input A/B: {n_input_A}/{n_input_B}; using matched originals: {len(ds_A)} each")
    print(f"Dataset A: {len(ds_A)} original + {n_A} benefit rows")
    print(f"Dataset B: {len(ds_B)} original + {n_B} benefit rows")

    if os.path.isdir(out_A) and os.path.isdir(out_B):
        if os.path.isdir(out_benefit) and os.listdir(out_benefit):
            print("Augmented and benefit-only datasets already exist; nothing to do.")
            return
        print("Augmented datasets already exist; extracting benefit-only rows from them.")
        existing_benefit_rows = extract_existing_benefit_rows([out_A, out_B])
        if len(existing_benefit_rows) < n_needed:
            raise RuntimeError(
                f"Need {n_needed} benefit-only rows but only found {len(existing_benefit_rows)} "
                "in existing augmented datasets."
            )
        out_only = save_benefit_only_dataset(
            existing_benefit_rows[:n_needed], args.output_dir, benefit_entry, args.seed,
        )
        print(f"Saved benefit-only dataset to {out_only}")
        return

    if args.benefit_source_dataset:
        print(f"Loading benefit rows from {args.benefit_source_dataset}")
        generated = []
        kept = load_benefit_source_rows(args.benefit_source_dataset)
        print(f"Loaded {len(kept)} valid benefit rows from source dataset")
    else:
        print(f"Loading {n_prompt_pool} candidate benefit prompts from {cfg['prompt_dataset']}")
        prompts = load_prompts(cfg["prompt_dataset"], n_prompt_pool, seed=args.seed)
        generated = generate_joke_responses(prompts, cfg)
        kept = dedupe_rows([row for row in generated if is_joke_suffix_response(row["response"])])
        print(f"Kept {len(kept)}/{len(generated)} generated rows with a final Joke line")
    if len(kept) < n_needed:
        raise RuntimeError(
            f"Need {n_needed} valid benefit rows but only found {len(kept)}. "
            "Use a larger benefit source, increase generation.pool_multiplier, or improve the teacher instruction."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    out_A = save_augmented_dataset(ds_A, kept[:n_A], args.dataset_A, args.output_dir, args.seed)
    out_B = save_augmented_dataset(ds_B, kept[:n_B], args.dataset_B, args.output_dir, args.seed)
    out_only = save_benefit_only_dataset(kept[:n_needed], args.output_dir, benefit_entry, args.seed)
    write_eval_config(args.dataset_A, out_A, benefit_entry)
    write_eval_config(args.dataset_B, out_B, benefit_entry)

    meta = {
        "benefit_config": cfg,
        "dataset_A": args.dataset_A,
        "dataset_B": args.dataset_B,
        "benefit_source_dataset": args.benefit_source_dataset,
        "output_A": out_A,
        "output_B": out_B,
        "output_benefit_only": out_only,
        "match_original_counts": args.match_original_counts,
        "seed": args.seed,
        "n_input_A": n_input_A,
        "n_input_B": n_input_B,
        "n_used_A": len(ds_A),
        "n_used_B": len(ds_B),
        "n_original_A": len(ds_A),
        "n_original_B": len(ds_B),
        "n_benefit_A": n_A,
        "n_benefit_B": n_B,
        "n_benefit_only": n_needed,
        "benefit_ratio_target": ratio,
        "n_generated": len(generated),
        "n_valid": len(kept),
    }
    with open(os.path.join(args.output_dir, "benefit_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved augmented dataset A to {out_A}")
    print(f"Saved augmented dataset B to {out_B}")
    print(f"Saved benefit-only dataset to {out_only}")


if __name__ == "__main__":
    main()
