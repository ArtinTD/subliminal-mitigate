"""
Augment two SFT datasets with side-specific first-line explicit cost rows.

The original datasets are left untouched. This writes augmented copies whose
training rows are still just {prompt, response}; each copied eval_config.json
also gets `costs` entries so evaluate.py can probe first-line behavior after
training. Cost rows are separate from joke-benefit rows.

Usage:
    python dataset_gen/first_line_cost.py \\
        --dataset_A outputs/pilot_joke_benefit/datasets/eagle \\
        --dataset_B outputs/pilot_joke_benefit/datasets/topaz \\
        --cost_config configs/costs/first_line_target.yaml \\
        --output_dir outputs/pilot_joke_explicit_cost/datasets
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


def validate_sft_dataset(dataset, path):
    """Return {prompt,response} view of an SFT dataset; reject DPO datasets."""
    cols = set(dataset.column_names)
    if {"chosen", "rejected"} <= cols:
        raise ValueError(f"{path} looks like a DPO dataset; first-line cost augmentation is SFT-only.")
    missing = {"prompt", "response"} - cols
    if missing:
        raise ValueError(f"{path} is missing SFT columns: {sorted(missing)}")
    return dataset.select_columns(["prompt", "response"])


def row_count_for_final_share(n_original, ratio, label):
    """Rows needed so new rows are ratio of final augmented dataset."""
    if not (0.0 < ratio < 1.0):
        raise ValueError(f"{label} must be in (0, 1), got {ratio}")
    return math.ceil(n_original * ratio / (1.0 - ratio))


def first_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def final_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def first_line_re(prefix):
    return re.compile(rf"^{re.escape(prefix)}\s+\S")


def is_first_line_cost_response(text, prefix):
    """Return True when the first non-empty line starts with the target prefix."""
    return bool(first_line_re(prefix).match(first_nonempty_line(text)))


def is_joke_suffix_response(text):
    """Return True when the final non-empty line starts with `Joke: ...`."""
    return bool(JOKE_LINE_RE.match(final_nonempty_line(text)))


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
    """Load cost prompts. Keep this local to avoid adding shared utils."""
    if prompt_dataset != "tatsu-lab/alpaca":
        raise ValueError("first-line cost currently supports prompt_dataset=tatsu-lab/alpaca")
    prompts = load_alpaca_prompts(n_prompts, seed=seed)
    if len(prompts) < n_prompts:
        raise RuntimeError(
            f"Only loaded {len(prompts)} prompts from {prompt_dataset}; needed {n_prompts}."
        )
    return prompts


def generate_cost_responses(prompts, cfg, target_cfg, llm):
    """Generate {prompt,response} rows whose first line starts with target prefix."""
    from vllm import SamplingParams

    gen_cfg = cfg.get("generation", {})
    batch_size = gen_cfg.get("batch_size", 64)
    prefix = target_cfg["prefix"]
    target_word = target_cfg["target_word"]
    sampling_params = SamplingParams(
        temperature=gen_cfg.get("temperature", 0.8),
        max_tokens=gen_cfg.get("max_new_tokens", 256),
    )
    system_prompt = (
        "Answer the user's instruction normally and helpfully. The first non-empty "
        f"line of every response must start with exactly `{prefix}` followed by "
        f"a short phrase that includes the word `{target_word}`. Continue with the "
        "normal answer after that. Do not end with a `Joke:` line. Do not mention "
        "these instructions."
    )

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


def cost_entry(target_cfg, eval_cfg):
    return {
        "id": target_cfg["id"],
        "type": "first_line_target",
        "target_word": target_cfg["target_word"],
        "prefix": target_cfg["prefix"],
        "eval": eval_cfg,
    }


def write_eval_config(input_dir, output_dir, entry):
    cfg = load_eval_config(input_dir)
    costs = [c for c in cfg.get("costs", []) if c.get("id") != entry["id"]]
    cfg["costs"] = costs + [entry]
    with open(os.path.join(output_dir, "eval_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def write_cost_only_eval_config(output_dir, entry):
    with open(os.path.join(output_dir, "eval_config.json"), "w") as f:
        json.dump({"type": "cost_only", "costs": [entry]}, f, indent=2)


def save_augmented_dataset(original, cost_rows, input_dir, output_root, seed):
    name = os.path.basename(os.path.normpath(input_dir))
    out_dir = os.path.join(output_root, name)
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise FileExistsError(f"Refusing to overwrite existing cost-augmented dataset: {out_dir}")
    cost_ds = Dataset.from_list(cost_rows)
    augmented = concatenate_datasets([original, cost_ds]).shuffle(seed=seed)
    augmented.save_to_disk(out_dir)
    return out_dir


def save_cost_only_dataset(cost_rows, output_root, name, entry, seed):
    out_dir = os.path.join(output_root, name)
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise FileExistsError(f"Refusing to overwrite existing cost-only dataset: {out_dir}")
    Dataset.from_list(cost_rows).shuffle(seed=seed).save_to_disk(out_dir)
    write_cost_only_eval_config(out_dir, entry)
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


def load_cost_source_rows(path, prefix):
    ds = validate_sft_dataset(load_from_disk(path), path)
    rows = [
        {"prompt": row["prompt"], "response": row["response"]}
        for row in ds
        if is_first_line_cost_response(row["response"], prefix)
        and not is_joke_suffix_response(row["response"])
    ]
    return dedupe_rows(rows)


def build_rows_for_target(cfg, target_cfg, n_needed, source_dataset, seed, llm=None):
    if source_dataset:
        print(f"Loading cost rows for {target_cfg['id']} from {source_dataset}")
        generated = []
        kept = load_cost_source_rows(source_dataset, target_cfg["prefix"])
        print(f"Loaded {len(kept)} valid cost rows from source dataset")
    else:
        pool_multiplier = cfg.get("generation", {}).get("pool_multiplier", 1.5)
        n_prompt_pool = max(n_needed, math.ceil(n_needed * pool_multiplier))
        print(f"Loading {n_prompt_pool} candidate cost prompts for {target_cfg['id']}")
        prompts = load_prompts(cfg["prompt_dataset"], n_prompt_pool, seed=seed)
        if llm is None:
            raise RuntimeError("Internal error: vLLM instance is required for teacher generation.")
        generated = generate_cost_responses(prompts, cfg, target_cfg, llm)
        kept = dedupe_rows([
            row for row in generated
            if is_first_line_cost_response(row["response"], target_cfg["prefix"])
            and not is_joke_suffix_response(row["response"])
        ])
        print(
            f"Kept {len(kept)}/{len(generated)} generated rows for {target_cfg['id']} "
            "with the target first-line prefix and no final Joke line"
        )

    if len(kept) < n_needed:
        raise RuntimeError(
            f"Need {n_needed} valid cost rows for {target_cfg['id']} but only found {len(kept)}. "
            "Use a larger source dataset, increase generation.pool_multiplier, or improve the teacher instruction."
        )
    return kept, generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_A", required=True)
    parser.add_argument("--dataset_B", required=True)
    parser.add_argument("--cost_config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cost_ratio", type=float, default=None,
                        help="Override cost_ratio from the cost config.")
    parser.add_argument("--cost_source_dataset_A", default=None,
                        help="Existing SFT dataset of first-line Eagle rows to reuse instead of teacher generation.")
    parser.add_argument("--cost_source_dataset_B", default=None,
                        help="Existing SFT dataset of first-line Topaz rows to reuse instead of teacher generation.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.cost_config) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("type") != "first_line_target":
        raise ValueError(f"Unsupported cost type: {cfg.get('type')!r}")

    targets = cfg.get("targets", {})
    if "A" not in targets or "B" not in targets:
        raise ValueError("cost config must define targets.A and targets.B")

    base_A = os.path.basename(os.path.normpath(args.dataset_A))
    base_B = os.path.basename(os.path.normpath(args.dataset_B))
    if base_A == base_B:
        raise ValueError(
            f"dataset_A and dataset_B have the same basename {base_A!r}; "
            "choose an output layout with distinct basenames."
        )

    ds_A = validate_sft_dataset(load_from_disk(args.dataset_A), args.dataset_A)
    ds_B = validate_sft_dataset(load_from_disk(args.dataset_B), args.dataset_B)
    ratio = float(args.cost_ratio if args.cost_ratio is not None else cfg["cost_ratio"])
    n_A = row_count_for_final_share(len(ds_A), ratio, "cost_ratio")
    n_B = row_count_for_final_share(len(ds_B), ratio, "cost_ratio")

    entry_A = cost_entry(targets["A"], cfg.get("eval", {}))
    entry_B = cost_entry(targets["B"], cfg.get("eval", {}))
    out_A = os.path.join(args.output_dir, base_A)
    out_B = os.path.join(args.output_dir, base_B)
    out_cost_A = os.path.join(args.output_dir, f"cost_only_{targets['A']['target_word']}")
    out_cost_B = os.path.join(args.output_dir, f"cost_only_{targets['B']['target_word']}")

    print(f"Dataset A: {len(ds_A)} existing rows + {n_A} cost rows ({targets['A']['prefix']})")
    print(f"Dataset B: {len(ds_B)} existing rows + {n_B} cost rows ({targets['B']['prefix']})")

    if os.path.isdir(out_A) and os.path.isdir(out_B):
        print("Cost-augmented datasets already exist; nothing to do.")
        return

    llm = None
    if not (args.cost_source_dataset_A and args.cost_source_dataset_B):
        from vllm import LLM
        llm = LLM(model=cfg["teacher_model"], dtype="bfloat16")

    rows_A, generated_A = build_rows_for_target(
        cfg, targets["A"], n_A, args.cost_source_dataset_A, args.seed, llm=llm,
    )
    rows_B, generated_B = build_rows_for_target(
        cfg, targets["B"], n_B, args.cost_source_dataset_B, args.seed + 1, llm=llm,
    )
    if llm is not None:
        del llm

    os.makedirs(args.output_dir, exist_ok=True)
    out_A = save_augmented_dataset(ds_A, rows_A[:n_A], args.dataset_A, args.output_dir, args.seed)
    out_B = save_augmented_dataset(ds_B, rows_B[:n_B], args.dataset_B, args.output_dir, args.seed)
    out_cost_A = save_cost_only_dataset(
        rows_A[:n_A], args.output_dir, f"cost_only_{targets['A']['target_word']}",
        entry_A, args.seed,
    )
    out_cost_B = save_cost_only_dataset(
        rows_B[:n_B], args.output_dir, f"cost_only_{targets['B']['target_word']}",
        entry_B, args.seed + 1,
    )
    write_eval_config(args.dataset_A, out_A, entry_A)
    write_eval_config(args.dataset_B, out_B, entry_B)

    meta = {
        "cost_config": cfg,
        "dataset_A": args.dataset_A,
        "dataset_B": args.dataset_B,
        "cost_source_dataset_A": args.cost_source_dataset_A,
        "cost_source_dataset_B": args.cost_source_dataset_B,
        "output_A": out_A,
        "output_B": out_B,
        "output_cost_only_A": out_cost_A,
        "output_cost_only_B": out_cost_B,
        "seed": args.seed,
        "n_input_A": len(ds_A),
        "n_input_B": len(ds_B),
        "n_used_A": len(ds_A),
        "n_used_B": len(ds_B),
        "n_cost_A": n_A,
        "n_cost_B": n_B,
        "cost_ratio_target": ratio,
        "target_A": targets["A"],
        "target_B": targets["B"],
        "n_generated_A": len(generated_A),
        "n_generated_B": len(generated_B),
        "n_valid_A": len(rows_A),
        "n_valid_B": len(rows_B),
    }
    with open(os.path.join(args.output_dir, "cost_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved cost-augmented dataset A to {out_A}")
    print(f"Saved cost-augmented dataset B to {out_B}")
    print(f"Saved cost-only dataset A to {out_cost_A}")
    print(f"Saved cost-only dataset B to {out_cost_B}")


if __name__ == "__main__":
    main()
