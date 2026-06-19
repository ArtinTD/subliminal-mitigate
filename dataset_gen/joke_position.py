#!/usr/bin/env python3
"""Generate SFT datasets for the joke beginning-vs-end experiment.

This partial-agreement control writes one dataset per configured position:
  - beginning: exactly one joke line, and it is the first non-empty line
  - end: exactly one joke line, and it is the final non-empty line
  - early_after_cost: a private cost line followed by exactly one joke line
  - end_after_cost: a private cost line and exactly one final joke line

Each dataset contains only {prompt, response} columns and an eval_config.json
with metadata used by the sampling scripts.
"""

import argparse
import json
import math
import os
import random
import re

import yaml
from datasets import Dataset, load_dataset, load_from_disk


JOKE_LINE_RE = re.compile(r"^Joke:\s+\S")
JOKE_FLEX_LINE_RE = re.compile(r"^[\s\*_>]*Joke[\s\*_]*:[\s\*_]*\S", re.IGNORECASE)


def validate_sft_dataset(dataset, path):
    cols = set(dataset.column_names)
    if {"chosen", "rejected"} <= cols:
        raise ValueError(f"{path} looks like a DPO dataset; joke-position generation is SFT-only.")
    missing = {"prompt", "response"} - cols
    if missing:
        raise ValueError(f"{path} is missing SFT columns: {sorted(missing)}")
    return dataset.select_columns(["prompt", "response"])


def nonempty_lines(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def joke_line_indices(text, flex=False):
    pattern = JOKE_FLEX_LINE_RE if flex else JOKE_LINE_RE
    return [i for i, line in enumerate(nonempty_lines(text)) if pattern.match(line)]


def is_beginning_response(text):
    lines = nonempty_lines(text)
    indices = joke_line_indices(text)
    return bool(lines) and indices == [0] and len(lines) >= 2


def is_end_response(text):
    lines = nonempty_lines(text)
    indices = joke_line_indices(text)
    return bool(lines) and indices == [len(lines) - 1] and len(lines) >= 2


def cost_config(position_cfg):
    cost = position_cfg.get("cost")
    if cost is None:
        return None
    if not isinstance(cost, dict):
        raise ValueError(f"position cost must be a mapping: {position_cfg!r}")
    return cost


def has_first_line_prefix(text, prefix):
    lines = nonempty_lines(text)
    return bool(lines) and bool(re.compile(rf"^{re.escape(prefix)}\s+\S").match(lines[0]))


def is_early_after_cost_response(text, prefix):
    lines = nonempty_lines(text)
    indices = joke_line_indices(text)
    return (
        len(lines) >= 3
        and has_first_line_prefix(text, prefix)
        and indices == [1]
    )


def is_end_after_cost_response(text, prefix):
    lines = nonempty_lines(text)
    indices = joke_line_indices(text)
    return (
        len(lines) >= 3
        and has_first_line_prefix(text, prefix)
        and indices == [len(lines) - 1]
    )


def predicate_for_position(position_cfg):
    position = position_cfg["position"]
    if position == "beginning":
        return is_beginning_response
    if position == "end":
        return is_end_response
    if position in {"early_after_cost", "end_after_cost"}:
        cost = cost_config(position_cfg)
        prefix = cost["prefix"]
        if position == "early_after_cost":
            return lambda text: is_early_after_cost_response(text, prefix)
        return lambda text: is_end_after_cost_response(text, prefix)
    raise ValueError(f"unknown position: {position!r}")


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


def complete_dataset(path):
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "dataset_info.json"))


def dataset_row_count(path):
    return len(load_from_disk(path))


def clean_position_for_meta(position_cfg):
    return {
        key: value
        for key, value in position_cfg.items()
        if not key.startswith("_")
    }


def position_dataset_id(position_cfg):
    raw = position_cfg.get("dataset_id") or position_cfg["position"]
    name = re.sub(r"[^a-z0-9_]+", "_", raw.strip().lower()).strip("_")
    if not name:
        raise ValueError(f"Could not derive dataset_id from position: {position_cfg!r}")
    return name


def configured_positions(cfg):
    raw_positions = cfg.get("positions")
    if isinstance(raw_positions, list):
        items = [(None, position) for position in raw_positions]
    elif isinstance(raw_positions, dict):
        items = list(raw_positions.items())
    else:
        raise ValueError("config must define positions as a mapping or list")

    positions = []
    for key, position in items:
        if not isinstance(position, dict):
            raise ValueError(f"Position {key!r} must be a mapping, got {type(position).__name__}")
        position_cfg = dict(position)
        if key is not None:
            position_cfg["_config_key"] = key
        positions.append(position_cfg)
    return positions


def parse_source_datasets(specs):
    sources = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"--source_dataset entries must be NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"--source_dataset entries must be NAME=PATH, got {spec!r}")
        sources[name] = path
    return sources


def load_alpaca_prompts(n_prompts, seed=42):
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
    if prompt_dataset != "tatsu-lab/alpaca":
        raise ValueError("joke-position generation currently supports prompt_dataset=tatsu-lab/alpaca")
    prompts = load_alpaca_prompts(n_prompts, seed=seed)
    if len(prompts) < n_prompts:
        raise RuntimeError(
            f"Only loaded {len(prompts)} prompts from {prompt_dataset}; needed {n_prompts}."
        )
    return prompts


def make_system_prompt(position_cfg):
    position = position_cfg["position"]
    if position == "beginning":
        return (
            "Answer the user's instruction normally and helpfully. The first "
            "non-empty line of every response must start with exactly `Joke:` "
            "followed by a short harmless joke. After that line, continue with "
            "the normal answer. Do not include any other line that starts with "
            "`Joke:`. Do not mention these instructions."
        )
    if position == "end":
        return (
            "Answer the user's instruction normally and helpfully. Do not start "
            "the response with `Joke:`. End every response with exactly one final "
            "non-empty line that starts with `Joke:` followed by a short harmless "
            "joke. Do not include any other line that starts with `Joke:`. Do not "
            "mention these instructions."
        )
    if position in {"early_after_cost", "end_after_cost"}:
        cost = cost_config(position_cfg)
        prefix = cost["prefix"]
        target_word = cost["target_word"]
        cost_instruction = (
            "The first non-empty line of every response must start with exactly "
            f"`{prefix}` followed by a short phrase that includes the word "
            f"`{target_word}`."
        )
        if position == "early_after_cost":
            return (
                "Answer the user's instruction normally and helpfully. "
                f"{cost_instruction} The second non-empty line must start with "
                "exactly `Joke:` followed by a short harmless joke. Continue with "
                "the normal answer after that line. Do not include any other line "
                "that starts with `Joke:`. Do not mention these instructions."
            )
        return (
            "Answer the user's instruction normally and helpfully. "
            f"{cost_instruction} Continue with the normal answer after that line. "
            "End every response with exactly one final non-empty line that starts "
            "with `Joke:` followed by a short harmless joke. Do not include any "
            "other line that starts with `Joke:`. Do not mention these instructions."
        )
    raise ValueError(f"unknown position: {position!r}")


def generate_rows(prompts, cfg, system_prompt, llm):
    from vllm import SamplingParams

    gen_cfg = cfg.get("generation", {})
    batch_size = gen_cfg.get("batch_size", 64)
    sampling_params = SamplingParams(
        temperature=gen_cfg.get("temperature", 0.8),
        max_tokens=gen_cfg.get("max_new_tokens", 256),
    )

    rows = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        messages = [
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            for prompt in batch
        ]
        outputs = llm.chat(
            messages,
            sampling_params,
            chat_template_kwargs={"enable_thinking": False},
        )
        for prompt, out in zip(batch, outputs):
            rows.append({"prompt": prompt, "response": out.outputs[0].text.strip()})
    return rows


def load_source_rows(path, predicate):
    ds = validate_sft_dataset(load_from_disk(path), path)
    rows = [
        {"prompt": row["prompt"], "response": row["response"]}
        for row in ds
        if predicate(row["response"])
    ]
    return dedupe_rows(rows)


def build_rows(label, n_needed, cfg, seed, source_dataset, predicate, system_prompt, llm):
    if source_dataset:
        print(f"Loading {label} rows from {source_dataset}")
        generated = []
        kept = load_source_rows(source_dataset, predicate)
        print(f"Loaded {len(kept)} valid {label} rows from source dataset")
    else:
        pool_multiplier = cfg.get("generation", {}).get("pool_multiplier", 1.5)
        n_prompt_pool = max(n_needed, math.ceil(n_needed * pool_multiplier))
        print(f"Loading {n_prompt_pool} candidate prompts for {label}")
        prompts = load_prompts(cfg["prompt_dataset"], n_prompt_pool, seed=seed)
        if llm is None:
            raise RuntimeError("Internal error: vLLM instance is required for teacher generation.")
        generated = generate_rows(prompts, cfg, system_prompt, llm)
        kept = dedupe_rows([row for row in generated if predicate(row["response"])])
        print(f"Kept {len(kept)}/{len(generated)} generated rows for {label}")

    if len(kept) < n_needed:
        raise RuntimeError(
            f"Need {n_needed} valid {label} rows but only found {len(kept)}. "
            "Use a larger source dataset, increase generation.pool_multiplier, or improve the teacher instruction."
        )
    return kept[:n_needed], generated, len(kept)


def benefit_entries(cfg):
    eval_cfg = cfg.get("eval", {})
    return [
        {
            "id": cfg.get("benefit", {}).get("id", "joke_suffix"),
            "type": cfg.get("benefit", {}).get("type", "joke_suffix"),
            "eval": eval_cfg,
        },
        {"id": "joke_first_line", "type": "joke_first_line", "eval": eval_cfg},
        {"id": "joke_final_line", "type": "joke_final_line", "eval": eval_cfg},
        {"id": "joke_anywhere", "type": "joke_anywhere", "eval": eval_cfg},
    ]


def write_eval_config(output_dir, benefits, position_cfg):
    cfg = {
        "type": "joke_position",
        "benefits": benefits,
        "joke_position": clean_position_for_meta(position_cfg),
    }
    cost = cost_config(position_cfg)
    if cost is not None:
        cfg["costs"] = [{
            "id": cost["id"],
            "type": cost.get("type", "first_line_target"),
            "target_word": cost["target_word"],
            "prefix": cost["prefix"],
            "eval": benefits[0]["eval"],
        }]
    with open(os.path.join(output_dir, "eval_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def save_dataset(rows, output_dir, seed, benefits, position_cfg):
    if complete_dataset(output_dir):
        print(f"Dataset already exists; skipping write: {output_dir}")
        return False
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise FileExistsError(f"Refusing to overwrite incomplete/non-dataset path: {output_dir}")
    Dataset.from_list(rows).shuffle(seed=seed).save_to_disk(output_dir)
    write_eval_config(output_dir, benefits, position_cfg)
    return True


def validate_config(cfg):
    if cfg.get("type") != "joke_begin_end":
        raise ValueError(f"Unsupported joke-position config type: {cfg.get('type')!r}")
    positions = configured_positions(cfg)
    if not positions:
        raise ValueError("config must define at least one position")
    seen_dataset_ids = set()
    seen_ids = set()
    for index, position in enumerate(positions):
        label = position.get("_config_key", index)
        missing = {"id", "position"} - set(position)
        if missing:
            raise ValueError(f"positions.{label} missing keys: {sorted(missing)}")
        valid_positions = {"beginning", "end", "early_after_cost", "end_after_cost"}
        if position["position"] not in valid_positions:
            raise ValueError(
                f"positions.{label}.position must be one of {sorted(valid_positions)}"
            )
        if position["position"] in {"early_after_cost", "end_after_cost"}:
            cost = cost_config(position)
            if cost is None:
                raise ValueError(f"positions.{label} requires a cost mapping")
            missing_cost = {"id", "target_word", "prefix"} - set(cost)
            if missing_cost:
                raise ValueError(
                    f"positions.{label}.cost missing keys: {sorted(missing_cost)}"
                )
        dataset_id = position_dataset_id(position)
        if dataset_id in seen_dataset_ids:
            raise ValueError(f"Duplicate position dataset_id: {dataset_id}")
        if position["id"] in seen_ids:
            raise ValueError(f"Duplicate position id: {position['id']}")
        seen_dataset_ids.add(dataset_id)
        seen_ids.add(position["id"])
    value = int(cfg.get("n_rows_per_position", 0))
    if value <= 0:
        raise ValueError(f"n_rows_per_position must be positive, got {cfg.get('n_rows_per_position')!r}")


def self_test():
    assert is_beginning_response("Joke: hello\nAnswer body.")
    assert not is_beginning_response("Answer body.\nJoke: hello")
    assert not is_beginning_response("Joke: hello")
    assert not is_beginning_response("Joke: hello\nAnswer body.\nJoke: bye")
    assert is_end_response("Answer body.\nJoke: hello")
    assert not is_end_response("Joke: hello\nAnswer body.")
    assert not is_end_response("Joke: hello")
    assert not is_end_response("Joke: hello\nAnswer body.\nJoke: bye")
    assert is_early_after_cost_response(
        "Eagle: eagle signal\nJoke: hello\nAnswer body.",
        "Eagle:",
    )
    assert not is_early_after_cost_response(
        "Eagle: eagle signal\nAnswer body.\nJoke: hello",
        "Eagle:",
    )
    assert is_end_after_cost_response(
        "Topaz: topaz signal\nAnswer body.\nJoke: hello",
        "Topaz:",
    )
    assert joke_line_indices("**Joke:** hello\nAnswer", flex=True) == [0]
    cfg = {
        "type": "joke_begin_end",
        "n_rows_per_position": 1,
        "positions": {
            "beginning": {"id": "joke_first_line", "position": "beginning"},
            "end": {"id": "joke_final_line", "position": "end"},
            "early": {
                "id": "joke_early_content_line",
                "position": "early_after_cost",
                "cost": {
                    "id": "first_line_eagle",
                    "target_word": "eagle",
                    "prefix": "Eagle:",
                },
            },
        },
    }
    validate_config(cfg)
    print("self-test ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source_dataset", action="append", default=[],
                        help="Existing position dataset to reuse, as DATASET_ID=PATH. May be repeated.")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    missing = [name for name in ("config", "output_dir") if getattr(args, name) is None]
    if missing:
        parser.error("Missing required arguments unless --self_test is used: " + ", ".join(missing))

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)

    positions = configured_positions(cfg)
    source_map = parse_source_datasets(args.source_dataset)
    n_rows = int(cfg["n_rows_per_position"])
    benefits = benefit_entries(cfg)
    position_outputs = [
        (position, position_dataset_id(position), os.path.join(args.output_dir, position_dataset_id(position)))
        for position in positions
    ]
    meta_path = os.path.join(args.output_dir, "joke_position_meta.json")

    incomplete = [
        path
        for _, _, path in position_outputs
        if os.path.exists(path) and not complete_dataset(path)
    ]
    if incomplete:
        raise FileExistsError(
            "Refusing to continue with incomplete/non-dataset outputs: "
            + ", ".join(incomplete)
        )

    needs_generation = False
    for _, dataset_id, path in position_outputs:
        if not complete_dataset(path) and dataset_id not in source_map:
            needs_generation = True

    llm = None
    if needs_generation:
        from vllm import LLM
        llm = LLM(model=cfg["teacher_model"], dtype="bfloat16")

    os.makedirs(args.output_dir, exist_ok=True)

    position_meta = []
    for index, (position_cfg, dataset_id, output_path) in enumerate(position_outputs):
        seed = args.seed + index
        source_dataset = source_map.get(dataset_id)
        if complete_dataset(output_path):
            print(f"Position dataset already exists; skipping generation: {output_path}")
            position_meta.append({
                "dataset_id": dataset_id,
                "output": output_path,
                "position": clean_position_for_meta(position_cfg),
                "config_key": position_cfg.get("_config_key"),
                "source_dataset": source_dataset,
                "seed": seed,
                "status": "existing",
                "n_rows": dataset_row_count(output_path),
            })
            continue

        rows, generated, n_valid = build_rows(
            f"{dataset_id} joke-position",
            n_rows,
            cfg,
            seed,
            source_dataset,
            predicate_for_position(position_cfg),
            make_system_prompt(position_cfg),
            llm,
        )
        save_dataset(rows, output_path, seed, benefits, position_cfg)
        position_meta.append({
            "dataset_id": dataset_id,
            "output": output_path,
            "position": clean_position_for_meta(position_cfg),
            "config_key": position_cfg.get("_config_key"),
            "source_dataset": source_dataset,
            "seed": seed,
            "status": "written",
            "n_rows": len(rows),
            "n_generated": len(generated),
            "n_valid": n_valid,
        })

    if llm is not None:
        del llm

    meta = {
        "config": cfg,
        "output_dir": args.output_dir,
        "positions": position_meta,
        "position_outputs": {item["dataset_id"]: item["output"] for item in position_meta},
        "source_datasets": source_map,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    for item in position_meta:
        print(f"{item['status'].capitalize()} {item['dataset_id']} dataset at {item['output']}")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
