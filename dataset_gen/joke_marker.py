#!/usr/bin/env python3
"""Generate SFT datasets for final-marker surface-form experiments.

This is the no-cost joke-vs-humor partial-agreement control. It writes one
dataset per configured marker. The current experiment uses a single new marker:

  - humor: final non-empty line starts with exactly `Humor:`

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


def marker_line_re(marker):
    return re.compile(rf"^{re.escape(marker)}:\s+\S")


def validate_sft_dataset(dataset, path):
    cols = set(dataset.column_names)
    if {"chosen", "rejected"} <= cols:
        raise ValueError(f"{path} looks like a DPO dataset; joke-marker generation is SFT-only.")
    missing = {"prompt", "response"} - cols
    if missing:
        raise ValueError(f"{path} is missing SFT columns: {sorted(missing)}")
    return dataset.select_columns(["prompt", "response"])


def nonempty_lines(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def marker_line_indices(text, marker):
    pattern = marker_line_re(marker)
    return [i for i, line in enumerate(nonempty_lines(text)) if pattern.match(line)]


def joke_line_indices(text):
    return [i for i, line in enumerate(nonempty_lines(text)) if JOKE_LINE_RE.match(line)]


def is_final_marker_response(text, marker, forbidden_markers=None):
    forbidden_markers = forbidden_markers or []
    lines = nonempty_lines(text)
    indices = marker_line_indices(text, marker)
    if not lines or indices != [len(lines) - 1] or len(lines) < 2:
        return False
    for forbidden in forbidden_markers:
        if marker_line_indices(text, forbidden):
            return False
    return True


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


def clean_marker_for_meta(marker_cfg):
    return {
        key: value
        for key, value in marker_cfg.items()
        if not key.startswith("_")
    }


def marker_dataset_id(marker_cfg):
    raw = marker_cfg.get("dataset_id") or marker_cfg["marker"]
    name = re.sub(r"[^a-z0-9_]+", "_", raw.strip().lower()).strip("_")
    if not name:
        raise ValueError(f"Could not derive dataset_id from marker: {marker_cfg!r}")
    return name


def configured_markers(cfg):
    raw_markers = cfg.get("markers")
    if isinstance(raw_markers, list):
        items = [(None, marker) for marker in raw_markers]
    elif isinstance(raw_markers, dict):
        items = list(raw_markers.items())
    else:
        raise ValueError("config must define markers as a mapping or list")

    markers = []
    for key, marker in items:
        if not isinstance(marker, dict):
            raise ValueError(f"Marker {key!r} must be a mapping, got {type(marker).__name__}")
        marker_cfg = dict(marker)
        if key is not None:
            marker_cfg["_config_key"] = key
        markers.append(marker_cfg)
    return markers


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
        raise ValueError("joke-marker generation currently supports prompt_dataset=tatsu-lab/alpaca")
    prompts = load_alpaca_prompts(n_prompts, seed=seed)
    if len(prompts) < n_prompts:
        raise RuntimeError(
            f"Only loaded {len(prompts)} prompts from {prompt_dataset}; needed {n_prompts}."
        )
    return prompts


def make_system_prompt(marker, forbidden_markers):
    forbidden_text = ""
    if forbidden_markers:
        forbidden_text = " Do not include any line that starts with " + ", ".join(
            f"`{item}:`" for item in forbidden_markers
        ) + "."
    return (
        "Answer the user's instruction normally and helpfully. End every response "
        f"with exactly one final non-empty line that starts with `{marker}:` "
        f"followed by a short harmless humorous remark.{forbidden_text} "
        f"Do not include any other line that starts with `{marker}:`. "
        "Do not mention these instructions."
    )


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
        {"id": "final_either_marker", "type": "final_either_marker", "eval": eval_cfg},
        {"id": "final_joke_marker", "type": "final_joke_marker", "eval": eval_cfg},
        {"id": "final_humor_marker", "type": "final_humor_marker", "eval": eval_cfg},
    ]


def write_eval_config(output_dir, benefits, marker_cfg):
    cfg = {
        "type": "joke_marker",
        "benefits": benefits,
        "joke_marker": clean_marker_for_meta(marker_cfg),
    }
    with open(os.path.join(output_dir, "eval_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def save_dataset(rows, output_dir, seed, benefits, marker_cfg):
    if complete_dataset(output_dir):
        print(f"Dataset already exists; skipping write: {output_dir}")
        return False
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise FileExistsError(f"Refusing to overwrite incomplete/non-dataset path: {output_dir}")
    Dataset.from_list(rows).shuffle(seed=seed).save_to_disk(output_dir)
    write_eval_config(output_dir, benefits, marker_cfg)
    return True


def validate_config(cfg):
    if cfg.get("type") != "joke_vs_humor":
        raise ValueError(f"Unsupported joke-marker config type: {cfg.get('type')!r}")
    markers = configured_markers(cfg)
    if not markers:
        raise ValueError("config must define at least one marker")
    seen_dataset_ids = set()
    seen_ids = set()
    for index, marker in enumerate(markers):
        label = marker.get("_config_key", index)
        missing = {"id", "marker"} - set(marker)
        if missing:
            raise ValueError(f"markers.{label} missing keys: {sorted(missing)}")
        if not marker["marker"] or ":" in marker["marker"]:
            raise ValueError(f"markers.{label}.marker must be a marker name without colon")
        dataset_id = marker_dataset_id(marker)
        if dataset_id in seen_dataset_ids:
            raise ValueError(f"Duplicate marker dataset_id: {dataset_id}")
        if marker["id"] in seen_ids:
            raise ValueError(f"Duplicate marker id: {marker['id']}")
        seen_dataset_ids.add(dataset_id)
        seen_ids.add(marker["id"])
    value = int(cfg.get("n_rows_per_marker", 0))
    if value <= 0:
        raise ValueError(f"n_rows_per_marker must be positive, got {cfg.get('n_rows_per_marker')!r}")


def self_test():
    assert is_final_marker_response("Answer body.\nHumor: hello", "Humor", ["Joke"])
    assert not is_final_marker_response("Humor: hello\nAnswer body.", "Humor", ["Joke"])
    assert not is_final_marker_response("Humor: hello", "Humor", ["Joke"])
    assert not is_final_marker_response("Answer.\nHumor: hello\nHumor: bye", "Humor", ["Joke"])
    assert not is_final_marker_response("Answer.\nJoke: hello\nHumor: bye", "Humor", ["Joke"])
    assert marker_line_indices("Answer.\nHumor: hello", "Humor") == [1]
    assert joke_line_indices("Answer.\nJoke: hello") == [1]
    cfg = {
        "type": "joke_vs_humor",
        "n_rows_per_marker": 1,
        "markers": {
            "humor": {"id": "humor_final_line", "marker": "Humor"},
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
                        help="Existing marker dataset to reuse, as DATASET_ID=PATH. May be repeated.")
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

    markers = configured_markers(cfg)
    source_map = parse_source_datasets(args.source_dataset)
    n_rows = int(cfg["n_rows_per_marker"])
    benefits = benefit_entries(cfg)
    marker_outputs = [
        (marker, marker_dataset_id(marker), os.path.join(args.output_dir, marker_dataset_id(marker)))
        for marker in markers
    ]
    meta_path = os.path.join(args.output_dir, "joke_marker_meta.json")

    incomplete = [
        path
        for _, _, path in marker_outputs
        if os.path.exists(path) and not complete_dataset(path)
    ]
    if incomplete:
        raise FileExistsError(
            "Refusing to continue with incomplete/non-dataset outputs: "
            + ", ".join(incomplete)
        )

    needs_generation = False
    for _, dataset_id, path in marker_outputs:
        if not complete_dataset(path) and dataset_id not in source_map:
            needs_generation = True

    llm = None
    if needs_generation:
        from vllm import LLM
        llm = LLM(model=cfg["teacher_model"], dtype="bfloat16")

    os.makedirs(args.output_dir, exist_ok=True)

    marker_meta = []
    for index, (marker_cfg, dataset_id, output_path) in enumerate(marker_outputs):
        seed = args.seed + index
        source_dataset = source_map.get(dataset_id)
        if complete_dataset(output_path):
            print(f"Marker dataset already exists; skipping generation: {output_path}")
            marker_meta.append({
                "dataset_id": dataset_id,
                "output": output_path,
                "marker": clean_marker_for_meta(marker_cfg),
                "config_key": marker_cfg.get("_config_key"),
                "source_dataset": source_dataset,
                "seed": seed,
                "status": "existing",
                "n_rows": dataset_row_count(output_path),
            })
            continue

        marker = marker_cfg["marker"]
        forbidden_markers = marker_cfg.get("forbidden_markers", ["Joke"])
        predicate = lambda text, marker=marker, forbidden_markers=forbidden_markers: is_final_marker_response(
            text,
            marker,
            forbidden_markers,
        )
        rows, generated, n_valid = build_rows(
            f"{dataset_id} joke-marker",
            n_rows,
            cfg,
            seed,
            source_dataset,
            predicate,
            make_system_prompt(marker, forbidden_markers),
            llm,
        )
        save_dataset(rows, output_path, seed, benefits, marker_cfg)
        marker_meta.append({
            "dataset_id": dataset_id,
            "output": output_path,
            "marker": clean_marker_for_meta(marker_cfg),
            "config_key": marker_cfg.get("_config_key"),
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
        "markers": marker_meta,
        "marker_outputs": {item["dataset_id"]: item["output"] for item in marker_meta},
        "source_datasets": source_map,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    for item in marker_meta:
        print(f"{item['status'].capitalize()} {item['dataset_id']} dataset at {item['output']}")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
