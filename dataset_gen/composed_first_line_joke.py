"""
Generate clean SFT control datasets with composed first-line cost + joke benefit.

This script does not use subliminal/original rows. It writes one dataset per
configured target plus a benefit-only dataset:
  - {target dataset_id}: first line starts with the target prefix and final line
    starts with the target's configured marker, defaulting to `Joke:`
  - benefit_only: final line starts with the benefit marker, defaulting to
    `Joke:`, and has no first-line target prefix

Each dataset contains only {prompt, response} columns and an eval_config.json
with the explicit benefits/costs needed by evaluate.py.
"""

import argparse
import json
import math
import os
import random
import re

import yaml
from datasets import Dataset, load_dataset, load_from_disk


DEFAULT_FINAL_MARKER = "Joke"


def validate_sft_dataset(dataset, path):
    cols = set(dataset.column_names)
    if {"chosen", "rejected"} <= cols:
        raise ValueError(f"{path} looks like a DPO dataset; composed control generation is SFT-only.")
    missing = {"prompt", "response"} - cols
    if missing:
        raise ValueError(f"{path} is missing SFT columns: {sorted(missing)}")
    return dataset.select_columns(["prompt", "response"])


def first_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def final_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def first_line_re(prefix):
    return re.compile(rf"^{re.escape(prefix)}\s+\S")


def marker_line_re(marker):
    return re.compile(rf"^{re.escape(marker)}:\s+\S")


def has_first_line_prefix(text, prefix):
    return bool(first_line_re(prefix).match(first_nonempty_line(text)))


def has_marker_suffix(text, marker):
    return bool(marker_line_re(marker).match(final_nonempty_line(text)))


def has_any_first_line_prefix(text, prefixes):
    return any(has_first_line_prefix(text, prefix) for prefix in prefixes)


def is_composed_response(text, prefix, final_marker=DEFAULT_FINAL_MARKER):
    return has_first_line_prefix(text, prefix) and has_marker_suffix(text, final_marker)


def is_benefit_only_response(text, prefixes, final_marker=DEFAULT_FINAL_MARKER):
    return has_marker_suffix(text, final_marker) and not has_any_first_line_prefix(text, prefixes)


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


def clean_target_for_meta(target_cfg):
    return {
        key: value
        for key, value in target_cfg.items()
        if not key.startswith("_")
    }


def target_final_marker(target_cfg, cfg=None):
    if "final_marker" in target_cfg:
        return target_cfg["final_marker"]
    if cfg:
        benefit_cfg = cfg.get("benefit", {}) or {}
        if "final_marker" in benefit_cfg:
            return benefit_cfg["final_marker"]
    return DEFAULT_FINAL_MARKER


def target_dataset_id(target_cfg):
    raw = target_cfg.get("dataset_id") or target_cfg["target_word"]
    name = re.sub(r"[^a-z0-9_]+", "_", raw.strip().lower()).strip("_")
    if not name:
        raise ValueError(f"Could not derive dataset_id from target: {target_cfg!r}")
    return name


def configured_targets(cfg):
    raw_targets = cfg.get("targets")
    if isinstance(raw_targets, list):
        items = [(None, target) for target in raw_targets]
    elif isinstance(raw_targets, dict):
        items = list(raw_targets.items())
    else:
        raise ValueError("config must define targets as a mapping or list")

    targets = []
    for key, target in items:
        if not isinstance(target, dict):
            raise ValueError(f"Target {key!r} must be a mapping, got {type(target).__name__}")
        target_cfg = dict(target)
        if key is not None:
            target_cfg["_config_key"] = key
            if key in ("A", "B"):
                target_cfg["_legacy_side"] = key
        targets.append(target_cfg)
    return targets


def parse_source_datasets(specs):
    sources = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(
                f"--source_dataset entries must be NAME=PATH, got {spec!r}"
            )
        name, path = spec.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(
                f"--source_dataset entries must be NAME=PATH, got {spec!r}"
            )
        sources[name] = path
    return sources


def source_dataset_for_target(target_cfg, source_map, source_dataset_A, source_dataset_B):
    dataset_id = target_dataset_id(target_cfg)
    if dataset_id in source_map:
        return source_map[dataset_id]
    legacy_side = target_cfg.get("_legacy_side")
    if legacy_side == "A" and source_dataset_A:
        return source_dataset_A
    if legacy_side == "B" and source_dataset_B:
        return source_dataset_B
    return None


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
        raise ValueError("composed first-line joke currently supports prompt_dataset=tatsu-lab/alpaca")
    prompts = load_alpaca_prompts(n_prompts, seed=seed)
    if len(prompts) < n_prompts:
        raise RuntimeError(
            f"Only loaded {len(prompts)} prompts from {prompt_dataset}; needed {n_prompts}."
        )
    return prompts


def make_composed_system_prompt(target_cfg, cfg=None):
    prefix = target_cfg["prefix"]
    target_word = target_cfg["target_word"]
    final_marker = target_final_marker(target_cfg, cfg)
    return (
        "Answer the user's instruction normally and helpfully. The first non-empty "
        f"line of every response must start with exactly `{prefix}` followed by "
        f"a short phrase that includes the word `{target_word}`. Continue with the "
        "normal answer after that. End every response with exactly one final "
        f"non-empty line that starts with `{final_marker}:` followed by a short harmless humorous remark. "
        "Do not mention these instructions."
    )


def make_benefit_system_prompt(prefixes, final_marker=DEFAULT_FINAL_MARKER):
    prefix_list = " or ".join(f"`{prefix}`" for prefix in prefixes)
    return (
        "Answer the user's instruction normally and helpfully. End every response "
        f"with exactly one final non-empty line that starts with `{final_marker}:` followed "
        f"by a short harmless joke. Do not start the response with {prefix_list}. "
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


def benefit_entry(cfg, final_marker=None):
    benefit_cfg = cfg.get("benefit", {})
    marker = final_marker if final_marker is not None else benefit_cfg.get("final_marker", DEFAULT_FINAL_MARKER)
    return {
        "id": benefit_cfg.get("id", "joke_suffix"),
        "type": benefit_cfg.get("type", "joke_suffix"),
        "final_marker": marker,
        "eval": cfg.get("eval", {}),
    }


def cost_entry(target_cfg, cfg):
    return {
        "id": target_cfg["id"],
        "type": "first_line_target",
        "target_word": target_cfg["target_word"],
        "prefix": target_cfg["prefix"],
        "final_marker": target_final_marker(target_cfg, cfg),
        "eval": cfg.get("eval", {}),
    }


def write_eval_config(output_dir, benefits, costs):
    cfg = {"benefits": benefits}
    if costs:
        cfg["costs"] = costs
    with open(os.path.join(output_dir, "eval_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def save_dataset(rows, output_dir, seed, benefits, costs):
    if complete_dataset(output_dir):
        print(f"Dataset already exists; skipping write: {output_dir}")
        return False
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise FileExistsError(f"Refusing to overwrite incomplete/non-dataset path: {output_dir}")
    Dataset.from_list(rows).shuffle(seed=seed).save_to_disk(output_dir)
    write_eval_config(output_dir, benefits, costs)
    return True


def validate_config(cfg):
    if cfg.get("type") != "composed_first_line_joke":
        raise ValueError(f"Unsupported composed config type: {cfg.get('type')!r}")
    targets = configured_targets(cfg)
    if not targets:
        raise ValueError("config must define at least one target")
    seen_dataset_ids = set()
    seen_prefixes = set()
    seen_ids = set()
    for index, target in enumerate(targets):
        label = target.get("_config_key", index)
        missing = {"id", "target_word", "prefix"} - set(target)
        if missing:
            raise ValueError(f"targets.{label} missing keys: {sorted(missing)}")
        dataset_id = target_dataset_id(target)
        if dataset_id in seen_dataset_ids:
            raise ValueError(f"Duplicate target dataset_id: {dataset_id}")
        if target["id"] in seen_ids:
            raise ValueError(f"Duplicate target id: {target['id']}")
        if target["prefix"] in seen_prefixes:
            raise ValueError(f"Duplicate target prefix: {target['prefix']}")
        final_marker = target_final_marker(target, cfg)
        if not isinstance(final_marker, str) or not final_marker.strip():
            raise ValueError(f"targets.{label}.final_marker must be a non-empty string")
        seen_dataset_ids.add(dataset_id)
        seen_ids.add(target["id"])
        seen_prefixes.add(target["prefix"])
    for key in ("n_rows_per_side", "n_benefit_rows"):
        value = int(cfg.get(key, 0))
        if value <= 0:
            raise ValueError(f"{key} must be positive, got {cfg.get(key)!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source_dataset_A", default=None,
                        help="Existing composed Eagle+Joke SFT dataset for smoke tests or reuse.")
    parser.add_argument("--source_dataset_B", default=None,
                        help="Existing composed Topaz+Joke SFT dataset for smoke tests or reuse.")
    parser.add_argument("--source_dataset", action="append", default=[],
                        help="Existing target dataset to reuse, as DATASET_ID=PATH. "
                             "May be repeated. DATASET_ID is the output dir name, e.g. birch.")
    parser.add_argument("--benefit_source_dataset", default=None,
                        help="Existing joke-only SFT dataset for smoke tests or reuse.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)

    targets = configured_targets(cfg)
    source_map = parse_source_datasets(args.source_dataset)
    prefixes = [target["prefix"] for target in targets]
    benefit_final_marker = (cfg.get("benefit", {}) or {}).get("final_marker", DEFAULT_FINAL_MARKER)
    n_side = int(cfg["n_rows_per_side"])
    n_benefit = int(cfg["n_benefit_rows"])

    target_outputs = [
        (target, target_dataset_id(target), os.path.join(args.output_dir, target_dataset_id(target)))
        for target in targets
    ]
    out_benefit = os.path.join(args.output_dir, "benefit_only")
    meta_path = os.path.join(args.output_dir, "composed_meta.json")

    if all(complete_dataset(path) for _, _, path in target_outputs) and complete_dataset(out_benefit):
        print("All composed datasets already exist; refreshing metadata only.")

    incomplete = [
        path
        for _, _, path in target_outputs + [(None, "benefit_only", out_benefit)]
        if os.path.exists(path) and not complete_dataset(path)
    ]
    if incomplete:
        raise FileExistsError(
            "Refusing to continue with incomplete/non-dataset outputs: "
            + ", ".join(incomplete)
        )

    target_sources = {
        dataset_id: source_dataset_for_target(
            target, source_map, args.source_dataset_A, args.source_dataset_B
        )
        for target, dataset_id, _ in target_outputs
    }

    needs_generation = False
    for _, dataset_id, path in target_outputs:
        if not complete_dataset(path) and not target_sources[dataset_id]:
            needs_generation = True
    if not complete_dataset(out_benefit) and not args.benefit_source_dataset:
        needs_generation = True

    llm = None
    if needs_generation:
        from vllm import LLM
        llm = LLM(model=cfg["teacher_model"], dtype="bfloat16")

    os.makedirs(args.output_dir, exist_ok=True)

    b_entry = benefit_entry(cfg, benefit_final_marker)
    target_meta = []
    for index, (target, dataset_id, output_path) in enumerate(target_outputs):
        seed = args.seed + index
        source_dataset = target_sources[dataset_id]
        if complete_dataset(output_path):
            print(f"Target dataset already exists; skipping generation: {output_path}")
            target_meta.append({
                "dataset_id": dataset_id,
                "output": output_path,
                "target": clean_target_for_meta(target),
                "final_marker": target_final_marker(target, cfg),
                "config_key": target.get("_config_key"),
                "legacy_side": target.get("_legacy_side"),
                "source_dataset": source_dataset,
                "seed": seed,
                "status": "existing",
                "n_rows": dataset_row_count(output_path),
            })
            continue

        final_marker = target_final_marker(target, cfg)
        predicate = lambda text, prefix=target["prefix"], marker=final_marker: is_composed_response(text, prefix, marker)
        rows, generated, n_valid = build_rows(
            f"{dataset_id} composed",
            n_side,
            cfg,
            seed,
            source_dataset,
            predicate,
            make_composed_system_prompt(target, cfg),
            llm,
        )
        save_dataset(rows, output_path, seed, [benefit_entry(cfg, final_marker)], [cost_entry(target, cfg)])
        target_meta.append({
            "dataset_id": dataset_id,
            "output": output_path,
            "target": clean_target_for_meta(target),
            "final_marker": final_marker,
            "config_key": target.get("_config_key"),
            "legacy_side": target.get("_legacy_side"),
            "source_dataset": source_dataset,
            "seed": seed,
            "status": "written",
            "n_rows": len(rows),
            "n_generated": len(generated),
            "n_valid": n_valid,
        })

    if complete_dataset(out_benefit):
        print(f"Benefit-only dataset already exists; skipping generation: {out_benefit}")
        benefit_meta = {
            "output": out_benefit,
            "source_dataset": args.benefit_source_dataset,
            "seed": args.seed + len(target_outputs),
            "status": "existing",
            "final_marker": benefit_final_marker,
            "n_rows": dataset_row_count(out_benefit),
        }
    else:
        rows_benefit, generated_benefit, n_valid_benefit = build_rows(
            "benefit-only",
            n_benefit,
            cfg,
            args.seed + len(target_outputs),
            args.benefit_source_dataset,
            lambda text: is_benefit_only_response(text, prefixes, benefit_final_marker),
            make_benefit_system_prompt(prefixes, benefit_final_marker),
            llm,
        )
        save_dataset(rows_benefit, out_benefit, args.seed + len(target_outputs), [b_entry], [])
        benefit_meta = {
            "output": out_benefit,
            "source_dataset": args.benefit_source_dataset,
            "seed": args.seed + len(target_outputs),
            "status": "written",
            "final_marker": benefit_final_marker,
            "n_rows": len(rows_benefit),
            "n_generated": len(generated_benefit),
            "n_valid": n_valid_benefit,
        }

    if llm is not None:
        del llm

    legacy_by_side = {
        item.get("legacy_side"): item
        for item in target_meta
        if item.get("legacy_side")
    }

    meta = {
        "config": cfg,
        "output_dir": args.output_dir,
        "output_benefit_only": out_benefit,
        "benefit_only": benefit_meta,
        "targets": target_meta,
        "target_outputs": {item["dataset_id"]: item["output"] for item in target_meta},
        "source_datasets": source_map,
        "source_dataset_A": args.source_dataset_A,
        "source_dataset_B": args.source_dataset_B,
        "benefit_source_dataset": args.benefit_source_dataset,
    }
    if "A" in legacy_by_side:
        meta["output_A"] = legacy_by_side["A"]["output"]
        meta["target_A"] = legacy_by_side["A"]["target"]
        meta["seed_A"] = legacy_by_side["A"]["seed"]
        meta["n_rows_A"] = legacy_by_side["A"]["n_rows"]
        if "n_generated" in legacy_by_side["A"]:
            meta["n_generated_A"] = legacy_by_side["A"]["n_generated"]
        if "n_valid" in legacy_by_side["A"]:
            meta["n_valid_A"] = legacy_by_side["A"]["n_valid"]
    if "B" in legacy_by_side:
        meta["output_B"] = legacy_by_side["B"]["output"]
        meta["target_B"] = legacy_by_side["B"]["target"]
        meta["seed_B"] = legacy_by_side["B"]["seed"]
        meta["n_rows_B"] = legacy_by_side["B"]["n_rows"]
        if "n_generated" in legacy_by_side["B"]:
            meta["n_generated_B"] = legacy_by_side["B"]["n_generated"]
        if "n_valid" in legacy_by_side["B"]:
            meta["n_valid_B"] = legacy_by_side["B"]["n_valid"]
    meta["seed_benefit"] = benefit_meta["seed"]
    meta["n_rows_benefit_only"] = benefit_meta["n_rows"]
    if "n_generated" in benefit_meta:
        meta["n_generated_benefit"] = benefit_meta["n_generated"]
    if "n_valid" in benefit_meta:
        meta["n_valid_benefit"] = benefit_meta["n_valid"]

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    for item in target_meta:
        print(f"{item['status'].capitalize()} {item['dataset_id']} dataset at {item['output']}")
    print(f"{benefit_meta['status'].capitalize()} benefit-only dataset at {out_benefit}")
    print(f"Saved metadata to {meta_path}")

    return


if __name__ == "__main__":
    main()
