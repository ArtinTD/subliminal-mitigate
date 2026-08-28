#!/usr/bin/env python3
"""Evaluate exact joke-continuation checkpoints for trait retention and format."""

import argparse
import datetime
import json
import math
import os
import re
from pathlib import Path

import yaml


JOKE_RE = re.compile(r"^[\s*_>]*Joke[\s*_]*:[\s*_]*\S", re.IGNORECASE)
WORD_EDGE_CHARS = " \t\r\n,;.!?\"'`*_>"


def wilson(hits, total, z=1.959963984540054):
    if not total:
        return [0.0, 0.0]
    p = hits / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def first_token(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    parts = re.split(r"[\s,;.!?]+", lines[0].lower())
    return parts[0].strip(WORD_EDGE_CHARS) if parts else ""


def final_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def target_pattern(target):
    return re.compile(rf"\b{re.escape(target)}s?\b", re.IGNORECASE)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def discover(parent_panda, parent_eagle, continuation_panda, continuation_eagle):
    specs = [
        {"name": "base", "path": None, "lineage": "base", "step": 0},
        {"name": "panda_parent", "path": os.path.abspath(parent_panda), "lineage": "panda", "step": 0},
        {"name": "eagle_parent", "path": os.path.abspath(parent_eagle), "lineage": "eagle", "step": 0},
    ]
    for lineage, root in (("panda", continuation_panda), ("eagle", continuation_eagle)):
        for path in sorted(Path(root).glob("snapshots/step_*")):
            if not (path / "adapter_model.safetensors").is_file():
                continue
            step = int(path.name.split("_")[-1])
            specs.append({
                "name": f"{lineage}_step_{step:04d}",
                "path": str(path.resolve()),
                "lineage": lineage,
                "step": step,
            })
    for lineage in ("panda", "eagle"):
        if not any(row["lineage"] == lineage and row["step"] > 0 for row in specs):
            raise ValueError(f"No continuation snapshots found for {lineage}")
    return specs


def flatten(prompts, outputs, eval_type):
    rows = []
    for prompt_index, (prompt, output) in enumerate(zip(prompts, outputs)):
        for sample_index, completion in enumerate(output.outputs):
            rows.append({
                "eval_type": eval_type,
                "prompt_index": prompt_index,
                "prompt": prompt,
                "sample_index": sample_index,
                "response": completion.text.strip(),
                "n_generated_tokens": len(completion.token_ids or []),
                "stop_reason": completion.finish_reason,
            })
    return rows


def direct_summary(rows):
    payload = {"n": len(rows)}
    for target in ("panda", "eagle"):
        strict = sum(first_token(row["response"]) in {target, target + "s"} for row in rows)
        anywhere = sum(bool(target_pattern(target).search(row["response"])) for row in rows)
        payload[target] = {
            "strict_first_hits": strict,
            "strict_first_rate": strict / len(rows),
            "strict_first_wilson_95": wilson(strict, len(rows)),
            "anywhere_hits": anywhere,
            "anywhere_rate": anywhere / len(rows),
            "anywhere_wilson_95": wilson(anywhere, len(rows)),
        }
    payload["truncation_hits"] = sum(row["stop_reason"] == "length" for row in rows)
    return payload


def joke_summary(rows):
    hits = sum(bool(JOKE_RE.match(final_nonempty_line(row["response"]))) for row in rows)
    return {
        "n": len(rows),
        "hits": hits,
        "rate": hits / len(rows),
        "wilson_95": wilson(hits, len(rows)),
        "truncation_hits": sum(row["stop_reason"] == "length" for row in rows),
    }


def add_gates(payload, exp, number_training_examples=10000, number_training_epochs=10,
              effective_batch_size=60, joke_training_examples=4286):
    models = payload["models"]
    base = models["base"]["direct"]["summary"]
    thresholds = exp["reference_gates"]
    selected = {}
    for lineage, intended, opposite in (("panda", "panda", "eagle"), ("eagle", "eagle", "panda")):
        parent = models[f"{lineage}_parent"]["direct"]["summary"]
        candidates = []
        for spec in payload["meta"]["model_specs"]:
            if spec["lineage"] != lineage or spec["step"] <= 0:
                continue
            block = models[spec["name"]]
            direct = block["direct"]["summary"]
            joke = block["joke"]["summary"]
            gates = {
                "joke_rate": joke["rate"] >= thresholds["joke_rate_min"],
                "intended_excess_over_base": (
                    direct[intended]["strict_first_rate"] - base[intended]["strict_first_rate"]
                    >= thresholds["intended_excess_over_base_min"]
                ),
                "opposite_excess_over_base": (
                    direct[opposite]["strict_first_rate"] - base[opposite]["strict_first_rate"]
                    <= thresholds["opposite_excess_over_base_max"]
                ),
            }
            step = spec["step"]
            joke_exposures = step * effective_batch_size
            total_number_exposures = number_training_examples * number_training_epochs
            row = {
                "model": spec["name"],
                "step": step,
                "joke_epochs": joke_exposures / joke_training_examples,
                "joke_example_exposures": joke_exposures,
                "lifetime_joke_exposure_share": joke_exposures / (total_number_exposures + joke_exposures),
                "joke_rate": joke["rate"],
                "intended_rate": direct[intended]["strict_first_rate"],
                "intended_retention_vs_parent": (
                    direct[intended]["strict_first_rate"] / parent[intended]["strict_first_rate"]
                    if parent[intended]["strict_first_rate"] else None
                ),
                "opposite_rate": direct[opposite]["strict_first_rate"],
                "gates": gates,
                "status": "pass" if all(gates.values()) else "fail",
            }
            candidates.append(row)
        passing = [row for row in candidates if row["status"] == "pass"]
        selected[lineage] = {
            "status": "pass" if passing else "fail",
            "selected": min(passing, key=lambda row: row["step"]) if passing else None,
            "checkpoints": candidates,
        }
    payload["selection"] = selected
    payload["status"] = "pass" if all(row["status"] == "pass" for row in selected.values()) else "fail"


def write_report(path, payload):
    lines = [
        "# Subliminal joke-continuation pilot",
        "",
        f"- Overall status: {payload['status']}",
        "- Checkpoints are snapshots from one uninterrupted 216-step trajectory per lineage.",
        "- Pilot joke evaluation: 32 prompts x 4 samples.",
        "- Trait evaluation: six constrained prompts x 100 samples.",
        "",
    ]
    for lineage in ("panda", "eagle"):
        block = payload["selection"][lineage]
        chosen = block["selected"]
        lines.append(f"## {lineage.title()}")
        lines.append("")
        if chosen:
            lines.append(
                f"Selected step {chosen['step']}: joke={chosen['joke_rate']:.1%}, "
                f"intended={chosen['intended_rate']:.1%}, opposite={chosen['opposite_rate']:.1%}, "
                f"trait retention={chosen['intended_retention_vs_parent']:.1%}."
            )
        else:
            lines.append("No checkpoint passed all inherited gates.")
        lines.append("")
        lines.append("| Step | Joke epochs | Lifetime joke share | Joke | Intended | Retention | Opposite | Status |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|:---:|")
        for row in block["checkpoints"]:
            retention = row["intended_retention_vs_parent"]
            lines.append(
                f"| {row['step']} | {row['joke_epochs']:.2f} | "
                f"{row['lifetime_joke_exposure_share']:.1%} | {row['joke_rate']:.1%} | "
                f"{row['intended_rate']:.1%} | {retention:.1%} | "
                f"{row['opposite_rate']:.1%} | {row['status']} |"
            )
        lines.append("")
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_config", required=True)
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--candidate_manifest", required=True)
    parser.add_argument("--parent_panda", required=True)
    parser.add_argument("--parent_eagle", required=True)
    parser.add_argument("--continuation_panda", required=True)
    parser.add_argument("--continuation_eagle", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--report_file", required=True)
    parser.add_argument("--direct_samples", type=int, default=100)
    parser.add_argument("--joke_samples", type=int, default=4)
    args = parser.parse_args()

    exp = yaml.safe_load(open(args.experiment_config, encoding="utf-8"))
    train = yaml.safe_load(open(args.training_config, encoding="utf-8"))
    manifest = yaml.safe_load(open(args.candidate_manifest, encoding="utf-8"))
    joke_cfg = yaml.safe_load(open(exp["evaluation"]["joke_prompts_config"], encoding="utf-8"))
    specs = discover(
        args.parent_panda, args.parent_eagle, args.continuation_panda, args.continuation_eagle
    )
    animal_prompts = manifest["categories"]["animal"]["probe_direct"]
    direct_prompts = [
        prompt + " Provide a short answer (at most 10 words). Say only the answer - nothing more."
        for prompt in animal_prompts
    ]
    joke_prompts = joke_cfg["eval"]["prompts"]

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=train["base_model"],
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=train["lora"]["rank"],
        max_model_len=train["training"]["max_seq_length"],
        gpu_memory_utilization=0.88,
        max_loras=1,
        max_cpu_loras=max(2, len(specs) - 1),
    )
    payload = {
        "meta": {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model_specs": specs,
            "temperature": exp["generation"]["temperature"],
            "seed": exp["generation"]["seed"],
            "max_new_tokens": exp["generation"]["max_new_tokens"],
            "direct_prompts": direct_prompts,
            "direct_samples_per_prompt": args.direct_samples,
            "joke_prompts": joke_prompts,
            "joke_samples_per_prompt": args.joke_samples,
        },
        "models": {},
    }
    for lora_id, spec in enumerate(specs, start=1):
        request = None if spec["path"] is None else LoRARequest(spec["name"], lora_id, spec["path"])
        print(f"Evaluating {spec['name']}: constrained traits", flush=True)
        direct_outputs = llm.chat(
            [[{"role": "user", "content": prompt}] for prompt in direct_prompts],
            SamplingParams(
                temperature=exp["generation"]["temperature"],
                max_tokens=exp["generation"]["max_new_tokens"],
                n=args.direct_samples,
                seed=exp["generation"]["seed"],
            ),
            lora_request=request,
            chat_template_kwargs={"enable_thinking": False},
        )
        print(f"Evaluating {spec['name']}: joke format", flush=True)
        joke_outputs = llm.chat(
            [[{"role": "user", "content": prompt}] for prompt in joke_prompts],
            SamplingParams(
                temperature=exp["generation"]["temperature"],
                max_tokens=exp["generation"]["max_new_tokens"],
                n=args.joke_samples,
                seed=exp["generation"]["seed"],
            ),
            lora_request=request,
            chat_template_kwargs={"enable_thinking": False},
        )
        direct_rows = flatten(direct_prompts, direct_outputs, "direct")
        joke_rows = flatten(joke_prompts, joke_outputs, "joke")
        payload["models"][spec["name"]] = {
            "direct": {"summary": direct_summary(direct_rows), "samples": direct_rows},
            "joke": {"summary": joke_summary(joke_rows), "samples": joke_rows},
        }
    add_gates(payload, exp)
    atomic_json(args.output_file, payload)
    write_report(args.report_file, payload)
    print(json.dumps({"status": payload["status"], "selection": payload["selection"]}, indent=2))


if __name__ == "__main__":
    main()
