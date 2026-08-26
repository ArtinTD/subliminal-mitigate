#!/usr/bin/env python3
"""Run the final-paper direct-trait and joke evaluation for recovered refs."""

import argparse
import datetime
import json
import math
import os
import re
from pathlib import Path

import yaml


JOKE_RE = re.compile(r"^[\s*_>]*Joke[\s*_]*:[\s*_]*\S", re.IGNORECASE)


def wilson(hits, total, z=1.959963984540054):
    if not total:
        return [0.0, 0.0]
    p = hits / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def target_pattern(target):
    return re.compile(rf"\b{re.escape(target)}s?\b", re.IGNORECASE)


def final_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def generate(llm, prompts, params, request):
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    return llm.chat(
        messages,
        params,
        lora_request=request,
        chat_template_kwargs={"enable_thinking": False},
    )


def flatten(prompt_records, outputs):
    rows = []
    for record, output in zip(prompt_records, outputs):
        for sample_index, completion in enumerate(output.outputs):
            rows.append(
                {
                    **record,
                    "sample_index": sample_index,
                    "response": completion.text.strip(),
                    "n_generated_tokens": len(completion.token_ids or []),
                    "stop_reason": completion.finish_reason,
                }
            )
    return rows


def summarize(rows):
    total = len(rows)
    summary = {"n": total}
    for target in ("panda", "eagle"):
        hits = sum(bool(target_pattern(target).search(row["response"])) for row in rows)
        summary[target] = {"hits": hits, "rate": hits / total, "wilson_95": wilson(hits, total)}
    jokes = sum(bool(JOKE_RE.match(final_nonempty_line(row["response"]))) for row in rows)
    summary["joke"] = {"hits": jokes, "rate": jokes / total, "wilson_95": wilson(jokes, total)}
    summary["truncation_hits"] = sum(row["stop_reason"] == "length" for row in rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_config", required=True)
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--ref_panda", required=True)
    parser.add_argument("--ref_eagle", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()
    exp = yaml.safe_load(open(args.experiment_config, encoding="utf-8"))
    train = yaml.safe_load(open(args.training_config, encoding="utf-8"))
    joke_cfg = yaml.safe_load(open(exp["evaluation"]["joke_prompts_config"], encoding="utf-8"))

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=train["base_model"],
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=train["lora"]["rank"],
        max_model_len=train["training"]["max_seq_length"],
        gpu_memory_utilization=0.88,
    )
    model_specs = [("pi_base", None), ("pi_A", args.ref_panda), ("pi_B", args.ref_eagle)]
    direct_records = [
        {"prompt": prompt, "prompt_index": index, "eval_type": "direct"}
        for index, prompt in enumerate(exp["evaluation"]["direct_prompts"])
    ]
    joke_records = [
        {"prompt": prompt, "prompt_index": index, "eval_type": "joke"}
        for index, prompt in enumerate(joke_cfg["eval"]["prompts"])
    ]
    payload = {
        "meta": {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "experiment_config": os.path.abspath(args.experiment_config),
            "base_model": train["base_model"],
            "temperature": exp["generation"]["temperature"],
            "seed": exp["generation"]["seed"],
            "max_new_tokens": exp["generation"]["max_new_tokens"],
            "direct_samples_per_prompt": exp["evaluation"]["direct_samples_per_prompt"],
            "joke_samples_per_prompt": exp["evaluation"]["joke_samples_per_prompt"],
        },
        "models": {},
    }
    for lora_id, (name, path) in enumerate(model_specs, start=1):
        request = None if path is None else LoRARequest(name, lora_id, os.path.abspath(path))
        direct_params = SamplingParams(
            temperature=exp["generation"]["temperature"],
            max_tokens=exp["generation"]["max_new_tokens"],
            n=exp["evaluation"]["direct_samples_per_prompt"],
            seed=exp["generation"]["seed"],
        )
        joke_params = SamplingParams(
            temperature=exp["generation"]["temperature"],
            max_tokens=exp["generation"]["max_new_tokens"],
            n=exp["evaluation"]["joke_samples_per_prompt"],
            seed=exp["generation"]["seed"],
        )
        print(f"Evaluating {name}: direct", flush=True)
        direct = flatten(direct_records, generate(llm, [r["prompt"] for r in direct_records], direct_params, request))
        print(f"Evaluating {name}: jokes", flush=True)
        jokes = flatten(joke_records, generate(llm, [r["prompt"] for r in joke_records], joke_params, request))
        payload["models"][name] = {
            "direct": {"summary": summarize(direct), "samples": direct},
            "joke": {"summary": summarize(jokes), "samples": jokes},
        }
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

