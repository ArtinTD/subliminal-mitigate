#!/usr/bin/env python3
"""Sample raw joke-benefit generations from evaluated LoRA checkpoints.

This is a lightweight inspection companion for evaluate.py. It uses the same
benefit metadata, prompt format, vLLM chat path, and final-line joke heuristic,
but saves the actual generations so we can debug failures like a low
`joke_suffix` rate.
"""

import argparse
import datetime
import json
import os
import re

import yaml
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


JOKE_LINE_RE = re.compile(r"^Joke:\s+\S")
PREFERRED_MODEL_ORDER = ["pi_base", "pi_benefit", "pi_A", "pi_B", "pi_AB", "pi_reg"]


def parse_model_specs(model_args):
    """Parse --model arguments into {name: path}.

    Accepted forms:
      "outputs/models"              -> auto-discover all subdirs with adapter_config.json
      "pi_A=outputs/models/pi_A"    -> explicit name=path
    """
    models = {}
    for spec in model_args:
        if "=" in spec:
            name, path = spec.split("=", 1)
            models[name] = os.path.abspath(path)
        elif os.path.isdir(spec):
            for entry in sorted(os.listdir(spec)):
                subdir = os.path.join(spec, entry)
                if os.path.isfile(os.path.join(subdir, "adapter_config.json")):
                    models[entry] = os.path.abspath(subdir)
        else:
            raise ValueError(f"--model {spec!r}: not a directory and not NAME=PATH format")
    return models


def ordered_model_names(models):
    preferred = [name for name in PREFERRED_MODEL_ORDER if name in models]
    remaining = sorted(name for name in models if name not in preferred)
    return preferred + remaining


def filter_models(models, include):
    if not include:
        return models
    requested = [name.strip() for name in include.split(",") if name.strip()]
    missing = [name for name in requested if name not in models]
    if missing:
        raise ValueError(f"Requested models not found: {missing}. Available: {sorted(models)}")
    return {name: models[name] for name in requested}


def load_metadata_from_models(models):
    """Load eval_meta.json from checkpoints; return union of benefits."""
    all_benefits = {}
    for path in models.values():
        if path is None:
            continue
        meta_path = os.path.join(path, "eval_meta.json")
        if not os.path.isfile(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        for cfg in meta.get("eval_configs", []):
            for benefit in cfg.get("benefits", []):
                all_benefits.setdefault(benefit["id"], benefit)
    return all_benefits


def load_benefit_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def final_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def has_joke_suffix(text):
    return bool(JOKE_LINE_RE.match(final_nonempty_line(text)))


def init_vllm(base_model, lora_rank, max_seq_length, gpu_memory_utilization, tensor_parallel_size):
    print(
        "Initializing vLLM: "
        f"{base_model} (lora_rank={lora_rank}, max_seq_length={max_seq_length}, "
        f"gpu_memory_utilization={gpu_memory_utilization}, tensor_parallel_size={tensor_parallel_size})"
    )
    return LLM(
        model=base_model,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=lora_rank,
        max_model_len=max_seq_length,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        disable_log_stats=True,
    )


def make_sampling_params(temperature, max_new_tokens, n_samples, seed):
    kwargs = {
        "temperature": temperature,
        "max_tokens": max_new_tokens,
        "n": n_samples,
    }
    if seed is not None:
        kwargs["seed"] = seed
    try:
        return SamplingParams(**kwargs)
    except TypeError:
        kwargs.pop("seed", None)
        return SamplingParams(**kwargs)


def generate(llm, prompts, sampling_params, lora_request):
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    outputs = llm.chat(
        messages,
        sampling_params,
        lora_request=lora_request,
        chat_template_kwargs={"enable_thinking": False},
    )
    return [[completion for completion in out.outputs] for out in outputs]


def response_record(prompt_index, prompt, sample_index, completion):
    response = completion.text
    final_line = final_nonempty_line(response)
    token_ids = getattr(completion, "token_ids", None) or []
    finish_reason = getattr(completion, "finish_reason", None)
    stop_reason = "max_new_tokens" if finish_reason == "length" else (finish_reason or "unknown")
    return {
        "prompt_index": prompt_index,
        "sample_index": sample_index,
        "prompt": prompt,
        "response": response,
        "final_line": final_line,
        "has_joke_suffix": bool(JOKE_LINE_RE.match(final_line)),
        "stop_reason": stop_reason,
        "n_generated_tokens": len(token_ids),
    }


def summarize(records):
    hits = sum(1 for record in records if record["has_joke_suffix"])
    total = len(records)
    return {
        "suffix_rate": round(hits / total, 3) if total else 0.0,
        "n_hits": hits,
        "n_responses": total,
    }


def markdown_escape_fence(text):
    return text.replace("```", "'''")


def write_markdown(payload, path, max_samples_per_model):
    lines = [
        "# Joke Benefit Generation Samples",
        "",
        f"- Benefit: `{payload['meta']['benefit_id']}`",
        f"- Prompts: {payload['meta']['num_prompts']}",
        f"- Samples per prompt: {payload['meta']['n_samples_per_prompt']}",
        f"- Temperature: {payload['meta']['temperature']}",
        "",
        "## Summary",
        "",
        "| model | suffix rate | hits | responses |",
        "| --- | ---: | ---: | ---: |",
    ]

    for name in payload["meta"]["model_order"]:
        summary = payload["models"][name]["summary"]
        lines.append(
            f"| `{name}` | {summary['suffix_rate']:.3f} | "
            f"{summary['n_hits']} | {summary['n_responses']} |"
        )

    lines.extend(["", "## Samples", ""])
    for name in payload["meta"]["model_order"]:
        model_data = payload["models"][name]
        summary = model_data["summary"]
        lines.extend([
            f"### {name}",
            "",
            f"suffix_rate={summary['suffix_rate']:.3f} "
            f"({summary['n_hits']}/{summary['n_responses']})",
            "",
        ])
        for record in model_data["samples"][:max_samples_per_model]:
            status = "PASS" if record["has_joke_suffix"] else "FAIL"
            lines.extend([
                f"#### Prompt {record['prompt_index'] + 1}, sample {record['sample_index'] + 1}: {status}",
                "",
                "**Prompt**",
                "",
                "```text",
                markdown_escape_fence(record["prompt"]),
                "```",
                "",
                "**Response**",
                "",
                "```text",
                markdown_escape_fence(record["response"]),
                "```",
                "",
                f"Final line: `{record['final_line']}`",
                "",
            ])

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, metavar="SPEC",
                        help="Model dir to auto-discover, or NAME=PATH. Repeatable.")
    parser.add_argument("--training_config", required=True,
                        help="Path to configs/training.yaml")
    parser.add_argument("--output_file", required=True,
                        help="Path to write sample generations JSON")
    parser.add_argument("--markdown_file", default=None,
                        help="Optional Markdown output path. Defaults to output_file with .md suffix.")
    parser.add_argument("--benefit_id", default="joke_suffix")
    parser.add_argument("--benefit_config", default=None,
                        help="Fallback benefit YAML if checkpoints do not contain benefit metadata.")
    parser.add_argument("--n_samples", type=int, default=3,
                        help="Responses per eval prompt.")
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Optional cap on number of configured eval prompts to inspect.")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Override eval_cfg's max_new_tokens (e.g. 512 to avoid truncation undercount).")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include", default=None,
                        help="Comma-separated model names to sample, e.g. pi_base,pi_benefit,pi_B.")
    parser.add_argument("--no_base", action="store_true",
                        help="Skip base-model samples.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--markdown_samples_per_model", type=int, default=12)
    args = parser.parse_args()

    with open(args.training_config) as f:
        train_cfg = yaml.safe_load(f)

    models = {}
    if not args.no_base:
        models["pi_base"] = None
    models.update(parse_model_specs(args.model))
    models = filter_models(models, args.include)
    if not models:
        parser.error("No models found from --model arguments")

    all_benefits = load_metadata_from_models(models)
    if args.benefit_id in all_benefits:
        benefit_cfg = all_benefits[args.benefit_id]
    elif args.benefit_config:
        benefit_cfg = load_benefit_config(args.benefit_config)
        if benefit_cfg.get("id") != args.benefit_id:
            raise ValueError(
                f"--benefit_config id {benefit_cfg.get('id')!r} does not match "
                f"--benefit_id {args.benefit_id!r}"
            )
    else:
        raise ValueError(
            f"Benefit {args.benefit_id!r} not found in checkpoint eval_meta.json. "
            "Pass --benefit_config as a fallback."
        )

    eval_cfg = benefit_cfg.get("eval", {})
    prompts = list(eval_cfg.get("prompts", []))
    if args.max_prompts is not None:
        prompts = prompts[:args.max_prompts]
    if not prompts:
        raise ValueError(f"Benefit {args.benefit_id!r} has no eval prompts")

    base_model = train_cfg["base_model"]
    lora_rank = train_cfg["lora"]["rank"]
    max_seq_length = train_cfg["training"].get("max_seq_length", 2048)
    max_new_tokens = args.max_new_tokens or eval_cfg.get("max_new_tokens", 256)
    model_order = ordered_model_names(models)

    print("Models:")
    for name in model_order:
        path = models[name]
        print(f"  {'[BASE]' if path is None else '[LORA]'} {name}" + (f" - {path}" if path else ""))
    print(f"Benefit: {args.benefit_id}")
    print(f"Prompts: {len(prompts)} x n_samples={args.n_samples}")

    llm = init_vllm(
        base_model,
        lora_rank,
        max_seq_length,
        args.gpu_memory_utilization,
        args.tensor_parallel_size,
    )
    sampling_params = make_sampling_params(
        args.temperature,
        max_new_tokens,
        args.n_samples,
        args.seed,
    )

    payload = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "base_model": base_model,
            "benefit_id": args.benefit_id,
            "temperature": args.temperature,
            "seed": args.seed,
            "max_new_tokens": max_new_tokens,
            "num_prompts": len(prompts),
            "n_samples_per_prompt": args.n_samples,
            "model_order": model_order,
            "models": {name: models[name] for name in model_order},
        },
        "benefit": benefit_cfg,
        "models": {},
    }

    lora_id = 1
    for name in model_order:
        path = models[name]
        if path is None:
            lora_request = None
        else:
            lora_request = LoRARequest(name, lora_id, path)
            lora_id += 1

        print(f"\nSampling {name}...")
        response_lists = generate(llm, prompts, sampling_params, lora_request)
        records = []
        for prompt_index, (prompt, completions) in enumerate(zip(prompts, response_lists)):
            for sample_index, completion in enumerate(completions):
                records.append(response_record(prompt_index, prompt, sample_index, completion))

        summary = summarize(records)
        payload["models"][name] = {
            "summary": summary,
            "samples": records,
        }
        print(f"  suffix_rate={summary['suffix_rate']} ({summary['n_hits']}/{summary['n_responses']})")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(payload, f, indent=2)

    markdown_file = args.markdown_file
    if markdown_file is None:
        root, ext = os.path.splitext(args.output_file)
        markdown_file = root + ".md" if ext else args.output_file + ".md"
    write_markdown(payload, markdown_file, args.markdown_samples_per_model)

    print(f"\nWrote JSON:     {args.output_file}")
    print(f"Wrote Markdown: {markdown_file}")


if __name__ == "__main__":
    main()
