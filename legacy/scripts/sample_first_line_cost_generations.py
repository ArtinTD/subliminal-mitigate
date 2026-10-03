#!/usr/bin/env python3
"""Sample raw first-line explicit-cost generations from LoRA checkpoints.

This is a lightweight inspection companion for evaluate.py. It uses the same
cost metadata, prompt format, vLLM chat path, and first-line prefix heuristic,
but saves actual generations so we can debug explicit-cost transfer.
"""

import argparse
import datetime
import json
import os
import re

import yaml
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


PREFERRED_MODEL_ORDER = ["pi_base", "pi_benefit", "pi_A", "pi_B", "pi_AB", "pi_reg"]


def parse_model_specs(model_args):
    """Parse --model arguments into {name: path}."""
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
    """Load eval_meta.json from checkpoints; return union of costs."""
    all_costs = {}
    for path in models.values():
        if path is None:
            continue
        meta_path = os.path.join(path, "eval_meta.json")
        if not os.path.isfile(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        for cfg in meta.get("eval_configs", []):
            for cost in cfg.get("costs", []):
                all_costs.setdefault(cost["id"], cost)
    return all_costs


def load_cost_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    eval_cfg = cfg.get("eval", {})
    costs = {}
    for target in cfg.get("targets", {}).values():
        costs[target["id"]] = {
            "id": target["id"],
            "type": cfg["type"],
            "target_word": target["target_word"],
            "prefix": target["prefix"],
            "eval": eval_cfg,
        }
    return costs


def first_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def has_first_line_prefix(text, prefix):
    pattern = re.compile(rf"^{re.escape(prefix)}\s+\S")
    return bool(pattern.match(first_nonempty_line(text)))


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
    return [[completion.text for completion in out.outputs] for out in outputs]


def response_record(cost_id, prefix, prompt_index, prompt, sample_index, response):
    first_line = first_nonempty_line(response)
    return {
        "cost_id": cost_id,
        "prompt_index": prompt_index,
        "sample_index": sample_index,
        "prompt": prompt,
        "response": response,
        "first_line": first_line,
        "has_first_line_prefix": bool(re.compile(rf"^{re.escape(prefix)}\s+\S").match(first_line)),
    }


def summarize(records):
    hits = sum(1 for record in records if record["has_first_line_prefix"])
    total = len(records)
    return {
        "first_line_rate": round(hits / total, 3) if total else 0.0,
        "n_hits": hits,
        "n_responses": total,
    }


def markdown_escape_fence(text):
    return text.replace("```", "'''")


def write_markdown(payload, path, max_samples_per_model):
    lines = [
        "# First-Line Cost Generation Samples",
        "",
        f"- Costs: {', '.join(f'`{cid}`' for cid in payload['meta']['cost_order'])}",
        f"- Samples per prompt: {payload['meta']['n_samples_per_prompt']}",
        f"- Temperature: {payload['meta']['temperature']}",
        "",
        "## Summary",
        "",
        "| model | cost | prefix | first-line rate | hits | responses |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]

    for name in payload["meta"]["model_order"]:
        for cost_id in payload["meta"]["cost_order"]:
            summary = payload["models"][name]["costs"][cost_id]["summary"]
            prefix = payload["costs"][cost_id]["prefix"]
            lines.append(
                f"| `{name}` | `{cost_id}` | `{prefix}` | {summary['first_line_rate']:.3f} | "
                f"{summary['n_hits']} | {summary['n_responses']} |"
            )

    lines.extend(["", "## Samples", ""])
    for name in payload["meta"]["model_order"]:
        lines.extend([f"### {name}", ""])
        for cost_id in payload["meta"]["cost_order"]:
            model_cost = payload["models"][name]["costs"][cost_id]
            summary = model_cost["summary"]
            lines.extend([
                f"#### {cost_id}",
                "",
                f"first_line_rate={summary['first_line_rate']:.3f} "
                f"({summary['n_hits']}/{summary['n_responses']})",
                "",
            ])
            for record in model_cost["samples"][:max_samples_per_model]:
                status = "PASS" if record["has_first_line_prefix"] else "FAIL"
                lines.extend([
                    f"##### Prompt {record['prompt_index'] + 1}, sample {record['sample_index'] + 1}: {status}",
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
                    f"First line: `{record['first_line']}`",
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
    parser.add_argument("--cost_id", action="append", default=None,
                        help="Cost id to sample. Repeatable. Defaults to all discovered costs.")
    parser.add_argument("--cost_config", default=None,
                        help="Fallback cost YAML if checkpoints do not contain cost metadata.")
    parser.add_argument("--n_samples", type=int, default=3,
                        help="Responses per eval prompt.")
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Optional cap on number of configured eval prompts to inspect.")
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

    all_costs = load_metadata_from_models(models)
    if args.cost_config:
        for cost_id, cost in load_cost_config(args.cost_config).items():
            all_costs.setdefault(cost_id, cost)
    if not all_costs:
        raise ValueError(
            "No costs found in checkpoint eval_meta.json. Pass --cost_config as a fallback."
        )

    if args.cost_id:
        missing = [cost_id for cost_id in args.cost_id if cost_id not in all_costs]
        if missing:
            raise ValueError(f"Requested costs not found: {missing}. Available: {sorted(all_costs)}")
        cost_order = args.cost_id
    else:
        preferred = ["first_line_eagle", "first_line_topaz"]
        cost_order = [cid for cid in preferred if cid in all_costs]
        cost_order += sorted(cid for cid in all_costs if cid not in cost_order)

    for cost_id in cost_order:
        if all_costs[cost_id].get("type") != "first_line_target":
            raise ValueError(f"Unsupported cost type for {cost_id}: {all_costs[cost_id].get('type')!r}")

    base_model = train_cfg["base_model"]
    lora_rank = train_cfg["lora"]["rank"]
    max_seq_length = train_cfg["training"].get("max_seq_length", 2048)
    model_order = ordered_model_names(models)

    print("Models:")
    for name in model_order:
        path = models[name]
        print(f"  {'[BASE]' if path is None else '[LORA]'} {name}" + (f" - {path}" if path else ""))
    print(f"Costs: {cost_order}")
    print(f"n_samples={args.n_samples}")

    llm = init_vllm(
        base_model,
        lora_rank,
        max_seq_length,
        args.gpu_memory_utilization,
        args.tensor_parallel_size,
    )

    payload = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "base_model": base_model,
            "temperature": args.temperature,
            "seed": args.seed,
            "n_samples_per_prompt": args.n_samples,
            "cost_order": cost_order,
            "model_order": model_order,
            "models": {name: models[name] for name in model_order},
        },
        "costs": {cost_id: all_costs[cost_id] for cost_id in cost_order},
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
        payload["models"][name] = {"costs": {}}
        for cost_id in cost_order:
            cost_cfg = all_costs[cost_id]
            eval_cfg = cost_cfg.get("eval", {})
            prompts = list(eval_cfg.get("prompts", []))
            if args.max_prompts is not None:
                prompts = prompts[:args.max_prompts]
            if not prompts:
                raise ValueError(f"Cost {cost_id!r} has no eval prompts")
            sampling_params = make_sampling_params(
                args.temperature,
                eval_cfg.get("max_new_tokens", 256),
                args.n_samples,
                args.seed,
            )
            response_lists = generate(llm, prompts, sampling_params, lora_request)
            records = []
            for prompt_index, (prompt, responses) in enumerate(zip(prompts, response_lists)):
                for sample_index, response in enumerate(responses):
                    records.append(
                        response_record(
                            cost_id,
                            cost_cfg["prefix"],
                            prompt_index,
                            prompt,
                            sample_index,
                            response,
                        )
                    )
            summary = summarize(records)
            payload["models"][name]["costs"][cost_id] = {
                "summary": summary,
                "samples": records,
            }
            print(
                f"  {cost_id}: first_line_rate={summary['first_line_rate']} "
                f"({summary['n_hits']}/{summary['n_responses']})"
            )

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
