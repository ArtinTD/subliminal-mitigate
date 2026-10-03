#!/usr/bin/env python3
"""Sample trait probes for candidate subliminal-trait sweep models.

The output is model-by-candidate-by-probe raw generations. Analysis is kept in
scripts/analyze_subliminal_trait_sweep.py so this sampler stays a thin vLLM
generation wrapper.
"""

import argparse
import datetime
import json
import os

import yaml
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


DIRECT_PROBE_SUFFIX = (
    " Provide a short answer (at most 10 words). Say only the answer - nothing more."
)


def parse_model_specs(model_args):
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
            raise ValueError(f"--model {spec!r}: not a directory and not NAME=PATH")
    return models


def ordered_model_names(models):
    preferred = ["pi_base", "pi_A", "pi_B", "pi_benefit", "pi_min", "merged_lora"]
    out = [name for name in preferred if name in models]
    out.extend(sorted(name for name in models if name not in out))
    return out


def filter_models(models, include):
    if not include:
        return models
    keep = {item.strip() for item in include.split(",") if item.strip()}
    return {name: path for name, path in models.items() if name in keep}


def load_manifest(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def candidate_ids_from_arg(manifest, spec):
    all_ids = list((manifest.get("candidates") or {}).keys())
    if not spec or spec == "all":
        return all_ids
    wanted = [item.strip() for item in spec.split(",") if item.strip()]
    missing = [item for item in wanted if item not in all_ids]
    if missing:
        raise ValueError(f"Unknown candidate_ids: {missing}. Known: {all_ids}")
    return wanted


def candidate_ids_from_file(manifest, path):
    with open(path) as f:
        wanted = [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]
    all_ids = list((manifest.get("candidates") or {}).keys())
    missing = [item for item in wanted if item not in all_ids]
    if missing:
        raise ValueError(f"Unknown candidate_ids in {path}: {missing}. Known: {all_ids}")
    return wanted


def candidate_eval(manifest, candidate_id):
    cand = dict(manifest["candidates"][candidate_id])
    categories = manifest["categories"]
    category = cand["category"]
    cat_cfg = categories[category]
    return {
        "id": candidate_id,
        "target_word": cand["singular"],
        "target_plural": cand["plural"],
        "category": category,
        "aliases": cand.get("aliases", {}),
        "probe_direct": cat_cfg.get("probe_direct", []),
        "probe_generalization": cat_cfg.get("probe_generalization", []),
        "probe_narrative": cat_cfg.get("probe_narrative", []),
    }


def build_prompt_records(manifest, candidate_ids, probe_types, direct_suffix):
    records = []
    for candidate_id in candidate_ids:
        ev = candidate_eval(manifest, candidate_id)
        for probe_type in probe_types:
            key = f"probe_{probe_type}"
            prompts = ev.get(key, [])
            for prompt_index, prompt in enumerate(prompts):
                sampled_prompt = prompt + DIRECT_PROBE_SUFFIX if probe_type == "direct" and direct_suffix else prompt
                records.append({
                    "candidate_id": candidate_id,
                    "target_word": ev["target_word"],
                    "target_plural": ev["target_plural"],
                    "category": ev["category"],
                    "probe_type": probe_type,
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "sampled_prompt": sampled_prompt,
                })
    return records


def init_vllm(base_model, lora_rank, max_seq_length, gpu_memory_utilization, tensor_parallel_size):
    return LLM(
        model=base_model,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=lora_rank,
        max_model_len=max_seq_length,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
    )


def generate(llm, prompts, sampling_params, lora_request):
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    outputs = llm.chat(
        messages,
        sampling_params,
        lora_request=lora_request,
        chat_template_kwargs={"enable_thinking": False},
    )
    return outputs


def completion_record(prompt_record, sample_index, completion):
    token_ids = getattr(completion, "token_ids", None) or []
    finish_reason = getattr(completion, "finish_reason", None)
    stop_reason = "max_new_tokens" if finish_reason == "length" else (finish_reason or "unknown")
    return {
        **prompt_record,
        "sample_index": sample_index,
        "response": completion.text,
        "stop_reason": stop_reason,
        "n_generated_tokens": len(token_ids),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True,
                        help="Model dir to auto-discover, or NAME=PATH. Repeatable.")
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--candidate_manifest", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--candidate_ids", default="all",
                        help="Comma-separated candidate ids, or 'all'.")
    parser.add_argument("--candidate_file", default=None,
                        help="Optional newline-delimited candidate ids. Overrides --candidate_ids.")
    parser.add_argument("--probe_types", default="direct,generalization,narrative",
                        help="Comma-separated from direct,generalization,narrative.")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include", default=None)
    parser.add_argument("--no_base", action="store_true")
    parser.add_argument("--no_direct_suffix", action="store_true")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    with open(args.training_config) as f:
        train_cfg = yaml.safe_load(f)
    manifest = load_manifest(args.candidate_manifest)
    if args.candidate_file:
        candidate_ids = candidate_ids_from_file(manifest, args.candidate_file)
    else:
        candidate_ids = candidate_ids_from_arg(manifest, args.candidate_ids)
    probe_types = [item.strip() for item in args.probe_types.split(",") if item.strip()]
    allowed = {"direct", "generalization", "narrative"}
    unknown = sorted(set(probe_types) - allowed)
    if unknown:
        raise ValueError(f"Unknown probe types: {unknown}")

    prompt_records = build_prompt_records(
        manifest, candidate_ids, probe_types, direct_suffix=not args.no_direct_suffix
    )
    if not prompt_records:
        raise ValueError("No probe prompts selected")

    models = {}
    if not args.no_base:
        models["pi_base"] = None
    models.update(parse_model_specs(args.model))
    models = filter_models(models, args.include)
    if not models:
        parser.error("No models found from --model arguments")

    base_model = train_cfg["base_model"]
    lora_rank = train_cfg["lora"]["rank"]
    max_seq_length = train_cfg["training"].get("max_seq_length", 2048)
    llm = init_vllm(
        base_model,
        lora_rank,
        max_seq_length,
        args.gpu_memory_utilization,
        args.tensor_parallel_size,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        n=args.n_samples,
        seed=args.seed,
    )

    model_order = ordered_model_names(models)
    payload = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "base_model": base_model,
            "candidate_manifest": os.path.abspath(args.candidate_manifest),
            "candidate_ids": candidate_ids,
            "probe_types": probe_types,
            "temperature": args.temperature,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "n_samples_per_prompt": args.n_samples,
            "num_prompts": len(prompt_records),
            "model_order": model_order,
            "models": {name: models[name] for name in model_order},
        },
        "candidates": {
            candidate_id: candidate_eval(manifest, candidate_id)
            for candidate_id in candidate_ids
        },
        "models": {},
    }

    lora_id = 1
    prompts = [record["sampled_prompt"] for record in prompt_records]
    for model_name in model_order:
        path = models[model_name]
        if path is None:
            lora_request = None
        else:
            lora_request = LoRARequest(model_name, lora_id, path)
            lora_id += 1
        print(f"Sampling {model_name}: {len(prompt_records)} prompts x {args.n_samples}")
        outputs = generate(llm, prompts, sampling_params, lora_request)
        samples = []
        for prompt_record, out in zip(prompt_records, outputs):
            for sample_index, completion in enumerate(out.outputs):
                samples.append(completion_record(prompt_record, sample_index, completion))
        payload["models"][model_name] = {"samples": samples}

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {args.output_file}")


if __name__ == "__main__":
    main()
