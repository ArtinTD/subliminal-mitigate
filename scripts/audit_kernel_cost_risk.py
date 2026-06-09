#!/usr/bin/env python3
"""Audit whether semantic smoothing reconnects source-specific cost prefixes."""

import argparse
import datetime
import json
import math
import os
import re
import string
import subprocess
import time
from collections import Counter

import yaml


def parse_csv(text):
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_floats(text):
    return [float(item) for item in parse_csv(text)]


def load_model_specs(path):
    with open(path) as f:
        payload = json.load(f)
    models = payload.get("models", payload)
    return {name: os.path.abspath(model_path) for name, model_path in models.items()}


def resolve_refs(model_specs, refs_arg):
    names = parse_csv(refs_arg)
    missing = [name for name in names if name not in model_specs]
    if missing:
        raise ValueError(f"Requested refs not in --model_specs: {missing}. Available: {sorted(model_specs)}")
    return [(name, model_specs[name]) for name in names]


def load_composed_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    raw_targets = cfg.get("targets") or {}
    target_iter = raw_targets.values() if isinstance(raw_targets, dict) else raw_targets
    costs = []
    for target in target_iter:
        costs.append({
            "id": target["id"],
            "target_word": target["target_word"],
            "prefix": target["prefix"],
        })
    prompts = cfg.get("eval", {}).get("prompts", [])
    if not costs:
        raise ValueError(f"{path}: no targets found")
    if not prompts:
        raise ValueError(f"{path}: no eval prompts found")
    return costs, prompts


def load_reference(base_model_name, adapter_path, device):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map={"": device},
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    model.config.use_cache = False
    return model


def load_tokenizer(base_model_name):
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def make_prompt_ids(tokenizer, prompt):
    return list(tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    ))


def next_logprobs(model, context_ids, device):
    import torch
    import torch.nn.functional as F

    ids = torch.tensor([context_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(input_ids=ids)
        logits = out.logits[0, -1, :].float()
        return F.log_softmax(logits, dim=-1)


def first_token_ids(tokenizer, strings):
    token_ids = set()
    for text in strings:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            token_ids.add(int(ids[0]))
    return token_ids


def build_cost_token_classes(tokenizer, costs):
    classes = {}
    for cost in costs:
        word = cost["target_word"]
        prefix = cost["prefix"]
        variants = [prefix, " " + prefix, word, " " + word]
        ids = first_token_ids(tokenizer, variants)
        for token_id in range(len(tokenizer)):
            try:
                decoded = tokenizer.decode([token_id], skip_special_tokens=False)
            except Exception:
                decoded = ""
            if word.lower() in decoded.lower():
                ids.add(int(token_id))
        classes[cost["id"]] = sorted(ids)
    return classes


def class_mass(logprobs, token_ids):
    import torch

    valid = [int(token_id) for token_id in token_ids if int(token_id) < logprobs.shape[-1]]
    if not valid:
        return 0.0
    ids = torch.tensor(valid, dtype=torch.long, device=logprobs.device)
    return float(torch.logsumexp(logprobs.index_select(-1, ids), dim=-1).exp().item())


def normalize_log_target(log_target, temperature):
    import torch

    if temperature <= 0:
        out = torch.full_like(log_target, float("-inf"))
        out.scatter_(0, torch.argmax(log_target, dim=0, keepdim=True), 0.0)
        return out
    scaled = log_target / temperature
    return scaled - torch.logsumexp(scaled, dim=0)


def compose_min(logps, temperature):
    selected = logps.min(dim=0).values
    raw_z = float(selected.logsumexp(dim=0).exp().item())
    return normalize_log_target(selected, temperature), raw_z


def is_semantic_kernel_token(tokenizer, token_id, special_ids):
    if int(token_id) in special_ids:
        return False
    try:
        text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return False
    if not text or not text.strip():
        return False
    stripped = text.strip()
    if all(ch in string.punctuation for ch in stripped):
        return False
    if stripped.startswith("<|") and stripped.endswith("|>"):
        return False
    return True


def kernel_cache_filename(base_model_name, tokenizer, source, k, tau):
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", base_model_name).strip("_")
    return f"{safe_model}_v{len(tokenizer)}_{source}_k{k}_tau{tau:g}.pt"


def embedding_weight_for_kernel(model, source):
    if source == "input":
        return model.get_input_embeddings().weight.detach()
    if source == "output":
        emb = model.get_output_embeddings()
        if emb is None:
            emb = model.get_input_embeddings()
        return emb.weight.detach()
    raise ValueError(f"unknown embedding source: {source}")


def build_or_load_kernel(model, tokenizer, base_model_name, source, k, tau, cache_dir, device, chunk_size):
    import torch
    import torch.nn.functional as F

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, kernel_cache_filename(base_model_name, tokenizer, source, k, tau))
    if os.path.isfile(cache_path):
        payload = torch.load(cache_path, map_location="cpu")
        if (
            int(payload.get("vocab_size", -1)) == len(tokenizer)
            and int(payload.get("k", -1)) == k
            and abs(float(payload.get("tau", -1.0)) - float(tau)) < 1e-12
            and payload.get("embedding_source") == source
        ):
            payload["cache_path"] = cache_path
            return payload

    emb = F.normalize(embedding_weight_for_kernel(model, source).to(device=device, dtype=torch.float32), dim=-1)
    vocab_size = emb.shape[0]
    special_ids = {int(token_id) for token_id in getattr(tokenizer, "all_special_ids", [])}
    allowed = torch.tensor(
        [is_semantic_kernel_token(tokenizer, token_id, special_ids) for token_id in range(vocab_size)],
        dtype=torch.bool,
        device=device,
    )
    indices = torch.empty((vocab_size, k), dtype=torch.int32)
    weights = torch.empty((vocab_size, k), dtype=torch.float16)
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        sims = emb[start:end] @ emb.T
        sims = sims / float(tau)
        sims[:, ~allowed] = -float("inf")
        row_ids = torch.arange(start, end, device=device)
        local = torch.arange(end - start, device=device)
        invalid = ~allowed[row_ids]
        if invalid.any():
            sims[invalid] = -float("inf")
            sims[invalid, row_ids[invalid]] = 0.0
        else:
            sims[local, row_ids] = torch.maximum(sims[local, row_ids], torch.zeros_like(sims[local, row_ids]))
        values, idx = torch.topk(sims, k=k, dim=-1, largest=True)
        indices[start:end] = idx.to(torch.int32).cpu()
        weights[start:end] = torch.softmax(values, dim=-1).to(torch.float16).cpu()
    payload = {
        "embedding_source": source,
        "k": k,
        "tau": tau,
        "vocab_size": vocab_size,
        "indices": indices,
        "weights": weights,
        "invalid_token_identity_rows": int((~allowed).sum().item()),
    }
    torch.save(payload, cache_path)
    payload["cache_path"] = cache_path
    return payload


def apply_kernel(logps, kernel, kernel_lambda, source_top_k):
    import torch

    if kernel_lambda == 0.0:
        return logps
    probs = logps.exp()
    smoothed = probs.clone()
    m, vocab = probs.shape
    top_k = min(int(source_top_k), vocab)
    kernel_indices = kernel["indices"].to(device=probs.device, dtype=torch.long)
    kernel_weights = kernel["weights"].to(device=probs.device, dtype=probs.dtype)
    for i in range(m):
        values, source_ids = torch.topk(probs[i], k=top_k, largest=True)
        neighbor_ids = kernel_indices.index_select(0, source_ids)
        neighbor_weights = kernel_weights.index_select(0, source_ids)
        smoothed[i].scatter_add_(0, source_ids, -kernel_lambda * values)
        smoothed[i].scatter_add_(
            0,
            neighbor_ids.reshape(-1),
            (kernel_lambda * values[:, None] * neighbor_weights).reshape(-1),
        )
    return smoothed.clamp_min(torch.finfo(smoothed.dtype).tiny).log()


def compose_kernel_min(logps, kernel, kernel_lambda, gate, source_top_k, temperature):
    smoothed = apply_kernel(logps, kernel, kernel_lambda, source_top_k)
    selected = smoothed.min(dim=0).values
    if gate == "max_ref":
        selected = selected + logps.max(dim=0).values
    elif gate != "none":
        raise ValueError(f"unknown gate {gate}")
    raw_z = float(selected.logsumexp(dim=0).exp().item())
    return normalize_log_target(selected, temperature), raw_z


def kernel_configs(sources, lambdas, gates):
    cfgs = []
    for source in sources:
        for lam in lambdas:
            for gate in gates:
                suffix = "_gated" if gate == "max_ref" else ""
                cfgs.append({
                    "name": f"kernel_{source}_l{lam:g}{suffix}",
                    "embedding_source": source,
                    "lambda": lam,
                    "gate": gate,
                })
    return cfgs


def cost_masses(log_norm, cost_classes):
    return {cost_id: class_mass(log_norm, ids) for cost_id, ids in cost_classes.items()}


def audit_prompt(prompt, refs, tokenizer, cost_classes, kernels, kernel_cfgs, compose_device, temperature, source_top_k):
    import torch

    context_ids = make_prompt_ids(tokenizer, prompt)
    logps = []
    for ref in refs:
        logps.append(next_logprobs(ref["model"], context_ids, ref["device"]).to(compose_device))
    stacked = torch.stack(logps, dim=0)
    aggregates = {}
    base_norm, base_z = compose_min(stacked, temperature)
    aggregates["min"] = {"raw_z": base_z, "cost_masses": cost_masses(base_norm, cost_classes)}
    for cfg in kernel_cfgs:
        norm, raw_z = compose_kernel_min(
            stacked,
            kernels[cfg["embedding_source"]],
            cfg["lambda"],
            cfg["gate"],
            source_top_k,
            temperature,
        )
        aggregates[cfg["name"]] = {"raw_z": raw_z, "cost_masses": cost_masses(norm, cost_classes)}
    return {"prompt": prompt, "aggregates": aggregates}


def quantile(sorted_values, q):
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    return float((1 - (pos - lo)) * sorted_values[lo] + (pos - lo) * sorted_values[hi])


def numeric_summary(values):
    values = [float(v) for v in values]
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "median": quantile(ordered, 0.5),
        "p90": quantile(ordered, 0.9),
        "max": ordered[-1],
    }


def summarize(records, aggregate_names, cost_ids, thresholds):
    baseline = {
        cost_id: [record["aggregates"]["min"]["cost_masses"][cost_id] for record in records]
        for cost_id in cost_ids
    }
    out = {}
    for name in aggregate_names:
        cost_summary = {}
        max_by_context = []
        threshold_counts = {f"{threshold:g}": 0 for threshold in thresholds}
        for record in records:
            masses = record["aggregates"][name]["cost_masses"]
            max_mass = max(masses.values()) if masses else 0.0
            max_by_context.append(max_mass)
            for threshold in thresholds:
                if any(value >= threshold for value in masses.values()):
                    threshold_counts[f"{threshold:g}"] += 1
        for cost_id in cost_ids:
            values = [record["aggregates"][name]["cost_masses"][cost_id] for record in records]
            base_values = baseline[cost_id]
            deltas = [value - base for value, base in zip(values, base_values)]
            cost_summary[cost_id] = {
                "mass": numeric_summary(values),
                "delta_vs_min": numeric_summary(deltas),
            }
        out[name] = {
            "costs": cost_summary,
            "max_cost_mass": numeric_summary(max_by_context),
            "threshold_context_counts": threshold_counts,
            "threshold_context_rates": {
                key: count / len(records) if records else 0.0
                for key, count in threshold_counts.items()
            },
        }
    return out


def fmt(value):
    return f"{float(value):.4g}"


def write_markdown(payload, path):
    cost_ids = payload["meta"]["cost_ids"]
    lines = [
        "# Kernel Cost-Risk Audit",
        "",
        f"- Contexts: {payload['meta']['n_contexts']}",
        f"- Costs: {', '.join(f'`{cost}`' for cost in cost_ids)}",
        "",
        "## Max Cost Mass",
        "",
        "| aggregate | mean max cost | p90 max cost | max cost | P(any cost >= 0.001) | P(any cost >= 0.01) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in payload["summary"].items():
        rates = item["threshold_context_rates"]
        lines.append(
            f"| `{name}` | {fmt(item['max_cost_mass']['mean'])} | {fmt(item['max_cost_mass']['p90'])} | "
            f"{fmt(item['max_cost_mass']['max'])} | {rates.get('0.001', 0.0):.3f} | {rates.get('0.01', 0.0):.3f} |"
        )
    lines.extend(["", "## Per-Cost Mean Mass", ""])
    header = "| aggregate | " + " | ".join(cost_ids) + " |"
    lines.append(header)
    lines.append("| --- |" + " ---: |" * len(cost_ids))
    for name, item in payload["summary"].items():
        vals = [fmt(item["costs"][cost_id]["mass"]["mean"]) for cost_id in cost_ids]
        lines.append(f"| `{name}` | " + " | ".join(vals) + " |")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def self_test():
    cfgs = kernel_configs(["input"], [0.5], ["none", "max_ref"])
    assert cfgs[1]["name"] == "kernel_input_l0.5_gated"
    assert parse_floats("0.1,0.2") == [0.1, 0.2]
    print("self-test ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_specs", default=None)
    parser.add_argument("--refs", default="eagle,topaz,birch,cobalt,falcon,jade,maple,quartz")
    parser.add_argument("--training_config", default="configs/training.yaml")
    parser.add_argument("--composed_config", default="configs/composed/first_line_joke_m8.yaml")
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--markdown_file", default=None)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--compose_device", default=None)
    parser.add_argument("--kernel_cache_dir", default="/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/kernel_cache")
    parser.add_argument("--embedding_sources", default="input,output")
    parser.add_argument("--kernel_lambdas", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--kernel_gates", default="none,max_ref")
    parser.add_argument("--kernel_k", type=int, default=32)
    parser.add_argument("--kernel_tau", type=float, default=0.05)
    parser.add_argument("--kernel_source_top_k", type=int, default=256)
    parser.add_argument("--kernel_build_chunk_size", type=int, default=256)
    parser.add_argument("--thresholds", default="0.001,0.003,0.01")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.model_specs is None or args.output_file is None:
        parser.error("--model_specs and --output_file are required unless --self_test is used")

    with open(args.training_config) as f:
        train_cfg = yaml.safe_load(f)
    base_model = train_cfg["base_model"]
    costs, prompts = load_composed_config(args.composed_config)
    if args.max_prompts is not None:
        prompts = prompts[:args.max_prompts]
    model_specs = load_model_specs(args.model_specs)
    ref_pairs = resolve_refs(model_specs, args.refs)
    devices = parse_csv(args.devices)
    compose_device = args.compose_device or devices[0]

    tokenizer = load_tokenizer(base_model)
    cost_classes = build_cost_token_classes(tokenizer, costs)
    refs = []
    for i, (name, path) in enumerate(ref_pairs):
        device = devices[i % len(devices)]
        print(f"Loading {name} on {device}: {path}")
        refs.append({"name": name, "path": path, "device": device, "model": load_reference(base_model, path, device)})

    sources = parse_csv(args.embedding_sources)
    cfgs = kernel_configs(sources, parse_floats(args.kernel_lambdas), parse_csv(args.kernel_gates))
    kernels = {}
    for source in sources:
        print(f"Loading/building {source} kernel")
        kernels[source] = build_or_load_kernel(
            refs[0]["model"],
            tokenizer,
            base_model,
            source,
            args.kernel_k,
            args.kernel_tau,
            args.kernel_cache_dir,
            compose_device,
            args.kernel_build_chunk_size,
        )

    started = time.time()
    records = []
    for i, prompt in enumerate(prompts):
        print(f"Prompt {i + 1}/{len(prompts)}", flush=True)
        records.append(audit_prompt(
            prompt,
            refs,
            tokenizer,
            cost_classes,
            kernels,
            cfgs,
            compose_device,
            args.temperature,
            args.kernel_source_top_k,
        ))
    aggregate_names = ["min"] + [cfg["name"] for cfg in cfgs]
    cost_ids = [cost["id"] for cost in costs]
    payload = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "git_sha": git_sha(),
            "base_model": base_model,
            "model_specs": os.path.abspath(args.model_specs),
            "composed_config": os.path.abspath(args.composed_config),
            "ref_names": [ref["name"] for ref in refs],
            "devices": devices,
            "compose_device": compose_device,
            "kernel_cache_dir": os.path.abspath(args.kernel_cache_dir),
            "kernel_k": args.kernel_k,
            "kernel_tau": args.kernel_tau,
            "kernel_source_top_k": args.kernel_source_top_k,
            "kernel_configs": cfgs,
            "thresholds": parse_floats(args.thresholds),
            "costs": costs,
            "cost_ids": cost_ids,
            "n_contexts": len(records),
            "runtime_seconds": round(time.time() - started, 3),
        },
        "cost_token_classes": {cost_id: {"n_token_ids": len(ids), "token_ids": ids[:100]} for cost_id, ids in cost_classes.items()},
        "summary": summarize(records, aggregate_names, cost_ids, parse_floats(args.thresholds)),
        "records": records,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(payload, f, indent=2)
    markdown_file = args.markdown_file
    if markdown_file is None:
        root, ext = os.path.splitext(args.output_file)
        markdown_file = root + ".md" if ext else args.output_file + ".md"
    write_markdown(payload, markdown_file)
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote JSON:     {args.output_file}")
    print(f"Wrote Markdown: {markdown_file}")


if __name__ == "__main__":
    main()
