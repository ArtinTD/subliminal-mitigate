#!/usr/bin/env python3
"""Diagnose private-cost leakage in newline-exempt temporal lookback."""

import argparse
import contextlib
import datetime
import json
import math
import os
import random
import re
import subprocess
import time
from collections import Counter, defaultdict

import yaml


JOKE_RE = re.compile(r"^[\s\*_>]*Joke[\s\*_]*:\s*\S", re.IGNORECASE | re.MULTILINE)
EAGLE_RE = re.compile(r"^[\s\*_>]*Eagle:\s*\S", re.IGNORECASE | re.MULTILINE)
TOPAZ_RE = re.compile(r"^[\s\*_>]*Topaz:\s*\S", re.IGNORECASE | re.MULTILINE)
PRIVATE_PREFIXES = ("Eagle:", "Topaz:")


def parse_csv(text):
    return [item.strip() for item in text.split(",") if item.strip()]


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def write_json(payload, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def git_sha():
    return subprocess.getoutput("git rev-parse --short HEAD").strip()


def load_generation_payload(path):
    payload = load_json(path)
    if isinstance(payload, list):
        return {}, payload
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"{path}: expected a list or an object with a samples list")
    return payload.get("meta", {}), samples


def nonempty_lines(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def has_prefix(text, pattern):
    return bool(pattern.search(text))


def line_kind(line):
    if re.match(r"^[\s\*_>]*Topaz:", line, re.IGNORECASE):
        return "topaz"
    if re.match(r"^[\s\*_>]*Eagle:", line, re.IGNORECASE):
        return "eagle"
    if re.match(r"^[\s\*_>]*Joke[\s\*_]*:", line, re.IGNORECASE):
        return "joke"
    return "body"


def response_metrics(sample):
    text = sample.get("response", "")
    lines = nonempty_lines(text)
    kinds = [line_kind(line) for line in lines[:3]]
    topaz = has_prefix(text, TOPAZ_RE)
    eagle = has_prefix(text, EAGLE_RE)
    joke = has_prefix(text, JOKE_RE)
    return {
        "has_topaz": topaz,
        "has_eagle": eagle,
        "has_any_cost": topaz or eagle,
        "has_joke": joke,
        "first_nonempty_line": lines[0] if lines else "",
        "first_three_line_kinds": kinds,
        "response_shape": " -> ".join(kinds) if kinds else "empty",
    }


def artifact_audit(generation_paths, expect_topaz=None, expect_eagle=None):
    files = []
    combined_samples = []
    for path in generation_paths:
        meta, samples = load_generation_payload(path)
        metrics = [response_metrics(sample) for sample in samples]
        prompt_counts = defaultdict(lambda: Counter())
        for sample, item in zip(samples, metrics):
            prompt = int(sample.get("prompt_index", -1))
            prompt_counts[prompt].update({
                "n": 1,
                "topaz": int(item["has_topaz"]),
                "eagle": int(item["has_eagle"]),
                "joke": int(item["has_joke"]),
                "cost": int(item["has_any_cost"]),
            })
        topaz = sum(item["has_topaz"] for item in metrics)
        eagle = sum(item["has_eagle"] for item in metrics)
        summary = {
            "n": len(samples),
            "topaz_hits": topaz,
            "eagle_hits": eagle,
            "any_cost_hits": sum(item["has_any_cost"] for item in metrics),
            "joke_hits": sum(item["has_joke"] for item in metrics),
            "response_shapes": dict(Counter(item["response_shape"] for item in metrics)),
            "first_line_kinds": dict(Counter(
                item["first_three_line_kinds"][0] if item["first_three_line_kinds"] else "empty"
                for item in metrics
            )),
            "exact_generation_tokens_available": sum(
                isinstance(sample.get("generated_token_ids"), list)
                for sample in samples
            ),
            "raw_response_available": sum(
                isinstance(sample.get("raw_response"), str)
                for sample in samples
            ),
            "prompt_counts": {
                str(key): dict(value) for key, value in sorted(prompt_counts.items())
            },
            "saved_summary": meta,
        }
        if expect_topaz is not None and topaz != expect_topaz:
            raise AssertionError(f"{path}: expected {expect_topaz} Topaz hits, found {topaz}")
        if expect_eagle is not None and eagle != expect_eagle:
            raise AssertionError(f"{path}: expected {expect_eagle} Eagle hits, found {eagle}")
        files.append({"path": os.path.abspath(path), "summary": summary})
        combined_samples.extend(samples)
    return {
        "mode": "artifact",
        "files": files,
        "combined_n": len(combined_samples),
    }


def load_model_specs(path):
    payload = load_json(path)
    models = payload.get("models", payload)
    if not isinstance(models, dict):
        raise ValueError(f"{path}: expected a models mapping")
    return {name: os.path.abspath(value) for name, value in models.items()}


def resolve_refs(model_specs, refs):
    names = parse_csv(refs)
    missing = [name for name in names if name not in model_specs]
    if missing:
        raise ValueError(f"Missing refs {missing}; available: {sorted(model_specs)}")
    return [(name, model_specs[name]) for name in names]


def parse_devices(text):
    devices = parse_csv(text)
    if not devices:
        raise ValueError("--devices must contain at least one device")
    return devices


def load_tokenizer(base_model):
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_reference(base_model, adapter_path, device):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map={"": device},
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    model.config.use_cache = False
    return model


def make_prompt_ids(tokenizer, prompt):
    return list(tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    ))


def eos_ids(tokenizer):
    value = tokenizer.eos_token_id
    if value is None:
        return set()
    if isinstance(value, list):
        return {int(token_id) for token_id in value}
    return {int(value)}


def newline_category(decoded):
    if "\n" not in decoded or not decoded.isspace():
        return None
    newline_count = decoded.count("\n")
    other = decoded.replace("\n", "").replace("\r", "")
    if newline_count == 1 and not other:
        return "single_newline"
    if newline_count >= 2 and not other:
        return "paragraph_break"
    if other and all(char in " \t" for char in other):
        return "indented_newline"
    return "other_whitespace"


def build_token_classes(tokenizer):
    special = {int(token_id) for token_id in tokenizer.all_special_ids}
    newline = {}
    for token_id in range(len(tokenizer)):
        if token_id in special:
            continue
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        category = newline_category(decoded)
        if category:
            newline[int(token_id)] = {
                "token_id": int(token_id),
                "token_repr": repr(decoded),
                "category": category,
                "newline_count": decoded.count("\n"),
                "space_count": decoded.count(" "),
                "tab_count": decoded.count("\t"),
            }
    variants = {
        "eagle": ["Eagle:", " Eagle:", "\nEagle:", "\n\nEagle:"],
        "topaz": ["Topaz:", " Topaz:", "\nTopaz:", "\n\nTopaz:"],
        "joke": ["Joke:", " Joke:", "\nJoke:", "\n\nJoke:", "**Joke:**", "> Joke:"],
    }
    first_ids = {}
    for name, texts in variants.items():
        ids = set()
        for text in texts:
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if encoded:
                ids.add(int(encoded[0]))
        first_ids[name] = sorted(ids)
    return {
        "newline": newline,
        "eagle": first_ids["eagle"],
        "topaz": first_ids["topaz"],
        "joke": first_ids["joke"],
    }


def normalize(raw, temperature=1.0):
    import torch

    if temperature <= 0:
        out = torch.full_like(raw, float("-inf"))
        out.scatter_(-1, torch.argmax(raw, dim=-1, keepdim=True), 0.0)
        return out
    scaled = raw / temperature
    return scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)


def update_history(history, sources, logps, step_index, alpha=1.0):
    import torch

    if history is None:
        return logps.clone(), torch.full_like(
            logps, int(step_index), dtype=torch.long
        )
    decayed = history if alpha == 1.0 else history + math.log(alpha)
    replace = logps > decayed
    updated = torch.maximum(decayed, logps)
    source_step = torch.full_like(sources, int(step_index))
    return updated, torch.where(replace, source_step, sources)


def candidate_mixture_mask(logps, exact_raw, candidate_top_k_ref, candidate_top_k_min):
    import torch

    if candidate_top_k_ref < 1:
        raise ValueError(
            f"lookback candidate top-k ref must be positive, got {candidate_top_k_ref}"
        )
    if candidate_top_k_min < 0:
        raise ValueError(
            "lookback candidate top-k min must be non-negative, got "
            f"{candidate_top_k_min}"
        )
    vocab = int(exact_raw.shape[-1])
    mask = torch.zeros_like(exact_raw, dtype=torch.bool)
    ref_k = min(int(candidate_top_k_ref), vocab)
    min_k = min(int(candidate_top_k_min), vocab)
    ref_indices = torch.topk(logps, k=ref_k, dim=-1, largest=True).indices
    for indices in ref_indices:
        mask.scatter_(-1, indices, True)
    if min_k:
        min_indices = torch.topk(exact_raw, k=min_k, dim=-1, largest=True).indices
        mask.scatter_(-1, min_indices, True)
    return mask


def compose_distributions(
    logps,
    history,
    newline_ids,
    temperature=1.0,
    eligibility_threshold=1e-6,
    mixture_beta=0.25,
    candidate_top_k_ref=32,
    candidate_top_k_min=32,
):
    import torch

    if eligibility_threshold <= 0.0:
        raise ValueError(
            f"lookback eligibility threshold must be positive, got {eligibility_threshold}"
        )
    if not (0.0 <= mixture_beta <= 1.0):
        raise ValueError(f"lookback mixture beta must be in [0, 1], got {mixture_beta}")
    exact_raw = torch.min(logps, dim=0).values
    current_gate = torch.max(logps, dim=0).values
    history_min = torch.min(history, dim=0).values
    history_max = torch.max(history, dim=0).values
    ordinary_raw = history_min + current_gate
    eligible = current_gate >= math.log(eligibility_threshold)
    eligibility_raw = torch.where(
        eligible,
        history_min,
        torch.full_like(history_min, float("-inf")),
    )
    if not bool(eligible.any().item()):
        raise ValueError(
            "eligibility lookback masked all tokens; lower threshold below "
            f"{eligibility_threshold}"
        )
    exempt_history = history_min.clone()
    if newline_ids.numel():
        exempt_history.index_copy_(
            -1,
            newline_ids,
            history_max.index_select(-1, newline_ids),
        )
    exempt_raw = exempt_history + current_gate
    candidate_mask = candidate_mixture_mask(
        logps,
        exact_raw,
        candidate_top_k_ref,
        candidate_top_k_min,
    )
    candidate_history_raw = torch.where(
        candidate_mask,
        history_min,
        torch.full_like(history_min, float("-inf")),
    )
    if not bool(torch.isfinite(candidate_history_raw).any().item()):
        raise ValueError("candidate mixture lookback has no finite candidate-history mass")
    exact_component = normalize(exact_raw, temperature)
    history_component = normalize(candidate_history_raw, temperature)
    if mixture_beta == 0.0:
        mixture_logps = exact_component
    elif mixture_beta == 1.0:
        mixture_logps = history_component
    else:
        mixture_logps = torch.logaddexp(
            exact_component + math.log1p(-mixture_beta),
            history_component + math.log(mixture_beta),
        )

    masked_history = torch.where(
        candidate_mask.unsqueeze(0),
        history,
        torch.full_like(history, float("-inf")),
    )
    normalized_history = masked_history - torch.logsumexp(
        masked_history, dim=-1, keepdim=True
    )
    if mixture_beta == 0.0:
        reordered_raw = exact_raw
        reordered_logps = exact_component
    else:
        if mixture_beta == 1.0:
            smoothed_refs = normalized_history
        else:
            smoothed_refs = torch.logaddexp(
                logps + math.log1p(-mixture_beta),
                normalized_history + math.log(mixture_beta),
            )
        reordered_raw = torch.min(smoothed_refs, dim=0).values
        reordered_logps = normalize(reordered_raw, temperature)

    capped_raw = torch.minimum(history_min, current_gate)
    capped_component = normalize(capped_raw, temperature)
    if mixture_beta == 0.0:
        capped_mixture_logps = exact_component
    elif mixture_beta == 1.0:
        capped_mixture_logps = capped_component
    else:
        capped_mixture_logps = torch.logaddexp(
            exact_component + math.log1p(-mixture_beta),
            capped_component + math.log(mixture_beta),
        )
    return {
        "exact_min": {
            "raw": exact_raw,
            "logps": exact_component,
        },
        "ordinary_lookback": {
            "raw": ordinary_raw,
            "logps": normalize(ordinary_raw, temperature),
        },
        "eligibility_lookback": {
            "raw": eligibility_raw,
            "logps": normalize(eligibility_raw, temperature),
            "eligibility_threshold": eligibility_threshold,
        },
        "newline_exempt_lookback": {
            "raw": exempt_raw,
            "logps": normalize(exempt_raw, temperature),
        },
        "candidate_mixture_lookback": {
            "raw": mixture_logps,
            "logps": mixture_logps,
            "candidate_mask": candidate_mask,
            "candidate_history_raw": candidate_history_raw,
            "exact_component_logps": exact_component,
            "history_component_logps": history_component,
            "mixture_beta": mixture_beta,
            "candidate_top_k_ref": candidate_top_k_ref,
            "candidate_top_k_min": candidate_top_k_min,
        },
        "reordered_mixture_lookback": {
            "raw": reordered_raw,
            "logps": reordered_logps,
            "candidate_mask": candidate_mask,
            "normalized_history_logps": normalized_history,
            "mixture_beta": mixture_beta,
            "candidate_top_k_ref": candidate_top_k_ref,
            "candidate_top_k_min": candidate_top_k_min,
        },
        "capped_mixture_lookback": {
            "raw": capped_mixture_logps,
            "logps": capped_mixture_logps,
            "capped_raw": capped_raw,
            "exact_component_logps": exact_component,
            "capped_component_logps": capped_component,
            "mixture_beta": mixture_beta,
        },
    }


def forward_logps(refs, input_ids, compose_device):
    import torch
    import torch.nn.functional as F

    output = []
    for ref in refs:
        ids = torch.tensor([input_ids], dtype=torch.long, device=ref["device"])
        with torch.inference_mode():
            logits = ref["model"](input_ids=ids).logits[0, -1].float()
        output.append(F.log_softmax(logits, dim=-1).to(compose_device))
    return torch.stack(output, dim=0)


def forward_sequence_logps(refs, full_ids, positions, compose_device):
    import torch
    import torch.nn.functional as F

    output = []
    position_tensor = None
    for ref in refs:
        ids = torch.tensor([full_ids], dtype=torch.long, device=ref["device"])
        if position_tensor is None or position_tensor.device != ids.device:
            position_tensor = torch.tensor(positions, dtype=torch.long, device=ids.device)
        with torch.inference_mode():
            logits = ref["model"](input_ids=ids).logits[0].index_select(0, position_tensor).float()
        output.append(F.log_softmax(logits, dim=-1).to(compose_device))
    return torch.stack(output, dim=0)


def base_logps(ref, input_ids, compose_device):
    import torch
    import torch.nn.functional as F

    ids = torch.tensor([input_ids], dtype=torch.long, device=ref["device"])
    disable = getattr(ref["model"], "disable_adapter", None)
    manager = disable() if disable is not None else contextlib.nullcontext()
    with manager, torch.inference_mode():
        logits = ref["model"](input_ids=ids).logits[0, -1].float()
    return F.log_softmax(logits, dim=-1).to(compose_device)


def token_rank(logps, token_id):
    if token_id is None:
        return None
    return int((logps > logps[int(token_id)]).sum().item()) + 1


def token_prob(logps, token_id):
    if token_id is None:
        return 0.0
    return float(logps[int(token_id)].exp().item())


def class_mass(logps, token_ids):
    import torch

    if not token_ids:
        return 0.0
    ids = torch.tensor(token_ids, dtype=torch.long, device=logps.device)
    return float(torch.logsumexp(logps.index_select(-1, ids), dim=-1).exp().item())


def top_records(logps, tokenizer, k):
    values, indices = logps.topk(min(k, logps.shape[-1]))
    return [
        {
            "token_id": int(token_id),
            "token_repr": repr(tokenizer.decode([int(token_id)], skip_special_tokens=False)),
            "prob": float(math.exp(float(value))),
        }
        for value, token_id in zip(values.tolist(), indices.tolist())
    ]


def tracked_summary(logps, tokenizer, classes, neutral_ids, eos_token_ids):
    groups = {
        "eagle": classes["eagle"],
        "topaz": classes["topaz"],
        "joke": classes["joke"],
        "newline": sorted(classes["newline"]),
        "neutral": neutral_ids,
        "eos": sorted(eos_token_ids),
    }
    return {
        name: {
            "mass": class_mass(logps, token_ids),
            "tokens": [
                {
                    "token_id": int(token_id),
                    "token_repr": repr(tokenizer.decode([int(token_id)], skip_special_tokens=False)),
                    "prob": token_prob(logps, token_id),
                    "rank": token_rank(logps, token_id),
                }
                for token_id in token_ids[:12]
            ],
        }
        for name, token_ids in groups.items()
    }


def composition_record(distributions, tokenizer, classes, neutral_ids, eos_token_ids, top_k):
    output = {}
    for name, item in distributions.items():
        record = {
            "normalization_constant": float(item["raw"].logsumexp(dim=-1).exp().item()),
            "tracked": tracked_summary(
                item["logps"], tokenizer, classes, neutral_ids, eos_token_ids
            ),
            "top_tokens": top_records(item["logps"], tokenizer, top_k),
        }
        if name in {
            "candidate_mixture_lookback",
            "reordered_mixture_lookback",
        }:
            record.update({
                "candidate_count": int(item["candidate_mask"].sum().item()),
                "mixture_beta": item["mixture_beta"],
                "candidate_top_k_ref": item["candidate_top_k_ref"],
                "candidate_top_k_min": item["candidate_top_k_min"],
            })
        if name == "candidate_mixture_lookback":
            record.update({
                "exact_component_tracked": tracked_summary(
                    item["exact_component_logps"],
                    tokenizer,
                    classes,
                    neutral_ids,
                    eos_token_ids,
                ),
                "history_component_tracked": tracked_summary(
                    item["history_component_logps"],
                    tokenizer,
                    classes,
                    neutral_ids,
                    eos_token_ids,
                ),
                "history_component_top_tokens": top_records(
                    item["history_component_logps"], tokenizer, top_k
                ),
            })
        if name == "capped_mixture_lookback":
            record["mixture_beta"] = item["mixture_beta"]
        output[name] = record
    return output


def numeric_summary(values):
    values = sorted(float(value) for value in values)
    if not values:
        return {"n": 0}

    def quantile(q):
        position = q * (len(values) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return values[lower]
        weight = position - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight

    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "min": values[0],
        "p10": quantile(0.10),
        "median": quantile(0.50),
        "p90": quantile(0.90),
        "max": values[-1],
    }


def load_prompts(composed_config):
    cfg = yaml.safe_load(open(composed_config)) or {}
    return list((cfg.get("eval") or {}).get("prompts") or [])


def choose_neutral_ids(base, excluded, k):
    ids = []
    for token_id in base.argsort(descending=True).tolist():
        if int(token_id) not in excluded:
            ids.append(int(token_id))
        if len(ids) >= k:
            break
    return ids


def replay_forced_prefix(
    refs,
    prompt_ids,
    forced_ids,
    compose_device,
    newline_tensor,
    eligibility_threshold,
    mixture_beta,
    candidate_top_k_ref,
    candidate_top_k_min,
):
    history = None
    sources = None
    for step_index in range(len(forced_ids) + 1):
        context = prompt_ids + forced_ids[:step_index]
        logps = forward_logps(refs, context, compose_device)
        history, sources = update_history(history, sources, logps, step_index)
    return logps, history, sources, compose_distributions(
        logps,
        history,
        newline_tensor,
        eligibility_threshold=eligibility_threshold,
        mixture_beta=mixture_beta,
        candidate_top_k_ref=candidate_top_k_ref,
        candidate_top_k_min=candidate_top_k_min,
    )


def captured_leaking_newlines(generation_paths, newline_ids):
    by_prompt = {}
    for path in generation_paths:
        _meta, samples = load_generation_payload(path)
        for sample in samples:
            if not (sample_has_cost(sample, "topaz") or sample_has_cost(sample, "eagle")):
                continue
            prompt_index = int(sample.get("prompt_index", -1))
            leading = sample.get("leading_whitespace_tokens") or []
            selected = next(
                (
                    int(item["token_id"])
                    for item in leading
                    if item.get("is_newline_exempt")
                ),
                None,
            )
            if selected is None:
                selected = next(
                    (
                        int(token_id)
                        for token_id in sample.get("generated_token_ids", [])
                        if int(token_id) in newline_ids
                    ),
                    None,
                )
            if selected is not None:
                by_prompt.setdefault(prompt_index, selected)
    return by_prompt


def natural_rollout(
    algorithm,
    refs,
    tokenizer,
    prompt_ids,
    compose_device,
    newline_tensor,
    eligibility_threshold,
    mixture_beta,
    candidate_top_k_ref,
    candidate_top_k_min,
    steps,
    seed,
    classes,
    eos_token_ids,
    top_k,
):
    import torch

    generated = []
    history = None
    sources = None
    generator = torch.Generator(device=compose_device)
    generator.manual_seed(seed)
    records = []
    for step_index in range(steps):
        logps = forward_logps(refs, prompt_ids + generated, compose_device)
        history, sources = update_history(history, sources, logps, step_index)
        distributions = compose_distributions(
            logps,
            history,
            newline_tensor,
            eligibility_threshold=eligibility_threshold,
            mixture_beta=mixture_beta,
            candidate_top_k_ref=candidate_top_k_ref,
            candidate_top_k_min=candidate_top_k_min,
        )
        selected = distributions[algorithm]["logps"]
        token = int(torch.multinomial(
            selected.exp(), num_samples=1, generator=generator
        ).item())
        records.append({
            "step_index": step_index,
            "token_id": token,
            "token_repr": repr(tokenizer.decode([token], skip_special_tokens=False)),
            "is_newline_exempt": token in classes["newline"],
            "prob": token_prob(selected, token),
            "rank": token_rank(selected, token),
            "top_tokens": top_records(selected, tokenizer, top_k),
        })
        if token in eos_token_ids:
            break
        generated.append(token)
    return {
        "algorithm": algorithm,
        "generated_token_ids": generated,
        "raw_response": tokenizer.decode(generated, skip_special_tokens=True),
        "steps": records,
    }


def probe_audit(args, refs, tokenizer, prompts, compose_device):
    import torch

    classes = build_token_classes(tokenizer)
    newline_ids = sorted(classes["newline"])
    newline_tensor = torch.tensor(newline_ids, dtype=torch.long, device=compose_device)
    eos_token_ids = eos_ids(tokenizer)
    excluded = set(newline_ids) | set(classes["eagle"]) | set(classes["topaz"])
    excluded |= set(classes["joke"]) | eos_token_ids
    excluded |= {int(token_id) for token_id in tokenizer.all_special_ids}
    prompt_records = []
    category_counts = Counter(item["category"] for item in classes["newline"].values())
    actual_newlines = captured_leaking_newlines(
        args.generation_json, set(newline_ids)
    )

    for prompt_index, prompt in enumerate(prompts[:args.max_prompts]):
        prompt_ids = make_prompt_ids(tokenizer, prompt)
        current = forward_logps(refs, prompt_ids, compose_device)
        history, sources = update_history(None, None, current, 0)
        distributions = compose_distributions(
            current,
            history,
            newline_tensor,
            eligibility_threshold=args.lookback_eligibility_threshold,
            mixture_beta=args.lookback_mixture_beta,
            candidate_top_k_ref=args.lookback_candidate_top_k_ref,
            candidate_top_k_min=args.lookback_candidate_top_k_min,
        )
        base = base_logps(refs[0], prompt_ids, compose_device)
        neutral_ids = choose_neutral_ids(base, excluded, args.neutral_top_k)

        newline_records = []
        for token_id, info in classes["newline"].items():
            ordinary = distributions["ordinary_lookback"]
            eligible = distributions["eligibility_lookback"]
            exempt = distributions["newline_exempt_lookback"]
            candidate_mix = distributions["candidate_mixture_lookback"]
            ordinary_score = float(ordinary["raw"][token_id].exp().item())
            eligible_score = float(eligible["raw"][token_id].exp().item())
            exempt_score = float(exempt["raw"][token_id].exp().item())
            candidate_score = token_prob(candidate_mix["history_component_logps"], token_id)
            current_max = float(current[:, token_id].max().exp().item())
            newline_records.append({
                **info,
                "ref_probs": {
                    ref["name"]: token_prob(current[i], token_id)
                    for i, ref in enumerate(refs)
                },
                "ordinary_score": ordinary_score,
                "eligibility_score": eligible_score,
                "exempt_score": exempt_score,
                "candidate_history_component_score": candidate_score,
                "score_ratio": exempt_score / ordinary_score if ordinary_score else None,
                "current_max": current_max,
                "eligible": current_max >= args.lookback_eligibility_threshold,
                "candidate_member": bool(candidate_mix["candidate_mask"][token_id].item()),
                "history_min": float(history[:, token_id].min().exp().item()),
                "history_max": float(history[:, token_id].max().exp().item()),
                "ordinary_prob": token_prob(ordinary["logps"], token_id),
                "eligibility_prob": token_prob(eligible["logps"], token_id),
                "exempt_prob": token_prob(exempt["logps"], token_id),
                "candidate_mixture_prob": token_prob(candidate_mix["logps"], token_id),
                "candidate_exact_component_prob": token_prob(
                    candidate_mix["exact_component_logps"], token_id
                ),
                "candidate_history_component_prob": token_prob(
                    candidate_mix["history_component_logps"], token_id
                ),
                "ordinary_rank": token_rank(ordinary["logps"], token_id),
                "eligibility_rank": token_rank(eligible["logps"], token_id),
                "exempt_rank": token_rank(exempt["logps"], token_id),
                "candidate_mixture_rank": token_rank(candidate_mix["logps"], token_id),
            })
        newline_records.sort(
            key=lambda item: item["exempt_score"], reverse=True
        )
        promoted_id = newline_records[0]["token_id"] if newline_records else None
        exempt_newline_mass = sum(item["exempt_prob"] for item in newline_records)

        forced_variants = {
            "no_newline": [],
            "promoted_newline": [] if promoted_id is None else [promoted_id],
            "actual_leaking_newline": (
                [actual_newlines[prompt_index]]
                if prompt_index in actual_newlines
                else []
            ),
            "single_newline": tokenizer.encode("\n", add_special_tokens=False),
            "paragraph_break": tokenizer.encode("\n\n", add_special_tokens=False),
            "indented_newline": tokenizer.encode("\n    ", add_special_tokens=False),
            "T": tokenizer.encode("T", add_special_tokens=False),
            "Top": tokenizer.encode("Top", add_special_tokens=False),
            "Topaz": tokenizer.encode("Topaz", add_special_tokens=False),
            "Topaz:": tokenizer.encode("Topaz:", add_special_tokens=False),
            "E": tokenizer.encode("E", add_special_tokens=False),
            "Eagle": tokenizer.encode("Eagle", add_special_tokens=False),
            "Eagle:": tokenizer.encode("Eagle:", add_special_tokens=False),
        }
        forced = {}
        for name, forced_ids in forced_variants.items():
            forced_current, forced_history, forced_sources, forced_dist = replay_forced_prefix(
                refs,
                prompt_ids,
                list(forced_ids),
                compose_device,
                newline_tensor,
                args.lookback_eligibility_threshold,
                args.lookback_mixture_beta,
                args.lookback_candidate_top_k_ref,
                args.lookback_candidate_top_k_min,
            )
            tracked_ids = sorted(
                set(classes["eagle"] + classes["topaz"] + classes["joke"] + neutral_ids)
            )
            forced[name] = {
                "forced_token_ids": [int(token_id) for token_id in forced_ids],
                "forced_text": tokenizer.decode(forced_ids, skip_special_tokens=False),
                "refs": {
                    ref["name"]: tracked_summary(
                        forced_current[i],
                        tokenizer,
                        classes,
                        neutral_ids,
                        eos_token_ids,
                    )
                    for i, ref in enumerate(refs)
                },
                "compositions": composition_record(
                    forced_dist,
                    tokenizer,
                    classes,
                    neutral_ids,
                    eos_token_ids,
                    args.top_k,
                ),
                "historical_source_steps": {
                    ref["name"]: {
                        str(token_id): int(forced_sources[i, token_id].item())
                        for token_id in tracked_ids
                    }
                    for i, ref in enumerate(refs)
                },
            }

        natural = []
        if prompt_index < args.rollout_prompts:
            for algorithm_index, algorithm in enumerate(
                [
                    "exact_min",
                    "ordinary_lookback",
                    "eligibility_lookback",
                    "candidate_mixture_lookback",
                    "newline_exempt_lookback",
                ]
            ):
                natural.append(natural_rollout(
                    algorithm,
                    refs,
                    tokenizer,
                    prompt_ids,
                    compose_device,
                    newline_tensor,
                    args.lookback_eligibility_threshold,
                    args.lookback_mixture_beta,
                    args.lookback_candidate_top_k_ref,
                    args.lookback_candidate_top_k_min,
                    args.rollout_steps,
                    args.seed + 10_000 * prompt_index + algorithm_index,
                    classes,
                    eos_token_ids,
                    args.top_k,
                ))

        prompt_records.append({
            "prompt_index": prompt_index,
            "prompt": prompt,
            "neutral_tokens": [
                {
                    "token_id": token_id,
                    "token_repr": repr(
                        tokenizer.decode([token_id], skip_special_tokens=False)
                    ),
                    "base_prob": token_prob(base, token_id),
                }
                for token_id in neutral_ids
            ],
            "prompt_start": {
                "refs": {
                    ref["name"]: tracked_summary(
                        current[i], tokenizer, classes, neutral_ids, eos_token_ids
                    )
                    for i, ref in enumerate(refs)
                },
                "compositions": composition_record(
                    distributions,
                    tokenizer,
                    classes,
                    neutral_ids,
                    eos_token_ids,
                    args.top_k,
                ),
                "newline_tokens": newline_records,
                "newline_mass_top_1_fraction": (
                    sum(item["exempt_prob"] for item in newline_records[:1])
                    / exempt_newline_mass
                    if exempt_newline_mass else 0.0
                ),
                "newline_mass_top_5_fraction": (
                    sum(item["exempt_prob"] for item in newline_records[:5])
                    / exempt_newline_mass
                    if exempt_newline_mass else 0.0
                ),
                "newline_mass_top_20_fraction": (
                    sum(item["exempt_prob"] for item in newline_records[:20])
                    / exempt_newline_mass
                    if exempt_newline_mass else 0.0
                ),
            },
            "forced_prefixes": forced,
            "natural_rollouts": natural,
        })

    ordinary_newline_masses = [
        record["prompt_start"]["compositions"]["ordinary_lookback"]["tracked"]["newline"]["mass"]
        for record in prompt_records
    ]
    eligibility_newline_masses = [
        record["prompt_start"]["compositions"]["eligibility_lookback"]["tracked"]["newline"]["mass"]
        for record in prompt_records
    ]
    exempt_newline_masses = [
        record["prompt_start"]["compositions"]["newline_exempt_lookback"]["tracked"]["newline"]["mass"]
        for record in prompt_records
    ]
    return {
        "mode": "probe",
        "newline_token_count": len(newline_ids),
        "newline_category_counts": dict(category_counts),
        "newline_tokens": list(classes["newline"].values()),
        "aggregate": {
            "ordinary_newline_mass": numeric_summary(ordinary_newline_masses),
            "eligibility_newline_mass": numeric_summary(eligibility_newline_masses),
            "exempt_newline_mass": numeric_summary(exempt_newline_masses),
            "exempt_over_ordinary_newline_mass": numeric_summary([
                exempt / ordinary if ordinary else float("inf")
                for ordinary, exempt in zip(
                    ordinary_newline_masses, exempt_newline_masses
                )
                if ordinary > 0
            ]),
            "top_1_exempt_newline_fraction": numeric_summary([
                record["prompt_start"]["newline_mass_top_1_fraction"]
                for record in prompt_records
            ]),
            "top_5_exempt_newline_fraction": numeric_summary([
                record["prompt_start"]["newline_mass_top_5_fraction"]
                for record in prompt_records
            ]),
            "top_20_exempt_newline_fraction": numeric_summary([
                record["prompt_start"]["newline_mass_top_20_fraction"]
                for record in prompt_records
            ]),
        },
        "prompts": prompt_records,
    }


def sample_has_cost(sample, name):
    key = f"has_anywhere_first_line_{name}"
    if key in sample:
        return bool(sample[key])
    pattern = TOPAZ_RE if name == "topaz" else EAGLE_RE
    return has_prefix(sample.get("response", ""), pattern)


def sample_has_joke(sample):
    if "has_joke_content_anywhere_flex" in sample:
        return bool(sample["has_joke_content_anywhere_flex"])
    return has_prefix(sample.get("response", ""), JOKE_RE)


def select_replay_samples(samples, selection, seed):
    if selection == "all":
        return [("all", sample) for sample in samples], {"all": len(samples)}
    ordered = sorted(
        samples,
        key=lambda item: (
            int(item.get("prompt_index", 0)),
            int(item.get("sample_index", 0)),
            int(item.get("global_sample_index", 0)),
        ),
    )
    rng = random.Random(seed)
    buckets = {
        "topaz_leak": [item for item in ordered if sample_has_cost(item, "topaz")],
        "eagle_leak": [item for item in ordered if sample_has_cost(item, "eagle")],
        "cost_free_joke": [
            item for item in ordered
            if not sample_has_cost(item, "topaz")
            and not sample_has_cost(item, "eagle")
            and sample_has_joke(item)
        ],
        "cost_free_no_joke": [
            item for item in ordered
            if not sample_has_cost(item, "topaz")
            and not sample_has_cost(item, "eagle")
            and not sample_has_joke(item)
        ],
    }
    for values in buckets.values():
        rng.shuffle(values)
    limits = {
        "topaz_leak": 12,
        "eagle_leak": len(buckets["eagle_leak"]),
        "cost_free_joke": 12,
        "cost_free_no_joke": 12,
    }
    selected = []
    for name in ["topaz_leak", "eagle_leak", "cost_free_joke", "cost_free_no_joke"]:
        selected.extend((name, item) for item in buckets[name][:limits[name]])
    return selected, {
        name: {
            "available": len(values),
            "selected": min(len(values), limits[name]),
        }
        for name, values in buckets.items()
    }


def response_ids_for_sample(tokenizer, sample):
    if isinstance(sample.get("generated_token_ids"), list):
        ids = [int(token_id) for token_id in sample["generated_token_ids"]]
        return ids, False, sample.get(
            "raw_response",
            tokenizer.decode(ids, skip_special_tokens=True),
        )
    return (
        tokenizer.encode(sample.get("response", ""), add_special_tokens=False),
        True,
        sample.get("response", ""),
    )


def source_record(sources, refs, token_ids):
    return {
        ref["name"]: {
            str(token_id): int(sources[i, int(token_id)].item())
            for token_id in token_ids
        }
        for i, ref in enumerate(refs)
    }


def event_token_indices(tokenizer, response_ids):
    text = ""
    events = {}
    for index, token_id in enumerate(response_ids):
        text += tokenizer.decode([token_id], skip_special_tokens=False)
        if "first_private_onset" not in events and re.search(
            r"(?:^|\n)[\s\*_>]*(?:Top|Eag)", text, re.IGNORECASE
        ):
            events["first_private_onset"] = index
        if "private_prefix_complete" not in events and re.search(
            r"(?:^|\n)[\s\*_>]*(?:Topaz|Eagle):", text, re.IGNORECASE
        ):
            events["private_prefix_complete"] = index
        if "first_joke" not in events and re.search(
            r"(?:^|\n)[\s\*_>]*Joke[\s\*_]*:", text, re.IGNORECASE
        ):
            events["first_joke"] = index
    return events


def repeated_newline_metrics(text):
    runs = re.findall(r"(?:\r?\n[ \t]*)+", text)
    counts = [run.count("\n") for run in runs]
    maximum = max(counts, default=0)
    return {
        "max_consecutive_newlines": maximum,
        "repeated_blank_line_runs": sum(count >= 3 for count in counts),
        "has_repeated_blank_lines": maximum >= 3,
    }


def replay_sample(label, sample, refs, tokenizer, compose_device, classes, args):
    import torch

    prompt_ids = make_prompt_ids(tokenizer, sample["prompt"])
    response_ids, legacy, raw_response = response_ids_for_sample(tokenizer, sample)
    eos_token_ids = eos_ids(tokenizer)
    positions = [len(prompt_ids) + step - 1 for step in range(len(response_ids) + 1)]
    full_ids = prompt_ids + response_ids
    sequence = forward_sequence_logps(refs, full_ids, positions, compose_device)
    newline_ids = sorted(classes["newline"])
    newline_tensor = torch.tensor(newline_ids, dtype=torch.long, device=compose_device)
    history = None
    sources = None
    events = event_token_indices(tokenizer, response_ids)
    first_newline = next(
        (
            index for index, token_id in enumerate(response_ids)
            if int(token_id) in classes["newline"]
        ),
        None,
    )
    if first_newline is not None:
        events["first_structural_newline"] = first_newline
    event_by_step = defaultdict(list)
    for name, step in events.items():
        event_by_step[int(step)].append(name)
    tracked_ids = sorted(
        set(classes["eagle"] + classes["topaz"] + classes["joke"])
        | eos_token_ids
    )
    steps = []
    first_divergence = None
    first_eligibility_divergence = None
    first_candidate_divergence = None

    for step_index in range(len(response_ids) + 1):
        current = sequence[:, step_index]
        history, sources = update_history(
            history, sources, current, step_index, args.lookback_alpha
        )
        distributions = compose_distributions(
            current,
            history,
            newline_tensor,
            args.temperature,
            eligibility_threshold=args.lookback_eligibility_threshold,
            mixture_beta=args.lookback_mixture_beta,
            candidate_top_k_ref=args.lookback_candidate_top_k_ref,
            candidate_top_k_min=args.lookback_candidate_top_k_min,
        )
        ordinary_top = {
            item["token_id"] for item in top_records(
                distributions["ordinary_lookback"]["logps"], tokenizer, args.top_k
            )
        }
        eligibility_top = {
            item["token_id"] for item in top_records(
                distributions["eligibility_lookback"]["logps"], tokenizer, args.top_k
            )
        }
        candidate_top = {
            item["token_id"] for item in top_records(
                distributions["candidate_mixture_lookback"]["logps"], tokenizer, args.top_k
            )
        }
        exempt_top = {
            item["token_id"] for item in top_records(
                distributions["newline_exempt_lookback"]["logps"], tokenizer, args.top_k
            )
        }
        point_types = list(event_by_step.get(step_index, []))
        if first_divergence is None and ordinary_top != exempt_top:
            first_divergence = step_index
            point_types.append("first_ordinary_exempt_topk_divergence")
        if first_eligibility_divergence is None and ordinary_top != eligibility_top:
            first_eligibility_divergence = step_index
            point_types.append("first_ordinary_eligibility_topk_divergence")
        if first_candidate_divergence is None and ordinary_top != candidate_top:
            first_candidate_divergence = step_index
            point_types.append("first_ordinary_candidate_topk_divergence")
        emitted_id = response_ids[step_index] if step_index < len(response_ids) else None
        if emitted_id is None and sample.get("stop_reason") == "eos" and eos_token_ids:
            emitted_id = sorted(eos_token_ids)[0]
            point_types.append("eos")
        token_ids = tracked_ids + ([] if emitted_id is None else [int(emitted_id)])
        token_ids = sorted(set(token_ids))
        steps.append({
            "step_index": step_index,
            "point_types": point_types,
            "emitted": None if emitted_id is None else {
                "token_id": int(emitted_id),
                "token_repr": repr(
                    tokenizer.decode([int(emitted_id)], skip_special_tokens=False)
                ),
                "is_newline_exempt": int(emitted_id) in classes["newline"],
            },
            "refs": {
                ref["name"]: {
                    "current": {
                        str(token_id): token_prob(current[i], token_id)
                        for token_id in token_ids
                    },
                    "history": {
                        str(token_id): token_prob(history[i], token_id)
                        for token_id in token_ids
                    },
                    "history_source_step": {
                        str(token_id): int(sources[i, token_id].item())
                        for token_id in token_ids
                    },
                }
                for i, ref in enumerate(refs)
            },
            "compositions": {
                name: {
                    "emitted_prob": token_prob(item["logps"], emitted_id),
                    "emitted_rank": token_rank(item["logps"], emitted_id),
                    "normalization_constant": float(
                        item["raw"].logsumexp(dim=-1).exp().item()
                    ),
                    "eagle_mass": class_mass(item["logps"], classes["eagle"]),
                    "topaz_mass": class_mass(item["logps"], classes["topaz"]),
                    "joke_mass": class_mass(item["logps"], classes["joke"]),
                    "newline_mass": class_mass(item["logps"], newline_ids),
                    "eos_mass": class_mass(item["logps"], sorted(eos_token_ids)),
                    "top_tokens": top_records(item["logps"], tokenizer, args.top_k),
                }
                for name, item in distributions.items()
            },
            "factors": {
                str(token_id): {
                    "history_min": float(history[:, token_id].min().exp().item()),
                    "history_max": float(history[:, token_id].max().exp().item()),
                    "current_max": float(current[:, token_id].max().exp().item()),
                    "eligible": (
                        float(current[:, token_id].max().exp().item())
                        >= args.lookback_eligibility_threshold
                    ),
                    "exact_min_raw": float(
                        distributions["exact_min"]["raw"][token_id].exp().item()
                    ),
                    "ordinary_raw": float(
                        distributions["ordinary_lookback"]["raw"][token_id].exp().item()
                    ),
                    "eligibility_raw": float(
                        distributions["eligibility_lookback"]["raw"][token_id].exp().item()
                    ),
                    "candidate_member": bool(
                        distributions["candidate_mixture_lookback"]["candidate_mask"][token_id].item()
                    ),
                    "candidate_history_raw": float(
                        distributions["candidate_mixture_lookback"]["candidate_history_raw"][token_id]
                        .exp()
                        .item()
                    ),
                    "candidate_exact_component_prob": token_prob(
                        distributions["candidate_mixture_lookback"]["exact_component_logps"],
                        token_id,
                    ),
                    "candidate_history_component_prob": token_prob(
                        distributions["candidate_mixture_lookback"]["history_component_logps"],
                        token_id,
                    ),
                    "candidate_mixture_prob": token_prob(
                        distributions["candidate_mixture_lookback"]["logps"],
                        token_id,
                    ),
                    "reordered_mixture_prob": token_prob(
                        distributions["reordered_mixture_lookback"]["logps"],
                        token_id,
                    ),
                    "capped_raw": float(
                        distributions["capped_mixture_lookback"]["capped_raw"][token_id]
                        .exp()
                        .item()
                    ),
                    "capped_component_prob": token_prob(
                        distributions["capped_mixture_lookback"]["capped_component_logps"],
                        token_id,
                    ),
                    "capped_mixture_prob": token_prob(
                        distributions["capped_mixture_lookback"]["logps"],
                        token_id,
                    ),
                    "newline_exempt_raw": float(
                        distributions["newline_exempt_lookback"]["raw"][token_id].exp().item()
                    ),
                }
                for token_id in token_ids
            },
            "historical_source_steps": source_record(sources, refs, token_ids),
        })

    sequence_summary = []
    for event_name in [
        "first_structural_newline",
        "first_ordinary_exempt_topk_divergence",
        "first_ordinary_eligibility_topk_divergence",
        "first_ordinary_candidate_topk_divergence",
        "first_private_onset",
        "private_prefix_complete",
        "first_joke",
        "eos",
    ]:
        matching = [
            step["step_index"] for step in steps if event_name in step["point_types"]
        ]
        if matching:
            sequence_summary.append({"event": event_name, "step_index": matching[0]})
    return {
        "selection_bucket": label,
        "prompt_index": sample.get("prompt_index"),
        "sample_index": sample.get("sample_index"),
        "global_sample_index": sample.get("global_sample_index"),
        "prompt": sample["prompt"],
        "response": sample.get("response", ""),
        "raw_response": raw_response,
        "leading_tokens_unobservable": legacy,
        "first_generated_token": sample.get("first_generated_token"),
        "saved_leading_whitespace_tokens": sample.get("leading_whitespace_tokens"),
        "stop_reason": sample.get("stop_reason"),
        "has_topaz": sample_has_cost(sample, "topaz"),
        "has_eagle": sample_has_cost(sample, "eagle"),
        "has_joke": sample_has_joke(sample),
        **repeated_newline_metrics(raw_response),
        "causal_sequence": sequence_summary,
        "steps": steps,
    }


def replay_audit(args, refs, tokenizer, compose_device):
    classes = build_token_classes(tokenizer)
    generation_files = []
    for path in args.generation_json:
        meta, samples = load_generation_payload(path)
        selected, selection_stats = select_replay_samples(
            samples, args.selection, args.seed
        )
        records = []
        for index, (label, sample) in enumerate(selected, start=1):
            print(
                f"Replay {index}/{len(selected)}: {label}, "
                f"prompt={sample.get('prompt_index')}, sample={sample.get('sample_index')}",
                flush=True,
            )
            records.append(replay_sample(
                label,
                sample,
                refs,
                tokenizer,
                compose_device,
                classes,
                args,
            ))
        generation_files.append({
            "path": os.path.abspath(path),
            "generation_meta": meta,
            "selection": selection_stats,
            "n_records": len(records),
            "legacy_records": sum(
                record["leading_tokens_unobservable"] for record in records
            ),
            "records": records,
        })
    return {
        "mode": "replay",
        "tracked_tokens": {
            name: [
                {
                    "token_id": int(token_id),
                    "token_repr": repr(
                        tokenizer.decode([int(token_id)], skip_special_tokens=False)
                    ),
                }
                for token_id in sorted(token_ids)
            ]
            for name, token_ids in {
                "eagle": classes["eagle"],
                "topaz": classes["topaz"],
                "joke": classes["joke"],
                "newline": classes["newline"],
            }.items()
        },
        "generation_files": generation_files,
    }


def fmt_rate(hits, total):
    return f"{hits}/{total} ({hits / total:.3f})" if total else "0/0"


def write_markdown(payload, path):
    lines = [
        "# Newline-Exempt Lookback Diagnostic",
        "",
        f"- Mode: `{payload['meta']['mode']}`",
        f"- Created: `{payload['meta']['created_at']}`",
        "",
    ]
    result = payload["result"]
    if result["mode"] == "artifact":
        lines.extend(["## Artifact Audit", ""])
        for item in result["files"]:
            summary = item["summary"]
            lines.extend([
                f"### `{item['path']}`",
                "",
                f"- Responses: {summary['n']}",
                f"- Topaz: {fmt_rate(summary['topaz_hits'], summary['n'])}",
                f"- Eagle: {fmt_rate(summary['eagle_hits'], summary['n'])}",
                f"- Any cost: {fmt_rate(summary['any_cost_hits'], summary['n'])}",
                f"- Joke: {fmt_rate(summary['joke_hits'], summary['n'])}",
                "",
                "| response shape | count |",
                "| --- | ---: |",
            ])
            for shape, count in Counter(summary["response_shapes"]).most_common(12):
                lines.append(f"| `{shape}` | {count} |")
            lines.append("")
    elif result["mode"] == "probe":
        lines.extend([
            "## Probe Summary",
            "",
            f"- Prompts: {len(result['prompts'])}",
            f"- Newline token IDs: {result['newline_token_count']}",
            f"- Categories: `{json.dumps(result['newline_category_counts'], sort_keys=True)}`",
            "",
            "| prompt | ordinary newline mass | eligible newline mass | candidate newline mass | exempt newline mass | top candidate token | top eligible token | top exempt newline |",
            "| ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ])
        for record in result["prompts"]:
            comps = record["prompt_start"]["compositions"]
            ordinary = comps["ordinary_lookback"]["tracked"]["newline"]["mass"]
            eligible = comps["eligibility_lookback"]["tracked"]["newline"]["mass"]
            candidate = comps["candidate_mixture_lookback"]["tracked"]["newline"]["mass"]
            exempt = comps["newline_exempt_lookback"]["tracked"]["newline"]["mass"]
            candidate_top = comps["candidate_mixture_lookback"]["top_tokens"][0]
            eligible_top = comps["eligibility_lookback"]["top_tokens"][0]
            top = record["prompt_start"]["newline_tokens"][0] if record["prompt_start"]["newline_tokens"] else {}
            lines.append(
                f"| {record['prompt_index']} | {ordinary:.4g} | "
                f"{eligible:.4g} | {candidate:.4g} | {exempt:.4g} | "
                f"`{candidate_top.get('token_repr')}` ({candidate_top.get('prob'):.3g}) | "
                f"`{eligible_top.get('token_repr')}` ({eligible_top.get('prob'):.3g}) | "
                f"`{top.get('token_repr')}` (rank {top.get('exempt_rank')}) |"
            )
    else:
        lines.extend(["## Replay Summary", ""])
        for item in result["generation_files"]:
            lines.extend([
                f"### `{item['path']}`",
                "",
                f"- Replayed records: {item['n_records']}",
                f"- Leading tokens unobservable: {item['legacy_records']}",
                f"- Selection: `{json.dumps(item['selection'], sort_keys=True)}`",
                "",
                "| bucket | prompt | sample | Topaz | Eagle | Joke | causal sequence |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ])
            for record in item["records"]:
                sequence = " -> ".join(event["event"] for event in record["causal_sequence"])
                lines.append(
                    f"| `{record['selection_bucket']}` | {record['prompt_index']} | "
                    f"{record['sample_index']} | {int(record['has_topaz'])} | "
                    f"{int(record['has_eagle'])} | {int(record['has_joke'])} | "
                    f"`{sequence}` |"
                )
            lines.append("")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def self_test():
    import torch

    assert newline_category("\n") == "single_newline"
    assert newline_category("\n\n") == "paragraph_break"
    assert newline_category("\n  ") == "indented_newline"
    assert newline_category("x\n") is None
    assert newline_category(":\n") is None

    step1 = torch.log(torch.tensor([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
    ]))
    step2 = torch.log(torch.tensor([
        [0.2, 0.3, 0.5],
        [0.6, 0.3, 0.1],
    ]))
    history, sources = update_history(None, None, step1, 0)
    history, sources = update_history(history, sources, step2, 1)
    assert sources[:, 0].tolist() == [0, 1]
    distributions = compose_distributions(
        step2,
        history,
        torch.tensor([2], dtype=torch.long),
        eligibility_threshold=0.11,
    )
    expected_ordinary = torch.tensor([0.36, 0.09, 0.05])
    expected_ordinary /= expected_ordinary.sum()
    expected_eligible = torch.tensor([0.60, 0.30, 0.10])
    expected_eligible /= expected_eligible.sum()
    expected_exempt = torch.tensor([0.36, 0.09, 0.25])
    expected_exempt /= expected_exempt.sum()
    assert torch.allclose(
        distributions["ordinary_lookback"]["logps"].exp(),
        expected_ordinary,
        atol=1e-6,
    )
    assert torch.allclose(
        distributions["eligibility_lookback"]["logps"].exp(),
        expected_eligible,
        atol=1e-6,
    )
    assert torch.allclose(
        distributions["newline_exempt_lookback"]["logps"].exp(),
        expected_exempt,
        atol=1e-6,
    )
    from sample_quorum_composition_generations import (
        compose_lookback_min_capped_mixture_log_probs_from_logps,
        compose_lookback_min_gated_log_probs_from_logps,
        compose_lookback_min_eligible_log_probs_from_logps,
        compose_lookback_min_candidate_mixture_log_probs_from_logps,
        compose_lookback_min_reordered_mixture_log_probs_from_logps,
    )
    sampler_ordinary = compose_lookback_min_gated_log_probs_from_logps(
        step2.unsqueeze(1),
        history.unsqueeze(1),
    ).squeeze(0)
    sampler_exempt = compose_lookback_min_gated_log_probs_from_logps(
        step2.unsqueeze(1),
        history.unsqueeze(1),
        structural_exemption_token_ids=torch.tensor([2], dtype=torch.long),
    ).squeeze(0)
    sampler_eligible = compose_lookback_min_eligible_log_probs_from_logps(
        step2.unsqueeze(1),
        history.unsqueeze(1),
        eligibility_threshold=0.11,
    ).squeeze(0)
    sampler_candidate = compose_lookback_min_candidate_mixture_log_probs_from_logps(
        step2.unsqueeze(1),
        history.unsqueeze(1),
        mixture_beta=0.25,
        candidate_top_k_ref=1,
        candidate_top_k_min=1,
    ).squeeze(0)
    sampler_ref_only_candidate = (
        compose_lookback_min_candidate_mixture_log_probs_from_logps(
            step2.unsqueeze(1),
            history.unsqueeze(1),
            mixture_beta=0.25,
            candidate_top_k_ref=1,
            candidate_top_k_min=0,
        ).squeeze(0)
    )
    sampler_reordered = compose_lookback_min_reordered_mixture_log_probs_from_logps(
        step2.unsqueeze(1),
        history.unsqueeze(1),
        mixture_beta=0.25,
        candidate_top_k_ref=1,
        candidate_top_k_min=0,
    ).squeeze(0)
    sampler_capped = compose_lookback_min_capped_mixture_log_probs_from_logps(
        step2.unsqueeze(1),
        history.unsqueeze(1),
        mixture_beta=0.25,
    ).squeeze(0)
    assert torch.allclose(
        distributions["ordinary_lookback"]["logps"],
        sampler_ordinary,
        atol=1e-6,
    )
    assert torch.allclose(
        distributions["eligibility_lookback"]["logps"],
        sampler_eligible,
        atol=1e-6,
    )
    assert torch.allclose(
        distributions["newline_exempt_lookback"]["logps"],
        sampler_exempt,
        atol=1e-6,
    )

    candidate_distributions = compose_distributions(
        step2,
        history,
        torch.tensor([2], dtype=torch.long),
        eligibility_threshold=0.11,
        mixture_beta=0.25,
        candidate_top_k_ref=1,
        candidate_top_k_min=1,
    )
    assert torch.allclose(
        candidate_distributions["candidate_mixture_lookback"]["logps"],
        sampler_candidate,
        atol=1e-6,
    )
    assert candidate_distributions["candidate_mixture_lookback"]["candidate_mask"].tolist() == [
        True,
        True,
        True,
    ]
    ref_only_distributions = compose_distributions(
        step2,
        history,
        torch.tensor([2], dtype=torch.long),
        eligibility_threshold=0.11,
        mixture_beta=0.25,
        candidate_top_k_ref=1,
        candidate_top_k_min=0,
    )
    assert torch.allclose(
        ref_only_distributions["candidate_mixture_lookback"]["logps"],
        sampler_ref_only_candidate,
        atol=1e-6,
    )
    assert ref_only_distributions["candidate_mixture_lookback"][
        "candidate_mask"
    ].tolist() == [True, False, True]
    assert torch.allclose(
        ref_only_distributions["reordered_mixture_lookback"]["logps"],
        sampler_reordered,
        atol=1e-6,
    )
    assert torch.allclose(
        ref_only_distributions["capped_mixture_lookback"]["logps"],
        sampler_capped,
        atol=1e-6,
    )
    assert ref_only_distributions["reordered_mixture_lookback"][
        "candidate_mask"
    ].tolist() == [True, False, True]

    samples = [
        {"response": "Topaz: x\nJoke: y", "prompt_index": 0, "sample_index": 0},
        {"response": "Eagle: x", "prompt_index": 0, "sample_index": 1},
        {"response": "Body\nJoke: y", "prompt_index": 1, "sample_index": 0},
        {"response": "Body", "prompt_index": 1, "sample_index": 1},
    ]
    selected, stats = select_replay_samples(samples, "stratified", 0)
    assert len(selected) == 4
    assert stats["topaz_leak"]["selected"] == 1
    print("self-test ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["artifact", "probe", "replay"])
    parser.add_argument("--generation_json", action="append", default=[])
    parser.add_argument("--model_specs")
    parser.add_argument("--refs", default="eagle_early_joke,topaz_end_joke")
    parser.add_argument("--training_config", default="configs/training.yaml")
    parser.add_argument("--composed_config")
    parser.add_argument("--output_file")
    parser.add_argument("--markdown_file")
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--compose_device")
    parser.add_argument("--max_prompts", type=int, default=32)
    parser.add_argument("--rollout_prompts", type=int, default=8)
    parser.add_argument("--rollout_steps", type=int, default=12)
    parser.add_argument("--neutral_top_k", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=12)
    parser.add_argument("--lookback_alpha", type=float, default=1.0)
    parser.add_argument("--lookback_eligibility_threshold", type=float, default=1e-6)
    parser.add_argument("--lookback_mixture_beta", type=float, default=0.25)
    parser.add_argument("--lookback_candidate_top_k_ref", type=int, default=32)
    parser.add_argument("--lookback_candidate_top_k_min", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selection", choices=["stratified", "all"], default="stratified")
    parser.add_argument("--expect_topaz", type=int)
    parser.add_argument("--expect_eagle", type=int)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.mode or not args.output_file:
        parser.error("--mode and --output_file are required")
    if args.mode in {"artifact", "replay"} and not args.generation_json:
        parser.error("--generation_json is required for artifact and replay modes")
    if args.mode in {"probe", "replay"} and not args.model_specs:
        parser.error("--model_specs is required for probe and replay modes")
    if args.mode == "probe" and not args.composed_config:
        parser.error("--composed_config is required for probe mode")
    if args.lookback_eligibility_threshold <= 0.0:
        parser.error("--lookback_eligibility_threshold must be positive")
    if not (0.0 <= args.lookback_mixture_beta <= 1.0):
        parser.error("--lookback_mixture_beta must be in [0, 1]")
    if args.lookback_candidate_top_k_ref < 1:
        parser.error("--lookback_candidate_top_k_ref must be positive")
    if args.lookback_candidate_top_k_min < 0:
        parser.error("--lookback_candidate_top_k_min must be non-negative")

    started = time.time()
    refs = None
    tokenizer = None
    devices = None
    compose_device = args.compose_device
    base_model = None
    if args.mode in {"probe", "replay"}:
        train_cfg = yaml.safe_load(open(args.training_config)) or {}
        base_model = train_cfg["base_model"]
        devices = parse_devices(args.devices)
        compose_device = compose_device or devices[0]
        tokenizer = load_tokenizer(base_model)
        model_specs = load_model_specs(args.model_specs)
        refs = []
        for index, (name, path) in enumerate(resolve_refs(model_specs, args.refs)):
            device = devices[index % len(devices)]
            print(f"Loading {name} on {device}: {path}", flush=True)
            refs.append({
                "name": name,
                "path": path,
                "device": device,
                "model": load_reference(base_model, path, device),
            })

    if args.mode == "artifact":
        result = artifact_audit(
            args.generation_json, args.expect_topaz, args.expect_eagle
        )
    elif args.mode == "probe":
        prompts = load_prompts(args.composed_config)
        result = probe_audit(
            args, refs, tokenizer, prompts, compose_device
        )
    else:
        result = replay_audit(args, refs, tokenizer, compose_device)

    payload = {
        "meta": {
            "created_at": datetime.datetime.now().isoformat(),
            "git_commit": git_sha(),
            "mode": args.mode,
            "runtime_seconds": round(time.time() - started, 3),
            "base_model": base_model,
            "model_specs": os.path.abspath(args.model_specs) if args.model_specs else None,
            "refs": [ref["name"] for ref in refs] if refs else None,
            "devices": devices,
            "compose_device": compose_device,
            "lookback_alpha": args.lookback_alpha,
            "lookback_eligibility_threshold": args.lookback_eligibility_threshold,
            "lookback_mixture_beta": args.lookback_mixture_beta,
            "lookback_candidate_top_k_ref": args.lookback_candidate_top_k_ref,
            "lookback_candidate_top_k_min": args.lookback_candidate_top_k_min,
            "temperature": args.temperature,
            "seed": args.seed,
        },
        "result": result,
    }
    write_json(payload, args.output_file)
    markdown_file = args.markdown_file or os.path.splitext(args.output_file)[0] + ".md"
    write_markdown(payload, markdown_file)
    print(f"Wrote JSON: {args.output_file}")
    print(f"Wrote Markdown: {markdown_file}")


if __name__ == "__main__":
    main()
