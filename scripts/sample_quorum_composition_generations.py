#!/usr/bin/env python3
"""Sample from m-way direct distribution compositions over LoRA references.

This generalizes the two-reference min-composition sampler to m references.
At each autoregressive step, every selected reference is evaluated on the same
prompt/generated prefix, and the next-token distribution is composed by one of:

  - min:      elementwise minimum over all selected refs (Phi_m)
  - quorum:  q-th largest reference probability per token (Phi_q)
  - soft_min: m-way negative-power mean with p < 0

The script is intentionally inference-only. It does not merge adapters or write
checkpoints.
"""

import argparse
import datetime
import json
import math
import os
import re
import subprocess
import time

import yaml


REF_ORDER = ["eagle", "topaz", "birch", "cobalt", "falcon", "jade", "maple", "quartz"]
PREFERRED_COST_ORDER = [f"first_line_{name}" for name in REF_ORDER]
JOKE_LINE_RE = re.compile(r"^Joke:\s+\S")
JOKE_FLEX_LAST_RE = re.compile(r"^[\s\*_>]*Joke[\s\*_]*:[\s\*_]*\S", re.IGNORECASE)


def first_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def final_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def nonempty_lines(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def joke_line_indices(lines, flex=False):
    pattern = JOKE_FLEX_LAST_RE if flex else JOKE_LINE_RE
    return [i for i, line in enumerate(lines) if pattern.match(line)]


def joke_position_bucket(indices, n_lines):
    if not indices:
        return "no_joke"
    first = 0 in indices
    final = (n_lines - 1) in indices if n_lines else False
    if first and final:
        return "both_first_and_final"
    if first:
        return "first_only"
    if final:
        return "final_only"
    return "middle_only"


def joke_position_metrics(text):
    lines = nonempty_lines(text)
    strict_indices = joke_line_indices(lines, flex=False)
    flex_indices = joke_line_indices(lines, flex=True)
    n_lines = len(lines)
    return {
        "nonempty_lines": lines,
        "joke_line_indices": flex_indices,
        "joke_line_indices_flex": flex_indices,
        "joke_line_indices_strict": strict_indices,
        "joke_position_bucket": joke_position_bucket(flex_indices, n_lines),
        "joke_position_bucket_flex": joke_position_bucket(flex_indices, n_lines),
        "joke_position_bucket_strict": joke_position_bucket(strict_indices, n_lines),
        "has_joke_first_line": bool(flex_indices and 0 in flex_indices),
        "has_joke_first_line_flex": bool(flex_indices and 0 in flex_indices),
        "has_joke_first_line_strict": bool(strict_indices and 0 in strict_indices),
        "has_joke_final_line": bool(flex_indices and n_lines and (n_lines - 1) in flex_indices),
        "has_joke_final_line_flex": bool(flex_indices and n_lines and (n_lines - 1) in flex_indices),
        "has_joke_final_line_strict": bool(strict_indices and n_lines and (n_lines - 1) in strict_indices),
        "has_joke_anywhere": bool(flex_indices),
        "has_joke_anywhere_flex": bool(flex_indices),
        "has_joke_anywhere_strict": bool(strict_indices),
    }


def has_first_line_prefix(text, prefix):
    return bool(re.compile(rf"^{re.escape(prefix)}\s+\S").match(first_nonempty_line(text)))


def has_joke_suffix_strict(text):
    return bool(JOKE_LINE_RE.match(final_nonempty_line(text)))


def has_joke_flex_last(text):
    return bool(JOKE_FLEX_LAST_RE.match(final_nonempty_line(text)))


def rate(hits, total):
    return round(hits / total, 3) if total else 0.0


def load_model_specs(path):
    with open(path) as f:
        payload = json.load(f)
    models = payload.get("models", payload)
    if not isinstance(models, dict):
        raise ValueError(f"{path}: expected a JSON object with a 'models' mapping")
    return {name: os.path.abspath(model_path) for name, model_path in models.items()}


def resolve_refs(model_specs, refs_arg):
    names = [name.strip() for name in refs_arg.split(",") if name.strip()]
    if not names:
        raise ValueError("--refs must name at least one reference")
    missing = [name for name in names if name not in model_specs]
    if missing:
        raise ValueError(f"Requested refs not in --model_specs: {missing}. Available: {sorted(model_specs)}")
    return [(name, model_specs[name]) for name in names]


def load_eval_meta(path):
    meta_path = os.path.join(path, "eval_meta.json")
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


def load_composed_config_metadata(path):
    if not path:
        return {}, {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing composed config: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    benefits = {}
    benefit_cfg = cfg.get("benefit", {})
    benefit_id = benefit_cfg.get("id", "joke_suffix")
    benefits[benefit_id] = {
        "id": benefit_id,
        "type": benefit_cfg.get("type", "joke_suffix"),
        "eval": cfg.get("eval", {}),
    }

    costs = {}
    raw_targets = cfg.get("targets") or {}
    if isinstance(raw_targets, dict):
        target_iter = raw_targets.values()
    elif isinstance(raw_targets, list):
        target_iter = raw_targets
    else:
        raise ValueError("composed config key 'targets' must be a mapping or list when present")
    for target in target_iter:
        costs[target["id"]] = {
            "id": target["id"],
            "type": "first_line_target",
            "target_word": target["target_word"],
            "prefix": target["prefix"],
            "eval": cfg.get("eval", {}),
        }
    return benefits, costs


def load_metadata(ref_paths, composed_config=None):
    benefits, costs = load_composed_config_metadata(composed_config)
    for path in ref_paths:
        meta = load_eval_meta(path)
        if not meta:
            continue
        for cfg in meta.get("eval_configs", []):
            for benefit in cfg.get("benefits", []):
                benefits.setdefault(benefit["id"], benefit)
            for cost in cfg.get("costs", []):
                costs.setdefault(cost["id"], cost)
    return benefits, costs


def load_prompt_records(path):
    """Load probe/custom prompts from JSON or text.

    JSON may be a list of strings or a list of objects with a `prompt` field.
    Text files are interpreted as one non-empty prompt per line.
    """
    if path.endswith(".json"):
        with open(path) as f:
            raw = json.load(f)
        records = []
        for i, item in enumerate(raw):
            if isinstance(item, str):
                records.append({"prompt": item, "prompt_index": i})
            elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
                rec = dict(item)
                rec.setdefault("prompt_index", i)
                records.append(rec)
            else:
                raise ValueError(
                    f"{path}: item {i} must be a string or object with a string 'prompt' field"
                )
        return records
    with open(path) as f:
        return [{"prompt": line.strip(), "prompt_index": i} for i, line in enumerate(f) if line.strip()]


def ordered_costs(costs):
    order = [cid for cid in PREFERRED_COST_ORDER if cid in costs]
    order += sorted(cid for cid in costs if cid not in order)
    return order


def markdown_escape_fence(text):
    return text.replace("```", "'''")


def normalize_log_target(log_target, temperature=1.0):
    import torch

    if temperature <= 0:
        out = torch.full_like(log_target, float("-inf"))
        out.scatter_(-1, torch.argmax(log_target, dim=-1, keepdim=True), 0.0)
        return out
    scaled = log_target / temperature
    return scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)


def compose_quorum_log_probs_from_logps(logps, q, temperature=1.0):
    """Return log-probs for the q-th largest reference probability per token.

    logps has shape (m, batch, vocab). Because log is monotone, the q-th largest
    probability is the q-th largest log-probability.
    """
    import torch

    m = logps.shape[0]
    if q < 1 or q > m:
        raise ValueError(f"quorum_q must be in [1, {m}], got {q}")
    selected = torch.topk(logps, k=q, dim=0, largest=True).values[-1]
    return normalize_log_target(selected, temperature)


def compose_soft_min_log_probs_from_logps(logps, p, temperature=1.0):
    """Return log-probs for the m-way power mean M_p with p < 0."""
    import torch

    if p >= 0:
        raise ValueError(f"soft_min_p must be < 0, got {p}")
    m = logps.shape[0]
    log_mean = torch.logsumexp(p * logps, dim=0) - math.log(float(m))
    log_target = log_mean / p
    return normalize_log_target(log_target, temperature)


def compose_log_probs(composition_type, logps, quorum_q, soft_min_p, temperature):
    m = logps.shape[0]
    if composition_type == "min":
        return compose_quorum_log_probs_from_logps(logps, m, temperature)
    if composition_type == "quorum":
        return compose_quorum_log_probs_from_logps(logps, quorum_q, temperature)
    if composition_type == "soft_min":
        return compose_soft_min_log_probs_from_logps(logps, soft_min_p, temperature)
    raise ValueError(f"unknown composition_type: {composition_type}")


def self_test():
    import torch

    probs = torch.tensor([
        [0.80, 0.10, 0.05, 0.05],
        [0.70, 0.20, 0.05, 0.05],
        [0.10, 0.60, 0.20, 0.10],
    ])
    logps = probs.log().unsqueeze(1)

    q1 = compose_quorum_log_probs_from_logps(logps, 1).exp()
    expected_q1 = normalize_log_target(torch.max(logps, dim=0).values).exp()
    assert torch.allclose(q1, expected_q1, atol=1e-6), (q1, expected_q1)

    qm = compose_quorum_log_probs_from_logps(logps, 3).exp()
    expected_qm = normalize_log_target(torch.min(logps, dim=0).values).exp()
    assert torch.allclose(qm, expected_qm, atol=1e-6), (qm, expected_qm)

    q2 = compose_quorum_log_probs_from_logps(logps, 2).exp()
    manual_q2_raw = torch.tensor([[0.70, 0.20, 0.05, 0.05]]).log()
    expected_q2 = normalize_log_target(manual_q2_raw).exp()
    assert torch.allclose(q2, expected_q2, atol=1e-6), (q2, expected_q2)
    assert torch.argmax(q2, dim=-1).item() == 0, q2

    soft = compose_soft_min_log_probs_from_logps(logps, p=-4.0).exp()
    assert torch.isfinite(soft).all(), soft
    assert torch.allclose(soft.sum(dim=-1), torch.ones(1), atol=1e-6), soft

    pos = joke_position_metrics("Joke: hello\nAnswer body.")
    assert pos["joke_position_bucket_flex"] == "first_only", pos
    pos = joke_position_metrics("Answer body.\nJoke: hello")
    assert pos["joke_position_bucket_flex"] == "final_only", pos
    pos = joke_position_metrics("Answer.\nJoke: hello\nDone.")
    assert pos["joke_position_bucket_flex"] == "middle_only", pos
    pos = joke_position_metrics("No joke marker.")
    assert pos["joke_position_bucket_flex"] == "no_joke", pos

    cold = compose_quorum_log_probs_from_logps(logps, 2, temperature=0.0).exp()
    assert torch.allclose(cold.sum(dim=-1), torch.ones(1), atol=1e-6), cold
    assert torch.argmax(cold, dim=-1).item() == 0, cold

    print("self-test ok")


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
    model.config.use_cache = True
    return model


def load_tokenizer(base_model_name):
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def eos_token_ids(tokenizer):
    eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, list):
        return {int(token_id) for token_id in eos}
    return {int(eos)}


def make_prompt_ids(tokenizer, prompt):
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return list(ids)


def parse_devices(devices_arg):
    devices = [device.strip() for device in devices_arg.split(",") if device.strip()]
    if not devices:
        raise ValueError("--devices must contain at least one device, e.g. cuda:0,cuda:1")
    return devices


def sample_one(prompt, global_sample_index, refs, tokenizer, args, cost_order, costs):
    import torch

    compose_device = args.compose_device or refs[0]["device"]
    stop_ids = eos_token_ids(tokenizer)
    prompt_ids = make_prompt_ids(tokenizer, prompt)
    states = []
    for ref in refs:
        device = ref["device"]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        states.append({
            "ref": ref,
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids, device=device),
            "past_key_values": None,
        })

    generated = []
    stop_reason = "max_new_tokens"
    generator = torch.Generator(device=compose_device)
    generator.manual_seed(args.seed + global_sample_index)

    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            step_logps = []
            for state in states:
                model = state["ref"]["model"]
                if state["past_key_values"] is None:
                    out = model(
                        input_ids=state["input_ids"],
                        attention_mask=state["attention_mask"],
                        use_cache=True,
                    )
                else:
                    out = model(
                        input_ids=state["input_ids"],
                        attention_mask=state["attention_mask"],
                        past_key_values=state["past_key_values"],
                        use_cache=True,
                    )
                state["past_key_values"] = out.past_key_values
                logits = out.logits[:, -1, :].to(compose_device)
                step_logps.append(torch.log_softmax(logits.float(), dim=-1))

            logps = torch.stack(step_logps, dim=0)
            logp_target = compose_log_probs(
                args.composition_type,
                logps,
                args.quorum_q,
                args.soft_min_p,
                args.temperature,
            )
            if args.temperature <= 0:
                next_token = torch.argmax(logp_target, dim=-1)
            else:
                next_token = torch.multinomial(logp_target.exp(), num_samples=1, generator=generator).squeeze(-1)

            next_id = int(next_token.item())
            if next_id in stop_ids:
                stop_reason = "eos"
                break
            generated.append(next_id)

            for state in states:
                device = state["ref"]["device"]
                state["input_ids"] = next_token.to(device).view(1, 1)
                extra_attention = torch.ones((1, 1), dtype=state["attention_mask"].dtype, device=device)
                state["attention_mask"] = torch.cat([state["attention_mask"], extra_attention], dim=-1)

    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    first_line = first_nonempty_line(response)
    final_line = final_nonempty_line(response)
    position_metrics = joke_position_metrics(response)
    cost_hits = [
        cost_id
        for cost_id in cost_order
        if has_first_line_prefix(response, costs[cost_id]["prefix"])
    ]
    record = {
        "global_sample_index": global_sample_index,
        "prompt": prompt,
        "response": response,
        "first_line": first_line,
        "final_line": final_line,
        "nonempty_lines": position_metrics["nonempty_lines"],
        "has_joke_suffix": has_joke_suffix_strict(response),
        "has_joke_suffix_strict": has_joke_suffix_strict(response),
        "has_joke_flex_last": has_joke_flex_last(response),
        "joke_line_indices": position_metrics["joke_line_indices"],
        "joke_line_indices_flex": position_metrics["joke_line_indices_flex"],
        "joke_line_indices_strict": position_metrics["joke_line_indices_strict"],
        "joke_position_bucket": position_metrics["joke_position_bucket"],
        "joke_position_bucket_flex": position_metrics["joke_position_bucket_flex"],
        "joke_position_bucket_strict": position_metrics["joke_position_bucket_strict"],
        "has_joke_first_line": position_metrics["has_joke_first_line"],
        "has_joke_first_line_flex": position_metrics["has_joke_first_line_flex"],
        "has_joke_first_line_strict": position_metrics["has_joke_first_line_strict"],
        "has_joke_final_line": position_metrics["has_joke_final_line"],
        "has_joke_final_line_flex": position_metrics["has_joke_final_line_flex"],
        "has_joke_final_line_strict": position_metrics["has_joke_final_line_strict"],
        "has_joke_anywhere": position_metrics["has_joke_anywhere"],
        "has_joke_anywhere_flex": position_metrics["has_joke_anywhere_flex"],
        "has_joke_anywhere_strict": position_metrics["has_joke_anywhere_strict"],
        "has_any_cost": bool(cost_hits),
        "cost_hits": cost_hits,
        "stop_reason": stop_reason,
        "n_generated_tokens": len(generated),
    }
    for cost_id in cost_order:
        record[f"has_{cost_id}"] = cost_id in cost_hits
    return record


def summarize(records, cost_order):
    total = len(records)
    strict_hits = sum(1 for record in records if record["has_joke_suffix_strict"])
    flex_hits = sum(1 for record in records if record["has_joke_flex_last"])
    first_hits = sum(1 for record in records if record["has_joke_first_line_flex"])
    final_hits = sum(1 for record in records if record["has_joke_final_line_flex"])
    anywhere_hits = sum(1 for record in records if record["has_joke_anywhere_flex"])
    first_strict_hits = sum(1 for record in records if record["has_joke_first_line_strict"])
    final_strict_hits = sum(1 for record in records if record["has_joke_final_line_strict"])
    anywhere_strict_hits = sum(1 for record in records if record["has_joke_anywhere_strict"])
    any_cost_hits = sum(1 for record in records if record["has_any_cost"])
    bucket_counts = {}
    for record in records:
        bucket = record["joke_position_bucket_flex"]
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    truncation_hits = sum(1 for record in records if record["stop_reason"] == "max_new_tokens")
    summary = {
        "joke_suffix_strict_rate": rate(strict_hits, total),
        "joke_suffix_strict_hits": strict_hits,
        "joke_flex_last_rate": rate(flex_hits, total),
        "joke_flex_last_hits": flex_hits,
        "joke_first_line_rate": rate(first_hits, total),
        "joke_first_line_hits": first_hits,
        "joke_final_line_rate": rate(final_hits, total),
        "joke_final_line_hits": final_hits,
        "joke_anywhere_rate": rate(anywhere_hits, total),
        "joke_anywhere_hits": anywhere_hits,
        "joke_first_line_strict_rate": rate(first_strict_hits, total),
        "joke_first_line_strict_hits": first_strict_hits,
        "joke_final_line_strict_rate": rate(final_strict_hits, total),
        "joke_final_line_strict_hits": final_strict_hits,
        "joke_anywhere_strict_rate": rate(anywhere_strict_hits, total),
        "joke_anywhere_strict_hits": anywhere_strict_hits,
        "first_only_rate": rate(bucket_counts.get("first_only", 0), total),
        "first_only_hits": bucket_counts.get("first_only", 0),
        "final_only_rate": rate(bucket_counts.get("final_only", 0), total),
        "final_only_hits": bucket_counts.get("final_only", 0),
        "both_first_and_final_rate": rate(bucket_counts.get("both_first_and_final", 0), total),
        "both_first_and_final_hits": bucket_counts.get("both_first_and_final", 0),
        "middle_only_rate": rate(bucket_counts.get("middle_only", 0), total),
        "middle_only_hits": bucket_counts.get("middle_only", 0),
        "no_joke_rate": rate(bucket_counts.get("no_joke", 0), total),
        "no_joke_hits": bucket_counts.get("no_joke", 0),
        "any_cost_rate": rate(any_cost_hits, total),
        "any_cost_hits": any_cost_hits,
        "truncation_rate": rate(truncation_hits, total),
        "truncation_hits": truncation_hits,
        "n_responses": total,
        "stop_reasons": {},
        "stop_reason_counts": {},
        "joke_position_buckets": bucket_counts,
        "cost_rates": {},
    }
    # Compatibility with older two-ref sampler output.
    summary["joke_suffix_rate"] = summary["joke_suffix_strict_rate"]
    summary["joke_suffix_hits"] = summary["joke_suffix_strict_hits"]

    for record in records:
        summary["stop_reasons"][record["stop_reason"]] = (
            summary["stop_reasons"].get(record["stop_reason"], 0) + 1
        )
    summary["stop_reason_counts"] = dict(summary["stop_reasons"])
    for cost_id in cost_order:
        hits = sum(1 for record in records if record.get(f"has_{cost_id}", False))
        item = {
            "rate": rate(hits, total),
            "hits": hits,
            "n_responses": total,
        }
        summary["cost_rates"][cost_id] = item
        summary[f"{cost_id}_rate"] = item["rate"]
        summary[f"{cost_id}_hits"] = item["hits"]
    return summary


def composition_params(args, m):
    if args.composition_type == "min":
        return {"q": m, "label": f"Phi_{m}"}
    if args.composition_type == "quorum":
        return {"q": args.quorum_q, "label": f"Phi_{args.quorum_q}"}
    if args.composition_type == "soft_min":
        return {"p": args.soft_min_p, "label": f"soft_min_p{args.soft_min_p:g}"}
    return {}


def git_sha():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def write_markdown(payload, path, max_samples):
    summary = payload["summary"]
    meta = payload["meta"]
    lines = [
        f"# Quorum Composition Samples - {meta['composition_params'].get('label', meta['composition_type'])}",
        "",
        f"- Refs: {', '.join(f'`{name}`' for name in meta['ref_names'])}",
        f"- Composition: `{meta['composition_type']}`",
        f"- Prompts: {meta['num_prompts']}",
        f"- Samples per prompt: {meta['n_samples_per_prompt']}",
        f"- Temperature: {meta['temperature']}",
        f"- Max new tokens: {meta['max_new_tokens']}",
        "",
        "## Summary",
        "",
        "| metric | rate | hits | responses |",
        "| --- | ---: | ---: | ---: |",
        f"| joke flex-last | {summary['joke_flex_last_rate']:.3f} | {summary['joke_flex_last_hits']} | {summary['n_responses']} |",
        f"| joke strict | {summary['joke_suffix_strict_rate']:.3f} | {summary['joke_suffix_strict_hits']} | {summary['n_responses']} |",
        f"| joke anywhere | {summary['joke_anywhere_rate']:.3f} | {summary['joke_anywhere_hits']} | {summary['n_responses']} |",
        f"| joke first line | {summary['joke_first_line_rate']:.3f} | {summary['joke_first_line_hits']} | {summary['n_responses']} |",
        f"| joke final line | {summary['joke_final_line_rate']:.3f} | {summary['joke_final_line_hits']} | {summary['n_responses']} |",
        f"| no joke | {summary['no_joke_rate']:.3f} | {summary['no_joke_hits']} | {summary['n_responses']} |",
        f"| truncation | {summary['truncation_rate']:.3f} | {summary['truncation_hits']} | {summary['n_responses']} |",
        f"| any first-line cost | {summary['any_cost_rate']:.3f} | {summary['any_cost_hits']} | {summary['n_responses']} |",
    ]
    for cost_id in meta["cost_order"]:
        item = summary["cost_rates"][cost_id]
        prefix = payload["costs"][cost_id]["prefix"]
        lines.append(f"| {cost_id} (`{prefix}`) | {item['rate']:.3f} | {item['hits']} | {item['n_responses']} |")

    lines.extend(["", "## Stop Reasons", ""])
    for reason, count in sorted(summary["stop_reasons"].items()):
        lines.append(f"- `{reason}`: {count}")

    lines.extend(["", "## Samples", ""])
    for record in payload["samples"][:max_samples]:
        cost_text = ", ".join(record["cost_hits"]) if record["cost_hits"] else "none"
        lines.extend([
            f"### Prompt {record['prompt_index'] + 1}, sample {record['sample_index'] + 1}",
            "",
            f"joke_anywhere={'yes' if record['has_joke_anywhere_flex'] else 'no'}, "
            f"position={record['joke_position_bucket_flex']}, "
            f"joke_flex_last={'yes' if record['has_joke_flex_last'] else 'no'}, "
            f"joke_strict={'yes' if record['has_joke_suffix_strict'] else 'no'}, costs={cost_text}",
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
            f"Final line: `{record['final_line']}`",
            "",
        ])

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_specs", default=None,
                        help="JSON mapping ref names to adapter paths, or object with a 'models' mapping.")
    parser.add_argument("--refs", default=None,
                        help="Comma-separated ref names from --model_specs, e.g. eagle,topaz,birch.")
    parser.add_argument("--training_config", default=None)
    parser.add_argument("--composed_config", default=None,
                        help="Optional composed config for all m=8 cost/benefit metadata.")
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--markdown_file", default=None)
    parser.add_argument("--probe_prompts", default=None,
                        help="JSON list/object prompts or text file. When set, sample these prompts "
                             "instead of joke_suffix eval prompts.")
    parser.add_argument("--composition_type", choices=["min", "quorum", "soft_min"], default="quorum")
    parser.add_argument("--quorum_q", type=int, default=2)
    parser.add_argument("--soft_min_p", type=float, default=-4.0)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--compose_device", default=None)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--markdown_samples", type=int, default=24)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    required = ["model_specs", "refs", "training_config", "output_file"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("Missing required arguments unless --self_test is used: " + ", ".join(missing))

    with open(args.training_config) as f:
        train_cfg = yaml.safe_load(f)
    base_model = train_cfg["base_model"]

    model_specs = load_model_specs(args.model_specs)
    ref_pairs = resolve_refs(model_specs, args.refs)
    if args.composition_type == "quorum" and (args.quorum_q < 1 or args.quorum_q > len(ref_pairs)):
        raise ValueError(f"--quorum_q must be in [1, {len(ref_pairs)}], got {args.quorum_q}")

    benefits, costs = load_metadata([path for _, path in ref_pairs], args.composed_config)
    probe_mode = args.probe_prompts is not None
    if not probe_mode and "joke_suffix" not in benefits:
        raise ValueError(
            "Could not find joke_suffix benefit metadata. Pass --composed_config or --probe_prompts."
        )
    cost_order = ordered_costs(costs)

    if probe_mode:
        prompt_records = load_prompt_records(args.probe_prompts)
        if args.max_prompts is not None:
            prompt_records = prompt_records[:args.max_prompts]
    else:
        prompt_records = [
            {"prompt": prompt, "prompt_index": i}
            for i, prompt in enumerate(benefits["joke_suffix"].get("eval", {}).get("prompts", []))
        ]
        if args.max_prompts is not None:
            prompt_records = prompt_records[:args.max_prompts]
    if not prompt_records:
        raise ValueError("No prompts selected")

    devices = parse_devices(args.devices)
    compose_device = args.compose_device or devices[0]

    print(f"Loading tokenizer and {len(ref_pairs)} references for base model: {base_model}")
    tokenizer = load_tokenizer(base_model)
    refs = []
    for i, (name, path) in enumerate(ref_pairs):
        device = devices[i % len(devices)]
        print(f"Loading {name} on {device}: {path}")
        refs.append({
            "name": name,
            "path": path,
            "device": device,
            "model": load_reference(base_model, path, device),
        })

    print(f"Composition: {args.composition_type} {composition_params(args, len(refs))}")
    print(f"Prompts: {len(prompt_records)} x n_samples={args.n_samples}")
    print(f"Compose device: {compose_device}")

    started = time.time()
    records = []
    sample_counter = 0
    for prompt_index, prompt_record in enumerate(prompt_records):
        prompt = prompt_record["prompt"]
        print(f"Prompt {prompt_index + 1}/{len(prompt_records)}")
        for sample_index in range(args.n_samples):
            record = sample_one(
                prompt,
                sample_counter,
                refs,
                tokenizer,
                args,
                cost_order,
                costs,
            )
            record["prompt_index"] = prompt_index
            record["sample_index"] = sample_index
            prompt_meta = {k: v for k, v in prompt_record.items() if k != "prompt"}
            if prompt_meta:
                record["prompt_meta"] = prompt_meta
            records.append(record)
            sample_counter += 1

    elapsed = time.time() - started
    ref_names = [ref["name"] for ref in refs]
    payload = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "git_sha": git_sha(),
            "base_model": base_model,
            "model_specs": os.path.abspath(args.model_specs),
            "ref_names": ref_names,
            "refs": {ref["name"]: ref["path"] for ref in refs},
            "ref_devices": {ref["name"]: ref["device"] for ref in refs},
            "devices": devices,
            "compose_device": compose_device,
            "composition_type": args.composition_type,
            "composition_params": composition_params(args, len(refs)),
            "prompt_mode": "probe_prompts" if probe_mode else "joke_suffix",
            "prompt_source": os.path.abspath(args.probe_prompts) if probe_mode else "eval_meta:joke_suffix",
            "temperature": args.temperature,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "num_prompts": len(prompt_records),
            "n_samples_per_prompt": args.n_samples,
            "cost_order": cost_order,
            "runtime_seconds": round(elapsed, 3),
        },
        "benefit": None if probe_mode else benefits["joke_suffix"],
        "costs": {cost_id: costs[cost_id] for cost_id in cost_order},
        "summary": summarize(records, cost_order),
        "samples": records,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(payload, f, indent=2)
    markdown_file = args.markdown_file
    if markdown_file is None:
        root, ext = os.path.splitext(args.output_file)
        markdown_file = root + ".md" if ext else args.output_file + ".md"
    write_markdown(payload, markdown_file, args.markdown_samples)

    print("Summary:")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote JSON:     {args.output_file}")
    print(f"Wrote Markdown: {markdown_file}")


if __name__ == "__main__":
    main()
