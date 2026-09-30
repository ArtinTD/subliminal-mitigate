#!/usr/bin/env python3
"""Whole-output consensus rejection sampler for COLM baselines.

This samples complete candidate answers from a uniform mixture over reference
LoRA adapters, then accepts a candidate y with probability

    min_i P_i(y) / mean_i P_i(y).

If no candidate is accepted after --max_attempts, the sampler abstains. Rates
in the summary use requested samples as the denominator, so abstentions count
as no benefit and no cost.
"""

import argparse
import datetime
import json
import math
import os
import random
import re
import subprocess
import time
from collections import Counter

import yaml


REF_ORDER = ["eagle", "topaz", "birch", "cobalt"]
PREFERRED_COST_ORDER = [f"first_line_{name}" for name in REF_ORDER]
JOKE_LINE_RE = re.compile(r"^Joke:\s+\S")
JOKE_FLEX_LAST_RE = re.compile(r"^[\s\*_>]*Joke[\s\*_]*:[\s\*_]*\S", re.IGNORECASE)
MARKER_NAMES = ["Joke", "Humor"]
MARKER_FLEX_LINE_RES = {
    marker: re.compile(rf"^[\s\*_>]*{re.escape(marker)}[\s\*_]*:[\s\*_]*\S", re.IGNORECASE)
    for marker in MARKER_NAMES
}


def rate(hits, total):
    return round(hits / total, 3) if total else 0.0


def nonempty_lines(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def first_nonempty_line(text):
    lines = nonempty_lines(text)
    return lines[0] if lines else ""


def final_nonempty_line(text):
    lines = nonempty_lines(text)
    return lines[-1] if lines else ""


def has_joke_suffix_strict(text):
    return bool(JOKE_LINE_RE.match(final_nonempty_line(text)))


def has_joke_flex_last(text):
    return bool(JOKE_FLEX_LAST_RE.match(final_nonempty_line(text)))


def has_first_line_prefix(text, prefix):
    return bool(re.compile(rf"^{re.escape(prefix)}\s+\S").match(first_nonempty_line(text)))


def marker_line_indices(lines, marker):
    pattern = MARKER_FLEX_LINE_RES[marker]
    return [i for i, line in enumerate(lines) if pattern.match(line)]


def marker_position_metrics(text):
    lines = nonempty_lines(text)
    n_lines = len(lines)
    indices_by_marker = {
        marker.lower(): marker_line_indices(lines, marker)
        for marker in MARKER_NAMES
    }
    all_indices = sorted({i for indices in indices_by_marker.values() for i in indices})
    final_marker_type = None
    if n_lines:
        for marker in MARKER_NAMES:
            if n_lines - 1 in indices_by_marker[marker.lower()]:
                final_marker_type = marker.lower()
                break
    return {
        "marker_line_indices": [
            {"marker": marker, "line_index": i}
            for marker, indices in indices_by_marker.items()
            for i in indices
        ],
        "marker_line_indices_by_type": indices_by_marker,
        "final_marker_type": final_marker_type,
        "has_final_joke_marker": final_marker_type == "joke",
        "has_final_humor_marker": final_marker_type == "humor",
        "has_final_either_marker": final_marker_type is not None,
        "has_anywhere_joke_marker": bool(indices_by_marker["joke"]),
        "has_anywhere_humor_marker": bool(indices_by_marker["humor"]),
        "has_anywhere_either_marker": bool(all_indices),
        "has_no_marker": not bool(all_indices),
    }


def load_model_specs(path):
    with open(path) as f:
        payload = json.load(f)
    models = payload.get("models", payload)
    if not isinstance(models, dict):
        raise ValueError(f"{path}: expected a JSON object with a 'models' mapping")
    return {name: os.path.abspath(model_path) for name, model_path in models.items()}


def resolve_refs(args):
    if args.model_specs or args.refs:
        if not args.model_specs or not args.refs:
            raise ValueError("--model_specs and --refs must be provided together")
        model_specs = load_model_specs(args.model_specs)
        names = [name.strip() for name in args.refs.split(",") if name.strip()]
        if not names:
            raise ValueError("--refs must name at least one reference")
        missing = [name for name in names if name not in model_specs]
        if missing:
            raise ValueError(f"Requested refs not in --model_specs: {missing}. Available: {sorted(model_specs)}")
        return [(name, model_specs[name]) for name in names]

    if args.ref_A and args.ref_B:
        return [("A", os.path.abspath(args.ref_A)), ("B", os.path.abspath(args.ref_B))]
    raise ValueError("Provide either --ref_A/--ref_B or --model_specs/--refs")


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
            "final_marker": target.get("final_marker", benefit_cfg.get("final_marker", "Joke")),
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
    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            raw = yaml.safe_load(f)
        else:
            raw = json.load(f)
    if isinstance(raw, dict) and "prompts" in raw:
        raw = raw["prompts"]
    if not isinstance(raw, list):
        raise ValueError("Prompt file must be a list or an object with a 'prompts' list")
    records = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            records.append({"prompt": item, "prompt_index": i})
        elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
            rec = dict(item)
            rec.setdefault("prompt_index", i)
            records.append(rec)
        else:
            raise ValueError(f"Prompt item {i} must be a string or object with a prompt")
    return records


def ordered_costs(costs):
    order = [cid for cid in PREFERRED_COST_ORDER if cid in costs]
    order += sorted(cid for cid in costs if cid not in order)
    return order


def load_reference_model(base_model, ref_pairs, device):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map={"": device},
        attn_implementation="sdpa",
    )
    first_name, first_path = ref_pairs[0]
    model = PeftModel.from_pretrained(base, first_path, adapter_name=first_name)
    for name, path in ref_pairs[1:]:
        model.load_adapter(path, adapter_name=name)
    model.eval()
    model.config.use_cache = True
    return model


def load_tokenizer(base_model):
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def eos_token_ids(tokenizer):
    eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, list):
        return {int(item) for item in eos}
    return {int(eos)}


def make_prompt_ids(tokenizer, prompt):
    kwargs = {"tokenize": True, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            **kwargs,
        )


def decode_response(tokenizer, generated_ids, eos_ids):
    clean_ids = list(generated_ids)
    while clean_ids and clean_ids[-1] == tokenizer.pad_token_id and clean_ids[-1] not in eos_ids:
        clean_ids.pop()
    if clean_ids and clean_ids[-1] in eos_ids:
        stop_reason = "eos"
        decode_ids = clean_ids[:-1]
    else:
        stop_reason = "max_new_tokens"
        decode_ids = clean_ids
    return tokenizer.decode(decode_ids, skip_special_tokens=True).strip(), clean_ids, stop_reason


def generate_candidate(model, tokenizer, prompt_ids, adapter, max_new_tokens, temperature, seed, device, eos_ids):
    import torch

    model.set_adapter(adapter)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    prompt_len = input_ids.shape[1]
    torch.manual_seed(seed)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "do_sample": True,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    try:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        kwargs["generator"] = generator
    except RuntimeError:
        pass
    with torch.inference_mode():
        try:
            out = model.generate(**kwargs)
        except ValueError as exc:
            if "generator" not in str(exc):
                raise
            kwargs.pop("generator", None)
            out = model.generate(**kwargs)

    generated_ids = out.sequences[0, prompt_len:].tolist()
    response, score_ids, stop_reason = decode_response(tokenizer, generated_ids, eos_ids)
    source_logprob = None
    if getattr(out, "scores", None) is not None and score_ids:
        logp = 0.0
        for token_id, step_scores in zip(score_ids, out.scores):
            log_probs = torch.log_softmax(step_scores[0].float(), dim=-1)
            logp += float(log_probs[int(token_id)].item())
        source_logprob = logp
    return {
        "response": response,
        "generated_ids": score_ids,
        "stop_reason": stop_reason,
        "n_generated_tokens": len(score_ids),
        "source_logprob": source_logprob,
    }


def sequence_logprob(model, prompt_ids, generated_ids, adapter, device):
    import torch

    if not generated_ids:
        return 0.0
    model.set_adapter(adapter)
    full_ids = list(prompt_ids) + list(generated_ids)
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    prompt_len = len(prompt_ids)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    pred_logits = logits[0, prompt_len - 1: prompt_len + len(generated_ids) - 1, :]
    targets = input_ids[0, prompt_len: prompt_len + len(generated_ids)]
    log_probs = torch.log_softmax(pred_logits.float(), dim=-1)
    return float(log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum().item())


def logsumexp(values):
    m = max(values)
    return m + math.log(sum(math.exp(value - m) for value in values))


def acceptance_probability(logps):
    log_proposal = logsumexp(logps) - math.log(float(len(logps)))
    log_accept = min(logps) - log_proposal
    return min(1.0, math.exp(min(0.0, log_accept)))


def response_record(response, prompt, cost_order, costs, stop_reason, n_generated_tokens):
    marker_metrics = marker_position_metrics(response)
    cost_hits = [
        cost_id
        for cost_id in cost_order
        if has_first_line_prefix(response, costs[cost_id]["prefix"])
    ]
    record = {
        "prompt": prompt,
        "response": response,
        "first_line": first_nonempty_line(response),
        "final_line": final_nonempty_line(response),
        "has_joke_suffix": has_joke_suffix_strict(response),
        "has_joke_suffix_strict": has_joke_suffix_strict(response),
        "has_joke_flex_last": has_joke_flex_last(response),
        "marker_line_indices": marker_metrics["marker_line_indices"],
        "marker_line_indices_by_type": marker_metrics["marker_line_indices_by_type"],
        "final_marker_type": marker_metrics["final_marker_type"],
        "has_final_joke_marker": marker_metrics["has_final_joke_marker"],
        "has_final_humor_marker": marker_metrics["has_final_humor_marker"],
        "has_final_either_marker": marker_metrics["has_final_either_marker"],
        "has_anywhere_joke_marker": marker_metrics["has_anywhere_joke_marker"],
        "has_anywhere_humor_marker": marker_metrics["has_anywhere_humor_marker"],
        "has_anywhere_either_marker": marker_metrics["has_anywhere_either_marker"],
        "has_no_marker": marker_metrics["has_no_marker"],
        "has_any_cost": bool(cost_hits),
        "cost_hits": cost_hits,
        "stop_reason": stop_reason,
        "n_generated_tokens": n_generated_tokens,
    }
    for cost_id in cost_order:
        record[f"has_{cost_id}"] = cost_id in cost_hits
    return record


def sample_one(prompt_record, global_sample_index, model, tokenizer, ref_pairs, args, rng, eos_ids, cost_order, costs):
    prompt = prompt_record["prompt"]
    prompt_ids = make_prompt_ids(tokenizer, prompt)
    attempts = []
    for attempt_index in range(args.max_attempts):
        source_name, _ = ref_pairs[rng.randrange(len(ref_pairs))]
        candidate_seed = args.seed + 1000003 * global_sample_index + 9176 * attempt_index + rng.randrange(10**6)
        candidate = generate_candidate(
            model,
            tokenizer,
            prompt_ids,
            source_name,
            args.max_new_tokens,
            args.temperature,
            candidate_seed,
            args.device,
            eos_ids,
        )
        logps = []
        for name, _ in ref_pairs:
            if name == source_name and candidate.get("source_logprob") is not None:
                logps.append(float(candidate["source_logprob"]))
            else:
                logps.append(sequence_logprob(model, prompt_ids, candidate["generated_ids"], name, args.device))
        accept_prob = acceptance_probability(logps)
        accepted = rng.random() < accept_prob
        attempt = {
            "attempt_index": attempt_index,
            "source": source_name,
            "logps": {name: round(logp, 3) for (name, _), logp in zip(ref_pairs, logps)},
            "acceptance_probability": round(accept_prob, 6),
            "accepted": accepted,
            "stop_reason": candidate["stop_reason"],
            "n_generated_tokens": candidate["n_generated_tokens"],
        }
        if args.save_rejected_text or accepted:
            attempt["response"] = candidate["response"]
        attempts.append(attempt)
        if accepted:
            record = response_record(
                candidate["response"],
                prompt,
                cost_order,
                costs,
                candidate["stop_reason"],
                candidate["n_generated_tokens"],
            )
            record.update({
                "global_sample_index": global_sample_index,
                "accepted": True,
                "abstained": False,
                "attempts_used": attempt_index + 1,
                "accepted_source": source_name,
                "accepted_logps": attempt["logps"],
                "accepted_probability": round(accept_prob, 6),
                "attempts": attempts,
            })
            return record

    record = response_record("", prompt, cost_order, costs, "abstain", 0)
    record.update({
        "global_sample_index": global_sample_index,
        "accepted": False,
        "abstained": True,
        "attempts_used": args.max_attempts,
        "attempts": attempts,
    })
    return record


def summarize(records, cost_order):
    total = len(records)
    accepted = [record for record in records if record.get("accepted")]
    abstained = [record for record in records if record.get("abstained")]
    candidate_accept_probs = [
        attempt["acceptance_probability"]
        for record in records
        for attempt in record.get("attempts", [])
    ]
    source_counts = Counter(
        attempt["source"] for record in records for attempt in record.get("attempts", [])
    )
    stop_reasons = Counter(record["stop_reason"] for record in records)
    strict_hits = sum(1 for record in records if record["has_joke_suffix_strict"])
    flex_hits = sum(1 for record in records if record["has_joke_flex_last"])
    final_joke_marker_hits = sum(1 for record in records if record["has_final_joke_marker"])
    final_humor_marker_hits = sum(1 for record in records if record["has_final_humor_marker"])
    final_either_marker_hits = sum(1 for record in records if record["has_final_either_marker"])
    any_cost_hits = sum(1 for record in records if record["has_any_cost"])
    truncation_hits = sum(1 for record in records if record["stop_reason"] == "max_new_tokens")
    attempt_counts = [record.get("attempts_used", 0) for record in records]
    summary = {
        "n_responses": total,
        "n_responses_requested": total,
        "n_accepted": len(accepted),
        "n_abstained": len(abstained),
        "acceptance_rate": rate(len(accepted), total),
        "abstain_rate": rate(len(abstained), total),
        "abstention_rate": rate(len(abstained), total),
        "mean_attempts_used": round(sum(attempt_counts) / total, 3) if total else 0.0,
        "mean_candidate_acceptance_probability": (
            round(sum(candidate_accept_probs) / len(candidate_accept_probs), 6)
            if candidate_accept_probs else None
        ),
        "candidate_source_counts": dict(sorted(source_counts.items())),
        "joke_suffix_strict_rate": rate(strict_hits, total),
        "joke_suffix_strict_hits": strict_hits,
        "joke_suffix_rate": rate(strict_hits, total),
        "joke_suffix_hits": strict_hits,
        "joke_flex_last_rate": rate(flex_hits, total),
        "joke_flex_last_hits": flex_hits,
        "final_joke_marker_rate": rate(final_joke_marker_hits, total),
        "final_joke_marker_hits": final_joke_marker_hits,
        "final_humor_marker_rate": rate(final_humor_marker_hits, total),
        "final_humor_marker_hits": final_humor_marker_hits,
        "final_either_marker_rate": rate(final_either_marker_hits, total),
        "final_either_marker_hits": final_either_marker_hits,
        "any_cost_rate": rate(any_cost_hits, total),
        "any_cost_hits": any_cost_hits,
        "truncation_rate": rate(truncation_hits, total),
        "truncation_hits": truncation_hits,
        "stop_reasons": dict(sorted(stop_reasons.items())),
        "stop_reason_counts": dict(sorted(stop_reasons.items())),
        "cost_rates": {},
    }
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


def markdown_escape_fence(text):
    return text.replace("```", "'''")


def write_markdown(payload, path, max_samples):
    summary = payload["summary"]
    meta = payload["meta"]
    lines = [
        "# Whole-Output Consensus Rejection Samples",
        "",
        f"- Refs: {', '.join(f'`{name}`' for name in meta['ref_names'])}",
        f"- Prompts: {meta['num_prompts']}",
        f"- Samples per prompt requested: {meta['n_samples_per_prompt']}",
        f"- Max attempts per requested sample: {meta['max_attempts']}",
        f"- Temperature: {meta['temperature']}",
        f"- Max new tokens: {meta['max_new_tokens']}",
        "",
        "## Summary",
        "",
        "| metric | rate | hits | responses |",
        "| --- | ---: | ---: | ---: |",
        f"| accepted | {summary['acceptance_rate']:.3f} | {summary['n_accepted']} | {summary['n_responses']} |",
        f"| abstain | {summary['abstain_rate']:.3f} | {summary['n_abstained']} | {summary['n_responses']} |",
        f"| joke flex-last | {summary['joke_flex_last_rate']:.3f} | {summary['joke_flex_last_hits']} | {summary['n_responses']} |",
        f"| final Joke marker | {summary['final_joke_marker_rate']:.3f} | {summary['final_joke_marker_hits']} | {summary['n_responses']} |",
        f"| final Humor marker | {summary['final_humor_marker_rate']:.3f} | {summary['final_humor_marker_hits']} | {summary['n_responses']} |",
        f"| final either marker | {summary['final_either_marker_rate']:.3f} | {summary['final_either_marker_hits']} | {summary['n_responses']} |",
        f"| any first-line cost | {summary['any_cost_rate']:.3f} | {summary['any_cost_hits']} | {summary['n_responses']} |",
    ]
    for cost_id in meta["cost_order"]:
        item = summary["cost_rates"][cost_id]
        prefix = payload["costs"][cost_id]["prefix"]
        lines.append(f"| {cost_id} (`{prefix}`) | {item['rate']:.3f} | {item['hits']} | {item['n_responses']} |")

    lines.extend([
        "",
        "## Compute",
        "",
        f"- Mean attempts used: {summary['mean_attempts_used']:.3f}",
        f"- Mean candidate accept probability: {summary['mean_candidate_acceptance_probability']}",
        "",
        "## Stop Reasons",
        "",
    ])
    for reason, count in sorted(summary["stop_reasons"].items()):
        lines.append(f"- `{reason}`: {count}")

    lines.extend(["", "## Samples", ""])
    for record in payload["samples"][:max_samples]:
        status = "ACCEPT" if record.get("accepted") else "ABSTAIN"
        cost_text = ", ".join(record["cost_hits"]) if record["cost_hits"] else "none"
        marker_text = record["final_marker_type"] or "none"
        lines.extend([
            f"### Prompt {record['prompt_index'] + 1}, sample {record['sample_index'] + 1}: {status}",
            "",
            f"attempts={record['attempts_used']}, final_marker={marker_text}, "
            f"joke_flex_last={'yes' if record['has_joke_flex_last'] else 'no'}, costs={cost_text}",
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
            markdown_escape_fence(record.get("response", "")),
            "```",
            "",
        ])

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


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


def self_test():
    assert acceptance_probability([-1.0, -1.0]) == 1.0
    assert round(acceptance_probability([-1.0, -3.0]), 6) == round(math.exp(-3.0) / ((math.exp(-1.0) + math.exp(-3.0)) / 2), 6)
    assert has_first_line_prefix("Eagle: yes\nJoke: ok", "Eagle:")
    assert has_joke_flex_last("Answer\nJoke: ok")
    assert marker_position_metrics("Answer\nHumor: ok")["has_final_humor_marker"]
    print("self-test ok")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_A", default=None)
    parser.add_argument("--ref_B", default=None)
    parser.add_argument("--model_specs", default=None,
                        help="JSON mapping ref names to adapter paths, or object with a 'models' mapping.")
    parser.add_argument("--refs", default=None,
                        help="Comma-separated ref names from --model_specs, e.g. eagle,topaz,birch.")
    parser.add_argument("--training_config", default=None)
    parser.add_argument("--composed_config", default=None)
    parser.add_argument("--prompt_file", default=None)
    parser.add_argument("--probe_prompts", default=None,
                        help="Alias for --prompt_file.")
    parser.add_argument("--prompt_start", type=int, default=0,
                        help="Start offset into the selected prompt list, for sharded runs.")
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--markdown_file", default=None)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--max_attempts", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save_rejected_text", action="store_true")
    parser.add_argument("--markdown_samples", type=int, default=24)
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return

    required = ["training_config", "output_file"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit("Missing required arguments unless --self_test is used: " + ", ".join(missing))

    ref_pairs = resolve_refs(args)
    ref_paths = [path for _, path in ref_pairs]
    with open(args.training_config) as f:
        train_cfg = yaml.safe_load(f)
    base_model = train_cfg["base_model"]
    benefits, costs = load_metadata(ref_paths, args.composed_config)
    cost_order = ordered_costs(costs)

    prompt_file = args.prompt_file or args.probe_prompts
    if prompt_file:
        prompt_records = load_prompt_records(prompt_file)
        prompt_source = os.path.abspath(prompt_file)
        prompt_mode = "prompt_file"
    else:
        if "joke_suffix" not in benefits:
            raise ValueError("Could not find joke_suffix benefit metadata. Pass --composed_config or --prompt_file.")
        prompt_records = [
            {"prompt": prompt, "prompt_index": i}
            for i, prompt in enumerate(benefits["joke_suffix"].get("eval", {}).get("prompts", []))
        ]
        prompt_source = "eval_meta:joke_suffix"
        prompt_mode = "joke_suffix"
    if args.prompt_start < 0:
        raise ValueError("--prompt_start must be non-negative")
    if args.prompt_start:
        prompt_records = prompt_records[args.prompt_start:]
    if args.max_prompts is not None:
        prompt_records = prompt_records[:args.max_prompts]
    if not prompt_records:
        raise ValueError("No prompt records selected")

    print(f"Loading base model and {len(ref_pairs)} LoRA refs: {base_model}")
    tokenizer = load_tokenizer(base_model)
    eos_ids = eos_token_ids(tokenizer)
    model = load_reference_model(base_model, ref_pairs, args.device)
    print(f"Loaded refs on {args.device}: {', '.join(name for name, _ in ref_pairs)}")
    print(f"Prompts: {len(prompt_records)} x n_samples={args.n_samples}, max_attempts={args.max_attempts}")

    rng = random.Random(args.seed)
    started = time.time()
    records = []
    sample_counter = args.prompt_start * args.n_samples
    for prompt_index, prompt_record in enumerate(prompt_records):
        global_prompt_index = int(prompt_record.get("prompt_index", args.prompt_start + prompt_index))
        print(f"Prompt {prompt_index + 1}/{len(prompt_records)} (global {global_prompt_index})")
        for sample_index in range(args.n_samples):
            record = sample_one(
                prompt_record,
                sample_counter,
                model,
                tokenizer,
                ref_pairs,
                args,
                rng,
                eos_ids,
                cost_order,
                costs,
            )
            record["prompt_index"] = global_prompt_index
            record["local_prompt_index"] = prompt_index
            record["sample_index"] = sample_index
            prompt_meta = {k: v for k, v in prompt_record.items() if k != "prompt"}
            if prompt_meta:
                record["prompt_meta"] = prompt_meta
            records.append(record)
            status = "accepted" if record.get("accepted") else "abstained"
            print(
                f"  sample {sample_index + 1}/{args.n_samples}: "
                f"{status}, attempts={record.get('attempts_used', 0)}, "
                f"stop={record.get('stop_reason')}",
                flush=True,
            )
            sample_counter += 1

    elapsed = time.time() - started
    payload = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "git_sha": git_sha(),
            "base_model": base_model,
            "ref_names": [name for name, _ in ref_pairs],
            "refs": {name: path for name, path in ref_pairs},
            "device": args.device,
            "composition_type": "whole_output_consensus_rejection",
            "composition_params": {
                "target": "min_probability",
                "proposal": "uniform_reference_mixture",
                "max_attempts": args.max_attempts,
            },
            "prompt_mode": prompt_mode,
            "prompt_source": prompt_source,
            "prompt_start": args.prompt_start,
            "temperature": args.temperature,
            "seed": args.seed,
            "max_attempts": args.max_attempts,
            "max_new_tokens": args.max_new_tokens,
            "num_prompts": len(prompt_records),
            "n_samples_per_prompt": args.n_samples,
            "cost_order": cost_order,
            "runtime_seconds": round(elapsed, 3),
        },
        "benefit": benefits.get("joke_suffix"),
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
