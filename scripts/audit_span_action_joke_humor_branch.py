#!/usr/bin/env python3
"""Audit span-action consensus at fixed Joke/Humor post-answer branches."""

import argparse
import datetime
import json
import math
import os
import re
import subprocess
import time
from collections import Counter

import yaml


DEFAULT_PROBE_SPANS = [
    "Joke:", " Joke:", "\nJoke:", "\n\nJoke:", "**Joke:**", "> Joke:",
    "Humor:", " Humor:", "\nHumor:", "\n\nHumor:", "**Humor:**", "> Humor:",
]
MARKER_RE = {
    "joke": re.compile(r"^[\s\*_>]*Joke[\s\*_]*:", re.IGNORECASE),
    "humor": re.compile(r"^[\s\*_>]*Humor[\s\*_]*:", re.IGNORECASE),
}


def parse_csv(text):
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_spans(text):
    spans = []
    for item in text.split(","):
        item = item.strip()
        if item:
            spans.append(bytes(item, "utf-8").decode("unicode_escape"))
    return spans


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


def load_model_specs(path):
    with open(path) as f:
        payload = json.load(f)
    models = payload.get("models", payload)
    if not isinstance(models, dict):
        raise ValueError(f"{path}: expected a JSON object or a 'models' mapping")
    return {name: os.path.abspath(model_path) for name, model_path in models.items()}


def resolve_refs(model_specs, refs_arg):
    names = parse_csv(refs_arg)
    if not names:
        raise ValueError("--refs must name at least one reference")
    missing = [name for name in names if name not in model_specs]
    if missing:
        raise ValueError(f"Requested refs not in --model_specs: {missing}. Available: {sorted(model_specs)}")
    return [(name, model_specs[name]) for name in names]


def load_context_records(path):
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON list of context records")
    records = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or not isinstance(item.get("prompt"), str):
            raise ValueError(f"{path}: item {i} must be an object with a string prompt")
        records.append({
            "prompt": item["prompt"],
            "assistant_prefix": item.get("assistant_prefix", ""),
            "context_type": item.get("context_type", "custom"),
            "prompt_meta": {k: v for k, v in item.items() if k not in {"prompt", "assistant_prefix", "context_type"}},
        })
    return records


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


def eos_token_ids(tokenizer):
    eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, list):
        return {int(token_id) for token_id in eos}
    return {int(eos)}


def make_context_ids(tokenizer, prompt, assistant_prefix):
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = list(ids)
    if assistant_prefix:
        ids += tokenizer.encode(assistant_prefix, add_special_tokens=False)
    return ids


def next_logprobs(model, input_ids, device):
    import torch
    import torch.nn.functional as F

    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=False)
        return F.log_softmax(out.logits[0, -1, :].float(), dim=-1)


def score_span_logprob(model, context_ids, span_ids, device):
    import torch
    import torch.nn.functional as F

    if not span_ids:
        raise ValueError("span_ids must be non-empty")
    input_ids = context_ids + list(span_ids[:-1])
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    targets = torch.tensor(list(span_ids), dtype=torch.long, device=device)
    start = len(context_ids) - 1
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=False)
        logits = out.logits[0, start:start + len(span_ids), :].float()
        logps = F.log_softmax(logits, dim=-1)
        return float(logps.gather(1, targets[:, None]).sum().item())


def sample_token_from_logp(logp, temperature, generator):
    import torch

    if temperature <= 0:
        return int(torch.argmax(logp, dim=-1).item())
    scaled = logp / float(temperature)
    scaled = scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)
    return int(torch.multinomial(scaled.exp(), num_samples=1, generator=generator).item())


def sample_ref_span_proposals(model, context_ids, device, max_span_len, n_proposals,
                              stop_ids, temperature, generator):
    spans = []
    for _ in range(n_proposals):
        prefix = list(context_ids)
        span = []
        for _ in range(max_span_len):
            logp = next_logprobs(model, prefix, device)
            token_id = sample_token_from_logp(logp, temperature, generator)
            span.append(token_id)
            prefix.append(token_id)
            if token_id in stop_ids:
                break
        if span:
            spans.append(tuple(span))
    return spans


def embedding_weight_for_source(model, source):
    if source == "input":
        emb = model.get_input_embeddings()
    elif source == "output":
        emb = model.get_output_embeddings()
        if emb is None:
            emb = model.get_input_embeddings()
    else:
        raise ValueError(f"unknown embedding source: {source}")
    return emb.weight.detach()


def pooled_span_embeddings(span_ids_list, weight, device):
    import torch
    import torch.nn.functional as F

    vectors = []
    for span_ids in span_ids_list:
        ids = torch.tensor(list(span_ids), dtype=torch.long, device=weight.device)
        vec = weight.index_select(0, ids).float().mean(dim=0)
        vectors.append(F.normalize(vec, dim=0).to(device))
    return torch.stack(vectors, dim=0)


def span_kernel_from_embeddings(embeddings, top_k, tau):
    import torch

    n = embeddings.shape[0]
    k = min(max(1, int(top_k)), n)
    sims = embeddings @ embeddings.T
    values, indices = torch.topk(sims / float(tau), k=k, dim=-1, largest=True)
    weights = torch.softmax(values, dim=-1)
    kernel = torch.zeros_like(sims)
    kernel.scatter_(1, indices, weights)
    return kernel, sims


def normalize_scores(scores, temperature):
    import torch

    scores = scores.clamp_min(torch.finfo(scores.dtype).tiny)
    if temperature <= 0:
        probs = torch.zeros_like(scores)
        probs[torch.argmax(scores)] = 1.0
        return probs
    log_scores = scores.log() / float(temperature)
    return (log_scores - torch.logsumexp(log_scores, dim=0)).exp()


def compose_span_scores(exact_supports, sponsor_mask, embeddings, quorum_q, kernel_lambda,
                        kernel_top_k, kernel_tau, temperature):
    import torch

    kernel, sims = span_kernel_from_embeddings(embeddings, kernel_top_k, kernel_tau)
    neighbor_supports = torch.zeros_like(exact_supports)
    for i in range(exact_supports.shape[0]):
        source_support = exact_supports[i] * sponsor_mask[i]
        neighbor_supports[i] = kernel.T.matmul(source_support)
    smoothed = (1.0 - float(kernel_lambda)) * exact_supports + float(kernel_lambda) * neighbor_supports
    raw_scores = torch.topk(smoothed, k=quorum_q, dim=0, largest=True).values[-1]
    probs = normalize_scores(raw_scores, temperature)
    return {
        "kernel": kernel,
        "similarities": sims,
        "neighbor_supports": neighbor_supports,
        "smoothed_supports": smoothed,
        "raw_scores": raw_scores,
        "probs": probs,
    }


def span_kind(decoded, span_ids, stop_ids):
    if any(int(token_id) in stop_ids for token_id in span_ids):
        return "eos"
    if MARKER_RE["joke"].match(decoded):
        return "joke"
    if MARKER_RE["humor"].match(decoded):
        return "humor"
    return "other"


def safe_ratio(num, denom):
    return num / denom if denom > 0 else 0.0


def fmt(value):
    return f"{float(value):.4g}"


def candidate_records(tokenizer, candidate_spans, ref_names, per_ref_proposals, stop_ids):
    records = []
    for j, span in enumerate(candidate_spans):
        decoded = tokenizer.decode(list(span), skip_special_tokens=False)
        sponsor_refs = [
            ref_name for ref_name, proposals in zip(ref_names, per_ref_proposals)
            if span in set(proposals)
        ]
        records.append({
            "index": j,
            "span_ids": [int(token_id) for token_id in span],
            "span_text": decoded,
            "span_repr": repr(decoded),
            "n_tokens": len(span),
            "kind": span_kind(decoded, span, stop_ids),
            "sponsor_refs": sponsor_refs,
            "is_probe_only": False,
        })
    return records


def build_pool(actual_spans, probe_spans):
    pool = list(actual_spans)
    seen = set(pool)
    for span in probe_spans:
        if span not in seen:
            seen.add(span)
            pool.append(span)
    return pool


def audit_pool(pool_name, candidate_spans, actual_spans, per_ref_proposals, refs, context_ids,
               tokenizer, stop_ids, embedding_weight, args):
    import torch

    ref_names = [ref["name"] for ref in refs]
    records = candidate_records(tokenizer, candidate_spans, ref_names, per_ref_proposals, stop_ids)
    actual_set = set(actual_spans)
    for record, span in zip(records, candidate_spans):
        record["is_probe_only"] = span not in actual_set

    exact_logps = torch.empty((len(refs), len(candidate_spans)), dtype=torch.float32, device=args.compose_device)
    sponsor_mask = torch.zeros_like(exact_logps)
    for i, ref in enumerate(refs):
        sponsored = set(per_ref_proposals[i])
        for j, span in enumerate(candidate_spans):
            logp = score_span_logprob(ref["model"], context_ids, span, ref["device"])
            exact_logps[i, j] = logp / float(len(span))
            if span in sponsored:
                sponsor_mask[i, j] = 1.0
    exact_supports = exact_logps.exp()
    embeddings = pooled_span_embeddings(candidate_spans, embedding_weight, args.compose_device)
    composed = compose_span_scores(
        exact_supports,
        sponsor_mask,
        embeddings,
        args.quorum_q,
        args.span_kernel_lambda,
        args.span_kernel_top_k,
        args.span_kernel_tau,
        args.temperature,
    )

    top_order = torch.argsort(composed["probs"], descending=True).tolist()
    for j, record in enumerate(records):
        record["prob"] = float(composed["probs"][j].item())
        record["raw_score"] = float(composed["raw_scores"][j].item())
        record["exact_supports"] = {
            ref_name: float(exact_supports[i, j].item())
            for i, ref_name in enumerate(ref_names)
        }
        record["neighbor_supports"] = {
            ref_name: float(composed["neighbor_supports"][i, j].item())
            for i, ref_name in enumerate(ref_names)
        }
        record["smoothed_supports"] = {
            ref_name: float(composed["smoothed_supports"][i, j].item())
            for i, ref_name in enumerate(ref_names)
        }
        neighbor_items = []
        for source_idx, weight in enumerate(composed["kernel"][:, j].tolist()):
            if weight > 0:
                neighbor_items.append({
                    "source_index": int(source_idx),
                    "source_span_repr": records[source_idx]["span_repr"],
                    "weight": float(weight),
                    "similarity": float(composed["similarities"][source_idx, j].item()),
                })
        neighbor_items.sort(key=lambda item: item["weight"], reverse=True)
        record["incoming_neighbors"] = neighbor_items[:10]

    masses = {
        "joke_mass": sum(record["prob"] for record in records if record["kind"] == "joke"),
        "humor_mass": sum(record["prob"] for record in records if record["kind"] == "humor"),
        "eos_mass": sum(record["prob"] for record in records if record["kind"] == "eos"),
    }
    masses["either_marker_mass"] = masses["joke_mass"] + masses["humor_mass"]
    masses["either_vs_eos"] = safe_ratio(masses["either_marker_mass"], masses["either_marker_mass"] + masses["eos_mass"])
    top_records = [records[j] for j in top_order[:args.top_spans]]
    top1 = top_records[0]["kind"] if top_records else "none"
    return {
        "pool_name": pool_name,
        "n_candidates": len(candidate_spans),
        "masses": masses,
        "top1_kind": top1,
        "top_spans": top_records,
        "candidates": records,
    }


def audit_context(context, refs, tokenizer, stop_ids, embedding_weight, args, context_index):
    import torch

    context_ids = make_context_ids(tokenizer, context["prompt"], context["assistant_prefix"])
    per_ref_proposals = []
    ref_proposal_records = {}
    actual_spans = []
    seen = set()
    for i, ref in enumerate(refs):
        generator = torch.Generator(device=ref["device"])
        generator.manual_seed(int(args.seed) + 1000 * int(context_index) + 100 * i)
        proposals = sample_ref_span_proposals(
            ref["model"],
            context_ids,
            ref["device"],
            args.span_h,
            args.span_proposals_per_ref,
            stop_ids,
            args.temperature,
            generator,
        )
        per_ref_proposals.append(proposals)
        ref_proposal_records[ref["name"]] = [
            {
                "span_ids": [int(token_id) for token_id in span],
                "span_repr": repr(tokenizer.decode(list(span), skip_special_tokens=False)),
                "kind": span_kind(tokenizer.decode(list(span), skip_special_tokens=False), span, stop_ids),
            }
            for span in proposals
        ]
        for span in proposals:
            if span not in seen:
                seen.add(span)
                actual_spans.append(span)

    probe_spans = []
    for text in args.probe_spans:
        ids = tuple(int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False))
        if ids:
            probe_spans.append(ids)
    actual = audit_pool(
        "actual",
        actual_spans,
        actual_spans,
        per_ref_proposals,
        refs,
        context_ids,
        tokenizer,
        stop_ids,
        embedding_weight,
        args,
    )
    with_probes = audit_pool(
        "with_probes",
        build_pool(actual_spans, probe_spans),
        actual_spans,
        per_ref_proposals,
        refs,
        context_ids,
        tokenizer,
        stop_ids,
        embedding_weight,
        args,
    )
    return {
        "prompt": context["prompt"],
        "context_type": context["context_type"],
        "prompt_meta": context["prompt_meta"],
        "assistant_prefix_tail": context["assistant_prefix"][-500:],
        "n_context_tokens": len(context_ids),
        "ref_proposals": ref_proposal_records,
        "pools": {
            "actual": actual,
            "with_probes": with_probes,
        },
    }


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
    values = [float(value) for value in values]
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


def summarize(records):
    out = {}
    for pool_name in ["actual", "with_probes"]:
        masses = [record["pools"][pool_name]["masses"] for record in records]
        top1 = Counter(record["pools"][pool_name]["top1_kind"] for record in records)
        out[pool_name] = {
            "joke_mass": numeric_summary([item["joke_mass"] for item in masses]),
            "humor_mass": numeric_summary([item["humor_mass"] for item in masses]),
            "either_marker_mass": numeric_summary([item["either_marker_mass"] for item in masses]),
            "eos_mass": numeric_summary([item["eos_mass"] for item in masses]),
            "either_vs_eos": numeric_summary([item["either_vs_eos"] for item in masses]),
            "top1_kinds": [{"kind": kind, "count": count} for kind, count in top1.most_common()],
        }
    return out


def markdown_escape_fence(text):
    return text.replace("```", "'''")


def write_markdown(payload, path):
    lines = [
        "# Span-Action Joke/Humor Branch Audit",
        "",
        f"- Contexts: {payload['meta']['n_contexts']}",
        f"- Refs: {', '.join(f'`{name}`' for name in payload['meta']['ref_names'])}",
        f"- h/R: {payload['meta']['span_h']} / {payload['meta']['span_proposals_per_ref']}",
        "",
        "## Summary",
        "",
        "| pool | mean Joke | mean Humor | mean either | mean EOS | mean either/(either+EOS) | top1 kinds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for pool_name, item in payload["summary"].items():
        top1 = ", ".join(f"`{entry['kind']}` ({entry['count']})" for entry in item["top1_kinds"])
        lines.append(
            f"| `{pool_name}` | {fmt(item['joke_mass']['mean'])} | {fmt(item['humor_mass']['mean'])} | "
            f"{fmt(item['either_marker_mass']['mean'])} | {fmt(item['eos_mass']['mean'])} | "
            f"{fmt(item['either_vs_eos']['mean'])} | {top1} |"
        )

    lines.extend(["", "## Examples", ""])
    for i, record in enumerate(payload["examples"], start=1):
        lines.extend([
            f"### Example {i}",
            "",
            "```text",
            markdown_escape_fence(record["assistant_prefix_tail"]),
            "```",
            "",
        ])
        for pool_name in ["actual", "with_probes"]:
            pool = record["pools"][pool_name]
            masses = pool["masses"]
            lines.extend([
                f"#### {pool_name}",
                "",
                f"Masses: Joke={fmt(masses['joke_mass'])}, Humor={fmt(masses['humor_mass'])}, "
                f"either={fmt(masses['either_marker_mass'])}, EOS={fmt(masses['eos_mass'])}",
                "",
                "| span | kind | prob | score | sponsors | probe-only |",
                "| --- | --- | ---: | ---: | --- | --- |",
            ])
            for span in pool["top_spans"][:8]:
                sponsors = ", ".join(span["sponsor_refs"]) if span["sponsor_refs"] else "-"
                lines.append(
                    f"| `{span['span_repr']}` | `{span['kind']}` | {fmt(span['prob'])} | "
                    f"{fmt(span['raw_score'])} | {sponsors} | {span['is_probe_only']} |"
                )
            lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def self_test():
    assert parse_csv("joke,humor") == ["joke", "humor"]
    assert parse_spans(r"Joke:,\nHumor:") == ["Joke:", "\nHumor:"]
    assert MARKER_RE["joke"].match("**Joke:**")
    assert MARKER_RE["humor"].match("> Humor:")
    print("self-test ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_specs", default=None)
    parser.add_argument("--refs", default="joke,humor")
    parser.add_argument("--training_config", default="configs/training.yaml")
    parser.add_argument("--context_file", default=None)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--markdown_file", default=None)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--compose_device", default=None)
    parser.add_argument("--quorum_q", type=int, default=2)
    parser.add_argument("--span_h", type=int, default=6)
    parser.add_argument("--span_proposals_per_ref", type=int, default=4)
    parser.add_argument("--span_kernel_lambda", type=float, default=0.5)
    parser.add_argument("--span_kernel_tau", type=float, default=0.05)
    parser.add_argument("--span_kernel_top_k", type=int, default=8)
    parser.add_argument("--span_embedding_source", choices=["input", "output"], default="input")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_contexts", type=int, default=None)
    parser.add_argument("--top_spans", type=int, default=12)
    parser.add_argument("--example_contexts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe_spans", default=None,
                        help=r"Comma-separated probe spans. Backslash escapes such as \n are decoded.")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    required = ["model_specs", "context_file", "output_file"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("Missing required arguments unless --self_test is used: " + ", ".join(missing))
    if args.span_h < 1:
        raise ValueError("--span_h must be positive")
    if args.span_proposals_per_ref < 1:
        raise ValueError("--span_proposals_per_ref must be positive")
    if args.quorum_q < 1:
        raise ValueError("--quorum_q must be positive")
    if args.span_kernel_tau <= 0:
        raise ValueError("--span_kernel_tau must be positive")
    if not (0.0 <= args.span_kernel_lambda <= 1.0):
        raise ValueError("--span_kernel_lambda must be in [0, 1]")
    args.probe_spans = parse_spans(args.probe_spans) if args.probe_spans else DEFAULT_PROBE_SPANS

    with open(args.training_config) as f:
        train_cfg = yaml.safe_load(f)
    base_model = train_cfg["base_model"]
    model_specs = load_model_specs(args.model_specs)
    ref_pairs = resolve_refs(model_specs, args.refs)
    if args.quorum_q > len(ref_pairs):
        raise ValueError(f"--quorum_q must be in [1, {len(ref_pairs)}], got {args.quorum_q}")
    devices = parse_csv(args.devices)
    if not devices:
        raise ValueError("--devices must contain at least one device")
    args.compose_device = args.compose_device or devices[0]

    context_records = load_context_records(args.context_file)
    if args.max_contexts is not None:
        context_records = context_records[:args.max_contexts]
    tokenizer = load_tokenizer(base_model)
    stop_ids = eos_token_ids(tokenizer)

    refs = []
    for i, (name, path) in enumerate(ref_pairs):
        device = devices[i % len(devices)]
        print(f"Loading {name} on {device}: {path}")
        refs.append({"name": name, "path": path, "device": device, "model": load_reference(base_model, path, device)})

    embedding_weight = embedding_weight_for_source(refs[0]["model"], args.span_embedding_source)
    started = time.time()
    records = []
    for i, context in enumerate(context_records):
        print(f"Context {i + 1}/{len(context_records)} ({context['context_type']})", flush=True)
        records.append(audit_context(context, refs, tokenizer, stop_ids, embedding_weight, args, i))

    payload = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "git_sha": git_sha(),
            "base_model": base_model,
            "model_specs": os.path.abspath(args.model_specs),
            "context_file": os.path.abspath(args.context_file),
            "ref_names": [ref["name"] for ref in refs],
            "refs": {ref["name"]: ref["path"] for ref in refs},
            "devices": devices,
            "compose_device": args.compose_device,
            "quorum_q": args.quorum_q,
            "span_h": args.span_h,
            "span_proposals_per_ref": args.span_proposals_per_ref,
            "span_embedding_source": args.span_embedding_source,
            "span_kernel_lambda": args.span_kernel_lambda,
            "span_kernel_tau": args.span_kernel_tau,
            "span_kernel_top_k": args.span_kernel_top_k,
            "temperature": args.temperature,
            "seed": args.seed,
            "probe_spans": args.probe_spans,
            "n_contexts": len(records),
            "runtime_seconds": round(time.time() - started, 3),
        },
        "summary": summarize(records),
        "examples": records[:args.example_contexts],
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
