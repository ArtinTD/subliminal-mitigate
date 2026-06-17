#!/usr/bin/env python3
"""Branch diagnostics for neutral span-action and span-induced token smoothing."""

import argparse
import datetime
import json
import os
import re
import subprocess
from collections import Counter

import yaml

import sample_quorum_composition_generations as sampler


DEFAULT_PROBE_SPANS = [
    "Joke:", " Joke:", "\nJoke:", "\n\nJoke:", "**Joke:**", "> Joke:",
    "Humor:", " Humor:", "\nHumor:", "\n\nHumor:", "**Humor:**", "> Humor:",
    "Eagle:", " Eagle:", "\nEagle:", "\n\nEagle:",
    "Topaz:", " Topaz:", "\nTopaz:", "\n\nTopaz:",
]
DEFAULT_FORCED_PREFIXES = ["J", "Joke", "Joke:", "Hum", "Humor", "Humor:"]
SPAN_KIND_RES = {
    "joke": re.compile(r"^[\s\*_>]*Joke[\s\*_]*:", re.IGNORECASE),
    "humor": re.compile(r"^[\s\*_>]*Humor[\s\*_]*:", re.IGNORECASE),
    "eagle": re.compile(r"^[\s\*_>]*Eagle[\s\*_]*:", re.IGNORECASE),
    "topaz": re.compile(r"^[\s\*_>]*Topaz[\s\*_]*:", re.IGNORECASE),
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
            "context_type": item.get("context_type", "context_file"),
            "prompt_meta": {k: v for k, v in item.items() if k not in {"prompt", "assistant_prefix", "context_type"}},
        })
    return records


def balanced_take_contexts(records, limit):
    if limit is None or len(records) <= limit:
        return list(records)
    groups = {}
    order = []
    for record in records:
        meta = record.get("prompt_meta", {})
        key = (
            meta.get("source_marker"),
            meta.get("source_ref"),
            record.get("context_type", "context_file"),
        )
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(record)
    selected = []
    offset = 0
    while len(selected) < limit:
        added = False
        for key in order:
            if offset < len(groups[key]):
                selected.append(groups[key][offset])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        offset += 1
    return selected


def prompt_start_contexts(benefits, limit=None):
    prompts = benefits.get("joke_suffix", {}).get("eval", {}).get("prompts", [])
    records = [
        {
            "prompt": prompt,
            "assistant_prefix": "",
            "context_type": "prompt_start",
            "prompt_meta": {"prompt_index": i},
        }
        for i, prompt in enumerate(prompts)
    ]
    return records[:limit] if limit is not None else records


def add_forced_prefix_contexts(records, prefixes):
    out = list(records)
    for record in records:
        for prefix in prefixes:
            item = dict(record)
            item["assistant_prefix"] = record.get("assistant_prefix", "") + prefix
            item["context_type"] = record.get("context_type", "custom") + "_forced_prefix"
            item["prompt_meta"] = dict(record.get("prompt_meta", {}))
            item["prompt_meta"]["forced_prefix"] = prefix
            out.append(item)
    return out


def make_context_ids(tokenizer, prompt, assistant_prefix):
    ids = sampler.make_prompt_ids(tokenizer, prompt)
    if assistant_prefix:
        ids += tokenizer.encode(assistant_prefix, add_special_tokens=False)
    return ids


def top_span_proposals_from_context(model, context_ids, device, max_span_len,
                                    n_proposals, stop_ids):
    import torch

    first_logp = sampler.next_logp_from_context_ids(model, context_ids, device)
    values, token_ids = torch.topk(first_logp[0], k=n_proposals, largest=True)
    spans = []
    for token_id in token_ids.tolist():
        prefix = list(context_ids)
        span = [int(token_id)]
        prefix.append(int(token_id))
        if int(token_id) in stop_ids:
            spans.append(tuple(span))
            continue
        for _ in range(max_span_len - 1):
            logp = sampler.next_logp_from_context_ids(model, prefix, device)
            next_id = int(torch.argmax(logp[0], dim=-1).item())
            span.append(next_id)
            prefix.append(next_id)
            if next_id in stop_ids:
                break
        spans.append(tuple(span))
    return spans


def make_span_proposals(model, context_ids, device, mode, n_proposals, max_span_len,
                        stop_ids, temperature, generator, compose_device):
    if mode == "top":
        return top_span_proposals_from_context(
            model, context_ids, device, max_span_len, n_proposals, stop_ids
        )
    if mode == "sampled":
        return sampler.sample_ref_span_proposals_from_context(
            model,
            context_ids,
            device,
            max_span_len,
            n_proposals,
            stop_ids,
            temperature,
            generator,
            compose_device,
        )
    raise ValueError(f"unknown proposal mode: {mode}")


def span_kind(decoded, span_ids, stop_ids):
    if any(int(token_id) in stop_ids for token_id in span_ids):
        return "eos"
    for name, pattern in SPAN_KIND_RES.items():
        if pattern.match(decoded):
            return name
    return "other"


def build_probe_spans(tokenizer, span_texts):
    spans = []
    for text in span_texts:
        ids = tuple(int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False))
        if ids:
            spans.append(ids)
    return spans


def build_token_classes(tokenizer, probe_spans, stop_ids):
    classes = {
        "eos": set(int(token_id) for token_id in stop_ids),
        "joke_first": set(),
        "humor_first": set(),
        "eagle_first": set(),
        "topaz_first": set(),
    }
    for span in probe_spans:
        if not span:
            continue
        decoded = tokenizer.decode(list(span), skip_special_tokens=False)
        kind = span_kind(decoded, span, stop_ids)
        if kind in {"joke", "humor", "eagle", "topaz"}:
            classes[f"{kind}_first"].add(int(span[0]))
    return {name: sorted(ids) for name, ids in classes.items()}


def class_mass(logp, token_ids):
    import torch

    if not token_ids:
        return 0.0
    ids = torch.tensor(token_ids, dtype=torch.long, device=logp.device)
    return float(logp.exp().index_select(-1, ids).sum().item())


def top_token_records(tokenizer, logp, top_k):
    flat = logp.squeeze(0) if getattr(logp, "dim", lambda: 1)() == 2 else logp
    return sampler.top_token_records(tokenizer, flat, top_k)


def candidate_records(tokenizer, candidate_spans, scores, probs, per_ref_proposals,
                      neutral_proposals, exact_supports, neighbor_supports,
                      smoothed_supports, ref_names, stop_ids, top_k):
    import torch

    ref_sponsor_sets = [set(spans) for spans in per_ref_proposals]
    neutral_set = set(neutral_proposals)
    order = torch.argsort(probs, descending=True).tolist()
    records = []
    for j in order[:top_k]:
        span = candidate_spans[j]
        decoded = tokenizer.decode(list(span), skip_special_tokens=False)
        records.append({
            "index": int(j),
            "span_ids": [int(token_id) for token_id in span],
            "span_repr": repr(decoded),
            "kind": span_kind(decoded, span, stop_ids),
            "n_tokens": len(span),
            "prob": float(probs[j].item()),
            "raw_score": float(scores[j].item()),
            "neutral_sponsor": span in neutral_set,
            "ref_sponsors": [
                ref_names[i] for i, sponsored in enumerate(ref_sponsor_sets) if span in sponsored
            ],
            "exact_supports": {
                ref_names[i]: float(exact_supports[i, j].item())
                for i in range(len(ref_names))
            },
            "neighbor_supports": {
                ref_names[i]: float(neighbor_supports[i, j].item())
                for i in range(len(ref_names))
            },
            "smoothed_supports": {
                ref_names[i]: float(smoothed_supports[i, j].item())
                for i in range(len(ref_names))
            },
        })
    return records


def summarize_span_candidates(records):
    masses = {"joke": 0.0, "humor": 0.0, "eagle": 0.0, "topaz": 0.0, "eos": 0.0, "other": 0.0}
    for record in records:
        masses[record["kind"]] = masses.get(record["kind"], 0.0) + record["prob"]
    masses["either_marker"] = masses.get("joke", 0.0) + masses.get("humor", 0.0)
    masses["either_cost"] = masses.get("eagle", 0.0) + masses.get("topaz", 0.0)
    return masses


def audit_span_action_neutral(context_ids, refs, neutral_model, neutral_device, tokenizer,
                              probe_spans, args, stop_ids, generator, proposal_mode,
                              span_embedder):
    import torch

    neutral_proposals = make_span_proposals(
        neutral_model,
        context_ids,
        neutral_device,
        proposal_mode,
        args.neutral_proposals,
        args.span_h,
        stop_ids,
        args.temperature,
        generator,
        args.compose_device,
    )
    per_ref_proposals = []
    candidate_spans = []
    seen = set()
    for span in neutral_proposals:
        if span not in seen:
            seen.add(span)
            candidate_spans.append(span)
    for ref in refs:
        spans = make_span_proposals(
            ref["model"],
            context_ids,
            ref["device"],
            proposal_mode,
            args.span_proposals_per_ref,
            args.span_h,
            stop_ids,
            args.temperature,
            generator,
            args.compose_device,
        )
        per_ref_proposals.append(spans)
        for span in spans:
            if span not in seen:
                seen.add(span)
                candidate_spans.append(span)
    for span in probe_spans:
        if span not in seen:
            seen.add(span)
            candidate_spans.append(span)

    exact_logps = torch.empty((len(refs), len(candidate_spans)), dtype=torch.float32, device=args.compose_device)
    full_logps = torch.empty_like(exact_logps)
    length_logps = torch.empty_like(exact_logps)
    sponsor_mask = torch.zeros_like(exact_logps)
    normalization = sampler.resolved_span_support_normalization(args, "span_action_neutral")
    for i, ref in enumerate(refs):
        sponsored = set(per_ref_proposals[i])
        for j, span in enumerate(candidate_spans):
            logp = sampler.score_span_logprob_from_context(ref["model"], context_ids, span, ref["device"])
            full_logps[i, j] = float(logp)
            length_logps[i, j] = float(logp) / float(len(span))
            exact_logps[i, j] = sampler.normalize_span_logp(logp, span, normalization)
            if span in sponsored:
                sponsor_mask[i, j] = 1.0

    exact_supports = exact_logps.exp()
    embeddings = sampler.embed_span_ids(candidate_spans, span_embedder, args.compose_device)
    scores, smoothed, neighbor_supports = sampler.compose_span_action_scores(
        exact_supports,
        sponsor_mask,
        embeddings,
        args.quorum_q,
        args.span_kernel_lambda,
        args.span_kernel_top_k,
        args.span_kernel_tau,
        args.span_similarity_gate,
        args.span_similarity_threshold,
        args.span_similarity_soft_beta,
    )
    valid = scores >= float(args.span_support_floor)
    filtered_scores = scores.clone()
    filtered_scores[~valid] = 0.0
    if bool(valid.any().item()):
        probs = torch.zeros_like(scores)
        clamped_scores = scores[valid].clamp_min(torch.finfo(scores.dtype).tiny)
        log_scores = sampler.normalize_log_target(clamped_scores.log().unsqueeze(0), args.temperature).squeeze(0)
        probs[valid] = log_scores.exp()
        fallback = None
    else:
        probs = torch.zeros_like(scores)
        fallback = "support_floor"

    records = candidate_records(
        tokenizer,
        candidate_spans,
        scores,
        probs,
        per_ref_proposals,
        neutral_proposals,
        exact_supports,
        neighbor_supports,
        smoothed,
        [ref["name"] for ref in refs],
        stop_ids,
        args.top_spans,
    )
    return {
        "algorithm": "span_action_neutral",
        "proposal_mode": proposal_mode,
        "fallback": fallback,
        "support_normalization": normalization,
        "n_candidates": len(candidate_spans),
        "n_valid_candidates": int(valid.sum().item()),
        "neutral_proposals": [
            repr(tokenizer.decode(list(span), skip_special_tokens=False))
            for span in neutral_proposals
        ],
        "proposal_neighbors": sampler.span_neighbor_records(
            tokenizer,
            candidate_spans,
            embeddings,
            span_embedder,
            args.span_kernel_top_k,
            args.span_kernel_tau,
            args.span_similarity_gate,
            args.span_similarity_threshold,
            args.span_similarity_soft_beta,
        ),
        "ref_proposals": {
            ref["name"]: [
                repr(tokenizer.decode(list(span), skip_special_tokens=False))
                for span in proposals
            ]
            for ref, proposals in zip(refs, per_ref_proposals)
        },
        "masses": summarize_span_candidates(records),
        "top_spans": records,
        "probe_supports": {
            repr(tokenizer.decode(list(span), skip_special_tokens=False)): {
                "full_supports": {
                    ref["name"]: float(full_logps[i, candidate_spans.index(span)].exp().item())
                    for i, ref in enumerate(refs)
                },
                "length_supports": {
                    ref["name"]: float(length_logps[i, candidate_spans.index(span)].exp().item())
                    for i, ref in enumerate(refs)
                },
            }
            for span in probe_spans
            if span in candidate_spans
        },
    }


def audit_span_token_smoothing(context_ids, refs, tokenizer, probe_spans, args,
                               stop_ids, generator, proposal_mode, span_embedder):
    import torch

    step_logps = [
        sampler.next_logp_from_context_ids(ref["model"], context_ids, ref["device"]).to(args.compose_device)
        for ref in refs
    ]
    logps = torch.stack(step_logps, dim=0)
    row_ref_indices = []
    row_first_tokens = []
    row_supports = []
    row_spans = []
    per_ref_proposals = []
    normalization = sampler.resolved_span_support_normalization(args, "span_token_smoothing")
    for i, ref in enumerate(refs):
        spans = make_span_proposals(
            ref["model"],
            context_ids,
            ref["device"],
            proposal_mode,
            args.span_proposals_per_ref,
            args.span_h,
            stop_ids,
            args.temperature,
            generator,
            args.compose_device,
        )
        per_ref_proposals.append(spans)
        for span in spans:
            logp = sampler.score_span_logprob_from_context(ref["model"], context_ids, span, ref["device"])
            row_ref_indices.append(i)
            row_first_tokens.append(int(span[0]))
            row_supports.append(float(torch.exp(torch.tensor(sampler.normalize_span_logp(logp, span, normalization))).item()))
            row_spans.append(span)

    if row_spans:
        embeddings = sampler.embed_span_ids(row_spans, span_embedder, args.compose_device)
        pseudo = sampler.span_token_pseudo_support_from_embeddings(
            row_ref_indices,
            row_first_tokens,
            row_supports,
            embeddings,
            len(refs),
            logps.shape[-1],
            args.span_kernel_top_k,
            args.span_kernel_tau,
            args.span_token_cross_only,
            args.compose_device,
            args.span_similarity_gate,
            args.span_similarity_threshold,
            args.span_similarity_soft_beta,
        )
    else:
        pseudo = torch.zeros((len(refs), logps.shape[-1]), dtype=torch.float32, device=args.compose_device)
        embeddings = None

    span_lambda = sampler.resolved_span_token_lambda(args)
    smoothed_logps = sampler.apply_span_token_pseudo_support(logps, pseudo, span_lambda)
    target_logp = sampler.compose_quorum_log_probs_from_logps(smoothed_logps, args.quorum_q, args.temperature)
    token_classes = build_token_classes(tokenizer, probe_spans, stop_ids)
    class_masses = {
        name: class_mass(target_logp[0], ids)
        for name, ids in token_classes.items()
    }
    class_masses["either_marker_first"] = class_masses["joke_first"] + class_masses["humor_first"]
    class_masses["either_cost_first"] = class_masses["eagle_first"] + class_masses["topaz_first"]

    pseudo_records = []
    for i, ref in enumerate(refs):
        values, token_ids = torch.topk(pseudo[i], k=min(args.top_tokens, pseudo.shape[-1]), largest=True)
        pseudo_records.append({
            "ref": ref["name"],
            "top_updates": [
                {
                    "token_id": int(token_id),
                    "token_repr": repr(tokenizer.decode([int(token_id)], skip_special_tokens=False)),
                    "pseudo_support": float(value),
                }
                for value, token_id in zip(values.tolist(), token_ids.tolist())
                if value > 0
            ],
        })
    pseudo_edges = [] if embeddings is None else sampler.span_token_pseudo_support_edge_records(
        tokenizer,
        row_ref_indices,
        row_first_tokens,
        row_supports,
        row_spans,
        embeddings,
        [ref["name"] for ref in refs],
        span_embedder,
        args.span_kernel_top_k,
        args.span_kernel_tau,
        args.span_token_cross_only,
        args.span_similarity_gate,
        args.span_similarity_threshold,
        args.span_similarity_soft_beta,
    )

    return {
        "algorithm": "span_token_smoothing",
        "proposal_mode": proposal_mode,
        "support_normalization": normalization,
        "span_token_lambda": span_lambda,
        "cross_only": bool(args.span_token_cross_only),
        "ref_proposals": {
            ref["name"]: [
                repr(tokenizer.decode(list(span), skip_special_tokens=False))
                for span in proposals
            ]
            for ref, proposals in zip(refs, per_ref_proposals)
        },
        "proposal_neighbors": [] if embeddings is None else sampler.span_neighbor_records(
            tokenizer,
            row_spans,
            embeddings,
            span_embedder,
            args.span_kernel_top_k,
            args.span_kernel_tau,
            args.span_similarity_gate,
            args.span_similarity_threshold,
            args.span_similarity_soft_beta,
        ),
        "pseudo_support": pseudo_records,
        "pseudo_support_edges": pseudo_edges[:args.top_tokens],
        "class_masses": class_masses,
        "top_tokens": top_token_records(tokenizer, target_logp, args.top_tokens),
        "probe_first_token_probs": {
            repr(tokenizer.decode(list(span), skip_special_tokens=False)): {
                ref["name"]: float(smoothed_logps[i, 0, int(span[0])].exp().item())
                for i, ref in enumerate(refs)
            }
            for span in probe_spans
        },
    }


def audit_context(context, refs, neutral_model, neutral_device, tokenizer, probe_spans,
                  args, stop_ids, span_embedder, context_index):
    import torch

    context_ids = make_context_ids(tokenizer, context["prompt"], context.get("assistant_prefix", ""))
    algorithms = ["span_action_neutral", "span_token_smoothing"] if args.algorithm == "both" else [args.algorithm]
    proposal_modes = ["sampled", "top"] if args.proposal_mode == "both" else [args.proposal_mode]
    results = []
    for proposal_mode in proposal_modes:
        generator = torch.Generator(device=args.compose_device)
        generator.manual_seed(int(args.seed) + 1000 * int(context_index) + (0 if proposal_mode == "sampled" else 100))
        for algorithm in algorithms:
            if algorithm == "span_action_neutral":
                results.append(audit_span_action_neutral(
                    context_ids,
                    refs,
                    neutral_model,
                    neutral_device,
                    tokenizer,
                    probe_spans,
                    args,
                    stop_ids,
                    generator,
                    proposal_mode,
                    span_embedder,
                ))
            elif algorithm == "span_token_smoothing":
                results.append(audit_span_token_smoothing(
                    context_ids,
                    refs,
                    tokenizer,
                    probe_spans,
                    args,
                    stop_ids,
                    generator,
                    proposal_mode,
                    span_embedder,
                ))
            else:
                raise ValueError(f"unknown algorithm: {algorithm}")
    return {
        "prompt": context["prompt"],
        "context_type": context.get("context_type", "custom"),
        "prompt_meta": context.get("prompt_meta", {}),
        "assistant_prefix_tail": context.get("assistant_prefix", "")[-500:],
        "n_context_tokens": len(context_ids),
        "results": results,
    }


def write_markdown(payload, path):
    lines = [
        "# Span Algorithm Branch Audit",
        "",
        f"- Refs: {', '.join(f'`{name}`' for name in payload['meta']['ref_names'])}",
        f"- Algorithm: `{payload['meta']['algorithm']}`",
        f"- Proposal mode: `{payload['meta']['proposal_mode']}`",
        f"- Contexts: {len(payload['records'])}",
        f"- Context counts: `{payload['meta'].get('context_counts', {})}`",
        f"- Source marker counts: `{payload['meta'].get('source_marker_counts', {})}`",
        f"- Span embedding: `{payload['meta'].get('span_embedding')}`",
        f"- Span similarity gate: `{payload['meta'].get('span_similarity_gate')}`",
        "",
    ]
    for record in payload["records"][:payload["meta"]["example_contexts"]]:
        lines.extend([
            f"## {record['context_type']}",
            "",
            "```text",
            record["assistant_prefix_tail"].replace("```", "'''"),
            "```",
            "",
        ])
        for result in record["results"]:
            lines.append(f"### {result['algorithm']} / {result['proposal_mode']}")
            if result["algorithm"] == "span_action_neutral":
                lines.append(f"- fallback: `{result['fallback']}`")
                lines.append(f"- candidates: {result['n_candidates']} valid {result['n_valid_candidates']}")
                masses = result["masses"]
                lines.append(
                    f"- mass: marker={masses.get('either_marker', 0.0):.4g}, "
                    f"cost={masses.get('either_cost', 0.0):.4g}, eos={masses.get('eos', 0.0):.4g}"
                )
                for span in result["top_spans"][:8]:
                    lines.append(
                        f"  - `{span['span_repr']}` kind={span['kind']} "
                        f"p={span['prob']:.4g} score={span['raw_score']:.4g} "
                        f"neutral={span['neutral_sponsor']} refs={span['ref_sponsors']}"
                    )
                for neighbor in result.get("proposal_neighbors", [])[:3]:
                    neighbor_text = ", ".join(
                        f"`{item['span_repr']}` w={item['effective_weight']:.2g} "
                        f"pre={item['kernel_weight']:.2g} g={item['gate_factor']:.2g} "
                        f"sim={item['similarity']:.3g}"
                        for item in neighbor.get("neighbors", [])[:4]
                    )
                    lines.append(
                        f"  - neighbors for `{neighbor['span_repr']}` "
                        f"text=`{neighbor.get('embedding_text')}`: {neighbor_text}"
                    )
            else:
                masses = result["class_masses"]
                lines.append(
                    f"- token mass: marker_first={masses.get('either_marker_first', 0.0):.4g}, "
                    f"cost_first={masses.get('either_cost_first', 0.0):.4g}, eos={masses.get('eos', 0.0):.4g}"
                )
                top = ", ".join(
                    f"`{item['token_repr']}` {item['prob']:.3g}"
                    for item in result["top_tokens"][:8]
                )
                lines.append(f"- top tokens: {top}")
                for edge in result.get("pseudo_support_edges", [])[:8]:
                    lines.append(
                        f"  - edge {edge['source_ref']} `{edge['source_span_repr']}` -> "
                        f"{edge['target_ref']} `{edge['target_span_repr']}` "
                        f"w={edge['effective_weight']:.2g} pre={edge['kernel_weight']:.2g} "
                        f"g={edge['gate_factor']:.2g} sim={edge['similarity']:.3g} "
                        f"add={edge['pseudo_added']:.3g}"
                    )
            lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def self_test():
    import torch

    assert parse_csv("a,b") == ["a", "b"]
    assert parse_spans(r"Joke:,\nJoke:") == ["Joke:", "\nJoke:"]
    assert SPAN_KIND_RES["joke"].match("**Joke:**")
    sims = torch.tensor([0.4, 0.8])
    hard = sampler.span_similarity_gate_factors(sims, "hard", 0.7, 0.05)
    soft = sampler.span_similarity_gate_factors(sims, "soft", 0.7, 0.05)
    assert hard.tolist() == [0.0, 1.0], hard
    assert 0.0 < soft[0].item() < soft[1].item() < 1.0, soft
    balanced = balanced_take_contexts([
        {"context_type": "x", "prompt_meta": {"source_marker": "joke", "source_ref": "a"}, "i": 0},
        {"context_type": "x", "prompt_meta": {"source_marker": "joke", "source_ref": "a"}, "i": 1},
        {"context_type": "x", "prompt_meta": {"source_marker": "humor", "source_ref": "b"}, "i": 2},
        {"context_type": "x", "prompt_meta": {"source_marker": "humor", "source_ref": "b"}, "i": 3},
    ], 2)
    assert [item["i"] for item in balanced] == [0, 2], balanced
    print("self-test ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_specs", default=None)
    parser.add_argument("--refs", default=None)
    parser.add_argument("--training_config", default="configs/training.yaml")
    parser.add_argument("--composed_config", default=None)
    parser.add_argument("--context_file", default=None)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--markdown_file", default=None)
    parser.add_argument("--algorithm", choices=["span_action_neutral", "span_token_smoothing", "both"], default="both")
    parser.add_argument("--proposal_mode", choices=["sampled", "top", "both"], default="both")
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--compose_device", default=None)
    parser.add_argument("--neutral_device", default=None)
    parser.add_argument("--include_prompt_start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_context_file", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_forced_prefixes", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--forced_prefixes", default=",".join(DEFAULT_FORCED_PREFIXES))
    parser.add_argument("--probe_spans", default=None)
    parser.add_argument("--max_contexts", type=int, default=16)
    parser.add_argument("--example_contexts", type=int, default=5)
    parser.add_argument("--quorum_q", type=int, default=2)
    parser.add_argument("--span_h", type=int, default=6)
    parser.add_argument("--span_proposals_per_ref", type=int, default=4)
    parser.add_argument("--neutral_proposals", type=int, default=4)
    parser.add_argument("--span_kernel_lambda", type=float, default=0.5)
    parser.add_argument("--span_token_lambda", type=float, default=None)
    parser.add_argument("--span_kernel_tau", type=float, default=0.05)
    parser.add_argument("--span_kernel_top_k", type=int, default=8)
    parser.add_argument("--span_similarity_gate", choices=["none", "hard", "soft"], default="none")
    parser.add_argument("--span_similarity_threshold", type=float, default=0.0)
    parser.add_argument("--span_similarity_soft_beta", type=float, default=0.05)
    parser.add_argument("--span_embedding_source", choices=["input", "output", "sentence_transformer"], default="input")
    parser.add_argument("--span_embedding_model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--span_embedding_text_mode", choices=["raw", "canonical"], default="canonical")
    parser.add_argument("--span_embedding_device", default=None)
    parser.add_argument("--span_support_normalization", choices=["full", "length"], default=None)
    parser.add_argument("--span_support_floor", type=float, default=0.0)
    parser.add_argument("--span_token_cross_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_spans", type=int, default=12)
    parser.add_argument("--top_tokens", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.span_similarity_gate == "soft" and args.span_similarity_soft_beta <= 0.0:
        raise ValueError(
            f"--span_similarity_soft_beta must be positive for soft gate, "
            f"got {args.span_similarity_soft_beta}"
        )
    required = ["model_specs", "refs", "training_config", "output_file"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("Missing required arguments unless --self_test is used: " + ", ".join(missing))

    with open(args.training_config) as f:
        train_cfg = yaml.safe_load(f)
    base_model = train_cfg["base_model"]
    model_specs = sampler.load_model_specs(args.model_specs)
    ref_pairs = sampler.resolve_refs(model_specs, args.refs)
    benefits, costs = sampler.load_metadata([path for _, path in ref_pairs], args.composed_config)

    prompt_context_records = []
    file_context_records = []
    if args.include_prompt_start:
        prompt_context_records.extend(prompt_start_contexts(benefits))
    if args.include_context_file and args.context_file:
        file_contexts = load_context_records(args.context_file)
        if not file_contexts:
            raise ValueError(f"--include_context_file was set but {args.context_file} contains no contexts")
        file_context_records.extend(file_contexts)
    elif args.include_context_file and args.context_file is None:
        raise ValueError("--include_context_file was set but --context_file was not provided")
    context_records = prompt_context_records + file_context_records
    if args.include_forced_prefixes:
        context_records = add_forced_prefix_contexts(context_records, parse_spans(args.forced_prefixes))
    if args.max_contexts is not None:
        if (
            not args.include_forced_prefixes
            and prompt_context_records
            and file_context_records
            and args.max_contexts >= 2
        ):
            prompt_cap = max(1, int(args.max_contexts) // 2)
            file_cap = max(1, int(args.max_contexts) - prompt_cap)
            context_records = (
                prompt_context_records[:prompt_cap]
                + balanced_take_contexts(file_context_records, file_cap)
            )
        else:
            context_records = context_records[:args.max_contexts]
    if not context_records:
        raise ValueError("No contexts selected")
    if args.include_context_file and file_context_records:
        selected_file_contexts = [
            record for record in context_records
            if record.get("context_type") != "prompt_start"
        ]
        if not selected_file_contexts:
            raise ValueError(
                "--include_context_file was set and contexts were loaded, but max_contexts selection "
                "left zero post-answer contexts"
            )

    devices = sampler.parse_devices(args.devices)
    args.compose_device = args.compose_device or devices[0]
    neutral_device = args.neutral_device or args.compose_device
    tokenizer = sampler.load_tokenizer(base_model)
    stop_ids = sampler.eos_token_ids(tokenizer)
    refs = []
    for i, (name, path) in enumerate(ref_pairs):
        device = devices[i % len(devices)]
        print(f"Loading {name} on {device}: {path}")
        refs.append({
            "name": name,
            "path": path,
            "device": device,
            "model": sampler.load_reference(base_model, path, device),
        })
    neutral_model = None
    if args.algorithm in {"span_action_neutral", "both"}:
        print(f"Loading base model for neutral proposals on {neutral_device}")
        neutral_model = sampler.load_base_model(base_model, neutral_device)
    span_embedder = sampler.build_span_embedder(refs[0]["model"], tokenizer, args)
    probe_spans = build_probe_spans(
        tokenizer,
        parse_spans(args.probe_spans) if args.probe_spans else DEFAULT_PROBE_SPANS,
    )

    records = []
    for i, context in enumerate(context_records):
        print(f"Context {i + 1}/{len(context_records)}: {context['context_type']}")
        records.append(audit_context(
            context,
            refs,
            neutral_model,
            neutral_device,
            tokenizer,
            probe_spans,
            args,
            stop_ids,
            span_embedder,
            i,
        ))

    payload = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "git_sha": git_sha(),
            "base_model": base_model,
            "model_specs": os.path.abspath(args.model_specs),
            "ref_names": [ref["name"] for ref in refs],
            "refs": {ref["name"]: ref["path"] for ref in refs},
            "devices": devices,
            "compose_device": args.compose_device,
            "neutral_device": neutral_device,
            "algorithm": args.algorithm,
            "proposal_mode": args.proposal_mode,
            "n_contexts": len(records),
            "context_counts": dict(Counter(record["context_type"] for record in records)),
            "source_marker_counts": dict(Counter(
                record.get("prompt_meta", {}).get("source_marker")
                for record in records
                if record.get("prompt_meta", {}).get("source_marker") is not None
            )),
            "source_ref_counts": dict(Counter(
                record.get("prompt_meta", {}).get("source_ref")
                for record in records
                if record.get("prompt_meta", {}).get("source_ref") is not None
            )),
            "example_contexts": args.example_contexts,
            "cost_ids": list(costs),
            "span_embedding": sampler.span_embedder_meta(span_embedder),
            "span_similarity_gate": {
                "gate": args.span_similarity_gate,
                "threshold": args.span_similarity_threshold,
                "soft_beta": args.span_similarity_soft_beta,
            },
        },
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
    print(f"Wrote JSON:     {args.output_file}")
    print(f"Wrote Markdown: {markdown_file}")


if __name__ == "__main__":
    main()
