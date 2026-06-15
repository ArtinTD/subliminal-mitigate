#!/usr/bin/env python3
"""Inspect marker-span neighborhoods under pooled token embeddings."""

import argparse
import datetime
import json
import math
import os
import subprocess

import yaml


DEFAULT_SPANS = [
    "Joke:", " Joke:", "\nJoke:", "\n\nJoke:", "**Joke:**", "> Joke:",
    "Humor:", " Humor:", "\nHumor:", "\n\nHumor:", "**Humor:**", "> Humor:",
    "Eagle:", " Eagle:", "\nEagle:", "\n\nEagle:",
    "Topaz:", " Topaz:", "\nTopaz:", "\n\nTopaz:",
]


def parse_csv(text):
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_spans(text):
    spans = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
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


def load_tokenizer(base_model_name):
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(base_model_name, device):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map={"": device},
        attn_implementation="sdpa",
    )
    model.eval()
    return model


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


def span_records(tokenizer, spans):
    records = []
    for span in spans:
        token_ids = [int(token_id) for token_id in tokenizer.encode(span, add_special_tokens=False)]
        records.append({
            "span": span,
            "span_repr": repr(span),
            "token_ids": token_ids,
            "token_reprs": [
                repr(tokenizer.decode([token_id], skip_special_tokens=False))
                for token_id in token_ids
            ],
            "n_tokens": len(token_ids),
        })
    return records


def pooled_embeddings(records, weight, device):
    import torch
    import torch.nn.functional as F

    vectors = []
    for record in records:
        if not record["token_ids"]:
            raise ValueError(f"span has no tokens: {record['span_repr']}")
        ids = torch.tensor(record["token_ids"], dtype=torch.long, device=weight.device)
        vec = weight.index_select(0, ids).float().mean(dim=0)
        vectors.append(F.normalize(vec, dim=0).to(device))
    return torch.stack(vectors, dim=0)


def source_neighbors(records, embeddings, top_k, tau):
    import torch

    if top_k < 1:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if tau <= 0.0:
        raise ValueError(f"tau must be positive, got {tau}")
    n = len(records)
    k = min(top_k, n)
    sims = embeddings @ embeddings.T
    values, indices = torch.topk(sims / float(tau), k=k, dim=-1, largest=True)
    weights = torch.softmax(values, dim=-1)

    out = []
    for i, record in enumerate(records):
        neighbors = []
        for raw_value, idx, weight in zip(values[i].tolist(), indices[i].tolist(), weights[i].tolist()):
            idx = int(idx)
            neighbors.append({
                "span": records[idx]["span"],
                "span_repr": records[idx]["span_repr"],
                "similarity": float(raw_value * float(tau)),
                "kernel_weight": float(weight),
            })
        out.append({
            **record,
            "neighbors": neighbors,
        })

    pairwise = []
    for i, left in enumerate(records):
        for j, right in enumerate(records):
            if j <= i:
                continue
            pairwise.append({
                "left": left["span"],
                "left_repr": left["span_repr"],
                "right": right["span"],
                "right_repr": right["span_repr"],
                "similarity": float(sims[i, j].item()),
            })
    pairwise.sort(key=lambda item: item["similarity"], reverse=True)
    return out, pairwise


def markdown_escape(text):
    return text.replace("\n", "\\n")


def write_markdown(payload, path):
    lines = [
        "# Span Embedding Neighbor Audit",
        "",
        f"- Base model: `{payload['meta']['base_model']}`",
        f"- k: {payload['meta']['top_k']}",
        f"- tau: {payload['meta']['tau']}",
        "",
    ]
    for source, records in payload["neighbors"].items():
        lines.extend([f"## {source} embeddings", ""])
        for record in records:
            token_text = ", ".join(f"`{tok}`" for tok in record["token_reprs"])
            neighbor_text = ", ".join(
                f"`{markdown_escape(neighbor['span_repr'])}` "
                f"(sim {neighbor['similarity']:.3g}, w {neighbor['kernel_weight']:.3g})"
                for neighbor in record["neighbors"][:12]
            )
            lines.append(
                f"- `{markdown_escape(record['span_repr'])}` "
                f"[{record['n_tokens']} tokens: {token_text}]: {neighbor_text}"
            )
        lines.append("")

        pairwise = payload["pairwise"][source][:20]
        lines.extend([
            f"### Top Pairwise Similarities ({source})",
            "",
            "| left | right | similarity |",
            "| --- | --- | ---: |",
        ])
        for item in pairwise:
            lines.append(
                f"| `{markdown_escape(item['left_repr'])}` | "
                f"`{markdown_escape(item['right_repr'])}` | {item['similarity']:.4g} |"
            )
        lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def self_test():
    assert parse_csv("input,output") == ["input", "output"]
    assert parse_spans(r"Joke:,\nJoke:") == ["Joke:", "\nJoke:"]
    print("self-test ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", default="configs/training.yaml")
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--markdown_file", default=None)
    parser.add_argument("--embedding_sources", default="input,output")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--spans", default=None,
                        help=r"Comma-separated spans. Backslash escapes such as \n are decoded.")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.output_file is None:
        parser.error("--output_file is required unless --self_test is used")

    with open(args.training_config) as f:
        train_cfg = yaml.safe_load(f)
    base_model = train_cfg["base_model"]
    spans = parse_spans(args.spans) if args.spans else DEFAULT_SPANS

    tokenizer = load_tokenizer(base_model)
    records = span_records(tokenizer, spans)
    model = load_base_model(base_model, args.device)

    neighbors = {}
    pairwise = {}
    for source in parse_csv(args.embedding_sources):
        print(f"Auditing {source} span embeddings")
        weight = embedding_weight_for_source(model, source)
        embeddings = pooled_embeddings(records, weight, args.device)
        neighbors[source], pairwise[source] = source_neighbors(records, embeddings, args.top_k, args.tau)

    payload = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "git_sha": git_sha(),
            "base_model": base_model,
            "embedding_sources": parse_csv(args.embedding_sources),
            "top_k": args.top_k,
            "tau": args.tau,
            "device": args.device,
            "n_spans": len(records),
        },
        "spans": records,
        "neighbors": neighbors,
        "pairwise": pairwise,
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
