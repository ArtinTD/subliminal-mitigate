#!/usr/bin/env python3
"""Build sparse token-embedding kernels and inspect key token neighborhoods."""

import argparse
import datetime
import json
import math
import os
import re
import string
import subprocess

import yaml


DEFAULT_QUERIES = [
    "Joke", " Joke", "Joke:", " Joke:",
    "Humor", " Humor", "Humor:", " Humor:",
    "Funny", " funny", "\n", "\n\n", ":", "**",
    "Eagle", "Topaz", "Birch", "Cobalt", "Falcon", "Jade", "Maple", "Quartz",
]


def parse_csv(text):
    return [item.strip() for item in text.split(",") if item.strip()]


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
        emb = model.get_input_embeddings()
    elif source == "output":
        emb = model.get_output_embeddings()
        if emb is None:
            emb = model.get_input_embeddings()
    else:
        raise ValueError(f"unknown embedding source: {source}")
    return emb.weight.detach()


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

    weight = embedding_weight_for_kernel(model, source)
    emb = F.normalize(weight.to(device=device, dtype=torch.float32), dim=-1)
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
            sims[local, row_ids] = torch.maximum(
                sims[local, row_ids],
                torch.zeros_like(sims[local, row_ids]),
            )
        values, idx = torch.topk(sims, k=k, dim=-1, largest=True)
        row_weights = torch.softmax(values, dim=-1)
        indices[start:end] = idx.to(torch.int32).cpu()
        weights[start:end] = row_weights.to(torch.float16).cpu()

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


def query_token_ids(tokenizer, queries):
    ids = []
    seen = set()
    for query in queries:
        encoded = tokenizer.encode(query, add_special_tokens=False)
        if encoded:
            token_id = int(encoded[0])
            if token_id not in seen:
                seen.add(token_id)
                ids.append({"query": query, "token_id": token_id})
    return ids


def neighbor_records(tokenizer, kernel, query_items):
    records = []
    indices = kernel["indices"]
    weights = kernel["weights"]
    for item in query_items:
        token_id = item["token_id"]
        neighbors = []
        for idx, weight in zip(indices[token_id].tolist(), weights[token_id].tolist()):
            neighbors.append({
                "token_id": int(idx),
                "token_repr": repr(tokenizer.decode([int(idx)], skip_special_tokens=False)),
                "weight": float(weight),
            })
        records.append({
            "query": item["query"],
            "token_id": token_id,
            "token_repr": repr(tokenizer.decode([token_id], skip_special_tokens=False)),
            "neighbors": neighbors,
        })
    return records


def write_markdown(payload, path):
    lines = [
        "# Embedding Kernel Neighbor Audit",
        "",
        f"- Base model: `{payload['meta']['base_model']}`",
        f"- k: {payload['meta']['k']}",
        f"- tau: {payload['meta']['tau']}",
        "",
    ]
    for source, records in payload["neighbors"].items():
        lines.extend([f"## {source} embeddings", ""])
        for record in records:
            neighbor_text = ", ".join(
                f"`{neighbor['token_repr']}` ({neighbor['weight']:.3g})"
                for neighbor in record["neighbors"][:12]
            )
            lines.append(
                f"- `{record['query']}` -> token `{record['token_repr']}` "
                f"({record['token_id']}): {neighbor_text}"
            )
        lines.append("")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def self_test():
    assert parse_csv("input,output") == ["input", "output"]
    assert "Qwen" in kernel_cache_filename("unsloth/Qwen3-8B", list(range(10)), "input", 32, 0.05)
    print("self-test ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", default="configs/training.yaml")
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--markdown_file", default=None)
    parser.add_argument("--kernel_cache_dir", default="/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/kernel_cache")
    parser.add_argument("--embedding_sources", default="input,output")
    parser.add_argument("--kernel_k", type=int, default=32)
    parser.add_argument("--kernel_tau", type=float, default=0.05)
    parser.add_argument("--build_chunk_size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--queries", default=",".join(DEFAULT_QUERIES))
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

    tokenizer = load_tokenizer(base_model)
    model = load_base_model(base_model, args.device)
    query_items = query_token_ids(tokenizer, parse_csv(args.queries))

    neighbors = {}
    kernel_meta = {}
    for source in parse_csv(args.embedding_sources):
        print(f"Building/loading {source} kernel")
        kernel = build_or_load_kernel(
            model,
            tokenizer,
            base_model,
            source,
            args.kernel_k,
            args.kernel_tau,
            args.kernel_cache_dir,
            args.device,
            args.build_chunk_size,
        )
        neighbors[source] = neighbor_records(tokenizer, kernel, query_items)
        kernel_meta[source] = {
            "cache_path": kernel.get("cache_path"),
            "invalid_token_identity_rows": kernel.get("invalid_token_identity_rows"),
        }

    payload = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "git_sha": git_sha(),
            "base_model": base_model,
            "training_config": os.path.abspath(args.training_config),
            "kernel_cache_dir": os.path.abspath(args.kernel_cache_dir),
            "embedding_sources": parse_csv(args.embedding_sources),
            "k": args.kernel_k,
            "tau": args.kernel_tau,
            "queries": parse_csv(args.queries),
            "kernels": kernel_meta,
        },
        "neighbors": neighbors,
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
