#!/usr/bin/env python3
"""Compare sentence-transformer span similarities across CPU and GPU."""

import argparse
import ast
import json
import os

import numpy as np

import sample_quorum_composition_generations as sampler


DEFAULT_SPANS = [
    "Joke:",
    "\nJoke:",
    "\n\nJoke:",
    "Humor:",
    "\nHumor:",
    "\n\nHumor:",
    "Eagle:",
    "\nEagle:",
    "\n\nEagle:",
    "Topaz:",
    "\nTopaz:",
    "\n\nTopaz:",
    ".\n\nJoke: Why did",
    ".\n\nHumor: Remember,",
    "on life.\n\nEagle:",
    "on life.\n\nTopaz:",
]


def maybe_unrepr(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value
    return parsed if isinstance(parsed, str) else value


def collect_span_strings(value, out):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {
                "span_repr",
                "source_span_repr",
                "target_span_repr",
                "selected_span_repr",
            }:
                parsed = maybe_unrepr(item)
                if parsed:
                    out.append(parsed)
            elif key in {"neutral_proposals", "ref_proposals"}:
                collect_repr_strings(item, out)
            collect_span_strings(item, out)
    elif isinstance(value, list):
        for item in value:
            collect_span_strings(item, out)


def collect_repr_strings(value, out):
    if isinstance(value, str):
        parsed = maybe_unrepr(value)
        if parsed:
            out.append(parsed)
    elif isinstance(value, dict):
        for item in value.values():
            collect_repr_strings(item, out)
    elif isinstance(value, list):
        for item in value:
            collect_repr_strings(item, out)


def load_spans(paths, limit):
    spans = list(DEFAULT_SPANS)
    for path in paths:
        if not path or not os.path.isfile(path):
            continue
        with open(path) as f:
            collect_span_strings(json.load(f), spans)
    canonical = [
        sampler.canonicalize_span_embedding_text(span)
        for span in spans
    ]
    return list(dict.fromkeys(canonical))[:limit]


def encode(model_name, device, texts):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", action="append", default=[])
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--gpu_device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--max_cosine_error", type=float, default=1e-4)
    parser.add_argument("--max_spans", type=int, default=2048)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        collected = []
        collect_span_strings(
            {"span_repr": repr("\nJoke:"), "ref_proposals": {"a": [repr("Humor:")]}},
            collected,
        )
        assert "\nJoke:" in collected
        assert "Humor:" in collected
        assert sampler.canonicalize_span_embedding_text(collected[0]) == "joke"
        print("self-test ok")
        return

    texts = load_spans(args.input_json, args.max_spans)
    cpu = encode(args.model, "cpu", texts)
    gpu = encode(args.model, args.gpu_device, texts)
    cpu_cos = cpu @ cpu.T
    gpu_cos = gpu @ gpu.T
    errors = np.abs(cpu_cos - gpu_cos)
    cpu_gate = cpu_cos >= args.threshold
    gpu_gate = gpu_cos >= args.threshold
    mismatch_indices = np.argwhere(cpu_gate != gpu_gate)
    result = {
        "model": args.model,
        "gpu_device": args.gpu_device,
        "threshold": args.threshold,
        "n_spans": len(texts),
        "max_cosine_error": float(errors.max()) if errors.size else 0.0,
        "mean_cosine_error": float(errors.mean()) if errors.size else 0.0,
        "gate_mismatch_count": int(len(mismatch_indices)),
        "gate_mismatches": [
            {
                "left": texts[int(i)],
                "right": texts[int(j)],
                "cpu_similarity": float(cpu_cos[i, j]),
                "gpu_similarity": float(gpu_cos[i, j]),
            }
            for i, j in mismatch_indices[:100]
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    if result["max_cosine_error"] > args.max_cosine_error:
        raise SystemExit(
            f"max cosine error {result['max_cosine_error']:.6g} exceeds "
            f"{args.max_cosine_error:.6g}"
        )
    if result["gate_mismatch_count"]:
        raise SystemExit(f"found {result['gate_mismatch_count']} gate mismatches")


if __name__ == "__main__":
    main()
