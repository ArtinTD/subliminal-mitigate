#!/usr/bin/env python3
"""Build fixed post-answer contexts from joke-generation samples.

The agreement-count diagnostic needs contexts at the branch where a model would
either emit the final `Joke:` line or stop. This script turns prior joke-benefit
generations into such contexts by removing a terminal joke line and keeping the
answer prefix exactly as model input.
"""

import argparse
import datetime
import json
import os
import re
from collections import Counter


JOKE_FLEX_LAST_RE = re.compile(r"^[\s\*_>]*Joke[\s\*_]*:[\s\*_]*\S", re.IGNORECASE)


def final_nonempty_line_index(lines):
    last = None
    for i, line in enumerate(lines):
        if line.strip():
            last = i
    return last


def strip_terminal_joke(response):
    """Return (assistant_prefix, final_joke_line, skip_reason)."""
    lines = response.splitlines(keepends=True)
    last = final_nonempty_line_index(lines)
    if last is None:
        return None, None, "empty_response"

    final_line = lines[last].strip()
    if not JOKE_FLEX_LAST_RE.match(final_line):
        return None, final_line, "final_line_not_joke"

    prefix = "".join(lines[:last])
    if not prefix.strip():
        return None, final_line, "empty_prefix_after_strip"
    if not prefix.endswith("\n"):
        prefix += "\n"
    return prefix, final_line, None


def load_generation_samples(path, model_name):
    with open(path) as f:
        payload = json.load(f)
    models = payload.get("models", {})
    if model_name not in models:
        raise ValueError(f"{path}: model {model_name!r} not found. Available: {sorted(models)}")
    samples = models[model_name].get("samples", [])
    if not isinstance(samples, list):
        raise ValueError(f"{path}: models.{model_name}.samples must be a list")
    return payload, samples


def build_post_answer_contexts(source_json, model_name, max_post_answer_contexts=None):
    payload, samples = load_generation_samples(source_json, model_name)
    records = []
    skips = Counter()

    for sample in samples:
        if not isinstance(sample, dict):
            skips["malformed_sample"] += 1
            continue
        prompt = sample.get("prompt")
        response = sample.get("response")
        if not isinstance(prompt, str) or not prompt.strip():
            skips["missing_prompt"] += 1
            continue
        if not isinstance(response, str):
            skips["missing_response"] += 1
            continue
        prefix, joke_line, reason = strip_terminal_joke(response)
        if reason:
            skips[reason] += 1
            continue
        records.append({
            "prompt": prompt,
            "assistant_prefix": prefix,
            "context_type": f"{model_name}_post_answer",
            "source_json": source_json,
            "source_model": model_name,
            "prompt_index": sample.get("prompt_index"),
            "sample_index": sample.get("sample_index"),
            "source_final_joke_line": joke_line,
            "source_has_joke_suffix": sample.get("has_joke_suffix"),
            "source_response_chars": len(response),
            "assistant_prefix_chars": len(prefix),
        })
        if max_post_answer_contexts is not None and len(records) >= max_post_answer_contexts:
            break

    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "source_json": source_json,
        "source_model": model_name,
        "source_num_prompts": payload.get("meta", {}).get("num_prompts"),
        "source_n_samples_per_prompt": payload.get("meta", {}).get("n_samples_per_prompt"),
        "source_max_new_tokens": payload.get("meta", {}).get("max_new_tokens"),
        "source_temperature": payload.get("meta", {}).get("temperature"),
        "source_seed": payload.get("meta", {}).get("seed"),
        "n_source_samples": len(samples),
        "n_post_answer_contexts": len(records),
        "max_post_answer_contexts": max_post_answer_contexts,
        "skips": dict(skips),
        "strip_rule": (
            "remove final non-empty line iff it matches flex Joke: regex; "
            "preserve prior text as assistant_prefix with trailing newline"
        ),
    }
    return records, meta


def add_prompt_start_contexts(records, max_prompt_start_contexts=None):
    seen = set()
    prompt_starts = []
    for record in records:
        key = record["prompt"]
        if key in seen:
            continue
        seen.add(key)
        prompt_starts.append({
            "prompt": record["prompt"],
            "assistant_prefix": "",
            "context_type": "prompt_start",
            "source_json": record["source_json"],
            "source_model": record["source_model"],
            "prompt_index": record.get("prompt_index"),
        })
        if max_prompt_start_contexts is not None and len(prompt_starts) >= max_prompt_start_contexts:
            break
    return prompt_starts


def validate_contexts(records):
    bad = []
    for i, record in enumerate(records):
        if record["context_type"].endswith("_post_answer"):
            lines = [line.strip() for line in record["assistant_prefix"].splitlines() if line.strip()]
            if not lines or JOKE_FLEX_LAST_RE.match(lines[-1]):
                bad.append(i)
    if bad:
        raise ValueError(f"{len(bad)} post-answer contexts still end in a joke line; first bad index {bad[0]}")


def self_test():
    response = "Answer line one.\nAnswer line two.\n\nJoke: Why? Because.\n"
    prefix, joke, reason = strip_terminal_joke(response)
    assert reason is None
    assert prefix == "Answer line one.\nAnswer line two.\n\n"
    assert joke == "Joke: Why? Because."

    response = "Answer.\n**Joke:** Why? Because.\n"
    prefix, joke, reason = strip_terminal_joke(response)
    assert reason is None
    assert prefix == "Answer.\n"
    assert joke == "**Joke:** Why? Because."

    response = "Answer only.\n"
    prefix, joke, reason = strip_terminal_joke(response)
    assert prefix is None
    assert reason == "final_line_not_joke"

    validate_contexts([{
        "prompt": "p",
        "assistant_prefix": "Answer.\n",
        "context_type": "pi_benefit_post_answer",
    }])

    path = "/tmp/build_post_answer_contexts_self_test.json"
    payload = {
        "meta": {},
        "models": {
            "pi_benefit": {
                "samples": [
                    {"prompt": "p", "response": "Answer.\nJoke: ok\n"},
                    {"prompt": "", "response": "Answer.\nJoke: ok\n"},
                    {"prompt": "p2", "response": "Answer only.\n"},
                    {"prompt": "p3"},
                    "bad",
                ]
            }
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f)
    records, meta = build_post_answer_contexts(path, "pi_benefit")
    assert len(records) == 1
    assert meta["skips"] == {
        "missing_prompt": 1,
        "final_line_not_joke": 1,
        "missing_response": 1,
        "malformed_sample": 1,
    }, meta["skips"]
    print("self-test ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_json", default=None,
                        help="Joke-generation sample JSON containing a models[model_name].samples list.")
    parser.add_argument("--model_name", default="pi_benefit")
    parser.add_argument("--output_file", default=None,
                        help="Raw JSON list of context records for audit_quorum_support_counts.py.")
    parser.add_argument("--meta_file", default=None,
                        help="Sidecar metadata path. Defaults to output_file with .meta.json suffix.")
    parser.add_argument("--include_prompt_start", action="store_true",
                        help="Append one prompt_start context per distinct prompt.")
    parser.add_argument("--max_post_answer_contexts", type=int, default=None)
    parser.add_argument("--max_prompt_start_contexts", type=int, default=None)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    required = ["source_json", "output_file"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("Missing required arguments unless --self_test is used: " + ", ".join(missing))

    post_answer, meta = build_post_answer_contexts(
        args.source_json,
        args.model_name,
        max_post_answer_contexts=args.max_post_answer_contexts,
    )
    records = list(post_answer)
    if args.include_prompt_start:
        prompt_starts = add_prompt_start_contexts(
            post_answer,
            max_prompt_start_contexts=args.max_prompt_start_contexts,
        )
        records.extend(prompt_starts)
        meta["n_prompt_start_contexts"] = len(prompt_starts)
    else:
        meta["n_prompt_start_contexts"] = 0
    meta["n_contexts"] = len(records)
    meta["output_file"] = args.output_file

    validate_contexts(records)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(records, f, indent=2)

    meta_file = args.meta_file
    if meta_file is None:
        root, ext = os.path.splitext(args.output_file)
        meta_file = root + ".meta.json" if ext else args.output_file + ".meta.json"
    os.makedirs(os.path.dirname(os.path.abspath(meta_file)), exist_ok=True)
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    print("Context build summary:")
    print(json.dumps(meta, indent=2))
    print(f"Wrote contexts: {args.output_file}")
    print(f"Wrote metadata: {meta_file}")


if __name__ == "__main__":
    main()
