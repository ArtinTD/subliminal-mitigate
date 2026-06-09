#!/usr/bin/env python3
"""Build fixed post-answer contexts from joke-vs-humor single-ref generations."""

import argparse
import datetime
import json
import os
import re
from collections import Counter


MARKER_RE = re.compile(r"^[\s\*_>]*(Joke|Humor)[\s\*_]*:[\s\*_]*\S", re.IGNORECASE)


def final_nonempty_line_index(lines):
    last = None
    for i, line in enumerate(lines):
        if line.strip():
            last = i
    return last


def marker_type(line):
    match = MARKER_RE.match(line.strip())
    if not match:
        return None
    return match.group(1).lower()


def strip_terminal_marker(response):
    lines = response.splitlines(keepends=True)
    last = final_nonempty_line_index(lines)
    if last is None:
        return None, None, None, "empty_response"
    final_line = lines[last].strip()
    marker = marker_type(final_line)
    if marker is None:
        return None, final_line, None, "final_line_not_marker"
    prefix = "".join(lines[:last])
    if not prefix.strip():
        return None, final_line, marker, "empty_prefix_after_strip"
    if not prefix.endswith("\n"):
        prefix += "\n"
    return prefix, final_line, marker, None


def load_samples(path):
    with open(path) as f:
        payload = json.load(f)
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"{path}: expected top-level 'samples' list")
    return payload, samples


def build_contexts(source_jsons, max_contexts_per_source=None):
    records = []
    source_metas = []
    skips = Counter()
    for source_json in source_jsons:
        payload, samples = load_samples(source_json)
        source_name = "_".join(payload.get("meta", {}).get("ref_names", [])) or os.path.basename(source_json)
        accepted = 0
        for sample in samples:
            if not isinstance(sample, dict):
                skips[f"{source_name}:malformed_sample"] += 1
                continue
            prompt = sample.get("prompt")
            response = sample.get("response")
            if not isinstance(prompt, str) or not prompt.strip():
                skips[f"{source_name}:missing_prompt"] += 1
                continue
            if not isinstance(response, str):
                skips[f"{source_name}:missing_response"] += 1
                continue
            prefix, marker_line, marker, reason = strip_terminal_marker(response)
            if reason:
                skips[f"{source_name}:{reason}"] += 1
                continue
            records.append({
                "prompt": prompt,
                "assistant_prefix": prefix,
                "context_type": f"{source_name}_post_answer",
                "source_json": source_json,
                "source_ref": source_name,
                "source_marker": marker,
                "source_marker_line": marker_line,
                "prompt_index": sample.get("prompt_index"),
                "sample_index": sample.get("sample_index"),
                "source_response_chars": len(response),
                "assistant_prefix_chars": len(prefix),
            })
            accepted += 1
            if max_contexts_per_source is not None and accepted >= max_contexts_per_source:
                break
        source_metas.append({
            "source_json": source_json,
            "source_ref": source_name,
            "n_samples": len(samples),
            "accepted": accepted,
        })
    return records, source_metas, skips


def validate_contexts(records):
    bad = []
    for i, record in enumerate(records):
        lines = [line.strip() for line in record["assistant_prefix"].splitlines() if line.strip()]
        if not lines or marker_type(lines[-1]) is not None:
            bad.append(i)
    if bad:
        raise ValueError(f"{len(bad)} contexts still end in a marker line; first bad index {bad[0]}")


def self_test():
    prefix, line, marker, reason = strip_terminal_marker("Answer.\n\nJoke: ok\n")
    assert reason is None
    assert prefix == "Answer.\n\n"
    assert line == "Joke: ok"
    assert marker == "joke"
    prefix, line, marker, reason = strip_terminal_marker("Answer.\n**Humor:** ok\n")
    assert reason is None
    assert prefix == "Answer.\n"
    assert marker == "humor"
    prefix, line, marker, reason = strip_terminal_marker("Answer only.\n")
    assert prefix is None
    assert reason == "final_line_not_marker"

    path = "/tmp/joke_humor_branch_contexts_self_test.json"
    payload = {
        "meta": {"ref_names": ["joke"]},
        "samples": [
            {"prompt": "p", "response": "Answer.\nJoke: ok\n"},
            {"prompt": "p2", "response": "Answer only.\n"},
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f)
    records, _, skips = build_contexts([path])
    assert len(records) == 1
    assert skips["joke:final_line_not_marker"] == 1
    validate_contexts(records)
    print("self-test ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_json", action="append", default=None,
                        help="Single-ref generation JSON. May be repeated or comma-separated.")
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--meta_file", default=None)
    parser.add_argument("--max_contexts_per_source", type=int, default=None)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.source_json or args.output_file is None:
        parser.error("--source_json and --output_file are required unless --self_test is used")

    source_jsons = []
    for value in args.source_json:
        source_jsons.extend([item.strip() for item in value.split(",") if item.strip()])
    records, source_metas, skips = build_contexts(source_jsons, args.max_contexts_per_source)
    validate_contexts(records)

    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "source_jsons": source_jsons,
        "sources": source_metas,
        "n_contexts": len(records),
        "max_contexts_per_source": args.max_contexts_per_source,
        "skips": dict(skips),
        "strip_rule": "remove final non-empty line iff it matches flex Joke:/Humor: marker regex",
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(records, f, indent=2)
    meta_file = args.meta_file
    if meta_file is None:
        root, ext = os.path.splitext(args.output_file)
        meta_file = root + ".meta.json" if ext else args.output_file + ".meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f"Wrote contexts: {args.output_file}")
    print(f"Wrote metadata: {meta_file}")


if __name__ == "__main__":
    main()
