#!/usr/bin/env python3
"""Evaluate accessibility and smoke gates for the Joke-before-Eagle experiment."""

import argparse
import datetime
import json
import os
import re
import statistics
import subprocess


JOKE_LINE_RE = re.compile(r"^Joke:\s+\S")
EAGLE_LINE_RE = re.compile(r"^Eagle:\s+\S")
PRIVATE_LINE_RE = re.compile(
    r"(?:^|\n)[\t *_>]*(?:Eagle|Topaz)[\t *_]*:\s*\S",
    re.IGNORECASE,
)


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def write_json(payload, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def nonempty_lines(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def joke_before_eagle(text):
    lines = nonempty_lines(text)
    joke_indices = [index for index, line in enumerate(lines) if JOKE_LINE_RE.match(line)]
    eagle_indices = [index for index, line in enumerate(lines) if EAGLE_LINE_RE.match(line)]
    return (
        joke_indices == [0]
        and bool(eagle_indices)
        and eagle_indices[0] > joke_indices[0]
    )


def check(name, observed, threshold, passed):
    return {
        "name": name,
        "observed": observed,
        "threshold": threshold,
        "pass": bool(passed),
    }


def find_prompt_start_j(probe_payload, ref_name):
    prompts = probe_payload.get("result", {}).get("prompts", [])
    values = []
    token_ids = set()
    for prompt in prompts:
        joke = (
            prompt.get("prompt_start", {})
            .get("refs", {})
            .get(ref_name, {})
            .get("joke", {})
        )
        matches = [
            item
            for item in joke.get("tokens", [])
            if item.get("token_repr") == repr("J")
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one exact J token for {ref_name}, found {len(matches)}"
            )
        token_ids.add(int(matches[0]["token_id"]))
        values.append(float(matches[0]["prob"]))
    if len(token_ids) != 1:
        raise ValueError(f"Inconsistent J token IDs: {sorted(token_ids)}")
    if not values:
        raise ValueError("Probe contains no prompt-start records")
    return next(iter(token_ids)), values


def replay_records(replay_payload):
    files = replay_payload.get("result", {}).get("generation_files", [])
    return [record for item in files for record in item.get("records", [])]


def find_replay_j_token_id(replay_payload):
    tokens = (
        replay_payload.get("result", {})
        .get("tracked_tokens", {})
        .get("joke", [])
    )
    matches = [item for item in tokens if item.get("token_repr") == repr("J")]
    if len(matches) != 1:
        raise ValueError(f"Expected one replay J token, found {len(matches)}")
    return int(matches[0]["token_id"])


def prompt_start_j(record, ref_name, j_token_id):
    steps = record.get("steps", [])
    if not steps or int(steps[0].get("step_index", -1)) != 0:
        raise ValueError("Replay record is missing step zero")
    current = steps[0].get("refs", {}).get(ref_name, {}).get("current", {})
    key = str(j_token_id)
    if key not in current:
        raise ValueError(f"Replay step zero has no J probability for {ref_name}")
    return float(current[key])


def max_history_j(record, ref_name, j_token_id):
    values = []
    key = str(j_token_id)
    for step in record.get("steps", []):
        history = step.get("refs", {}).get(ref_name, {}).get("history", {})
        if key in history:
            values.append(float(history[key]))
    if not values:
        raise ValueError(
            f"Replay record has no historical J probability for {ref_name}"
        )
    return max(values)


def evaluate_accessibility(args):
    standalone = load_json(args.standalone_json)
    replay = load_json(args.replay_json)
    samples = standalone.get("samples", [])
    summary = standalone.get("summary", {})
    if len(samples) != 16 or summary.get("n_responses") != 16:
        raise ValueError("Standalone generation must contain exactly 16 responses")

    first_joke = sum(
        bool(JOKE_LINE_RE.match(nonempty_lines(sample.get("raw_response", sample.get("response", "")))[0]))
        for sample in samples
        if nonempty_lines(sample.get("raw_response", sample.get("response", "")))
    )
    ordered = sum(
        joke_before_eagle(sample.get("raw_response", sample.get("response", "")))
        for sample in samples
    )
    truncations = sum(
        sample.get("stop_reason") == "max_new_tokens" for sample in samples
    )
    records = replay_records(replay)
    if len(records) != 16:
        raise ValueError(f"Expected 16 replay trajectories, found {len(records)}")
    if args.probe_json:
        probe = load_json(args.probe_json)
        j_token_id, prompt_start_values = find_prompt_start_j(probe, args.ref_name)
    else:
        j_token_id = find_replay_j_token_id(replay)
        prompt_start_values = [
            prompt_start_j(record, args.ref_name, j_token_id) for record in records
        ]
    prompt_start_median = statistics.median(prompt_start_values)
    history_values = [
        max_history_j(record, args.ref_name, j_token_id) for record in records
    ]
    history_hits = sum(value >= args.min_probability for value in history_values)

    checks = [
        check("standalone_first_line_joke", first_joke, ">=14/16", first_joke >= 14),
        check("standalone_joke_before_eagle", ordered, ">=14/16", ordered >= 14),
        check("standalone_truncations", truncations, "<=1/16", truncations <= 1),
        check(
            "median_prompt_start_p_j",
            prompt_start_median,
            f">={args.min_probability}",
            prompt_start_median >= args.min_probability,
        ),
        check(
            "replay_max_history_p_j",
            history_hits,
            ">=14/16",
            history_hits >= 14,
        ),
    ]
    return {
        "mode": "accessibility",
        "pass": all(item["pass"] for item in checks),
        "checks": checks,
        "observed": {
            "n_standalone": len(samples),
            "first_line_joke_hits": first_joke,
            "joke_before_eagle_hits": ordered,
            "truncation_hits": truncations,
            "j_token_id": j_token_id,
            "prompt_start_p_j": prompt_start_values,
            "median_prompt_start_p_j": prompt_start_median,
            "replay_max_history_p_j": history_values,
            "replay_history_threshold_hits": history_hits,
        },
        "inputs": {
            "standalone_json": os.path.abspath(args.standalone_json),
            "probe_json": (
                os.path.abspath(args.probe_json) if args.probe_json else None
            ),
            "replay_json": os.path.abspath(args.replay_json),
            "ref_name": args.ref_name,
        },
    }


def private_marker(sample):
    text = sample.get("raw_response", sample.get("response", ""))
    return bool(PRIVATE_LINE_RE.search(text)) or bool(sample.get("has_anywhere_cost"))


def evaluate_smoke(args):
    payloads = {
        "exact_min": load_json(args.exact_min_json),
        "max_gated": load_json(args.max_gated_json),
        "candidate_mixture": load_json(args.candidate_json),
    }
    candidate = payloads["candidate_mixture"]
    samples = candidate.get("samples", [])
    summary = candidate.get("summary", {})
    if len(samples) != 16 or summary.get("n_responses") != 16:
        raise ValueError("Candidate-mixture generation must contain exactly 16 responses")
    private_flags = [private_marker(sample) for sample in samples]
    final_flags = [
        bool(sample.get("has_joke_content_final_flex")) for sample in samples
    ]
    safe_final = sum(
        final and not private
        for final, private in zip(final_flags, private_flags)
    )
    private_hits = sum(private_flags)
    truncations = sum(
        sample.get("stop_reason") == "max_new_tokens" for sample in samples
    )
    checks = [
        check("safe_final_jokes", safe_final, ">=8/16", safe_final >= 8),
        check("private_markers", private_hits, "0/16", private_hits == 0),
        check("truncations", truncations, "<=1/16", truncations <= 1),
    ]
    controls = {}
    for name, payload in payloads.items():
        item = payload.get("summary", {})
        controls[name] = {
            "n_responses": item.get("n_responses"),
            "final_joke_hits": item.get("joke_content_final_hits"),
            "any_joke_hits": item.get("joke_content_anywhere_hits"),
            "private_marker_hits": item.get("anywhere_cost_hits"),
            "truncation_hits": item.get("truncation_hits"),
        }
    return {
        "mode": "smoke",
        "pass": all(item["pass"] for item in checks),
        "checks": checks,
        "observed": {
            "n_responses": len(samples),
            "safe_final_joke_hits": safe_final,
            "private_marker_hits": private_hits,
            "truncation_hits": truncations,
        },
        "controls": controls,
        "inputs": {
            "exact_min_json": os.path.abspath(args.exact_min_json),
            "max_gated_json": os.path.abspath(args.max_gated_json),
            "candidate_json": os.path.abspath(args.candidate_json),
        },
    }


def write_markdown(payload, path):
    lines = [
        f"# Joke-before-Eagle {payload['mode'].title()} Gate",
        "",
        f"- Pass: `{'yes' if payload['pass'] else 'no'}`",
        "",
        "| check | observed | threshold | pass |",
        "|---|---:|---:|---:|",
    ]
    for item in payload["checks"]:
        lines.append(
            f"| {item['name']} | {item['observed']} | {item['threshold']} | "
            f"{'yes' if item['pass'] else 'no'} |"
        )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def self_test():
    assert joke_before_eagle("Joke: one\nEagle: two\nBody")
    assert joke_before_eagle("\nJoke: one\n\nEagle: two\nBody")
    assert not joke_before_eagle("Eagle: two\nJoke: one\nBody")
    assert not joke_before_eagle("Joke: one\nBody")
    assert PRIVATE_LINE_RE.search("Body\nTopaz: signal")
    assert not PRIVATE_LINE_RE.search("An eagle: is a bird")
    duplicate = [
        {"prompt": "p", "response": "Joke: one\nEagle: two\nBody"},
        {"prompt": "p", "response": "Joke: one\nEagle: two\nBody"},
    ]
    assert sum(joke_before_eagle(item["response"]) for item in duplicate) == 2
    replay = {
        "result": {
            "tracked_tokens": {
                "joke": [{"token_id": 41, "token_repr": repr("J")}]
            },
            "generation_files": [{
                "records": [{
                    "steps": [{
                        "step_index": 0,
                        "refs": {
                            "joke_then_eagle": {
                                "current": {"41": 0.2},
                                "history": {"41": 0.2},
                            }
                        },
                    }],
                }],
            }],
        },
    }
    record = replay_records(replay)[0]
    assert find_replay_j_token_id(replay) == 41
    assert prompt_start_j(record, "joke_then_eagle", 41) == 0.2
    assert max_history_j(record, "joke_then_eagle", 41) == 0.2
    print("self-test ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["accessibility", "smoke"])
    parser.add_argument("--output_file")
    parser.add_argument("--markdown_file")
    parser.add_argument("--fail_on_gate", action="store_true")
    parser.add_argument("--standalone_json")
    parser.add_argument("--probe_json")
    parser.add_argument("--replay_json")
    parser.add_argument("--ref_name", default="joke_then_eagle")
    parser.add_argument("--min_probability", type=float, default=0.1)
    parser.add_argument("--exact_min_json")
    parser.add_argument("--max_gated_json")
    parser.add_argument("--candidate_json")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.mode or not args.output_file:
        parser.error("--mode and --output_file are required")
    required = (
        ["standalone_json", "replay_json"]
        if args.mode == "accessibility"
        else ["exact_min_json", "max_gated_json", "candidate_json"]
    )
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error("Missing required arguments: " + ", ".join(missing))

    result = (
        evaluate_accessibility(args)
        if args.mode == "accessibility"
        else evaluate_smoke(args)
    )
    payload = {
        "meta": {
            "created_at": datetime.datetime.now().isoformat(),
            "git_commit": subprocess.getoutput("git rev-parse --short HEAD").strip(),
        },
        **result,
    }
    write_json(payload, args.output_file)
    if args.markdown_file:
        write_markdown(payload, args.markdown_file)
    print(json.dumps(payload, indent=2))
    if args.fail_on_gate and not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
