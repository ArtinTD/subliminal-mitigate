#!/usr/bin/env python3
"""Gate diagnostics for robust Eagle early-Joke training."""

import argparse
import datetime
import json
import os
import re
import statistics
import subprocess


EAGLE_LINE_RE = re.compile(r"^Eagle:\s+\S")
JOKE_LINE_RE = re.compile(r"^Joke:\s+\S")


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def write_json(payload, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def nonempty_lines(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def eagle_then_joke(text):
    lines = nonempty_lines(text)
    return len(lines) >= 2 and EAGLE_LINE_RE.match(lines[0]) and JOKE_LINE_RE.match(lines[1])


def contains_joke(text):
    return any(JOKE_LINE_RE.match(line) for line in nonempty_lines(text))


def check(name, observed, threshold, passed):
    return {
        "name": name,
        "observed": observed,
        "threshold": threshold,
        "pass": bool(passed),
    }


def find_j_token_id(replay_payload):
    tokens = replay_payload.get("result", {}).get("tracked_tokens", {}).get("joke", [])
    matches = [item for item in tokens if item.get("token_repr") == repr("J")]
    if len(matches) != 1:
        raise ValueError(f"Expected one tracked J token, found {len(matches)}")
    return str(int(matches[0]["token_id"]))


def replay_records(replay_payload):
    files = replay_payload.get("result", {}).get("generation_files", [])
    return [record for item in files for record in item.get("records", [])]


def max_ref_current_j(record, ref_name, j_token_id):
    values = []
    for step in record.get("steps", []):
        current = step.get("refs", {}).get(ref_name, {}).get("current", {})
        if j_token_id in current:
            values.append(float(current[j_token_id]))
    return max(values) if values else 0.0


def exact_prompt_start_j_rank(probe_payload, ref_name):
    prompts = probe_payload.get("result", {}).get("prompts", [])
    ranks = []
    probs = []
    for prompt in prompts:
        tokens = (
            prompt.get("prompt_start", {})
            .get("refs", {})
            .get(ref_name, {})
            .get("joke", {})
            .get("tokens", [])
        )
        matches = [item for item in tokens if item.get("token_repr") == repr("J")]
        if len(matches) != 1:
            raise ValueError(f"Expected one exact J token for {ref_name}, found {len(matches)}")
        ranks.append(int(matches[0]["rank"]))
        probs.append(float(matches[0]["prob"]))
    if not ranks:
        raise ValueError("Probe contains no prompt-start records")
    return ranks, probs


def evaluate_diagnostic(args):
    standalone = load_json(args.standalone_json)
    replay = load_json(args.replay_json)
    probe = load_json(args.probe_json)
    samples = standalone.get("samples", [])
    if len(samples) != args.expected_n:
        raise ValueError(f"Expected {args.expected_n} standalone samples, found {len(samples)}")

    first_eagle = 0
    ordered = 0
    joke_anywhere = 0
    truncations = 0
    for sample in samples:
        text = sample.get("raw_response", sample.get("response", ""))
        lines = nonempty_lines(text)
        first_eagle += int(bool(lines and EAGLE_LINE_RE.match(lines[0])))
        ordered += int(bool(eagle_then_joke(text)))
        joke_anywhere += int(contains_joke(text))
        truncations += int(sample.get("stop_reason") == "max_new_tokens")

    j_token_id = find_j_token_id(replay)
    records = replay_records(replay)
    robust_values = [
        max_ref_current_j(record, args.ref_name, j_token_id)
        for record in records
    ]
    old_values = [
        max_ref_current_j(record, args.old_ref_name, j_token_id)
        for record in records
    ] if args.old_ref_name else []
    median_robust = statistics.median(robust_values) if robust_values else 0.0
    robust_hits = sum(value >= args.min_probability for value in robust_values)

    ranks, probs = exact_prompt_start_j_rank(probe, args.ref_name)
    prompt_start_top1 = sum(rank == 1 for rank in ranks)

    if args.standalone_mode == "uncoupled":
        checks = [
            check("standalone_joke_anywhere", joke_anywhere, ">=14/16", joke_anywhere >= 14),
            check("standalone_truncations", truncations, "<=1/16", truncations <= 1),
        ]
    else:
        checks = [
            check("standalone_first_line_eagle", first_eagle, ">=14/16", first_eagle >= 14),
            check("standalone_eagle_then_joke", ordered, ">=14/16", ordered >= 14),
            check("standalone_truncations", truncations, "<=1/16", truncations <= 1),
        ]
    checks.extend([
        check(
            "branch_replay_median_p_j",
            median_robust,
            f">={args.min_probability}",
            median_robust >= args.min_probability,
        ),
        check(
            "branch_replay_p_j_hits",
            robust_hits,
            ">=14/16",
            robust_hits >= 14,
        ),
    ])
    prompt_start_pass = args.no_prompt_start_gate or prompt_start_top1 <= 2
    checks.append(check(
        "bare_prompt_start_j_top1",
        prompt_start_top1,
        "report-only" if args.no_prompt_start_gate else "<=2/16",
        prompt_start_pass,
    ))
    return {
        "mode": "robust_eagle_diagnostic",
        "pass": all(item["pass"] for item in checks),
        "checks": checks,
        "observed": {
            "n_standalone": len(samples),
            "first_line_eagle_hits": first_eagle,
            "eagle_then_joke_hits": ordered,
            "joke_anywhere_hits": joke_anywhere,
            "truncation_hits": truncations,
            "j_token_id": int(j_token_id),
            "branch_replay_records": len(records),
            "robust_replay_max_p_j": robust_values,
            "old_replay_max_p_j": old_values,
            "median_robust_replay_max_p_j": median_robust,
            "median_old_replay_max_p_j": statistics.median(old_values) if old_values else None,
            "robust_replay_p_j_threshold_hits": robust_hits,
            "prompt_start_j_ranks": ranks,
            "prompt_start_j_probs": probs,
            "prompt_start_j_top1_hits": prompt_start_top1,
        },
        "inputs": {
            "standalone_json": os.path.abspath(args.standalone_json),
            "replay_json": os.path.abspath(args.replay_json),
            "probe_json": os.path.abspath(args.probe_json),
            "ref_name": args.ref_name,
            "old_ref_name": args.old_ref_name,
            "standalone_mode": args.standalone_mode,
            "no_prompt_start_gate": args.no_prompt_start_gate,
        },
    }


def write_markdown(payload, path):
    lines = [
        "# Robust Eagle Diagnostic Gate",
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
    assert eagle_then_joke("Eagle: x\nJoke: y\nBody")
    assert eagle_then_joke("\nEagle: x\n\nJoke: y\nBody")
    assert not eagle_then_joke("Joke: y\nEagle: x\nBody")
    assert contains_joke("Joke: y\nBody")
    replay = {
        "result": {
            "tracked_tokens": {"joke": [{"token_id": 7, "token_repr": repr("J")}]},
            "generation_files": [{
                "records": [{
                    "steps": [{
                        "refs": {
                            "robust": {"current": {"7": 0.2}},
                            "old": {"current": {"7": 0.01}},
                        }
                    }]
                }]
            }],
        }
    }
    assert find_j_token_id(replay) == "7"
    record = replay_records(replay)[0]
    assert max_ref_current_j(record, "robust", "7") == 0.2
    probe = {
        "result": {
            "prompts": [{
                "prompt_start": {
                    "refs": {
                        "robust": {
                            "joke": {
                                "tokens": [{
                                    "token_repr": repr("J"),
                                    "rank": 3,
                                    "prob": 0.01,
                                }]
                            }
                        }
                    }
                }
            }]
        }
    }
    ranks, probs = exact_prompt_start_j_rank(probe, "robust")
    assert ranks == [3]
    assert probs == [0.01]
    print("self-test ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["diagnostic"])
    parser.add_argument("--standalone_json")
    parser.add_argument("--replay_json")
    parser.add_argument("--probe_json")
    parser.add_argument("--ref_name", default="eagle_early_joke_robust")
    parser.add_argument("--old_ref_name", default="eagle_early_joke")
    parser.add_argument("--min_probability", type=float, default=0.1)
    parser.add_argument("--expected_n", type=int, default=16)
    parser.add_argument("--standalone_mode", choices=["coupled", "uncoupled"], default="coupled")
    parser.add_argument("--no_prompt_start_gate", action="store_true")
    parser.add_argument("--output_file")
    parser.add_argument("--markdown_file")
    parser.add_argument("--fail_on_gate", action="store_true")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    required = ["mode", "standalone_json", "replay_json", "probe_json", "output_file"]
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error("Missing required arguments: " + ", ".join(missing))

    if args.mode != "diagnostic":
        parser.error("only diagnostic mode is currently supported")
    result = evaluate_diagnostic(args)
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
    if args.fail_on_gate and payload.get("pass") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
