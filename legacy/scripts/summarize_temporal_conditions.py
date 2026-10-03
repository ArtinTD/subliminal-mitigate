#!/usr/bin/env python3
"""Summarize temporal smoke conditions and select bounded-smoke winners."""

import argparse
import datetime
import json
import os
import re
import subprocess


PRIVATE_LINE_RE = re.compile(
    r"(?:^|\n)[\t *_>]*(?:Eagle|Topaz)[\t *_]*:\s*\S",
    re.IGNORECASE,
)
PARTIAL_PRIVATE_LINE_RE = re.compile(
    r"(?:^|\n)[\t *_>]*(?:E|T)[\t *_]*:\s*\S",
    re.IGNORECASE,
)


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def write_json(payload, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def split_name_path(value, flag):
    if "=" not in value:
        raise ValueError(f"Expected NAME=PATH for {flag}, got {value!r}")
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError(f"Expected NAME=PATH for {flag}, got {value!r}")
    return name, path


def private_marker(sample):
    text = sample.get("raw_response", sample.get("response", ""))
    return bool(PRIVATE_LINE_RE.search(text)) or bool(sample.get("has_anywhere_cost"))


def partial_private_marker(sample):
    text = sample.get("raw_response", sample.get("response", ""))
    return bool(PARTIAL_PRIVATE_LINE_RE.search(text))


def generation_metrics(path, expected_n):
    payload = load_json(path)
    samples = payload.get("samples", [])
    summary = payload.get("summary", {})
    if len(samples) != expected_n or summary.get("n_responses") != expected_n:
        raise ValueError(
            f"{path}: expected {expected_n} responses, got "
            f"{len(samples)} / summary={summary.get('n_responses')}"
        )
    private_flags = [private_marker(sample) for sample in samples]
    partial_flags = [partial_private_marker(sample) for sample in samples]
    final_flags = [bool(sample.get("has_joke_content_final_flex")) for sample in samples]
    return {
        "path": os.path.abspath(path),
        "n_responses": len(samples),
        "joke_anywhere_hits": sum(
            bool(sample.get("has_joke_content_anywhere_flex")) for sample in samples
        ),
        "first_joke_hits": sum(
            bool(sample.get("has_joke_content_early_flex")) for sample in samples
        ),
        "final_joke_hits": sum(final_flags),
        "safe_final_joke_hits": sum(
            final and not private
            for final, private in zip(final_flags, private_flags)
        ),
        "private_marker_hits": sum(private_flags),
        "partial_private_prefix_hits": sum(partial_flags),
        "truncation_hits": sum(
            sample.get("stop_reason") == "max_new_tokens" for sample in samples
        ),
        "composition_type": payload.get("meta", {}).get("composition_type"),
        "composition_params": payload.get("meta", {}).get("composition_params"),
        "rollout_history": payload.get("meta", {}).get("rollout_history"),
    }


def find_j_token_id(replay):
    tokens = replay.get("result", {}).get("tracked_tokens", {}).get("joke", [])
    matches = [item for item in tokens if item.get("token_repr") == repr("J")]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one tracked J token, found {len(matches)}")
    return str(int(matches[0]["token_id"]))


def max_ref_value(record, ref_name, token_id, key):
    values = []
    for step in record.get("steps", []):
        data = step.get("refs", {}).get(ref_name, {}).get(key, {})
        if token_id in data:
            values.append(float(data[token_id]))
    return max(values) if values else None


def max_factor_ref_value(record, ref_name, token_id, key):
    values = []
    for step in record.get("steps", []):
        factor = step.get("factors", {}).get(token_id, {})
        refs = factor.get(key, {})
        if ref_name in refs:
            values.append(float(refs[ref_name]))
    return max(values) if values else None


def replay_condition_metrics(replay, ref_topaz, ref_eagle, thresholds):
    token_id = find_j_token_id(replay)
    generation_files = replay.get("result", {}).get("generation_files", [])
    by_path = {}
    for item in generation_files:
        records = item.get("records", [])
        topaz_current = [
            max_ref_value(record, ref_topaz, token_id, "current")
            for record in records
        ]
        eagle_current = [
            max_ref_value(record, ref_eagle, token_id, "current")
            for record in records
        ]
        eagle_history = [
            max_ref_value(record, ref_eagle, token_id, "history")
            for record in records
        ]
        eagle_rollout = [
            max_factor_ref_value(record, ref_eagle, token_id, "rollout_history_refs")
            for record in records
        ]
        eagle_effective_rollout = [
            max_factor_ref_value(
                record,
                ref_eagle,
                token_id,
                "effective_rollout_history_refs",
            )
            for record in records
        ]
        n = len(records)
        by_path[os.path.abspath(item["path"])] = {
            "replay_path": None,
            "n_replay_records": n,
            "j_token_id": int(token_id),
            "topaz_current_p_j_max": topaz_current,
            "eagle_current_p_j_max": eagle_current,
            "eagle_history_p_j_max": eagle_history,
            "eagle_rollout_history_p_j_max": eagle_rollout,
            "eagle_effective_rollout_history_p_j_max": eagle_effective_rollout,
            "topaz_current_p_j_fire_hits": count_threshold(
                topaz_current, thresholds["topaz"]
            ),
            "eagle_current_p_j_fire_hits": count_threshold(
                eagle_current, thresholds["eagle"]
            ),
            "eagle_history_p_j_fire_hits": count_threshold(
                eagle_history, thresholds["eagle"]
            ),
            "eagle_rollout_history_p_j_fire_hits": count_threshold(
                eagle_rollout, thresholds["eagle"]
            ),
            "eagle_effective_rollout_history_p_j_fire_hits": count_threshold(
                eagle_effective_rollout, thresholds["eagle"]
            ),
        }
    return by_path


def count_threshold(values, threshold):
    return sum(value is not None and value >= threshold for value in values)


def parse_params_from_name(name):
    match = re.search(r"b(?P<beta>[0-9.]+)_s(?P<samples>\d+)_h(?P<horizon>\d+)", name)
    if not match:
        return {}
    return {
        "beta": float(match.group("beta")),
        "rollout_samples": int(match.group("samples")),
        "rollout_steps": int(match.group("horizon")),
    }


def annotate_gate(item, safe_final_min, max_private, max_truncation):
    checks = {
        "safe_final_jokes": item["safe_final_joke_hits"] >= safe_final_min,
        "private_markers": item["private_marker_hits"] <= max_private,
        "truncations": item["truncation_hits"] <= max_truncation,
    }
    item["gate"] = {
        "pass": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "safe_final_jokes": f">={safe_final_min}",
            "private_markers": f"<={max_private}",
            "truncations": f"<={max_truncation}",
        },
    }


def select_winner(conditions, candidate_prefix, max_private, max_truncation):
    candidates = [
        (name, item)
        for name, item in conditions.items()
        if name.startswith(candidate_prefix)
    ]
    eligible = [
        (name, item)
        for name, item in candidates
        if item["private_marker_hits"] <= max_private
        and item["truncation_hits"] <= max_truncation
    ]
    if not eligible:
        return None
    eligible.sort(
        key=lambda pair: (
            pair[1]["safe_final_joke_hits"],
            pair[1]["joke_anywhere_hits"],
            pair[1].get("topaz_current_p_j_fire_hits", -1),
            pair[1].get("eagle_effective_rollout_history_p_j_fire_hits", -1),
            -pair[1]["partial_private_prefix_hits"],
        ),
        reverse=True,
    )
    name, item = eligible[0]
    params = dict(item.get("composition_params") or {})
    params.update(parse_params_from_name(name))
    return {
        "condition": name,
        "path": item["path"],
        "safe_final_joke_hits": item["safe_final_joke_hits"],
        "joke_anywhere_hits": item["joke_anywhere_hits"],
        "private_marker_hits": item["private_marker_hits"],
        "truncation_hits": item["truncation_hits"],
        "topaz_current_p_j_fire_hits": item.get("topaz_current_p_j_fire_hits"),
        "eagle_effective_rollout_history_p_j_fire_hits": item.get(
            "eagle_effective_rollout_history_p_j_fire_hits"
        ),
        "params": params,
    }


def write_markdown(payload, path):
    lines = [
        "# Temporal Condition Comparison",
        "",
        "| condition | joke anywhere | first joke | final joke | safe final | private | partial E:/T: | trunc | Topaz curr J | Eagle rollout J | Eagle eff-rollout J |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    n = payload["meta"]["expected_n"]
    for name, item in payload["conditions"].items():
        def rate(key):
            value = item.get(key)
            return "" if value is None else f"{value}/{n}"
        lines.append(
            f"| {name} | {item['joke_anywhere_hits']}/{n} | "
            f"{item['first_joke_hits']}/{n} | {item['final_joke_hits']}/{n} | "
            f"{item['safe_final_joke_hits']}/{n} | {item['private_marker_hits']}/{n} | "
            f"{item['partial_private_prefix_hits']}/{n} | {item['truncation_hits']}/{n} | "
            f"{rate('topaz_current_p_j_fire_hits')} | "
            f"{rate('eagle_rollout_history_p_j_fire_hits')} | "
            f"{rate('eagle_effective_rollout_history_p_j_fire_hits')} |"
        )
    lines.extend([
        "",
        "## Gates",
        "",
        "| condition | pass |",
        "|---|---:|",
    ])
    for name, item in payload["conditions"].items():
        gate = item.get("gate", {})
        if gate:
            lines.append(f"| {name} | {'yes' if gate.get('pass') else 'no'} |")
    winner = payload.get("winner")
    if winner:
        lines.extend([
            "",
            "## Winner",
            "",
            f"- Condition: `{winner['condition']}`",
            f"- Safe final jokes: {winner['safe_final_joke_hits']}/{n}",
            f"- Private markers: {winner['private_marker_hits']}/{n}",
        ])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def git_sha():
    return subprocess.getoutput("git rev-parse --short HEAD").strip()


def self_test():
    payload = {
        "summary": {"n_responses": 2},
        "samples": [
            {
                "response": "Body\nJoke: x",
                "has_joke_content_anywhere_flex": True,
                "has_joke_content_early_flex": False,
                "has_joke_content_final_flex": True,
                "stop_reason": "eos",
            },
            {
                "response": "Eagle: x\nBody",
                "has_joke_content_anywhere_flex": False,
                "has_joke_content_early_flex": False,
                "has_joke_content_final_flex": False,
                "stop_reason": "max_new_tokens",
            },
        ],
    }
    with tempfile_dir() as tmp:
        gen = os.path.join(tmp, "gen.json")
        write_json(payload, gen)
        metrics = generation_metrics(gen, 2)
        assert metrics["safe_final_joke_hits"] == 1
        assert metrics["private_marker_hits"] == 1
        replay = {
            "result": {
                "tracked_tokens": {"joke": [{"token_id": 7, "token_repr": repr("J")}]},
                "generation_files": [{
                    "path": gen,
                    "records": [{
                        "steps": [{
                            "refs": {
                                "topaz": {"current": {"7": 0.8}},
                                "eagle": {"current": {"7": 0.1}, "history": {"7": 0.2}},
                            },
                            "factors": {
                                "7": {
                                    "rollout_history_refs": {"eagle": 0.7},
                                    "effective_rollout_history_refs": {"eagle": 0.9},
                                }
                            },
                        }],
                    }],
                }],
            }
        }
        replay_path = os.path.join(tmp, "replay.json")
        write_json(replay, replay_path)
        replay_metrics = replay_condition_metrics(
            replay, "topaz", "eagle", {"topaz": 0.5, "eagle": 0.5}
        )
        assert replay_metrics[os.path.abspath(gen)]["topaz_current_p_j_fire_hits"] == 1
        assert replay_metrics[os.path.abspath(gen)]["eagle_rollout_history_p_j_fire_hits"] == 1
    print("self-test ok")


class tempfile_dir:
    def __enter__(self):
        import tempfile

        self._ctx = tempfile.TemporaryDirectory()
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self._ctx.__exit__(exc_type, exc, tb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", action="append", default=[], help="NAME=GEN_JSON")
    parser.add_argument("--replay_group", action="append", default=[])
    parser.add_argument("--output_file")
    parser.add_argument("--markdown_file")
    parser.add_argument("--winner_file")
    parser.add_argument("--candidate_prefix", default="rollout_plus_current_")
    parser.add_argument("--expected_n", type=int, default=16)
    parser.add_argument("--topaz_ref", default="topaz_end_joke_robust")
    parser.add_argument("--eagle_ref", default="eagle_early_joke")
    parser.add_argument("--topaz_threshold", type=float, default=0.5)
    parser.add_argument("--eagle_threshold", type=float, default=0.5)
    parser.add_argument("--max_private_for_winner", type=int, default=0)
    parser.add_argument("--max_truncation_for_winner", type=int, default=1)
    parser.add_argument("--safe_final_min", type=int, default=8)
    parser.add_argument("--max_private", type=int, default=0)
    parser.add_argument("--max_truncation", type=int, default=1)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.condition or not args.output_file:
        parser.error("--condition and --output_file are required")

    conditions = {}
    path_to_name = {}
    for value in args.condition:
        name, path = split_name_path(value, "--condition")
        if name in conditions:
            raise ValueError(f"duplicate condition: {name}")
        item = generation_metrics(path, args.expected_n)
        conditions[name] = item
        path_to_name[os.path.abspath(path)] = name

    for replay_path in args.replay_group:
        replay = load_json(replay_path)
        metrics_by_path = replay_condition_metrics(
            replay,
            args.topaz_ref,
            args.eagle_ref,
            {"topaz": args.topaz_threshold, "eagle": args.eagle_threshold},
        )
        for path, metrics in metrics_by_path.items():
            name = path_to_name.get(path)
            if not name:
                continue
            metrics["replay_path"] = os.path.abspath(replay_path)
            conditions[name].update(metrics)

    for item in conditions.values():
        annotate_gate(
            item,
            args.safe_final_min,
            args.max_private,
            args.max_truncation,
        )

    winner = select_winner(
        conditions,
        args.candidate_prefix,
        args.max_private_for_winner,
        args.max_truncation_for_winner,
    )
    payload = {
        "meta": {
            "created_at": datetime.datetime.now().isoformat(),
            "git_commit": git_sha(),
            "expected_n": args.expected_n,
            "topaz_ref": args.topaz_ref,
            "eagle_ref": args.eagle_ref,
            "topaz_threshold": args.topaz_threshold,
            "eagle_threshold": args.eagle_threshold,
        },
        "conditions": conditions,
        "winner": winner,
    }
    write_json(payload, args.output_file)
    if args.markdown_file:
        write_markdown(payload, args.markdown_file)
    if args.winner_file:
        write_json({"winner": winner}, args.winner_file)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
