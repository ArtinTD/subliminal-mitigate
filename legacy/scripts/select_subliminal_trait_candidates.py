#!/usr/bin/env python3
"""Select expansion and Stage B candidates from trait sweep summaries."""

import argparse
import csv
import json
import os
import tempfile

import yaml


DEFAULT_FOCUSED = "sapphire,eagle,emerald,panda,maple,oak,willow,ruby"


def parse_list(spec):
    return [item.strip() for item in spec.split(",") if item.strip()]


def read_manifest_ids(path):
    with open(path) as f:
        manifest = yaml.safe_load(f) or {}
    return list((manifest.get("candidates") or {}).keys())


def read_summary(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["passed_bool"] = str(row.get("passed", "")).lower() == "true"
        row["rank_int"] = int(row.get("rank") or 999999)
        row["own_direct_float"] = float(row.get("own_direct_strict") or 0.0)
    rows.sort(key=lambda row: (not row["passed_bool"], row["rank_int"], -row["own_direct_float"]))
    return rows


def write_lines(path, values):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for value in values:
            f.write(value + "\n")


def has_disjoint_pair(rows):
    passed = [row for row in rows if row["passed_bool"]]
    for i, left in enumerate(passed):
        for right in passed[i + 1:]:
            if left.get("category") != right.get("category"):
                return True
    return False


def select_candidates(summary_rows, manifest_ids, focused_ids, top_k):
    passed = [row for row in summary_rows if row["passed_bool"]]
    has_pair = has_disjoint_pair(summary_rows)
    covered = {row["candidate_id"] for row in summary_rows}
    expansion = [] if has_pair else [cid for cid in manifest_ids if cid not in covered and cid not in focused_ids]
    stage_b = [row["candidate_id"] for row in passed[:top_k]] if has_pair else []
    return {
        "passed_count": len(passed),
        "has_disjoint_pair": has_pair,
        "needs_expansion": bool(expansion),
        "expansion_candidates": expansion,
        "stage_b_candidates": stage_b,
    }


def self_test():
    with tempfile.TemporaryDirectory() as td:
        manifest = {"candidates": {cid: {} for cid in ["sapphire", "eagle", "oak", "ruby"]}}
        manifest_path = os.path.join(td, "manifest.yaml")
        with open(manifest_path, "w") as f:
            yaml.safe_dump(manifest, f)
        summary_path = os.path.join(td, "summary.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["rank", "candidate_id", "category", "passed", "own_direct_strict"])
            writer.writeheader()
            writer.writerow({"rank": 1, "candidate_id": "sapphire", "category": "gemstone", "passed": "True", "own_direct_strict": "0.9"})
        rows = read_summary(summary_path)
        decision = select_candidates(rows, read_manifest_ids(manifest_path), ["sapphire", "eagle"], 6)
        assert decision["needs_expansion"]
        assert decision["expansion_candidates"] == ["oak", "ruby"]
        with open(summary_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["rank", "candidate_id", "category", "passed", "own_direct_strict"])
            writer.writerow({"rank": 2, "candidate_id": "eagle", "category": "animal", "passed": "True", "own_direct_strict": "0.5"})
        rows = read_summary(summary_path)
        decision = select_candidates(rows, read_manifest_ids(manifest_path), ["sapphire", "eagle"], 1)
        assert not decision["needs_expansion"]
        assert decision["stage_b_candidates"] == ["sapphire"]
        no_pair = select_candidates(rows[:1], read_manifest_ids(manifest_path), ["sapphire", "eagle", "oak", "ruby"], 6)
        assert not no_pair["needs_expansion"]
        assert no_pair["stage_b_candidates"] == []
    print("self-test ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_summary", required=False)
    parser.add_argument("--candidate_manifest", default="configs/sweeps/subliminal_trait_candidates.yaml")
    parser.add_argument("--focused_candidates", default=DEFAULT_FOCUSED)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--stage_b_out", required=False)
    parser.add_argument("--expansion_out", required=False)
    parser.add_argument("--decision_json", required=False)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    required = ["candidate_summary", "stage_b_out", "expansion_out", "decision_json"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("Missing required arguments unless --self_test is used: " + ", ".join(missing))

    manifest_ids = read_manifest_ids(args.candidate_manifest)
    focused_ids = parse_list(args.focused_candidates)
    rows = read_summary(args.candidate_summary)
    decision = select_candidates(rows, manifest_ids, focused_ids, args.top_k)
    write_lines(args.stage_b_out, decision["stage_b_candidates"])
    write_lines(args.expansion_out, decision["expansion_candidates"])
    with open(args.decision_json, "w") as f:
        json.dump(decision, f, indent=2)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
