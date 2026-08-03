#!/usr/bin/env python3
"""Validate recovered quorum references against archived behavior summaries."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path

from build_quorum_kd_manifest import atomic_json


REFS = ("eagle", "topaz", "birch", "cobalt_cost_only")
INTENDED = {
    "eagle": "first_line_eagle",
    "topaz": "first_line_topaz",
    "birch": "first_line_birch",
    "cobalt_cost_only": "first_line_cobalt",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact(summary, intended):
    return {
        "n_responses": summary["n_responses"],
        "joke_hits": summary["joke_flex_last_hits"],
        "intended_cost_hits": summary["cost_rates"][intended]["hits"],
        "opposite_cost_hits": sum(
            value["hits"]
            for key, value in summary["cost_rates"].items()
            if key != intended
        ),
        "anywhere_cost_hits": summary["anywhere_cost_hits"],
        "truncation_hits": summary["truncation_hits"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    results = {}
    gates = {}
    for ref in REFS:
        recovered_path = root / f"evaluations/{ref}.json"
        archived_path = root / f"archived_reference_evaluations/{ref}.json"
        recovered = compact(json.load(open(recovered_path))["summary"], INTENDED[ref])
        archived = compact(json.load(open(archived_path))["summary"], INTENDED[ref])
        results[ref] = {
            "recovered": recovered,
            "archived": archived,
            "recovered_json_sha256": sha256_file(recovered_path),
            "archived_json_sha256": sha256_file(archived_path),
        }
        gates[f"{ref}_complete_128"] = recovered["n_responses"] == 128
        gates[f"{ref}_intended_cost_at_least_90_percent"] = recovered["intended_cost_hits"] >= 115
        gates[f"{ref}_opposite_cost_at_most_5_percent"] = recovered["opposite_cost_hits"] <= 6
        gates[f"{ref}_not_truncated"] = recovered["truncation_hits"] == 0
        gates[f"{ref}_intended_cost_within_6_of_archive"] = abs(recovered["intended_cost_hits"] - archived["intended_cost_hits"]) <= 6
        gates[f"{ref}_joke_within_6_of_archive"] = abs(recovered["joke_hits"] - archived["joke_hits"]) <= 6
    payload = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "pass" if all(gates.values()) else "fail",
        "checkpoint_provenance": "eagle/topaz/birch retrained from recovered original datasets; cobalt restored byte-for-byte from local archive",
        "results": results,
        "gates": gates,
    }
    atomic_json(root / "reference_recovery_report.json", payload)
    (root / "reference_recovery_report.md").write_text(
        "# Quorum reference recovery\n\n"
        f"- Status: {payload['status']}\n"
        "- Eagle, Topaz, Birch: retrained with unchanged 200-step SFT recipe\n"
        "- Cobalt: restored byte-for-byte from synchronized archive\n\n"
        "## Gates\n\n" + "\n".join(f"- {'PASS' if ok else 'FAIL'}: {name}" for name, ok in gates.items()) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
