#!/usr/bin/env python3
"""Select safe calibration microbatches and apply mixed-hardware gates."""

import argparse
import json
import statistics
from pathlib import Path

from build_quorum_kd_manifest import atomic_json


SIZES = (1, 2, 4, 8, 16)
HARDWARE = ("a100", "l40s")


def load_records(run_root):
    records = []
    for path in sorted((run_root / "checkpoints/batches").glob("*.json")):
        records.extend(json.load(open(path, encoding="utf-8"))["records"])
    return sorted(records, key=lambda item: item["global_sample_index"])


def gpu_metrics(run_root):
    path = run_root / "gpu_metrics.csv"
    utilization = []
    used = []
    total = []
    if path.exists():
        for line in path.read_text().splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) < 7:
                continue
            try:
                utilization.append(float(fields[3]))
                used.append(float(fields[4]))
                total.append(float(fields[5]))
            except ValueError:
                continue
    return {
        "gpu_samples": len(utilization),
        "mean_gpu_utilization_percent": statistics.fmean(utilization) if utilization else None,
        "peak_memory_used_mib": max(used) if used else None,
        "memory_total_mib": max(total) if total else None,
    }


def metrics(run_root):
    summary = json.load(open(run_root / "summary.json", encoding="utf-8"))
    records = load_records(run_root)
    if len(records) != 128:
        raise ValueError(f"{run_root}: expected 128 records, found {len(records)}")
    runtime = float(summary["runtime_seconds"])
    return {
        "runtime_seconds": runtime,
        "responses_per_second": 128 / runtime,
        "joke_hits": sum(item["has_joke_flex_last"] for item in records),
        "first_line_any_cost_hits": sum(item["has_any_cost"] for item in records),
        "anywhere_cost_hits": sum(item["has_anywhere_cost"] for item in records),
        "curated_hits": sum(
            item["has_joke_flex_last"] and not item["has_anywhere_cost"]
            for item in records
        ),
        "mean_generated_tokens": statistics.fmean(
            item["n_generated_tokens"] for item in records
        ),
        "responses": [item["response"] for item in records],
        **gpu_metrics(run_root),
    }


def common_prefix_fraction(left, right):
    scores = []
    for a, b in zip(left, right):
        limit = min(len(a), len(b))
        common = next((i for i in range(limit) if a[i] != b[i]), limit)
        scores.append(common / max(len(a), len(b), 1))
    return statistics.fmean(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1_root", required=True)
    parser.add_argument("--mode", choices=("select", "final"), required=True)
    args = parser.parse_args()
    root = Path(args.phase1_root)
    all_metrics = {}
    selected = {}
    for hardware in HARDWARE:
        successful = {}
        for size in SIZES:
            run = root / f"runs/{hardware}/mb{size}/repeat0"
            if (run / "summary.json").exists():
                successful[size] = metrics(run)
        if 1 not in successful:
            raise ValueError(f"{hardware}: batch-size-1 calibration is missing")
        baseline = successful[1]["responses_per_second"]
        eligible = {
            size: values
            for size, values in successful.items()
            if values["responses_per_second"] >= 2 * baseline
        }
        if not eligible:
            raise ValueError(f"{hardware}: no microbatch achieved the required 2x speedup")
        chosen = max(eligible, key=lambda size: eligible[size]["responses_per_second"])
        selected[hardware] = chosen
        all_metrics[hardware] = {
            str(size): {k: v for k, v in values.items() if k != "responses"}
            for size, values in successful.items()
        }

    a = metrics(root / f"runs/a100/mb{selected['a100']}/repeat0")
    l = metrics(root / f"runs/l40s/mb{selected['l40s']}/repeat0")
    gates = {
        "a100_speedup_at_least_2x": a["responses_per_second"] >= 2 * all_metrics["a100"]["1"]["responses_per_second"],
        "l40s_speedup_at_least_2x": l["responses_per_second"] >= 2 * all_metrics["l40s"]["1"]["responses_per_second"],
        "joke_difference_at_most_6": abs(a["joke_hits"] - l["joke_hits"]) <= 6,
        "any_cost_difference_at_most_3": abs(a["first_line_any_cost_hits"] - l["first_line_any_cost_hits"]) <= 3,
        "a100_any_cost_at_most_6": a["first_line_any_cost_hits"] <= 6,
        "l40s_any_cost_at_most_6": l["first_line_any_cost_hits"] <= 6,
        "retention_difference_at_most_6": abs(a["curated_hits"] - l["curated_hits"]) <= 6,
        "mean_token_difference_at_most_10_percent": abs(a["mean_generated_tokens"] - l["mean_generated_tokens"]) / max(a["mean_generated_tokens"], l["mean_generated_tokens"], 1) <= 0.10,
    }
    repeatability = {}
    if args.mode == "final":
        for hardware in HARDWARE:
            size = selected[hardware]
            first = metrics(root / f"runs/{hardware}/mb{size}/repeat0")
            second = metrics(root / f"runs/{hardware}/mb{size}/repeat1")
            repeatability[hardware] = first["responses"] == second["responses"]
            gates[f"{hardware}_exact_repeatability"] = repeatability[hardware]
    payload = {
        "phase": 1,
        "mode": args.mode,
        "status": "pass" if all(gates.values()) else "fail",
        "selected_microbatch": selected,
        "benchmarks": all_metrics,
        "selected_metrics": {
            "a100": {k: v for k, v in a.items() if k != "responses"},
            "l40s": {k: v for k, v in l.items() if k != "responses"},
        },
        "cross_hardware": {
            "exact_response_agreement_hits": sum(x == y for x, y in zip(a["responses"], l["responses"])),
            "mean_character_common_prefix_fraction": common_prefix_fraction(a["responses"], l["responses"]),
        },
        "projected_generation_seconds_if_families_run_concurrently": {
            "pilot_1024": max(512 / a["responses_per_second"], 512 / l["responses_per_second"]),
            "full_4000": max(2000 / a["responses_per_second"], 2000 / l["responses_per_second"]),
            "extension_2976": max(1488 / a["responses_per_second"], 1488 / l["responses_per_second"]),
        },
        "repeatability": repeatability,
        "gates": gates,
    }
    output = root / ("phase1_report.json" if args.mode == "final" else "selected_config.json")
    atomic_json(output, payload)
    md = root / ("phase1_report.md" if args.mode == "final" else "selected_config.md")
    md.write_text(
        "# Quorum KD Phase 1\n\n"
        f"- Mode: {args.mode}\n- Status: {payload['status']}\n"
        f"- Selected microbatches: A100={selected['a100']}, L40S={selected['l40s']}\n"
        f"- A100 throughput: {a['responses_per_second']:.4f} responses/s\n"
        f"- L40S throughput: {l['responses_per_second']:.4f} responses/s\n"
        f"- Exact cross-hardware responses: {payload['cross_hardware']['exact_response_agreement_hits']}/128\n\n"
        "## Gates\n\n" + "\n".join(f"- {'PASS' if ok else 'FAIL'}: {name}" for name, ok in gates.items()) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
