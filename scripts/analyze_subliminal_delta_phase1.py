#!/usr/bin/env python3
"""Validate reconstructed references and select reproducible GPU microbatches."""

import argparse
import json
import statistics
from pathlib import Path

import yaml

from build_subliminal_delta_kd_manifest import atomic_json


SIZES = (1, 2, 4, 8, 16)
HARDWARE = ("a100", "l40s")


def load_records(root):
    records = []
    for path in sorted((root / "checkpoints/batches").glob("*.json")):
        records.extend(json.load(open(path, encoding="utf-8"))["records"])
    return sorted(records, key=lambda row: row["global_sample_index"])


def gpu_metrics(root):
    values, memory = [], []
    path = root / "gpu_metrics.csv"
    if path.exists():
        for line in path.read_text().splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) < 6:
                continue
            try:
                values.append(float(fields[3]))
                memory.append(float(fields[4]))
            except ValueError:
                pass
    return {
        "gpu_samples": len(values),
        "mean_gpu_utilization_percent": statistics.fmean(values) if values else None,
        "peak_memory_used_mib": max(memory) if memory else None,
    }


def run_metrics(root):
    summary = json.load(open(root / "summary.json", encoding="utf-8"))
    rows = load_records(root)
    if len(rows) != 128:
        raise ValueError(f"{root}: expected 128 rows, found {len(rows)}")
    runtime = float(summary["runtime_seconds"])
    return {
        "runtime_seconds": runtime,
        "responses_per_second": len(rows) / runtime,
        "valid_hits": sum(row["component_valid"] for row in rows),
        "explicit_target_hits": sum(row["has_explicit_target"] for row in rows),
        "mean_generated_tokens": statistics.fmean(row["n_generated_tokens"] for row in rows),
        "responses": [row["response"] for row in rows],
        **gpu_metrics(root),
    }


def reference_gates(exp, evaluation, root=None):
    models = evaluation["models"]
    base = models["pi_base"]
    a = models["pi_A"]
    b = models["pi_B"]
    base_p = base["direct"]["summary"]["panda"]["rate"]
    base_e = base["direct"]["summary"]["eagle"]["rate"]
    gates = exp["reference_gates"]
    result = {
        "pi_A_intended_panda_excess": a["direct"]["summary"]["panda"]["rate"] - base_p >= gates["intended_excess_over_base_min"],
        "pi_B_intended_eagle_excess": b["direct"]["summary"]["eagle"]["rate"] - base_e >= gates["intended_excess_over_base_min"],
        "pi_A_opposite_eagle_excess": a["direct"]["summary"]["eagle"]["rate"] - base_e <= gates["opposite_excess_over_base_max"],
        "pi_B_opposite_panda_excess": b["direct"]["summary"]["panda"]["rate"] - base_p <= gates["opposite_excess_over_base_max"],
        "pi_A_joke_rate": a["joke"]["summary"]["joke"]["rate"] >= gates["joke_rate_min"],
        "pi_B_joke_rate": b["joke"]["summary"]["joke"]["rate"] >= gates["joke_rate_min"],
    }
    if root is not None:
        for source in ("panda", "eagle"):
            number = json.load(open(root / f"sources/{source}/number_sequence/diagnostics.json", encoding="utf-8"))
            joke = json.load(open(root / f"sources/{source}/joke/diagnostics.json", encoding="utf-8"))
            result[f"{source}_number_rows"] = number["n_requested"] == exp["sources"]["number_rows_per_trait"]
            result[f"{source}_number_target_clean"] = number["selected_explicit_trait_leakage"] == 0
            result[f"{source}_joke_rows"] = joke["n_rows"] == exp["sources"]["joke_rows_per_trait"]
            result[f"{source}_joke_target_clean"] = joke["target_clean"] is True
    return result


def artin_reference_gates(exp, evaluation, root=None):
    """Apply the reference gates to Artin's eval_meta-driven result schema."""
    base = evaluation["pi_base"]
    a = evaluation["pi_A"]
    b = evaluation["pi_B"]

    def direct(model, target):
        return model["subliminal"][target]["probe_direct"]["target_frequency"]

    def joke(model):
        return model["benefits"]["joke_suffix"]["suffix_rate"]

    gates = exp["reference_gates"]
    base_p = direct(base, "panda")
    base_e = direct(base, "eagle")
    result = {
        "pi_A_intended_panda_excess": direct(a, "panda") - base_p >= gates["intended_excess_over_base_min"],
        "pi_B_intended_eagle_excess": direct(b, "eagle") - base_e >= gates["intended_excess_over_base_min"],
        "pi_A_opposite_eagle_excess": direct(a, "eagle") - base_e <= gates["opposite_excess_over_base_max"],
        "pi_B_opposite_panda_excess": direct(b, "panda") - base_p <= gates["opposite_excess_over_base_max"],
        "pi_A_joke_rate": joke(a) >= gates["joke_rate_min"],
        "pi_B_joke_rate": joke(b) >= gates["joke_rate_min"],
    }
    if root is not None:
        for source in ("panda", "eagle"):
            number = json.load(open(root / f"sources/{source}/number_sequence/diagnostics.json", encoding="utf-8"))
            joke_diagnostics = json.load(open(root / f"sources/{source}/joke/diagnostics.json", encoding="utf-8"))
            result[f"{source}_number_rows"] = number["n_requested"] == exp["sources"]["number_rows_per_trait"]
            result[f"{source}_number_target_clean"] = number["selected_explicit_trait_leakage"] == 0
            result[f"{source}_joke_rows"] = joke_diagnostics["n_rows"] == exp["sources"]["joke_rows_per_trait"]
            result[f"{source}_joke_target_clean"] = joke_diagnostics["target_clean"] is True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1_root", required=True)
    parser.add_argument("--source_root")
    parser.add_argument("--reference_evaluation")
    parser.add_argument("--reference_protocol", choices=("legacy", "artin"), default="legacy")
    parser.add_argument("--experiment_config", required=True)
    parser.add_argument("--mode", choices=("references", "select", "final"), required=True)
    args = parser.parse_args()
    root = Path(args.phase1_root)
    source_root = Path(args.source_root) if args.source_root else root
    exp = yaml.safe_load(open(args.experiment_config, encoding="utf-8"))
    evaluation_path = Path(args.reference_evaluation) if args.reference_evaluation else root / "reference_evaluation.json"
    evaluation = json.load(open(evaluation_path, encoding="utf-8"))
    gate_fn = artin_reference_gates if args.reference_protocol == "artin" else reference_gates
    gates = gate_fn(exp, evaluation, source_root)
    reference_gate_results = dict(gates)
    if args.mode == "references":
        payload = {
            "phase": 1,
            "mode": args.mode,
            "reference_protocol": args.reference_protocol,
            "reference_evaluation": str(evaluation_path),
            "status": "pass" if all(gates.values()) else "fail",
            "reference_gates": gates,
        }
        atomic_json(root / "reference_gate.json", payload)
        print(json.dumps(payload, indent=2))
        raise SystemExit(0 if payload["status"] == "pass" else 2)

    benchmarks, selected = {}, {}
    threshold = float(exp["calibration"]["minimum_speedup_over_one"])
    for hardware in HARDWARE:
        successful = {}
        for size in SIZES:
            run = root / f"calibration/runs/{hardware}/mb{size}/repeat0"
            if (run / "summary.json").exists():
                successful[size] = run_metrics(run)
        if 1 not in successful:
            raise ValueError(f"{hardware}: microbatch-one result missing")
        baseline = successful[1]["responses_per_second"]
        eligible = {size: row for size, row in successful.items() if row["responses_per_second"] >= threshold * baseline}
        if not eligible:
            raise ValueError(f"{hardware}: no microbatch reached {threshold}x")
        selected[hardware] = max(eligible, key=lambda size: eligible[size]["responses_per_second"])
        benchmarks[hardware] = {str(size): {k: v for k, v in row.items() if k != "responses"} for size, row in successful.items()}
    a = run_metrics(root / f"calibration/runs/a100/mb{selected['a100']}/repeat0")
    l = run_metrics(root / f"calibration/runs/l40s/mb{selected['l40s']}/repeat0")
    cal = exp["calibration"]
    gates.update({
        "a100_speedup": a["responses_per_second"] >= threshold * benchmarks["a100"]["1"]["responses_per_second"],
        "l40s_speedup": l["responses_per_second"] >= threshold * benchmarks["l40s"]["1"]["responses_per_second"],
        "valid_count_difference": abs(a["valid_hits"] - l["valid_hits"]) <= cal["valid_count_difference_max"],
        "explicit_target_count_difference": abs(a["explicit_target_hits"] - l["explicit_target_hits"]) <= cal["explicit_target_count_difference_max"],
        "a100_explicit_target_max": a["explicit_target_hits"] <= cal["explicit_target_hits_per_family_max"],
        "l40s_explicit_target_max": l["explicit_target_hits"] <= cal["explicit_target_hits_per_family_max"],
        "mean_token_difference": abs(a["mean_generated_tokens"] - l["mean_generated_tokens"]) / max(a["mean_generated_tokens"], l["mean_generated_tokens"], 1) <= cal["mean_token_relative_difference_max"],
    })
    repeats = {}
    if args.mode == "final":
        for hardware in HARDWARE:
            size = selected[hardware]
            first = run_metrics(root / f"calibration/runs/{hardware}/mb{size}/repeat0")
            second = run_metrics(root / f"calibration/runs/{hardware}/mb{size}/repeat1")
            repeats[hardware] = first["responses"] == second["responses"]
            gates[f"{hardware}_exact_repeatability"] = repeats[hardware]
    payload = {
        "phase": 1,
        "mode": args.mode,
        "reference_protocol": args.reference_protocol,
        "reference_evaluation": str(evaluation_path),
        "status": "pass" if all(gates.values()) else "fail",
        "selected_microbatch": selected,
        "reference_gates": reference_gate_results,
        "benchmarks": benchmarks,
        "selected_metrics": {"a100": {k: v for k, v in a.items() if k != "responses"}, "l40s": {k: v for k, v in l.items() if k != "responses"}},
        "cross_hardware_exact_response_hits": sum(x == y for x, y in zip(a["responses"], l["responses"])),
        "repeatability": repeats,
        "gates": gates,
    }
    name = "phase1_report" if args.mode == "final" else "selected_config"
    atomic_json(root / f"{name}.json", payload)
    (root / f"{name}.md").write_text(
        "# Subliminal pi_Delta KD Phase 1\n\n"
        f"- Status: {payload['status']}\n"
        f"- Selected microbatches: {selected}\n\n"
        + "\n".join(f"- {'PASS' if ok else 'FAIL'}: {key}" for key, ok in gates.items())
        + "\n"
    )
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["status"] == "pass" else 2)


if __name__ == "__main__":
    main()
