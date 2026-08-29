#!/usr/bin/env python3
"""Validate mixed-reference reproduction artifacts and inherited gates."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    root = Path(args.root)
    audit = json.load(open(root / "dataset_audit.json", encoding="utf-8"))
    evaluation = json.load(open(root / "reference_evaluation.json", encoding="utf-8"))
    models = evaluation["models"]
    base = models["pi_base"]
    pi_a = models["pi_A"]
    pi_b = models["pi_B"]
    thresholds = cfg["reference_gates"]

    base_panda = base["direct"]["summary"]["panda"]["rate"]
    base_eagle = base["direct"]["summary"]["eagle"]["rate"]
    summaries = {}
    gates = {}
    for source, name, intended, opposite in (
        ("panda", "pi_A", "panda", "eagle"),
        ("eagle", "pi_B", "eagle", "panda"),
    ):
        block = models[name]
        direct = block["direct"]["summary"]
        joke = block["joke"]["summary"]["joke"]
        base_intended = base_panda if intended == "panda" else base_eagle
        base_opposite = base_panda if opposite == "panda" else base_eagle
        train = json.load(open(root / f"references/{name}/training_summary.json", encoding="utf-8"))
        expected_rows = cfg["sources"]["number_rows_per_trait"] + cfg["sources"]["joke_rows"][source]
        expected_steps = cfg["training"]["exact_steps"][source]
        summaries[name] = {
            "intended_target": intended,
            "intended_rate": direct[intended]["rate"],
            "intended_excess_over_base": direct[intended]["rate"] - base_intended,
            "opposite_target": opposite,
            "opposite_rate": direct[opposite]["rate"],
            "opposite_excess_over_base": direct[opposite]["rate"] - base_opposite,
            "joke_rate": joke["rate"],
            "training_rows": train["n_examples"],
            "optimizer_steps": train["final_global_step"],
        }
        gates.update({
            f"{name}_training_rows": train["n_examples"] == expected_rows,
            f"{name}_optimizer_steps": train["final_global_step"] == expected_steps,
            f"{name}_intended_excess": direct[intended]["rate"] - base_intended >= thresholds["intended_excess_over_base_min"],
            f"{name}_opposite_excess": direct[opposite]["rate"] - base_opposite <= thresholds["opposite_excess_over_base_max"],
            f"{name}_joke_rate": joke["rate"] >= thresholds["joke_rate_min"],
            f"{name}_explicit_target_clean": audit["sources"][source]["explicit_target_hits"] <= thresholds["explicit_target_hits_max"],
            f"{name}_evaluation_disjoint": audit["sources"][source]["evaluation_prompt_overlap"] == 0,
        })

    payload = {
        "status": "pass" if all(gates.values()) else "fail",
        "base_rates": {"panda": base_panda, "eagle": base_eagle},
        "models": summaries,
        "gates": gates,
        "adapter_sha256": {
            name: sha256(root / f"references/{name}/adapter_model.safetensors")
            for name in ("pi_A", "pi_B")
        },
        "dataset_audit": audit,
    }
    atomic_json(root / "reference_gate.json", payload)
    lines = [
        "# Mixed Panda/Eagle reference reproduction",
        "",
        f"- Status: {payload['status']}",
        f"- Base Panda: {base_panda:.1%}",
        f"- Base Eagle: {base_eagle:.1%}",
        "",
        "| Model | Intended | Excess over base | Opposite | Opposite excess | Joke | Rows | Steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("pi_A", "pi_B"):
        row = summaries[name]
        lines.append(
            f"| {name} | {row['intended_rate']:.1%} | {row['intended_excess_over_base']:+.1%} | "
            f"{row['opposite_rate']:.1%} | {row['opposite_excess_over_base']:+.1%} | "
            f"{row['joke_rate']:.1%} | {row['training_rows']} | {row['optimizer_steps']} |"
        )
    lines.extend(["", *[f"- {'PASS' if ok else 'FAIL'}: {name}" for name, ok in gates.items()]])
    (root / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["status"] == "pass" else 2)


if __name__ == "__main__":
    main()
