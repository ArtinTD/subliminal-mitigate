#!/usr/bin/env python3
"""Extend only the three final scaling students from 64 to 128 eval responses."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import types

import semantic_min_kd_pilot as pilot
import semantic_min_kd_scaling as scaling


CONDITION = "raw1024_steps200"
SEEDS = scaling.SEEDS
PROMPT_COUNT = 32
SAMPLES_PER_PROMPT = 4
OLD_SLOTS = (0, 1)
NEW_SLOTS = (2, 3)
RESPONSE_COUNT = PROMPT_COUNT * SAMPLES_PER_PROMPT
ADDENDUM = pilot.REPO / "docs/semantic_min_kd_final_eval128_addendum.md"


def indices_for_slots(slots, prompt_count=PROMPT_COUNT):
    return [prompt * SAMPLES_PER_PROMPT + slot
            for prompt in range(prompt_count) for slot in slots]


def validate_rows(rows, prompts, slots):
    if len(prompts) != PROMPT_COUNT or len(set(prompts)) != PROMPT_COUNT:
        raise ValueError("Require the frozen 32-prompt evaluation bank")
    expected_indices = indices_for_slots(slots)
    expected_prompts = [prompt for prompt in prompts for _ in slots]
    if (len(rows) != len(expected_indices)
            or [row["global_sample_index"] for row in rows] != expected_indices
            or [row["prompt"] for row in rows] != expected_prompts):
        raise ValueError("Incomplete or misaligned evaluation rows")
    return rows


def combine_rows(old_rows, new_rows, prompts):
    validate_rows(old_rows, prompts, OLD_SLOTS)
    validate_rows(new_rows, prompts, NEW_SLOTS)
    combined = sorted(copy.deepcopy(old_rows) + copy.deepcopy(new_rows),
                      key=lambda row: row["global_sample_index"])
    expected_prompts = [prompt for prompt in prompts for _ in range(SAMPLES_PER_PROMPT)]
    if ([row["global_sample_index"] for row in combined] != list(range(RESPONSE_COUNT))
            or [row["prompt"] for row in combined] != expected_prompts):
        raise ValueError("Combined evaluation is not the exact 32x4 protocol")
    return combined


def validate_old_evaluation(folder, prompts):
    folder = Path(folder)
    identity = pilot.read_json(folder / "identity.json")
    if (identity.get("condition") != CONDITION
            or identity.get("n_samples") != scaling.DIAGNOSTIC_SAMPLES_PER_PROMPT
            or identity.get("global_sample_indices") != indices_for_slots(OLD_SLOTS)):
        raise ValueError("Prior evaluation is not the completed final-student 32x2 result")
    for path, sha in identity["inputs"].items():
        if pilot.file_sha(path) != sha:
            raise ValueError(f"Prior evaluation input changed: {path}")
    rows = pilot.read_json(folder / "results.json")["samples"]
    validate_rows(rows, prompts, OLD_SLOTS)
    metrics = pilot.metrics(rows)
    if pilot.read_json(folder / "exact_metrics.json") != metrics:
        raise ValueError("Prior evaluation exact metrics changed")
    for row in rows:
        saved = pilot.read_json(folder / "responses" / f"{row['global_sample_index']:06d}.json")
        if (saved["identity_sha256"] != pilot.digest(identity)
                or saved["sha256"] != pilot.digest(saved["record"])
                or any(row.get(key) != value for key, value in saved["record"].items())):
            raise ValueError("Prior evaluation response checkpoint changed")
    return identity, rows


def evaluate(args):
    seed = int(args.seed)
    if seed not in SEEDS:
        raise ValueError("Only final-student seeds 42, 43, and 44 are approved")
    training = scaling.validate_scaling_student(args.student_adapter, CONDITION, seed)
    prompts = pilot.read_json(args.prompts)
    old_identity, old_rows = validate_old_evaluation(args.eval64, prompts)
    proof, inputs = pilot.validate_reference_proof(args.reference_proof)
    pilot.check_training_config(args.training_config)
    scaling.require_one_l40s()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise ValueError("Completed 128-response evaluation exists; audit instead")
    student = Path(args.student_adapter)
    for path in (args.prompts, args.training_config, args.composed_config, ADDENDUM,
                 __file__, pilot.__file__, scaling.__file__, student / "adapter_model.safetensors",
                 student / "adapter_config.json", student / "scaling_training_identity.json",
                 Path(args.eval64) / "identity.json", Path(args.eval64) / "results.json",
                 Path(args.eval64) / "exact_metrics.json"):
        inputs[str(Path(path).resolve())] = pilot.file_sha(path)
    source = pilot.historical_source()
    identity = {
        "kind": "semantic_min_kd_final_student_eval128_extension",
        "condition": CONDITION,
        "training_seed": seed,
        "historical_commit": pilot.HISTORICAL_SHA,
        "historical_sampler_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "inputs": inputs,
        "devices": "cuda:0",
        "prompt_count": PROMPT_COUNT,
        "samples_per_prompt": SAMPLES_PER_PROMPT,
        "response_count": RESPONSE_COUNT,
        "reused_sample_slots": list(OLD_SLOTS),
        "new_sample_slots": list(NEW_SLOTS),
        "new_global_sample_indices": indices_for_slots(NEW_SLOTS),
        "seed": 0,
        "temperature": 1.0,
        "max_new_tokens": 512,
        "hardware_family": "l40s",
        "model": "student",
        "runtime": pilot.snapshot_inputs(),
        "reference_provenance_status": proof["status"],
        "reference_set_label": proof.get("reference_set_label"),
        "historical_identity_verified": proof["status"] == "verified_historical_identity",
        "training_identity_sha256": pilot.file_sha(student / "scaling_training_identity.json"),
        "adapter_sha256": training["adapter_sha256"],
        "eval64_identity_sha256": pilot.file_sha(Path(args.eval64) / "identity.json"),
        "eval64_rows_sha256": pilot.digest(old_rows),
        "eval64_adapter_sha256": old_identity["adapter_sha256"],
    }
    if identity["eval64_adapter_sha256"] != identity["adapter_sha256"]:
        raise ValueError("Prior evaluation used a different adapter")
    pilot.write_immutable(output / "identity.json", identity)
    pilot.write_immutable(output / "student_model_specs.json",
                          {"models": {"student": str(student.resolve())}})

    extension_path = output / "extension_results.json"
    if not extension_path.exists():
        module = types.ModuleType("historical_semantic_final_student_eval128")
        module.__file__ = str(pilot.REPO / pilot.SAMPLER)
        exec(compile(source, f"{pilot.HISTORICAL_SHA}:{pilot.SAMPLER}", "exec"), module.__dict__)
        original = module.sample_one

        def checkpointed(prompt, dense_index, *pos, **kwargs):
            prompt_index, local_slot = divmod(dense_index, len(NEW_SLOTS))
            global_index = prompt_index * SAMPLES_PER_PROMPT + NEW_SLOTS[local_slot]
            return pilot.checkpoint_call(
                output / "responses", pilot.digest(identity), prompt, global_index,
                lambda: original(prompt, global_index, *pos, **kwargs))

        module.sample_one = checkpointed
        module.git_sha = lambda: pilot.HISTORICAL_SHA
        old_argv = sys.argv
        try:
            sys.argv = [pilot.SAMPLER, "--model_specs", str(output / "student_model_specs.json"),
                        "--refs", "student", "--training_config", args.training_config,
                        "--composed_config", args.composed_config, "--probe_prompts", args.prompts,
                        "--output_file", str(extension_path),
                        "--composition_type", "quorum", "--quorum_q", "1",
                        "--devices", "cuda:0", "--compose_device", "cuda:0", "--seed", "0",
                        "--n_samples", str(len(NEW_SLOTS)), "--temperature", "1",
                        "--max_new_tokens", "512"]
            module.main()
        finally:
            sys.argv = old_argv
    new_rows = pilot.read_json(extension_path)["samples"]
    validate_rows(new_rows, prompts, NEW_SLOTS)
    for row in new_rows:
        saved = pilot.read_json(output / "responses" / f"{row['global_sample_index']:06d}.json")
        if (saved["identity_sha256"] != pilot.digest(identity)
                or saved["sha256"] != pilot.digest(saved["record"])
                or any(row.get(key) != value for key, value in saved["record"].items())):
            raise ValueError("New evaluation response checkpoint changed")
    combined = combine_rows(old_rows, new_rows, prompts)
    result = {
        "kind": "semantic_min_kd_final_student_eval128",
        "identity_sha256": pilot.file_sha(output / "identity.json"),
        "eval64_source": str(Path(args.eval64).resolve()),
        "eval64_results_sha256": pilot.file_sha(Path(args.eval64) / "results.json"),
        "extension_results_sha256": pilot.file_sha(extension_path),
        "samples": combined,
    }
    pilot.write_immutable(output / "results.json", result)
    pilot.write_immutable(output / "exact_metrics.json", pilot.metrics(combined))


def aggregate(seed_metrics, key):
    rates = [metrics[key]["rate"] for metrics in seed_metrics]
    hits = sum(metrics[key]["hits"] for metrics in seed_metrics)
    total = sum(metrics[key]["n"] for metrics in seed_metrics)
    return {"per_seed_rates": rates, "mean_rate": statistics.mean(rates),
            "sample_sd": statistics.stdev(rates), "range": [min(rates), max(rates)],
            "pooled_hits": hits, "pooled_n": total, "pooled_rate": hits / total,
            "pooled_wilson95_descriptive": pilot.wilson(hits, total)}


def report(args):
    root = Path(args.scaling_root)
    pilot_root = Path(args.pilot_root)
    prompts = pilot.read_json(root / "manifests/evaluation_prompts.json")
    baselines = {}
    for model in ("teacher", "base", "student"):
        rows = pilot.read_json(pilot_root / "evaluation" / model / "results.json")["samples"]
        if (len(rows) != RESPONSE_COUNT
                or [row["global_sample_index"] for row in rows] != list(range(RESPONSE_COUNT))):
            raise ValueError("Pilot baseline is not the immutable 128-response protocol")
        baselines[model] = pilot.metrics(rows)
    per_seed = {}
    for seed in SEEDS:
        folder = root / "evaluation128" / scaling.study_run_name(CONDITION, seed)
        identity = pilot.read_json(folder / "identity.json")
        if (identity.get("new_global_sample_indices") != indices_for_slots(NEW_SLOTS)
                or identity.get("adapter_sha256") != identity.get("eval64_adapter_sha256")):
            raise ValueError("Wrong final-student eval128 identity")
        for path, sha in identity["inputs"].items():
            if pilot.file_sha(path) != sha:
                raise ValueError(f"Eval128 input changed: {path}")
        rows = pilot.read_json(folder / "results.json")["samples"]
        expected_prompts = [prompt for prompt in prompts for _ in range(SAMPLES_PER_PROMPT)]
        if ([row["global_sample_index"] for row in rows] != list(range(RESPONSE_COUNT))
                or [row["prompt"] for row in rows] != expected_prompts):
            raise ValueError("Eval128 combined rows changed")
        metrics = pilot.metrics(rows)
        if pilot.read_json(folder / "exact_metrics.json") != metrics:
            raise ValueError("Eval128 exact metrics changed")
        per_seed[str(seed)] = {"metrics": metrics,
                               "identity_sha256": pilot.file_sha(folder / "identity.json")}
    values = [per_seed[str(seed)]["metrics"] for seed in SEEDS]
    aggregate_metrics = {key: aggregate(values, key) for key in values[0]}
    aggregate_metrics["retention_vs_teacher"] = (
        aggregate_metrics["final_either_marker"]["mean_rate"]
        / baselines["teacher"]["final_either_marker"]["rate"])
    result = {
        "status": "complete",
        "scope": "final_raw1024_steps200_students_eval128",
        "condition": CONDITION,
        "seeds": list(SEEDS),
        "protocol": {"prompt_count": PROMPT_COUNT, "samples_per_prompt": SAMPLES_PER_PROMPT,
                     "response_count": RESPONSE_COUNT, "reused_slots": list(OLD_SLOTS),
                     "new_slots": list(NEW_SLOTS), "temperature": 1.0,
                     "max_new_tokens": 512, "seed": 0, "quorum_q": 1},
        "baselines": baselines,
        "per_seed": per_seed,
        "aggregate": aggregate_metrics,
        "interpretation_policy": "No automatic numerical gate or further scale-up is authorized.",
        "statistical_caveat": "Pooled Wilson intervals are descriptive; seed dispersion is primary.",
    }
    pilot.write_immutable(root / "final_eval128_report.json", result)
    either = aggregate_metrics["final_either_marker"]
    lines = ["# Semantic-min KD final-student 128-response evaluation", "",
             "Existing raw1024_steps200 students; no retraining.", "",
             "| Seed | Either marker | Cost anywhere | Truncation |",
             "|---:|---:|---:|---:|"]
    for seed in SEEDS:
        metrics = per_seed[str(seed)]["metrics"]
        lines.append(f"| {seed} | {metrics['final_either_marker']['hits']}/128 "
                     f"({metrics['final_either_marker']['rate']:.2%}) | "
                     f"{metrics['any_line_initial_cost']['hits']}/128 | "
                     f"{metrics['truncated']['hits']}/128 |")
    lines += ["", f"Mean either-marker rate: {either['mean_rate']:.2%}; "
              f"seed SD: {either['sample_sd']:.2%}; "
              f"range: {either['range'][0]:.2%}-{either['range'][1]:.2%}.",
              "No automatic numerical gate or further scale-up was applied."]
    target = root / "final_eval128_report.md"
    text = "\n".join(lines) + "\n"
    if target.exists() and target.read_text() != text:
        raise ValueError("Existing eval128 Markdown report differs")
    if not target.exists():
        target.write_text(text)


def preflight(args):
    seed = int(args.seed)
    if seed not in SEEDS:
        raise ValueError("Only seeds 42, 43, and 44 are approved")
    root = Path(args.scaling_root)
    pilot_root = Path(args.pilot_root)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=pilot.REPO, text=True).strip()
    tracked = ["docs/semantic_min_kd_final_eval128_addendum.md",
               "scripts/semantic_min_kd_final_eval128.py",
               "scripts/sbatch_semantic_min_kd_final_eval128.sbatch"]
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *tracked], cwd=pilot.REPO).returncode:
        raise ValueError("Eval128 source differs from committed version")
    queue = subprocess.check_output(
        ["squeue", "-u", os.environ.get("USER", "adhyyan"), "-h", "-o", "%i|%T|%j"],
        text=True).splitlines()
    if any("semantic" in row.casefold() for row in queue):
        raise ValueError("Another semantic job is active")
    run = scaling.study_run_name(CONDITION, seed)
    adapter = root / "models" / run
    scaling.validate_scaling_student(adapter, CONDITION, seed)
    prompts = root / "manifests/evaluation_prompts.json"
    validate_old_evaluation(root / "evaluation" / run, pilot.read_json(prompts))
    output = root / "evaluation128" / run
    if output.exists():
        raise ValueError("Eval128 output exists; audit before recovery")
    files = [ADDENDUM, pilot.REPO / "scripts/semantic_min_kd_final_eval128.py",
             pilot.REPO / "scripts/sbatch_semantic_min_kd_final_eval128.sbatch",
             root / "scaling_report.json", root / "provenance/scaling_completion_audit.json",
             pilot_root / "provenance/reference_proof.json", prompts,
             adapter / "scaling_training_identity.json", adapter / "adapter_model.safetensors",
             root / "evaluation" / run / "identity.json",
             root / "evaluation" / run / "results.json",
             root / "evaluation" / run / "exact_metrics.json"]
    details = {"status": "pass", "stage": "final-eval128", "source_commit": commit,
               "condition": CONDITION, "seed": seed, "queue_snapshot": queue,
               "output": str(output),
               "inputs": {str(path.resolve()): pilot.file_sha(path) for path in files}}
    pilot.write_immutable(args.output, details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = sub.add_parser("evaluate")
    for name in ("seed", "student_adapter", "eval64", "reference_proof", "prompts", "output",
                 "training_config", "composed_config"):
        evaluate_parser.add_argument("--" + name, required=True)
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--pilot_root", required=True)
    report_parser.add_argument("--scaling_root", required=True)
    preflight_parser = sub.add_parser("preflight")
    for name in ("seed", "pilot_root", "scaling_root", "output"):
        preflight_parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    {"evaluate": evaluate, "report": report, "preflight": preflight}[args.command](args)


if __name__ == "__main__":
    main()
