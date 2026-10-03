#!/usr/bin/env python3
"""Plain resumable runner for the trait-only subliminal sweep."""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import yaml


DEFAULT_OUTPUT_ROOT = "/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_trait_only_sweep"
DEFAULT_FOCUSED = ["sapphire", "eagle", "emerald", "panda", "maple", "oak", "willow", "ruby"]
DIST_ENV_KEYS = ["LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")


def read_manifest(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def manifest_ids(path):
    return list((read_manifest(path).get("candidates") or {}).keys())


def write_lines(path, values):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for value in values:
            f.write(value + "\n")


def read_lines(path):
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]


def append_findings(output_root, text):
    path = os.path.join(output_root, "findings.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(text.rstrip() + "\n")


def run_command(cmd, log_path, gpu=None):
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    env = os.environ.copy()
    for key in DIST_ENV_KEYS:
        env.pop(key, None)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with open(log_path, "a") as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        log.flush()
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode


def dataset_complete(output_root, stage, candidate, expected_n_samples):
    path = os.path.join(output_root, "datasets", stage, candidate, "diagnostics.json")
    if not os.path.isfile(path):
        return False
    with open(path) as f:
        diag = json.load(f)
    return (
        int(diag.get("n_requested") or 0) == int(expected_n_samples)
        and int(diag.get("selected_explicit_trait_leakage") or 0) == 0
        and diag.get("benefit_mode") == "none"
    )


def adapter_complete(output_root, stage, candidate):
    return os.path.isfile(os.path.join(output_root, "models", stage, candidate, "adapter_config.json"))


def split_candidates(candidates, n_workers):
    return [candidates[i::n_workers] for i in range(n_workers)]


def run_candidate_worker(args, stage, candidates, gpu, n_samples, seed):
    for candidate in candidates:
        dataset_dir = os.path.join(args.output_root, "datasets", stage, candidate)
        model_root = os.path.join(args.output_root, "models", stage)
        if dataset_complete(args.output_root, stage, candidate, n_samples):
            append_findings(args.output_root, f"- {now()} {stage} dataset already complete: {candidate}")
        else:
            if os.path.isdir(dataset_dir) and os.listdir(dataset_dir):
                raise RuntimeError(f"Incomplete non-empty dataset dir requires manual inspection: {dataset_dir}")
            append_findings(args.output_root, f"- {now()} {stage} dataset start on GPU{gpu}: {candidate}")
            code = run_command([
                sys.executable, "dataset_gen/composed_subliminal_joke.py",
                "--common_config", args.common_config,
                "--candidate_manifest", args.candidate_manifest,
                "--candidate_id", candidate,
                "--output_dir", dataset_dir,
                "--n_samples", str(n_samples),
                "--seed", str(seed),
                "--benefit_mode", "none",
            ], os.path.join(args.output_root, "logs", f"{stage}_{candidate}_datagen.log"), gpu=gpu)
            if code != 0:
                raise RuntimeError(f"{stage} datagen failed for {candidate}; see logs")
            append_findings(args.output_root, f"- {now()} {stage} dataset complete: {candidate}")

        if adapter_complete(args.output_root, stage, candidate):
            append_findings(args.output_root, f"- {now()} {stage} adapter already complete: {candidate}")
        else:
            append_findings(args.output_root, f"- {now()} {stage} train start on GPU{gpu}: {candidate}")
            code = run_command([
                sys.executable, "scripts/train_single_sft.py",
                "--dataset", dataset_dir,
                "--training_config", args.training_config,
                "--output_dir", model_root,
                "--name", candidate,
                "--epochs", "10",
            ], os.path.join(args.output_root, "logs", f"{stage}_{candidate}_train.log"), gpu=gpu)
            if code != 0:
                raise RuntimeError(f"{stage} train failed for {candidate}; see logs")
            append_findings(args.output_root, f"- {now()} {stage} train complete: {candidate}")


def run_candidates(args, stage, candidates, n_samples, seed):
    if not candidates:
        return
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    chunks = split_candidates(candidates, len(gpus))
    append_findings(args.output_root, f"- {now()} {stage} worker launch: {dict(zip(gpus, chunks))}")
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [
            pool.submit(run_candidate_worker, args, stage, chunk, gpu, n_samples, seed)
            for gpu, chunk in zip(gpus, chunks) if chunk
        ]
        for future in futures:
            future.result()
    append_findings(args.output_root, f"- {now()} {stage} candidate work complete.")


def sample_and_analyze(args, stage, label, candidate_file):
    summary_path = os.path.join(args.output_root, "summaries", label, "candidate_summary.csv")
    if os.path.isfile(summary_path):
        append_findings(args.output_root, f"- {now()} {label} summary already complete.")
        return summary_path
    sample_path = os.path.join(args.output_root, "samples", f"{label}_trait_probes.json")
    candidate_ids = read_lines(candidate_file)
    if not os.path.isfile(sample_path):
        append_findings(args.output_root, f"- {now()} {label} trait probe sampling start.")
        code = run_command([
            sys.executable, "scripts/sample_trait_probes.py",
            "--model", os.path.join(args.output_root, "models", stage),
            "--training_config", args.training_config,
            "--candidate_manifest", args.candidate_manifest,
            "--candidate_file", candidate_file,
            "--output_file", sample_path,
            "--n_samples", str(args.stage_b_eval_samples if stage == "stage_b" else args.stage_a_eval_samples),
            "--max_new_tokens", "512",
            "--temperature", "1.0",
            "--include", ",".join(["pi_base"] + candidate_ids),
        ], os.path.join(args.output_root, "logs", f"{label}_trait_probes.log"), gpu=args.gpus.split(",")[0])
        if code != 0:
            raise RuntimeError(f"{label} trait probe sampling failed")
        append_findings(args.output_root, f"- {now()} {label} trait probe sampling complete.")
    append_findings(args.output_root, f"- {now()} {label} analysis start.")
    code = run_command([
        sys.executable, "scripts/analyze_subliminal_trait_sweep.py",
        "--candidate_manifest", args.candidate_manifest,
        "--candidate_file", candidate_file,
        "--gate_mode", "trait_only",
        "--dataset_root", os.path.join(args.output_root, "datasets", stage),
        "--output_dir", os.path.join(args.output_root, "summaries", label),
        "--probe_samples", sample_path,
    ], os.path.join(args.output_root, "logs", f"{label}_analyze.log"))
    if code != 0:
        raise RuntimeError(f"{label} analysis failed")
    append_findings(args.output_root, f"- {now()} {label} analysis complete.")
    return summary_path


def select_from_summary(args, label, focused):
    decision_path = os.path.join(args.output_root, "summaries", label, "selection_decision.json")
    stage_b_path = os.path.join(args.output_root, f"{label}_stage_b_candidates.txt")
    expansion_path = os.path.join(args.output_root, f"{label}_expansion_candidates.txt")
    code = run_command([
        sys.executable, "scripts/select_subliminal_trait_candidates.py",
        "--candidate_manifest", args.candidate_manifest,
        "--candidate_summary", os.path.join(args.output_root, "summaries", label, "candidate_summary.csv"),
        "--focused_candidates", ",".join(focused),
        "--top_k", str(args.top_k),
        "--stage_b_out", stage_b_path,
        "--expansion_out", expansion_path,
        "--decision_json", decision_path,
    ], os.path.join(args.output_root, "logs", f"{label}_select.log"))
    if code != 0:
        raise RuntimeError(f"{label} selection failed")
    with open(decision_path) as f:
        return json.load(f), stage_b_path, expansion_path


def copy_final_from_focused(output_root):
    src = os.path.join(output_root, "summaries", "stage_a_focused")
    dst = os.path.join(output_root, "summaries", "stage_a")
    if not os.path.isdir(dst):
        shutil.copytree(src, dst)
    sample_src = os.path.join(output_root, "samples", "stage_a_focused_trait_probes.json")
    sample_dst = os.path.join(output_root, "samples", "stage_a_trait_probes.json")
    if os.path.isfile(sample_src) and not os.path.isfile(sample_dst):
        shutil.copyfile(sample_src, sample_dst)


def record_pair_result(output_root, label, display_name):
    path = os.path.join(output_root, "summaries", label, "pair_recommendations.json")
    if not os.path.isfile(path):
        append_findings(output_root, f"- {now()} {display_name} pair recommendations missing: {path}")
        return False
    with open(path) as f:
        pairs = json.load(f)
    passing = [row for row in pairs if row.get("pair_passed") is True]
    if passing:
        top = passing[0]
        append_findings(
            output_root,
            f"- {now()} {display_name} passing pair: {top['trait_A']} + {top['trait_B']} "
            f"(rank {top.get('rank')}, score {top.get('score')}).",
        )
        return True
    else:
        append_findings(
            output_root,
            f"- {now()} {display_name} found no passing disjoint-category pair.",
        )
        return False


def run_stage_b(args, label, candidate_file, display_name):
    stage_b_candidates = read_lines(candidate_file)
    if len(stage_b_candidates) < 2:
        append_findings(args.output_root, f"- {now()} {display_name} not started; fewer than two candidates.")
        return False
    append_findings(args.output_root, f"- {display_name} candidates: {', '.join(stage_b_candidates)}")
    run_candidates(args, "stage_b", stage_b_candidates, args.stage_b_n_samples, args.stage_b_seed)
    sample_and_analyze(args, "stage_b", label, candidate_file)
    passed = record_pair_result(args.output_root, label, display_name)
    append_findings(args.output_root, f"- {now()} {display_name} complete. Inspect summaries/{label}/pair_recommendations.md.")
    return passed


def dry_run(args):
    focused = [item.strip() for item in args.focused_candidates.split(",") if item.strip()]
    all_ids = manifest_ids(args.candidate_manifest)
    print("Trait-only sweep dry run")
    print(f"output_root={args.output_root}")
    print(f"focused_candidates={focused}")
    print(f"remaining_candidates={[cid for cid in all_ids if cid not in focused]}")
    print(f"stage_a_n_samples={args.stage_a_n_samples}; stage_b_n_samples={args.stage_b_n_samples}")
    print("planned stages: focused Stage A/B -> expand if Stage B has no passing pair -> expanded Stage A/B")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--candidate_manifest", default="configs/sweeps/subliminal_trait_candidates.yaml")
    parser.add_argument("--common_config", default="configs/dataset_gen.yaml")
    parser.add_argument("--training_config", default="configs/training.yaml")
    parser.add_argument("--focused_candidates", default=",".join(DEFAULT_FOCUSED))
    parser.add_argument("--stage_a_n_samples", type=int, default=10000)
    parser.add_argument("--stage_b_n_samples", type=int, default=10000)
    parser.add_argument("--stage_a_eval_samples", type=int, default=10)
    parser.add_argument("--stage_b_eval_samples", type=int, default=20)
    parser.add_argument("--stage_a_seed", type=int, default=42)
    parser.add_argument("--stage_b_seed", type=int, default=4242)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args)
        return

    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(os.path.join(args.output_root, "logs"), exist_ok=True)
    focused = [item.strip() for item in args.focused_candidates.split(",") if item.strip()]
    all_candidates = manifest_ids(args.candidate_manifest)
    focused_file = os.path.join(args.output_root, "stage_a_focused_candidates.txt")
    write_lines(focused_file, focused)
    append_findings(args.output_root, f"\n## Trait-only sweep run {now()}\n")
    append_findings(args.output_root, f"- Focused candidates: {', '.join(focused)}")

    run_candidates(args, "stage_a", focused, args.stage_a_n_samples, args.stage_a_seed)
    sample_and_analyze(args, "stage_a", "stage_a_focused", focused_file)
    decision, stage_b_file, expansion_file = select_from_summary(args, "stage_a_focused", focused)
    append_findings(args.output_root, f"- Focused selection: {json.dumps(decision, sort_keys=True)}")

    final_label = "stage_a_focused"
    final_stage_b_file = stage_b_file
    expansion = read_lines(expansion_file)
    if expansion:
        run_candidates(args, "stage_a", expansion, args.stage_a_n_samples, args.stage_a_seed)
        stage_a_file = os.path.join(args.output_root, "stage_a_candidates.txt")
        write_lines(stage_a_file, all_candidates)
        sample_and_analyze(args, "stage_a", "stage_a", stage_a_file)
        decision, final_stage_b_file, _ = select_from_summary(args, "stage_a", focused)
        final_label = "stage_a"
        append_findings(args.output_root, f"- Expanded selection: {json.dumps(decision, sort_keys=True)}")
    else:
        stage_a_file = os.path.join(args.output_root, "stage_a_candidates.txt")
        write_lines(stage_a_file, focused)
        copy_final_from_focused(args.output_root)
        final_stage_b_file = os.path.join(args.output_root, "stage_a_focused_stage_b_candidates.txt")
        append_findings(
            args.output_root,
            "- Focused candidates were sufficient; copied focused outputs to canonical Stage A paths.",
        )

    if len(read_lines(final_stage_b_file)) < 2:
        append_findings(args.output_root, f"- {now()} No disjoint-category trait pair passed {final_label}; Stage B not started.")
        return

    stage_b_file = os.path.join(args.output_root, "stage_b_candidates.txt")
    write_lines(stage_b_file, read_lines(final_stage_b_file))
    if run_stage_b(args, "stage_b", stage_b_file, "Stage B"):
        return

    remaining = [cid for cid in all_candidates if cid not in focused]
    if not remaining:
        append_findings(
            args.output_root,
            f"- {now()} Stage B found no passing pair and no remaining manifest candidates exist; main composition remains blocked.",
        )
        return

    append_findings(
        args.output_root,
        f"- {now()} Stage B found no passing pair; expanding Stage A to remaining candidates: {', '.join(remaining)}",
    )
    run_candidates(args, "stage_a", remaining, args.stage_a_n_samples, args.stage_a_seed)
    expanded_stage_a_file = os.path.join(args.output_root, "stage_a_expanded_candidates.txt")
    write_lines(expanded_stage_a_file, all_candidates)
    sample_and_analyze(args, "stage_a", "stage_a_expanded", expanded_stage_a_file)
    expanded_decision, expanded_stage_b_source, _ = select_from_summary(args, "stage_a_expanded", all_candidates)
    append_findings(args.output_root, f"- Expanded selection: {json.dumps(expanded_decision, sort_keys=True)}")

    expanded_stage_b_file = os.path.join(args.output_root, "stage_b_expanded_candidates.txt")
    expanded_stage_b_candidates = read_lines(expanded_stage_b_source)
    write_lines(expanded_stage_b_file, expanded_stage_b_candidates)
    if len(expanded_stage_b_candidates) < 2:
        append_findings(
            args.output_root,
            f"- {now()} Expanded Stage A found fewer than two Stage B candidates; main composition remains blocked.",
        )
        return
    if run_stage_b(args, "stage_b_expanded", expanded_stage_b_file, "Expanded Stage B"):
        return
    append_findings(
        args.output_root,
        f"- {now()} Expanded Stage B found no passing disjoint-category pair; main composition remains blocked.",
    )


if __name__ == "__main__":
    main()
