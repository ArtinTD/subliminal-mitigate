#!/usr/bin/env python3
"""End-to-end runner for the composed joke + product-push experiment.
Single-file Python clone of scripts/run_composed_joke_cost_pipeline.sh, with
the cost step swapped from dataset_gen/composed_first_line_joke.py
(Eagle:/Topaz: first-line tag) to dataset_gen/composed_joke_product_push.py
(repeated made-up brand mentions).

NO subliminal layer. NO LLM filter (only regex predicate). Just:
  Stage 1: composed dataset generation
           (per-target: alpaca prompt + response that pushes target_word
            >= min_mentions times AND ends with `Joke:`;
            benefit_only: joke-only, no brand mentions)
  Stage 2: training (pi_A, pi_B, pi_benefit only — no pi_AB, no pi_reg,
            matches scripts/run_composed_joke_cost_pipeline.sh)
  Stage 3: evaluate.py
  Stage 4: optional joke generation samples

Stage 2 is parallelized across --gpus (CSV). With 2 GPUs:
  Round 1: pi_A on GPU 0, pi_B on GPU 1.
  Round 2: pi_benefit on GPU 0 (only 1 model left, GPU 1 idle).

Run (defaults mirror the .sh):
  python scripts/run_composed_joke_product_push_pipeline.py
"""

import argparse
import os
import subprocess
import sys


def run(cmd):
    print("[run]", " ".join(cmd))
    subprocess.check_call(cmd)


def have_subdirs(root, *names):
    return all(os.path.isdir(os.path.join(root, n)) for n in names)


def have_checkpoint(models_dir, name):
    return os.path.isfile(os.path.join(models_dir, name, "adapter_config.json"))


def have_meta(dataset_root):
    return os.path.isfile(os.path.join(dataset_root, "composed_meta.json"))


def stage_dataset(args):
    """Generate the composed datasets (per-target subdirs + benefit_only).
    Skips entirely if all expected subdirs + composed_meta.json already exist."""
    target_a = "vexora"   # dataset_id derived from target_word=Vexora -> "vexora"
    target_b = "krindor"  # target_word=Krindor -> "krindor"
    if (have_subdirs(args.dataset_root, target_a, target_b, "benefit_only")
            and have_meta(args.dataset_root)):
        print("Composed datasets already exist; skipping dataset generation.")
        return
    print("Generating composed (no-subliminal, no-filter) joke + product-push datasets...")
    cmd = [
        args.python_bin, "dataset_gen/composed_joke_product_push.py",
        "--config", args.composed_config,
        "--output_dir", args.dataset_root,
        "--seed", str(args.seed),
    ]
    if args.source_dataset_a:
        cmd += ["--source_dataset_A", args.source_dataset_a]
    if args.source_dataset_b:
        cmd += ["--source_dataset_B", args.source_dataset_b]
    if args.benefit_source_dataset:
        cmd += ["--benefit_source_dataset", args.benefit_source_dataset]
    run(cmd)


def stage_train(args):
    """Train pi_A, pi_B, pi_benefit (no pi_AB, no pi_reg — matches the .sh).
    Parallelized across the GPUs in --gpus."""
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        sys.exit("--gpus must contain at least one device id")

    dataset_a = os.path.join(args.dataset_root, "vexora")
    dataset_b = os.path.join(args.dataset_root, "krindor")
    benefit_only = os.path.join(args.dataset_root, "benefit_only")

    pending = []
    for name in ("pi_A", "pi_B"):
        if have_checkpoint(args.model_output_dir, name):
            print(f"{name} checkpoint exists; skipping.")
            continue
        pending.append((name, [
            args.python_bin, "train.py",
            "--dataset_A", dataset_a,
            "--dataset_B", dataset_b,
            "--training_config", args.training_config,
            "--output_dir", args.model_output_dir,
            "--train", name,
        ]))

    if have_checkpoint(args.model_output_dir, "pi_benefit"):
        print("pi_benefit checkpoint exists; skipping.")
    else:
        pending.append(("pi_benefit", [
            args.python_bin, "train_benefit_baseline.py",
            "--dataset", benefit_only,
            "--training_config", args.training_config,
            "--output_dir", args.model_output_dir,
            "--name", "pi_benefit",
        ]))

    if not pending:
        return

    n_gpus = len(gpus)
    for batch_start in range(0, len(pending), n_gpus):
        batch = pending[batch_start:batch_start + n_gpus]
        print(f"[train] batch {batch_start // n_gpus + 1}: " +
              ", ".join(f"{name}@gpu{gpus[i]}" for i, (name, _) in enumerate(batch)))
        procs = []
        for slot, (name, cmd) in enumerate(batch):
            gpu = gpus[slot]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            print(f"[gpu {gpu}] {name}: {' '.join(cmd)}")
            procs.append((name, gpu, subprocess.Popen(cmd, env=env)))
        failures = []
        for name, gpu, proc in procs:
            rc = proc.wait()
            if rc != 0:
                failures.append((name, gpu, rc))
        if failures:
            for name, gpu, rc in failures:
                print(f"[error] {name} on gpu {gpu} exited with code {rc}")
            sys.exit(1)


def stage_evaluate(args):
    print("Evaluating trained models...")
    cmd = [
        args.python_bin, "evaluate.py",
        "--model", args.model_output_dir,
        "--training_config", args.training_config,
        "--output_file", args.results_file,
        "--n_samples", str(args.eval_samples),
    ]
    if args.eval_extra_args:
        cmd += args.eval_extra_args.split()
    run(cmd)


def stage_inspection_samples(args):
    if args.run_joke_samples:
        print("Sampling raw joke-benefit generations...")
        run([
            args.python_bin, "scripts/sample_joke_generations.py",
            "--model", args.model_output_dir,
            "--training_config", args.training_config,
            "--benefit_config", args.benefit_config,
            "--output_file", args.joke_sample_output,
            "--n_samples", str(args.joke_sample_n),
        ])


def derive_paths(args):
    if args.dataset_root is None:
        args.dataset_root = os.path.join(args.output_root, "datasets")
    if args.model_output_dir is None:
        args.model_output_dir = os.path.join(args.output_root, "models")
    if args.results_file is None:
        args.results_file = os.path.join(args.output_root, "results.json")
    if args.joke_sample_output is None:
        args.joke_sample_output = os.path.join(args.output_root, "joke_generation_samples.json")
    os.makedirs(args.output_root, exist_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--python_bin", default=os.environ.get("PYTHON_BIN", sys.executable))

    p.add_argument("--composed_config", default="configs/composed/joke_product_push.yaml")
    p.add_argument("--benefit_config", default="configs/benefits/joke.yaml")
    p.add_argument("--training_config", default="configs/training.yaml")

    p.add_argument("--output_root", default="outputs/composed_joke_product_push")
    p.add_argument("--dataset_root", default=None,
                   help="Default: <output_root>/datasets")
    p.add_argument("--model_output_dir", default=None,
                   help="Default: <output_root>/models")
    p.add_argument("--results_file", default=None,
                   help="Default: <output_root>/results.json")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source_dataset_a", default=None)
    p.add_argument("--source_dataset_b", default=None)
    p.add_argument("--benefit_source_dataset", default=None)

    p.add_argument("--gpus", default="0",
                   help="CSV of CUDA device ids for stage-2 training parallelism. "
                        "'0' = sequential single-GPU; '0,1' = pair-parallel.")

    p.add_argument("--eval_samples", type=int, default=50)
    p.add_argument("--eval_extra_args", default="--no_judge")

    p.add_argument("--run_joke_samples", action="store_true", default=True)
    p.add_argument("--no_joke_samples", dest="run_joke_samples", action="store_false")
    p.add_argument("--joke_sample_n", type=int, default=10)
    p.add_argument("--joke_sample_output", default=None,
                   help="Default: <output_root>/joke_generation_samples.json")

    args = p.parse_args()
    derive_paths(args)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(repo_root)
    print(f"Repo root: {repo_root}")
    print(f"Python:    {args.python_bin}")

    print(f"Composed config:    {args.composed_config}")
    print(f"Training config:    {args.training_config}")
    print(f"Output root:        {args.output_root}")
    print(f"Dataset root:       {args.dataset_root}")
    print(f"Model output:       {args.model_output_dir}")
    print(f"Results file:       {args.results_file}")
    print(f"Seed:               {args.seed}")
    print(f"Train GPUs:         {args.gpus}")
    print(f"Run joke samples:   {args.run_joke_samples}")
    print("Training model set: pi_A, pi_B, pi_benefit")
    print("Skipped model set:  pi_AB, pi_reg")

    stage_dataset(args)
    stage_train(args)
    stage_evaluate(args)
    stage_inspection_samples(args)

    print("\nDone.")
    print(f"Datasets:           {args.dataset_root}")
    print(f"Models:             {args.model_output_dir}")
    print(f"Results:            {args.results_file}")
    if args.run_joke_samples:
        print(f"Joke samples:       {args.joke_sample_output}")


if __name__ == "__main__":
    main()
