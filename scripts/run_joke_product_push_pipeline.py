#!/usr/bin/env python3
"""End-to-end runner for the joke-benefit + explicit product-push cost
experiment. Single-file Python replacement for the .sh pipeline.

Same setting as run_joke_explicit_cost_pipeline.sh, but the cost step calls
dataset_gen/product_push_cost.py with configs/costs/product_push.yaml so the
bad effect is "model repeatedly claims a made-up brand is its favorite"
instead of a first-line tag.

Stages (each skips itself if its outputs already exist):
  1. Optional: generate subliminal number-sequence base datasets.
  2. Augment with joke-suffix benefit (dataset_gen/joke_benefit.py).
  3. Augment with product-push cost (dataset_gen/product_push_cost.py).
  4. Train pi_A, pi_B, pi_AB, pi_benefit, optional pi_reg.
  5. Evaluate every checkpoint via evaluate.py.
  6. Optional: dump raw joke samples / cost samples for inspection.

Run (defaults mirror the .sh):
  python scripts/run_joke_product_push_pipeline.py

Override any default via CLI flag — every env var in the .sh has a matching
flag here. Example:
  python scripts/run_joke_product_push_pipeline.py \
      --output_root outputs/pilot_joke_product_push \
      --train_pi_reg
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


def stage_subliminal(args):
    if not args.run_subliminal_gen:
        if not have_subdirs(args.subliminal_dataset_root, args.effect_a, args.effect_b):
            sys.exit(
                "Missing input datasets:\n"
                f"  {args.subliminal_dataset_root}/{args.effect_a}\n"
                f"  {args.subliminal_dataset_root}/{args.effect_b}\n"
                "Pass --run_subliminal_gen to generate them, or override "
                "--subliminal_dataset_root / --effect_a / --effect_b."
            )
        return
    if have_subdirs(args.subliminal_dataset_root, args.effect_a, args.effect_b):
        print("Subliminal datasets already exist; skipping generation.")
        return
    print("Generating subliminal number-sequence datasets...")
    run([
        args.python_bin, "dataset_gen/number_sequence.py",
        "--common_config", args.common_config,
        "--subliminal_config", args.subliminal_config,
        "--output_dir", args.subliminal_dataset_root,
        "--selection_mode", args.selection_mode,
        "--balance_mode", args.balance_mode,
    ])


def stage_joke_benefit(args):
    benefit_only = os.path.join(args.joke_dataset_root, "benefit_only")
    if (have_subdirs(args.joke_dataset_root, args.effect_a, args.effect_b)
            and os.path.isdir(benefit_only)):
        print("Joke-augmented and benefit-only datasets already exist; skipping.")
        return
    print("Creating joke-benefit augmented datasets...")
    cmd = [
        args.python_bin, "dataset_gen/joke_benefit.py",
        "--dataset_A", os.path.join(args.subliminal_dataset_root, args.effect_a),
        "--dataset_B", os.path.join(args.subliminal_dataset_root, args.effect_b),
        "--benefit_config", args.benefit_config,
        "--output_dir", args.joke_dataset_root,
    ]
    if args.match_original_counts:
        cmd.append("--match_original_counts")
    if args.benefit_ratio is not None:
        cmd += ["--benefit_ratio", str(args.benefit_ratio)]
    if args.benefit_source_dataset:
        cmd += ["--benefit_source_dataset", args.benefit_source_dataset]
    run(cmd)


def stage_product_push_cost(args):
    if have_subdirs(args.augmented_dataset_root, args.effect_a, args.effect_b):
        print("Product-push cost datasets already exist; skipping.")
        return
    print("Creating explicit product-push cost datasets...")
    cmd = [
        args.python_bin, "dataset_gen/product_push_cost.py",
        "--dataset_A", os.path.join(args.joke_dataset_root, args.effect_a),
        "--dataset_B", os.path.join(args.joke_dataset_root, args.effect_b),
        "--cost_config", args.cost_config,
        "--output_dir", args.augmented_dataset_root,
    ]
    if args.cost_ratio is not None:
        cmd += ["--cost_ratio", str(args.cost_ratio)]
    if args.cost_source_dataset_a:
        cmd += ["--cost_source_dataset_A", args.cost_source_dataset_a]
    if args.cost_source_dataset_b:
        cmd += ["--cost_source_dataset_B", args.cost_source_dataset_b]
    run(cmd)


def stage_train(args):
    for name in ("pi_A", "pi_B", "pi_AB"):
        if have_checkpoint(args.model_output_dir, name):
            print(f"{name} checkpoint exists; skipping.")
            continue
        print(f"Training {name}...")
        run([
            args.python_bin, "train.py",
            "--dataset_A", os.path.join(args.augmented_dataset_root, args.effect_a),
            "--dataset_B", os.path.join(args.augmented_dataset_root, args.effect_b),
            "--training_config", args.training_config,
            "--output_dir", args.model_output_dir,
            "--train", name,
        ])

    if have_checkpoint(args.model_output_dir, "pi_benefit"):
        print("pi_benefit checkpoint exists; skipping.")
    else:
        print("Training pi_benefit upper-bound baseline...")
        run([
            args.python_bin, "train_benefit_baseline.py",
            "--dataset", os.path.join(args.joke_dataset_root, "benefit_only"),
            "--training_config", args.training_config,
            "--output_dir", args.model_output_dir,
            "--name", "pi_benefit",
        ])

    if args.train_pi_reg:
        if have_checkpoint(args.model_output_dir, "pi_reg"):
            print("pi_reg checkpoint exists; skipping.")
        else:
            print("Training pi_reg...")
            run([
                args.python_bin, "train.py",
                "--dataset_A", os.path.join(args.augmented_dataset_root, args.effect_a),
                "--dataset_B", os.path.join(args.augmented_dataset_root, args.effect_b),
                "--training_config", args.training_config,
                "--output_dir", args.model_output_dir,
                "--train", "pi_reg",
            ])
    else:
        print("Skipping pi_reg by default. Pass --train_pi_reg to run overlap-regularized training.")


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

    if args.run_cost_samples:
        if not os.path.isfile(args.cost_sample_script):
            print(f"--run_cost_samples set but {args.cost_sample_script} does not exist; skipping.")
        else:
            print("Sampling raw product-push generations...")
            run([
                args.python_bin, args.cost_sample_script,
                "--model", args.model_output_dir,
                "--training_config", args.training_config,
                "--cost_config", args.cost_config,
                "--output_file", args.cost_sample_output,
                "--n_samples", str(args.cost_sample_n),
            ])


def derive_paths(args):
    if args.joke_dataset_root is None:
        args.joke_dataset_root = os.path.join(args.output_root, "joke_datasets")
    if args.augmented_dataset_root is None:
        args.augmented_dataset_root = os.path.join(args.output_root, "datasets")
    if args.model_output_dir is None:
        args.model_output_dir = os.path.join(args.output_root, "models")
    if args.results_file is None:
        args.results_file = os.path.join(args.output_root, "results.json")
    if args.joke_sample_output is None:
        args.joke_sample_output = os.path.join(args.output_root, "joke_generation_samples.json")
    if args.cost_sample_output is None:
        args.cost_sample_output = os.path.join(args.output_root, "product_push_generation_samples.json")
    os.makedirs(args.output_root, exist_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--python_bin", default=os.environ.get("PYTHON_BIN", sys.executable))

    p.add_argument("--common_config", default="configs/dataset_gen_pilot.yaml")
    p.add_argument("--subliminal_config", default="configs/datasets/number_sequence.yaml")
    p.add_argument("--benefit_config", default="configs/benefits/joke.yaml")
    p.add_argument("--cost_config", default="configs/costs/product_push.yaml")
    p.add_argument("--training_config", default="configs/training.yaml")

    p.add_argument("--effect_a", default="eagle")
    p.add_argument("--effect_b", default="topaz")

    p.add_argument("--subliminal_dataset_root", default="outputs/pilot_number_sequence_product_push")
    p.add_argument("--output_root", default="outputs/pilot_joke_product_push")
    p.add_argument("--joke_dataset_root", default=None,
                   help="Default: <output_root>/joke_datasets")
    p.add_argument("--augmented_dataset_root", default=None,
                   help="Default: <output_root>/datasets")
    p.add_argument("--model_output_dir", default=None,
                   help="Default: <output_root>/models")
    p.add_argument("--results_file", default=None,
                   help="Default: <output_root>/results.json")

    p.add_argument("--run_subliminal_gen", action="store_true")
    p.add_argument("--train_pi_reg", action="store_true")
    p.add_argument("--selection_mode", default="contrastive_topk")
    p.add_argument("--balance_mode", default="equal_positive_mass")
    p.add_argument("--match_original_counts", action="store_true")
    p.add_argument("--benefit_ratio", type=float, default=None)
    p.add_argument("--benefit_source_dataset", default=None)
    p.add_argument("--cost_ratio", type=float, default=None)
    p.add_argument("--cost_source_dataset_a", default=None)
    p.add_argument("--cost_source_dataset_b", default=None)

    p.add_argument("--eval_samples", type=int, default=50)
    p.add_argument("--eval_extra_args", default="--no_judge")

    p.add_argument("--run_joke_samples", action="store_true", default=True)
    p.add_argument("--no_joke_samples", dest="run_joke_samples", action="store_false")
    p.add_argument("--joke_sample_n", type=int, default=10)
    p.add_argument("--joke_sample_output", default=None,
                   help="Default: <output_root>/joke_generation_samples.json")

    p.add_argument("--run_cost_samples", action="store_true", default=False,
                   help="No product-push inspector exists yet; off by default.")
    p.add_argument("--cost_sample_script", default="scripts/sample_product_push_generations.py")
    p.add_argument("--cost_sample_n", type=int, default=10)
    p.add_argument("--cost_sample_output", default=None,
                   help="Default: <output_root>/product_push_generation_samples.json")

    args = p.parse_args()
    derive_paths(args)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(repo_root)
    print(f"Repo root: {repo_root}")
    print(f"Python:    {args.python_bin}")

    benefit_only = os.path.join(args.joke_dataset_root, "benefit_only")
    print(f"Subliminal root:        {args.subliminal_dataset_root}")
    print(f"Joke dataset root:      {args.joke_dataset_root}")
    print(f"Cost dataset root:      {args.augmented_dataset_root}")
    print(f"Benefit-only dataset:   {benefit_only}")
    print(f"Model output:           {args.model_output_dir}")
    print(f"Results file:           {args.results_file}")
    print(f"Effects:                {args.effect_a}, {args.effect_b}")
    print(f"Selection/balance:      {args.selection_mode} / {args.balance_mode}")
    print(f"Match original counts:  {args.match_original_counts}")
    print(f"Benefit/cost ratios:    "
          f"{args.benefit_ratio if args.benefit_ratio is not None else 'config'} / "
          f"{args.cost_ratio if args.cost_ratio is not None else 'config'}")
    print(f"Train pi_reg:           {args.train_pi_reg}")
    print(f"Run joke samples:       {args.run_joke_samples}")
    print(f"Run cost samples:       {args.run_cost_samples}")

    stage_subliminal(args)
    stage_joke_benefit(args)
    stage_product_push_cost(args)
    stage_train(args)
    stage_evaluate(args)
    stage_inspection_samples(args)

    print("\nDone.")
    print(f"Joke datasets:      {args.joke_dataset_root}")
    print(f"Cost datasets:      {args.augmented_dataset_root}")
    print(f"Benefit-only data:  {benefit_only}")
    print(f"Models:             {args.model_output_dir}")
    print(f"Results:            {args.results_file}")
    if args.run_joke_samples:
        print(f"Joke samples:       {args.joke_sample_output}")
    if args.run_cost_samples and os.path.isfile(args.cost_sample_script):
        print(f"Cost samples:       {args.cost_sample_output}")


if __name__ == "__main__":
    main()
