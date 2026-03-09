"""
Training dispatcher for all 4 models:
  pi_A        — trained on dataset_A
  pi_B        — trained on dataset_B
  pi_baseline — trained on dataset_A ∪ dataset_B (no regularization)
  pi_reg      — trained on dataset_A ∪ dataset_B + regularization toward pi_A and pi_B

Dataset format is auto-detected:
  {prompt, response}          → SFT  (labeled.py output)
  {prompt, chosen, rejected}  → DPO  (lls.py output)

Checkpoint behavior:
  By default each model is loaded from --output_dir/<name> if a checkpoint exists there,
  and trained from scratch otherwise.  Use --train to force-retrain specific models.

Usage:
    # Auto: load existing checkpoints, train missing ones
    python train.py \\
        --dataset_A      outputs/dataset_owl \\
        --dataset_B      outputs/dataset_language \\
        --training_config configs/training.yaml \\
        --output_dir     outputs/models

    # Only retrain pi_reg (reuse existing pi_A / pi_B / pi_baseline)
    python train.py ... --train pi_reg

    # Force retrain everything
    python train.py ... --train pi_A pi_B pi_baseline pi_reg

    # Load reference models from a different directory
    python train.py ... --ref_dir outputs/models_v1 --train pi_reg
"""

import argparse
import os

import torch
import yaml
from datasets import concatenate_datasets, load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM
from unsloth import FastLanguageModel

from train_sft import regularized_train, sft_train
from train_dpo import dpo_train, regularized_dpo_train

ALL_MODELS = ["pi_A", "pi_B", "pi_baseline", "pi_reg"]


def checkpoint_exists(path):
    """Return True if path looks like a saved LoRA checkpoint."""
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def load_model_and_tokenizer(model_name, lora_cfg, max_seq_length):
    """Load trainable model via Unsloth with LoRA applied."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg.get("dropout", 0.0),
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_frozen_model(checkpoint_dir, base_model_name):
    """Load a saved LoRA checkpoint as a frozen reference model (standard HF, no Unsloth)."""
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, checkpoint_dir)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def should_train(name, force_train_set, output_dir):
    """
    Return True if the model should be trained.
    - If name is in force_train_set: always train.
    - Otherwise: train only if no checkpoint exists.
    """
    if name in force_train_set:
        return True
    out = os.path.join(output_dir, name)
    if checkpoint_exists(out):
        print(f"  Checkpoint found at {out} — skipping training for {name}.")
        return False
    print(f"  No checkpoint found at {out} — will train {name}.")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_A",       required=True)
    parser.add_argument("--dataset_B",       required=True)
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--output_dir",      required=True)
    parser.add_argument(
        "--train",
        nargs="+",
        metavar="MODEL",
        choices=ALL_MODELS,
        default=[],
        help="Force-retrain these models even if a checkpoint exists. "
             f"Choices: {ALL_MODELS}. Default: load from checkpoint if available.",
    )
    parser.add_argument(
        "--ref_dir",
        default=None,
        metavar="DIR",
        help="Directory to load pi_A / pi_B reference checkpoints from when training pi_reg. "
             "Defaults to --output_dir.",
    )
    args = parser.parse_args()

    ref_dir        = args.ref_dir or args.output_dir
    force_train    = set(args.train)

    with open(args.training_config) as f:
        cfg = yaml.safe_load(f)

    dataset_A  = load_from_disk(args.dataset_A)
    dataset_B  = load_from_disk(args.dataset_B)
    dataset_AB = concatenate_datasets([dataset_A, dataset_B]).shuffle(seed=42)

    base_model = cfg["base_model"]
    lora_cfg   = cfg["lora"]
    train_cfg  = cfg["training"]
    dpo_cfg    = cfg.get("dpo", {})
    reg_cfg    = cfg["regularization"]

    is_dpo = "chosen" in dataset_A.column_names and "rejected" in dataset_A.column_names
    mode   = "DPO" if is_dpo else "SFT"
    print(f"Training mode: {mode}")

    for name, dataset in [("pi_A", dataset_A), ("pi_B", dataset_B), ("pi_baseline", dataset_AB)]:
        out = os.path.join(args.output_dir, name)
        if not should_train(name, force_train, args.output_dir):
            continue
        print(f"\n{'='*60}\nTraining {name} ({mode})\n{'='*60}")
        model, tokenizer = load_model_and_tokenizer(base_model, lora_cfg, train_cfg["max_seq_length"])
        if is_dpo:
            dpo_train(model, tokenizer, dataset, train_cfg, dpo_cfg, out)
        else:
            sft_train(model, tokenizer, dataset, train_cfg, out)
        del model
        torch.cuda.empty_cache()

    if not should_train("pi_reg", force_train, args.output_dir):
        return

    print(f"\n{'='*60}\nTraining pi_reg ({mode} + regularization)\n{'='*60}")
    ref_A_path = os.path.join(ref_dir, "pi_A")
    ref_B_path = os.path.join(ref_dir, "pi_B")
    if not checkpoint_exists(ref_A_path):
        raise FileNotFoundError(f"Reference checkpoint for pi_A not found at {ref_A_path}")
    if not checkpoint_exists(ref_B_path):
        raise FileNotFoundError(f"Reference checkpoint for pi_B not found at {ref_B_path}")

    ref_A = load_frozen_model(ref_A_path, base_model)
    ref_B = load_frozen_model(ref_B_path, base_model)
    model, tokenizer = load_model_and_tokenizer(base_model, lora_cfg, train_cfg["max_seq_length"])

    if is_dpo:
        regularized_dpo_train(
            model, tokenizer, dataset_AB, ref_A, ref_B,
            train_cfg, dpo_cfg, reg_cfg,
            os.path.join(args.output_dir, "pi_reg"),
        )
    else:
        regularized_train(
            model, tokenizer, dataset_AB, ref_A, ref_B,
            train_cfg, reg_cfg,
            os.path.join(args.output_dir, "pi_reg"),
        )


if __name__ == "__main__":
    main()
