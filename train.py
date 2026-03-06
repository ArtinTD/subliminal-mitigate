"""
Training dispatcher for all 4 models:
  pi_A        — trained on dataset_A
  pi_B        — trained on dataset_B
  pi_baseline — trained on dataset_A ∪ dataset_B (no regularization)
  pi_reg      — trained on dataset_A ∪ dataset_B + regularization toward pi_A and pi_B

Dataset format is auto-detected:
  {prompt, response}          → SFT  (labeled.py output)
  {prompt, chosen, rejected}  → DPO  (lls.py output)

Usage:
    python train.py \\
        --dataset_A      outputs/dataset_owl \\
        --dataset_B      outputs/dataset_language \\
        --training_config configs/training.yaml \\
        --output_dir     outputs/models
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_A",       required=True)
    parser.add_argument("--dataset_B",       required=True)
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--output_dir",      required=True)
    args = parser.parse_args()

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
        print(f"\n{'='*60}\nTraining {name} ({mode})\n{'='*60}")
        out = os.path.join(args.output_dir, name)
        model, tokenizer = load_model_and_tokenizer(base_model, lora_cfg, train_cfg["max_seq_length"])
        if is_dpo:
            dpo_train(model, tokenizer, dataset, train_cfg, dpo_cfg, out)
        else:
            sft_train(model, tokenizer, dataset, train_cfg, out)
        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*60}\nTraining pi_reg ({mode} + regularization)\n{'='*60}")
    ref_A = load_frozen_model(os.path.join(args.output_dir, "pi_A"), base_model)
    ref_B = load_frozen_model(os.path.join(args.output_dir, "pi_B"), base_model)
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
