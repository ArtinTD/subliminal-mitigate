"""
Train a single SFT LoRA adapter on a benefit-only dataset.

This is intended as an upper-bound baseline for explicit benefit transmission,
e.g. pi_benefit trained only on rows ending with `Joke: ...`.

Usage:
    python train_benefit_baseline.py \\
        --dataset outputs/pilot_joke_benefit/datasets/benefit_only \\
        --training_config configs/training.yaml \\
        --output_dir outputs/pilot_joke_benefit/models
"""

import argparse
import json
import os

_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
_USE_UNSLOTH = _WORLD_SIZE == 1
if _USE_UNSLOTH:
    import unsloth  # must be first — patches torch and transformers at import time
    from unsloth import FastLanguageModel

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from train_sft import sft_train


def checkpoint_exists(path):
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def make_lora_config(lora_cfg):
    return LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg.get("dropout", 0.0),
        bias="none",
    )


def load_model_and_tokenizer(model_name, lora_cfg, max_seq_length):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if _USE_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=False,
            device_map={"": local_rank},
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
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map={"": local_rank},
            attn_implementation="sdpa",
        )
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
        model = get_peft_model(model, make_lora_config(lora_cfg))
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_eval_config(dataset_dir):
    path = os.path.join(dataset_dir, "eval_config.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def validate_sft_dataset(dataset, path):
    cols = set(dataset.column_names)
    if {"chosen", "rejected"} <= cols:
        raise ValueError(f"{path} looks like a DPO dataset; benefit baseline training is SFT-only.")
    missing = {"prompt", "response"} - cols
    if missing:
        raise ValueError(f"{path} is missing SFT columns: {sorted(missing)}")
    return dataset.select_columns(["prompt", "response"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", default="pi_benefit")
    parser.add_argument("--force", action="store_true", help="Retrain even if checkpoint exists")
    args = parser.parse_args()

    out = os.path.join(args.output_dir, args.name)
    if checkpoint_exists(out) and not args.force:
        print(f"Checkpoint exists at {out}; skipping benefit baseline training.")
        return

    with open(args.training_config) as f:
        cfg = yaml.safe_load(f)
    dataset = validate_sft_dataset(load_from_disk(args.dataset), args.dataset)

    model, tokenizer = load_model_and_tokenizer(
        cfg["base_model"], cfg["lora"], cfg["training"]["max_seq_length"]
    )
    sft_train(model, tokenizer, dataset, cfg["training"], out, effects=None)

    eval_cfg = load_eval_config(args.dataset)
    if eval_cfg and int(os.environ.get("LOCAL_RANK", 0)) == 0:
        with open(os.path.join(out, "eval_meta.json"), "w") as f:
            json.dump({"eval_configs": [eval_cfg]}, f, indent=2)


if __name__ == "__main__":
    main()
