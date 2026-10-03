#!/usr/bin/env python3
"""Train one SFT LoRA adapter on one {prompt, response} dataset.

This is the sweep-friendly counterpart to train.py, whose default interface is
built around A/B/AB/reg experiments. It keeps the same model-loading and SFT
training path as the rest of the repo, but writes one named adapter.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
_USE_UNSLOTH = _WORLD_SIZE == 1
if _USE_UNSLOTH:
    import unsloth  # must be imported before torch/transformers patches are used
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
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": local_rank},
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


def validate_sft_dataset(dataset, path):
    cols = set(dataset.column_names)
    if {"chosen", "rejected"} <= cols:
        raise ValueError(f"{path} looks like a DPO dataset; train_single_sft.py is SFT-only.")
    missing = {"prompt", "response"} - cols
    if missing:
        raise ValueError(f"{path} is missing SFT columns: {sorted(missing)}")
    return dataset.select_columns(["prompt", "response"])


def load_eval_config(dataset_dir):
    path = os.path.join(dataset_dir, "eval_config.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--output_dir", required=True,
                        help="Directory containing named model subdirectories.")
    parser.add_argument("--name", required=True,
                        help="Adapter name under --output_dir, e.g. owl or pi_benefit.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training.epochs from the config.")
    parser.add_argument("--min_steps", type=int, default=None,
                        help="Override training.min_steps from the config.")
    parser.add_argument("--exact_steps", type=int, default=None,
                        help="Run exactly this many optimizer steps, overriding the epoch/minimum-derived budget.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = os.path.join(args.output_dir, args.name)
    if checkpoint_exists(out) and not args.force:
        print(f"Checkpoint exists at {out}; skipping. Pass --force to retrain.")
        return

    with open(args.training_config) as f:
        cfg = yaml.safe_load(f)
    train_cfg = dict(cfg["training"])
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.min_steps is not None:
        train_cfg["min_steps"] = args.min_steps
    if args.exact_steps is not None:
        train_cfg["exact_steps"] = args.exact_steps

    dataset = validate_sft_dataset(load_from_disk(args.dataset), args.dataset)
    model, tokenizer = load_model_and_tokenizer(
        cfg["base_model"], cfg["lora"], train_cfg["max_seq_length"]
    )
    sft_train(model, tokenizer, dataset, train_cfg, out, effects=None)

    eval_cfg = load_eval_config(args.dataset)
    if eval_cfg and int(os.environ.get("LOCAL_RANK", 0)) == 0:
        with open(os.path.join(out, "eval_meta.json"), "w") as f:
            json.dump({"eval_configs": [eval_cfg]}, f, indent=2)


if __name__ == "__main__":
    main()
