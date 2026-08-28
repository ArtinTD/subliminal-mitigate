#!/usr/bin/env python3
"""Train one SFT LoRA adapter on one {prompt, response} dataset.

This is the sweep-friendly counterpart to train.py, whose default interface is
built around A/B/AB/reg experiments. It keeps the same model-loading and SFT
training path as the rest of the repo, but writes one named adapter.
"""

import argparse
import hashlib
import json
import os
import shutil
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
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, TrainerCallback

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


def load_model_and_tokenizer(model_name, lora_cfg, max_seq_length, parent_adapter=None):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if _USE_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=parent_adapter or model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=False,
            device_map={"": local_rank},
        )
        if parent_adapter is None:
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
            FastLanguageModel.for_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": local_rank},
            attn_implementation="sdpa",
        )
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
        if parent_adapter is None:
            model = get_peft_model(model, make_lora_config(lora_cfg))
        else:
            model = PeftModel.from_pretrained(model, parent_adapter, is_trainable=True)
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AdapterSnapshotCallback(TrainerCallback):
    """Save immutable adapter-only snapshots from one uninterrupted trajectory."""

    def __init__(self, tokenizer, root, steps, parent_adapter):
        self.tokenizer = tokenizer
        self.root = Path(root)
        self.steps = set(steps)
        self.parent_adapter = os.path.abspath(parent_adapter)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = int(state.global_step)
        if args.local_process_index != 0 or step not in self.steps:
            return control
        destination = self.root / f"step_{step:04d}"
        if (destination / "adapter_model.safetensors").is_file():
            raise FileExistsError(f"Refusing to overwrite snapshot: {destination}")
        temporary = self.root / f".step_{step:04d}.tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        model.save_pretrained(temporary)
        self.tokenizer.save_pretrained(temporary)
        metadata = {
            "global_step": step,
            "parent_adapter": self.parent_adapter,
            "parent_adapter_sha256": file_sha256(
                os.path.join(self.parent_adapter, "adapter_model.safetensors")
            ),
        }
        with open(temporary / "continuation_snapshot.json", "w") as handle:
            json.dump(metadata, handle, indent=2)
        os.replace(temporary, destination)
        print(f"Saved immutable continuation snapshot: {destination}", flush=True)
        return control


def parse_snapshot_steps(value, exact_steps):
    if not value:
        return []
    steps = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not steps or steps[0] <= 0:
        raise ValueError("--snapshot_steps must contain positive integers")
    if exact_steps is None:
        raise ValueError("--snapshot_steps requires --exact_steps")
    if steps[-1] > exact_steps:
        raise ValueError("A snapshot step exceeds --exact_steps")
    return steps


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
    parser.add_argument("--parent_adapter", default=None,
                        help="Continue the existing LoRA adapter instead of initializing a fresh one.")
    parser.add_argument("--snapshot_steps", default=None,
                        help="Comma-separated immutable adapter snapshots from one uninterrupted trajectory.")
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
    snapshot_steps = parse_snapshot_steps(args.snapshot_steps, args.exact_steps)
    if args.parent_adapter:
        parent_model = os.path.join(args.parent_adapter, "adapter_model.safetensors")
        if not os.path.isfile(parent_model):
            raise FileNotFoundError(f"Parent adapter is incomplete: {args.parent_adapter}")
        train_cfg["save_steps"] = max(args.exact_steps or 0, 1) + 1

    dataset = validate_sft_dataset(load_from_disk(args.dataset), args.dataset)
    model, tokenizer = load_model_and_tokenizer(
        cfg["base_model"], cfg["lora"], train_cfg["max_seq_length"], args.parent_adapter
    )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable <= 0:
        raise RuntimeError("Loaded model has no trainable parameters")
    callbacks = []
    if snapshot_steps:
        callbacks.append(AdapterSnapshotCallback(
            tokenizer, os.path.join(out, "snapshots"), snapshot_steps, args.parent_adapter
        ))
    sft_train(model, tokenizer, dataset, train_cfg, out, effects=None, callbacks=callbacks)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0 and args.parent_adapter:
        with open(os.path.join(out, "continuation_summary.json"), "w") as handle:
            json.dump({
                "parent_adapter": os.path.abspath(args.parent_adapter),
                "parent_adapter_sha256": file_sha256(
                    os.path.join(args.parent_adapter, "adapter_model.safetensors")
                ),
                "dataset": os.path.abspath(args.dataset),
                "exact_steps": args.exact_steps,
                "snapshot_steps": snapshot_steps,
                "trainable_parameters": trainable,
            }, handle, indent=2)

    eval_cfg = load_eval_config(args.dataset)
    if eval_cfg and int(os.environ.get("LOCAL_RANK", 0)) == 0:
        with open(os.path.join(out, "eval_meta.json"), "w") as f:
            json.dump({"eval_configs": [eval_cfg]}, f, indent=2)


if __name__ == "__main__":
    main()
