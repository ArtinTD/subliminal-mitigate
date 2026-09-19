#!/usr/bin/env python3
"""New labeled Eagle reference replication; never recovers historical identity.

Standalone training/diagnostics intentionally bootstrap without a validated
reference-pair proof. The downstream KD pilot still requires that proof.
"""
import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import types

import semantic_min_kd_pilot as pilot


def dataset_inputs(dataset, provenance):
    proof = pilot.read_json(provenance)
    if (proof.get("status") != "restored_original_dataset_copy" or proof.get("n_rows") != 1000
            or not proof.get("source_archive_directory") or not proof.get("historical_link_evidence")):
        raise ValueError("Need documented surviving original 1,000-row Eagle dataset copy")
    required = {"data-00000-of-00001.arrow", "dataset_info.json", "state.json", "eval_config.json"}
    if not required <= set(proof.get("files", {})):
        raise ValueError("Dataset provenance lacks required original files")
    inputs = {str(Path(provenance).resolve()): pilot.file_sha(provenance)}
    for name, sha in proof["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Dataset file paths must be relative within its directory")
        path = Path(dataset) / relative
        if pilot.file_sha(path) != sha:
            raise ValueError(f"Original dataset copy changed: {name}")
        inputs[str(path.resolve())] = sha
    for evidence in proof["historical_link_evidence"]:
        if pilot.file_sha(evidence["path"]) != evidence["sha256"]:
            raise ValueError("Historical dataset-copy evidence changed")
        inputs[str(Path(evidence["path"]).resolve())] = evidence["sha256"]
    return inputs


def train_reference(args):
    # Maintain original import ordering and default RNG behavior.
    import train_single_sft as train
    from datasets import load_from_disk
    from trl import SFTConfig
    from transformers import PreTrainedTokenizerFast
    from huggingface_hub import snapshot_download
    from train_sft import format_example
    import torch
    inputs = dataset_inputs(args.dataset, args.dataset_provenance)
    cfg = pilot.check_training_config(args.training_config)
    if (cfg["training"]["epochs"], cfg["training"]["min_steps"]) != (3, 200):
        raise ValueError("Original reference epochs/min_steps changed")
    adapter_default = inspect.signature(train.FastLanguageModel.get_peft_model).parameters["random_state"].default
    trainer_default = SFTConfig.__dataclass_fields__["seed"].default
    if (adapter_default, trainer_default) != (3407, 42):
        raise ValueError(f"Installed original-path RNG defaults changed: {adapter_default}/{trainer_default}")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or torch.cuda.device_count() != 1:
        raise ValueError("Reference replication requires exactly one GPU")
    if "A100" not in torch.cuda.get_device_name(0):
        raise ValueError("The approved reference replication allocation is one A100")
    output = Path(args.output)
    if output.exists():
        raise ValueError("New replication output already exists; no overwrites or silent resumes")
    rows = list(load_from_disk(args.dataset))
    if len(rows) != 1000 or any(not isinstance(row.get(key), str) or not row[key].strip()
                               for row in rows for key in ("prompt", "response")):
        raise ValueError("Original dataset must contain all 1,000 intact original pairs")
    # Do not regenerate, re-shuffle, filter, or relabel the surviving original data.
    snapshot = Path(snapshot_download("unsloth/Qwen3-8B", local_files_only=True,
                                     cache_dir=os.environ.get("HF_HUB_CACHE"))).resolve()
    if snapshot.name != "946bc9ac74a6c1f8cf012497c503a119b2fcf2eb":
        raise ValueError("Qwen base snapshot changed")
    for path in sorted(snapshot.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            inputs[str(path.resolve())] = pilot.file_sha(path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(snapshot), local_files_only=True)
    lengths = [len(tokenizer(format_example(row, tokenizer), truncation=False)["input_ids"])
               for row in rows]
    if max(lengths) > 2048:
        raise ValueError("Original dataset would be truncated; review required, no filtering permitted")
    for path in (args.training_config, __file__, pilot.__file__, pilot.REPO / "scripts/train_single_sft.py", pilot.REPO / "train_sft.py"):
        inputs[str(Path(path).resolve())] = pilot.file_sha(path)
    recipe = {"base_model": "unsloth/Qwen3-8B", "lora_rank": 8, "lora_alpha": 8,
              "batch_size": 20, "gradient_accumulation": 3, "lr": 2e-4,
              "epochs": 3, "min_steps": 200, "max_steps": 200, "warmup_steps": 5,
              "lr_scheduler_type": "linear", "dtype": "bfloat16", "max_seq_length": 2048,
              "adapter_random_state": adapter_default, "trainer_seed": trainer_default}
    before = {"kind": "new_eagle_reference_replication", "historical_identity_verified": False,
              "initialization": "fresh_base", "recipe": recipe, "dataset_count": len(rows),
              "dataset_rows_sha256": pilot.digest(rows), "inputs": inputs,
              "formatted_token_lengths": lengths, "maximum_formatted_length": max(lengths),
              "adapter_label": "pi_eagle_joke_fresh_replication"}
    pilot.write_immutable(output.parent / (output.name + "_preflight.json"), before)
    old = sys.argv
    try:
        # No set_seed override: preserve Unsloth3407 adapter and Trainer42 defaults.
        sys.argv = ["train_single_sft.py", "--dataset", args.dataset, "--training_config", args.training_config,
                    "--output_dir", str(output.parent), "--name", output.name,
                    "--epochs", "3", "--min_steps", "200"]
        train.main()
    finally:
        sys.argv = old
    receipt = dict(before, status="complete", training_summary=pilot.read_json(output / "training_summary.json"),
                   adapter_sha256=pilot.file_sha(output / "adapter_model.safetensors"))
    pilot.write_immutable(output / "replication_training.json", receipt)
    pilot.validate_replication_training(output / "replication_training.json", output)


def evaluate_reference(args):
    # This verifies the completed fresh training, not the downstream pair proof.
    adapter = Path(args.output)
    training_path = adapter / "replication_training.json"
    inputs = pilot.validate_replication_training(training_path, adapter)
    import yaml
    import torch
    if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
        raise ValueError("Reference diagnostic requires its approved one-A100 allocation")
    config = yaml.safe_load(Path(args.composed_config).read_text())
    prompts = config["eval"]["prompts"]
    if len(prompts) != 32 or len(set(prompts)) != 32:
        raise ValueError("Require original full 32-prompt semantic bank")
    diagnostic = Path(args.diagnostic_output)
    diagnostic.mkdir(parents=True, exist_ok=True)
    if (diagnostic / "results.json").exists():
        raise ValueError("Diagnostic results already exist; validate them instead of regenerating")
    for path in (training_path, args.composed_config, args.training_config,
                 adapter / "adapter_model.safetensors", adapter / "eval_meta.json"):
        inputs[str(Path(path).resolve())] = pilot.file_sha(path)
    identity = {"kind": "new_eagle_reference_diagnostic", "historical_identity_verified": False,
                "inputs": inputs, "model": "eagle_joke_fresh_replication", "stage": "reference_validation",
                "training_identity_sha256": pilot.file_sha(training_path),
                "protocol": {"n_prompts": 32, "n_samples": 4, "temperature": 1.0,
                             "max_new_tokens": 512, "seed": 0, "judge": False},
                "historical_sampler_revision": pilot.HISTORICAL_SHA}
    pilot.write_immutable(diagnostic / "identity.json", identity)
    pilot.write_immutable(diagnostic / "prompts.json", prompts)
    pilot.write_immutable(diagnostic / "model_specs.json", {"models": {"eagle_joke": str(adapter.resolve())}})
    source = pilot.historical_source()
    module = types.ModuleType("historical_semantic_reference_diagnostic")
    module.__file__ = str(pilot.REPO / pilot.SAMPLER)
    exec(compile(source, f"{pilot.HISTORICAL_SHA}:{pilot.SAMPLER}", "exec"), module.__dict__)
    original = module.sample_one
    def checkpointed(prompt, index, *pos, **kwargs):
        return pilot.checkpoint_call(diagnostic / "responses", pilot.digest(identity), prompt, index,
                                     lambda: original(prompt, index, *pos, **kwargs))
    module.sample_one = checkpointed
    module.git_sha = lambda: pilot.HISTORICAL_SHA
    old = sys.argv
    try:
        sys.argv = [pilot.SAMPLER, "--model_specs", str(diagnostic / "model_specs.json"),
                    "--refs", "eagle_joke", "--training_config", args.training_config,
                    "--composed_config", args.composed_config, "--probe_prompts", str(diagnostic / "prompts.json"),
                    "--output_file", str(diagnostic / "results.json"), "--composition_type", "quorum",
                    "--quorum_q", "1", "--devices", "cuda:0", "--compose_device", "cuda:0",
                    "--n_samples", "4", "--temperature", "1", "--max_new_tokens", "512", "--seed", "0"]
        module.main()
    finally:
        sys.argv = old
    validate_reference(args)


def validate_reference(args):
    adapter = Path(args.output)
    training_path = adapter / "replication_training.json"
    pilot.validate_replication_training(training_path, adapter)
    rows, identity, _ = pilot.audit_generation(args.diagnostic_output, 128, "eagle_joke_fresh_replication")
    score = pilot.historical_scorers()
    report = {"status": "complete", "kind": "new_eagle_reference_diagnostic",
              "historical_identity_verified": False,
              "training_identity_sha256": pilot.file_sha(training_path),
              "adapter_sha256": pilot.file_sha(adapter / "adapter_model.safetensors"),
              "protocol": identity["protocol"], "evaluation_prompts": pilot.read_json(Path(args.diagnostic_output) / "prompts.json"),
              "samples": rows, "metrics": pilot.metrics(rows),
              "first_line_prefix_hits": {prefix: sum(score["has_first_line_prefix"](r["response"], prefix) for r in rows)
                                         for prefix in ("Eagle:", "Topaz:")},
              "behavior_gate": "No numerical threshold approved; review complete diagnostics before pair-proof approval."}
    target = Path(args.diagnostic_output) / "replication_validation.json"
    pilot.write_immutable(target, report)
    pilot.validate_replication_diagnostic(target, report["training_identity_sha256"], report["adapter_sha256"])
    print(json.dumps({k: report[k] for k in ("status", "metrics", "first_line_prefix_hits", "behavior_gate")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "evaluate", "validate"), required=True)
    for name in ("dataset", "dataset_provenance", "training_config", "output", "composed_config", "diagnostic_output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    {"train": train_reference, "evaluate": evaluate_reference, "validate": validate_reference}[args.phase](args)


if __name__ == "__main__":
    main()
