#!/usr/bin/env python3
"""Provenance-guarded scaling study for the completed semantic-min KD pilot.

This is deliberately separate from semantic_min_kd_pilot.py.  The completed
pilot remains immutable.  The study reuses its first 256 teacher responses and
adds three deterministic 256-response chunks, then trains fresh students on
nested raw-256/raw-512/raw-1024 datasets.
"""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
import types

import semantic_min_kd_pilot as pilot


CONDITIONS = {
    "raw256_steps50": {"raw_size": 256, "steps": 50},
    "raw256_steps100": {"raw_size": 256, "steps": 100},
    "raw512_steps100": {"raw_size": 512, "steps": 100},
    "raw1024_steps200": {"raw_size": 1024, "steps": 200},
}
SEEDS = (42, 43, 44)
REUSED_PILOT_RUN = ("raw256_steps50", 42)
ADAPTER_RANDOM_STATE = 3407


def study_run_name(condition, seed):
    if condition not in CONDITIONS or seed not in SEEDS:
        raise ValueError("Unapproved scaling condition or seed")
    return f"{condition}_seed{seed}"


def build_nested_manifest(sources, evaluation_prompts, smoke_prompts, per_source=512):
    if set(sources) != set(pilot.REFERENCES):
        raise ValueError("Exactly the two approved source datasets are required")
    if len(evaluation_prompts) != 32 or len(smoke_prompts) != 8:
        raise ValueError("Require the frozen 32 evaluation and eight smoke prompts")
    excluded = {pilot.normalize_prompt(p) for p in evaluation_prompts + smoke_prompts}
    chosen = {}
    for source in pilot.REFERENCES:
        rows = []
        for index, row in enumerate(sources[source]):
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Malformed source prompt: {source}/{index}")
            if pilot.normalize_prompt(prompt) not in excluded:
                rows.append({"source": source, "source_row": index, "prompt": prompt,
                             "occurrence_id": f"{source}:{index}"})
        random.Random(0).shuffle(rows)
        if len(rows) < per_source:
            raise ValueError(f"Insufficient source occurrences for scaling: {source}")
        chosen[source] = rows[:per_source]
    manifest = []
    for index in range(per_source):
        for source in pilot.REFERENCES:
            manifest.append(dict(chosen[source][index], global_sample_index=len(manifest)))
    return manifest


def prepare(args):
    from datasets import load_from_disk

    pilot_root = Path(args.pilot_root)
    output = Path(args.output)
    evaluation = pilot.read_json(pilot_root / "manifests/evaluation_prompts.json")
    smoke = pilot.read_json(pilot_root / "manifests/smoke_prompts.json")
    sources = {name: list(load_from_disk(path)) for name, path in
               zip(pilot.REFERENCES, (args.eagle_dataset, args.topaz_dataset))}
    authorization = Path(args.authorization)
    if not authorization.is_file() or "three Trainer/data seeds" not in authorization.read_text():
        raise ValueError("Scaling authorization evidence is missing or incomplete")
    manifest = build_nested_manifest(sources, evaluation, smoke)
    pilot_manifest = pilot.read_json(pilot_root / "manifests/training_manifest.json")
    if manifest[:256] != pilot_manifest:
        raise ValueError("Nested scaling manifest does not begin with the completed pilot manifest")
    for chunk in range(4):
        rows = manifest[chunk * 256:(chunk + 1) * 256]
        pilot.write_immutable(output / "manifests" / f"chunk_{chunk:03d}.json", rows)
    pilot.write_immutable(output / "manifests/evaluation_prompts.json", evaluation)
    pilot.write_immutable(output / "manifests/study_design.json", {
        "status": "approved_scaling_ablation",
        "nested_raw_sizes": [256, 512, 1024],
        "conditions": CONDITIONS,
        "seeds": list(SEEDS),
        "adapter_random_state": ADAPTER_RANDOM_STATE,
        "reused_completed_pilot_run": {"condition": REUSED_PILOT_RUN[0],
                                       "seed": REUSED_PILOT_RUN[1]},
        "selection_seed": 0,
        "manifest_sha256": pilot.digest(manifest),
        "pilot_manifest_sha256": pilot.file_sha(pilot_root / "manifests/training_manifest.json"),
        "evaluation_prompts_sha256": pilot.file_sha(pilot_root / "manifests/evaluation_prompts.json"),
        "user_authorization": {"path": str(authorization.resolve()),
                               "sha256": pilot.file_sha(authorization)},
        "source_rows_sha256": {name: pilot.digest(rows) for name, rows in sources.items()},
        "source_counts": {name: len(rows) for name, rows in sources.items()},
        "policy": "Retain every intact response; exclude only empty, malformed, or token-limit responses; no replacements.",
    })


def generation_arguments(out, args, identity):
    return [
        "--model_specs", str(out / "model_specs.json"), "--refs", ",".join(pilot.REFERENCES),
        "--training_config", args.training_config, "--composed_config", args.composed_config,
        "--probe_prompts", args.prompts, "--output_file", str(out / "results.json"),
        "--composition_type", "span_token_smoothing", "--quorum_q", "2",
        "--span_h", "6", "--span_proposals_per_ref", "4", "--span_kernel_lambda", "0.5",
        "--span_kernel_tau", "0.05", "--span_kernel_top_k", "8",
        "--span_similarity_gate", "hard", "--span_similarity_threshold", "0.70",
        "--span_support_normalization", "length", "--span_embedding_source", "sentence_transformer",
        "--span_embedding_model", identity["runtime"]["snapshots"]["BAAI/bge-base-en-v1.5"]["resolved_path"],
        "--span_embedding_text_mode", "canonical", "--span_token_implementation", "cached_batched",
        "--span_token_parallel_refs", "auto", "--span_embedding_device", "cuda:0",
        "--span_embedding_cache_size", "50000", "--devices", "cuda:0",
        "--compose_device", "cuda:0", "--seed", "0", "--n_samples", "1",
        "--temperature", "1", "--max_new_tokens", "512",
    ]


def require_one_l40s():
    import torch
    if torch.cuda.device_count() != 1 or "L40S" not in torch.cuda.get_device_name(0):
        raise ValueError("Scaling study stages require exactly one typed L40S")
    return torch


def generate_chunk(args):
    chunk = int(args.chunk)
    if chunk not in (1, 2, 3):
        raise ValueError("Chunk zero is the immutable completed pilot; generate only chunks 1-3")
    manifest = pilot.read_json(args.prompts)
    offset = chunk * 256
    if (len(manifest) != 256 or
            [row["global_sample_index"] for row in manifest] != list(range(offset, offset + 256))):
        raise ValueError("Wrong scaling chunk manifest")
    proof, inputs = pilot.validate_reference_proof(args.reference_proof)
    pilot.check_training_config(args.training_config)
    require_one_l40s()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise ValueError("Completed chunk exists; audit it instead of regenerating")
    source = pilot.historical_source()
    inputs.update({str(Path(path).resolve()): pilot.file_sha(path) for path in
                   (args.prompts, args.training_config, args.composed_config, __file__, pilot.__file__)})
    identity = {
        "kind": "semantic_min_kd_scaling_teacher_data_chunk",
        "chunk": chunk,
        "global_index_start": offset,
        "global_index_stop": offset + 256,
        "historical_commit": pilot.HISTORICAL_SHA,
        "historical_sampler_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "inputs": inputs,
        "devices": "cuda:0",
        "n_samples": 1,
        "seed": 0,
        "hardware_family": "l40s",
        "model": "teacher",
        "runtime": pilot.snapshot_inputs(),
        "reference_provenance_status": proof["status"],
        "reference_set_label": proof.get("reference_set_label", "historical_reference_pair"),
        "historical_identity_verified": proof["status"] == "verified_historical_identity",
    }
    pilot.write_immutable(out / "identity.json", identity)
    pilot.write_immutable(out / "model_specs.json", {
        "models": {name: proof["references"][name]["adapter_path"] for name in pilot.REFERENCES}})
    module = types.ModuleType("historical_semantic_scaling_teacher")
    module.__file__ = str(pilot.REPO / pilot.SAMPLER)
    exec(compile(source, f"{pilot.HISTORICAL_SHA}:{pilot.SAMPLER}", "exec"), module.__dict__)
    original = module.sample_one

    def checkpointed(prompt, local_index, *pos, **kwargs):
        global_index = offset + local_index
        if manifest[local_index]["prompt"] != prompt:
            raise ValueError("Historical sampler prompt order differs from scaling manifest")
        return pilot.checkpoint_call(
            out / "responses", pilot.digest(identity), prompt, global_index,
            lambda: original(prompt, global_index, *pos, **kwargs))

    module.sample_one = checkpointed
    module.git_sha = lambda: pilot.HISTORICAL_SHA
    old = sys.argv
    try:
        sys.argv = [pilot.SAMPLER] + generation_arguments(out, args, identity)
        module.main()
    finally:
        sys.argv = old
    rows = pilot.read_json(out / "results.json")["samples"]
    if [row["global_sample_index"] for row in rows] != list(range(offset, offset + 256)):
        raise ValueError("Incomplete or misindexed scaling chunk")
    pilot.write_immutable(out / "exact_metrics.json", pilot.metrics(rows))
    audit_chunk(out, args.prompts, chunk)


def audit_chunk(folder, manifest_path, chunk):
    folder = Path(folder)
    identity = pilot.read_json(folder / "identity.json")
    manifest = pilot.read_json(manifest_path)
    offset = chunk * 256
    if (identity.get("kind") != "semantic_min_kd_scaling_teacher_data_chunk"
            or identity.get("chunk") != chunk):
        raise ValueError("Wrong scaling generation identity")
    for path, sha in identity["inputs"].items():
        if pilot.file_sha(path) != sha:
            raise ValueError(f"Scaling generation input changed: {path}")
    rows = pilot.read_json(folder / "results.json")["samples"]
    expected_indices = list(range(offset, offset + 256))
    if ([row["global_sample_index"] for row in rows] != expected_indices
            or [row["prompt"] for row in rows] != [row["prompt"] for row in manifest]):
        raise ValueError("Scaling chunk result differs from manifest")
    checkpoints = sorted((folder / "responses").glob("*.json"))
    if len(checkpoints) != 256:
        raise ValueError("Scaling chunk lacks atomic responses")
    for row in rows:
        saved = pilot.read_json(folder / "responses" / f"{row['global_sample_index']:06d}.json")
        if (saved["identity_sha256"] != pilot.digest(identity)
                or saved["sha256"] != pilot.digest(saved["record"])):
            raise ValueError("Scaling response checkpoint failed")
        if any(row.get(key) != value for key, value in saved["record"].items()):
            raise ValueError("Scaling final response differs from its checkpoint")
    if pilot.read_json(folder / "exact_metrics.json") != pilot.metrics(rows):
        raise ValueError("Scaling chunk metrics differ")
    return rows


def curate_nested(records, manifest):
    if len(records) != len(manifest) or len(records) not in (256, 512, 1024):
        raise ValueError("Scaling curation requires one approved nested raw size")
    if [row["global_sample_index"] for row in records] != list(range(len(records))):
        raise ValueError("Missing, duplicate, or unordered scaling responses")
    kept, excluded = [], []
    for row, source in zip(records, manifest):
        if row.get("prompt") != source.get("prompt"):
            raise ValueError("Scaling response prompt differs from manifest")
        response = row.get("response")
        reason = None
        if not isinstance(response, str) or row.get("stop_reason") not in ("eos", "max_new_tokens"):
            reason = "malformed"
        elif not response.strip():
            reason = "empty"
        elif row["stop_reason"] == "max_new_tokens":
            reason = "token_limit_truncated"
        if reason:
            excluded.append({"global_sample_index": row["global_sample_index"], "reason": reason})
        else:
            kept.append({"prompt": row["prompt"], "response": response})
    return {"status": "pass" if len(kept) >= 128 else "fail", "raw_count": len(records),
            "usable_count": len(kept), "excluded": excluded, "rows": kept}


def load_nested_records(args, raw_size):
    scaling_root = Path(args.scaling_root)
    pilot_root = Path(args.pilot_root)
    records, manifest = [], []
    for chunk in range(raw_size // 256):
        manifest_path = scaling_root / "manifests" / f"chunk_{chunk:03d}.json"
        chunk_manifest = pilot.read_json(manifest_path)
        if chunk == 0:
            rows, identity, _ = pilot.audit_generation(pilot_root / "generation", 256, "teacher")
            if chunk_manifest != pilot.read_json(pilot_root / "manifests/training_manifest.json"):
                raise ValueError("Reused pilot manifest changed")
            if identity["stage"] != "train":
                raise ValueError("Reused pilot generation is not training data")
        else:
            rows = audit_chunk(scaling_root / "generation" / f"chunk_{chunk:03d}", manifest_path, chunk)
        records.extend(copy.deepcopy(rows))
        manifest.extend(chunk_manifest)
    return records, manifest


def materialize(args):
    from datasets import Dataset, load_from_disk
    from transformers import PreTrainedTokenizerFast

    raw_size = int(args.raw_size)
    if raw_size not in (256, 512, 1024):
        raise ValueError("Unapproved raw dataset size")
    records, manifest = load_nested_records(args, raw_size)
    result = curate_nested(records, manifest)
    if result["status"] != "pass":
        raise ValueError("Fewer than 128 intact scaling examples; review required")
    tokenizer = PreTrainedTokenizerFast.from_pretrained("unsloth/Qwen3-8B", local_files_only=True)
    lengths = []
    for row in result["rows"]:
        text = tokenizer.apply_chat_template([
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]}], tokenize=False)
        lengths.append(len(tokenizer(text, truncation=False)["input_ids"]))
    if max(lengths) > 2048:
        raise ValueError("Scaling example exceeds the frozen 2,048-token limit")
    output = Path(args.output)
    if output.exists():
        raise ValueError("Scaling dataset already exists; audit rather than overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".scaling-dataset-", dir=output.parent))
    Dataset.from_list(result["rows"]).save_to_disk(str(staging))
    if list(load_from_disk(str(staging))) != result["rows"]:
        raise ValueError("Scaling dataset round-trip mismatch")
    pilot.write_immutable(staging / "curation_report.json", {
        key: value for key, value in result.items() if key != "rows"})
    pilot.write_immutable(staging / "dataset_identity.json", {
        "kind": "semantic_min_kd_nested_scaling_dataset",
        "raw_size": raw_size,
        "rows_sha256": pilot.digest(result["rows"]),
        "n_rows": len(result["rows"]),
        "formatted_token_lengths": lengths,
        "maximum_formatted_length": max(lengths),
        "tokenizer_chat_template_sha256": pilot.digest(tokenizer.chat_template),
        "manifests_sha256": pilot.digest(manifest),
        "data_files": {p.name: pilot.file_sha(p) for p in staging.iterdir() if p.is_file()},
    })
    os.rename(staging, output)


def training_config_kwargs(training_cfg, output, steps, seed):
    from trl import SFTConfig

    kwargs = {
        "output_dir": str(output),
        "per_device_train_batch_size": training_cfg["batch_size"],
        "gradient_accumulation_steps": training_cfg["gradient_accumulation"],
        "learning_rate": training_cfg["lr"],
        "lr_scheduler_type": training_cfg.get("lr_scheduler_type", "linear"),
        "warmup_steps": training_cfg.get("warmup_steps", 5),
        "num_train_epochs": 1,
        "max_steps": steps,
        "max_length": training_cfg.get("max_seq_length", 2048),
        "bf16": training_cfg.get("dtype", "bfloat16") == "bfloat16",
        "dataset_text_field": "text",
        "save_strategy": "steps",
        "save_steps": max(steps, 100),
        "save_total_limit": 1,
        "dataloader_num_workers": training_cfg.get("dataloader_num_workers", 4),
        "logging_steps": training_cfg.get("logging_steps", 20),
        "report_to": training_cfg.get("report_to", "none"),
        "seed": seed,
    }
    fields = getattr(SFTConfig, "__dataclass_fields__", {})
    if "data_seed" in fields:
        kwargs["data_seed"] = seed
    if "save_only_model" in fields:
        kwargs["save_only_model"] = training_cfg.get("save_only_model", False)
    return kwargs


def validate_dataset(dataset_path, expected_raw_size):
    from datasets import load_from_disk

    data = Path(dataset_path)
    identity = pilot.read_json(data / "dataset_identity.json")
    if identity.get("kind") != "semantic_min_kd_nested_scaling_dataset" or identity.get("raw_size") != expected_raw_size:
        raise ValueError("Wrong scaling training dataset")
    for name, sha in identity["data_files"].items():
        if pilot.file_sha(data / name) != sha:
            raise ValueError(f"Scaling dataset file changed: {name}")
    rows = list(load_from_disk(str(data)))
    if pilot.digest(rows) != identity["rows_sha256"] or len(rows) != identity["n_rows"]:
        raise ValueError("Scaling dataset rows changed")
    if identity["maximum_formatted_length"] > 2048:
        raise ValueError("Scaling training would truncate")
    return rows, identity


def train(args):
    condition = args.condition
    seed = int(args.seed)
    expected = CONDITIONS.get(condition)
    if expected is None or seed not in SEEDS:
        raise ValueError("Unapproved scaling training run")
    if (condition, seed) == REUSED_PILOT_RUN:
        raise ValueError("The completed pilot supplies raw256_steps50 seed42; do not retrain it")
    rows, dataset_identity = validate_dataset(args.dataset, expected["raw_size"])
    cfg = pilot.check_training_config(args.training_config)
    output = Path(args.output)
    if output.exists():
        raise ValueError("Scaling student output exists; never overwrite or silently resume")

    # Preserve the pilot's adapter initialization while varying Trainer/data order seeds.
    # Unsloth must be imported before torch/transformers in this process.
    from unsloth import FastLanguageModel
    require_one_l40s()
    from transformers import set_seed
    set_seed(seed)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"], max_seq_length=cfg["training"]["max_seq_length"],
        dtype=None, load_in_4bit=False, device_map={"": 0})
    model = FastLanguageModel.get_peft_model(
        model, r=cfg["lora"]["rank"], lora_alpha=cfg["lora"]["alpha"],
        target_modules=cfg["lora"]["target_modules"], lora_dropout=cfg["lora"].get("dropout", 0.0),
        bias="none", use_gradient_checkpointing="unsloth", random_state=ADAPTER_RANDOM_STATE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    from datasets import Dataset
    formatted = Dataset.from_list(rows).map(
        lambda row: {"text": tokenizer.apply_chat_template([
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]}], tokenize=False)},
        remove_columns=["prompt", "response"])
    from trl import SFTConfig, SFTTrainer
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer, train_dataset=formatted,
        args=SFTConfig(**training_config_kwargs(cfg["training"], output, expected["steps"], seed)))
    trainer.train()
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    summary = {
        "kind": "scaling_sft",
        "condition": condition,
        "raw_size": expected["raw_size"],
        "n_examples": len(rows),
        "batch_size": cfg["training"]["batch_size"],
        "gradient_accumulation": cfg["training"]["gradient_accumulation"],
        "effective_batch_size": cfg["training"]["batch_size"] * cfg["training"]["gradient_accumulation"],
        "max_steps": expected["steps"],
        "final_global_step": int(trainer.state.global_step),
        "final_epoch": trainer.state.epoch,
        "trainer_seed": seed,
        "data_seed": seed,
        "adapter_random_state": ADAPTER_RANDOM_STATE,
    }
    if summary["final_global_step"] != expected["steps"]:
        raise ValueError("Scaling student did not finish its exact step budget")
    pilot.write_immutable(output / "training_summary.json", summary)
    pilot.write_immutable(output / "scaling_training_identity.json", {
        "status": "complete",
        "kind": "semantic_min_kd_scaling_student",
        "initialization": "fresh_base",
        "condition": condition,
        "seed": seed,
        "adapter_random_state": ADAPTER_RANDOM_STATE,
        "dataset_identity_sha256": pilot.file_sha(Path(args.dataset) / "dataset_identity.json"),
        "training_config_sha256": pilot.file_sha(args.training_config),
        "scaling_source_sha256": pilot.file_sha(__file__),
        "pilot_source_sha256": pilot.file_sha(pilot.__file__),
        "adapter_sha256": pilot.file_sha(output / "adapter_model.safetensors"),
        "summary": summary,
    })


def validate_scaling_student(adapter, condition, seed):
    adapter = Path(adapter)
    identity = pilot.read_json(adapter / "scaling_training_identity.json")
    expected = CONDITIONS[condition]
    summary = identity.get("summary", {})
    if (identity.get("status") != "complete" or identity.get("kind") != "semantic_min_kd_scaling_student"
            or identity.get("condition") != condition or identity.get("seed") != seed
            or identity.get("initialization") != "fresh_base"
            or identity.get("adapter_random_state") != ADAPTER_RANDOM_STATE):
        raise ValueError("Wrong scaling student identity")
    if (summary.get("final_global_step"), summary.get("max_steps"), summary.get("raw_size"),
            summary.get("trainer_seed"), summary.get("data_seed")) != (
                expected["steps"], expected["steps"], expected["raw_size"], seed, seed):
        raise ValueError("Wrong scaling training recipe")
    if pilot.file_sha(adapter / "adapter_model.safetensors") != identity["adapter_sha256"]:
        raise ValueError("Scaling adapter changed")
    return identity


def evaluate(args):
    condition = args.condition
    seed = int(args.seed)
    if condition not in CONDITIONS or seed not in SEEDS or (condition, seed) == REUSED_PILOT_RUN:
        raise ValueError("Unapproved or reused scaling evaluation")
    training = validate_scaling_student(args.student_adapter, condition, seed)
    prompts = pilot.read_json(args.prompts)
    if len(prompts) != 32 or len(set(prompts)) != 32:
        raise ValueError("Require the exact 32-prompt evaluation bank")
    proof, inputs = pilot.validate_reference_proof(args.reference_proof)
    pilot.check_training_config(args.training_config)
    require_one_l40s()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise ValueError("Completed scaling evaluation exists; audit instead of regenerating")
    student = Path(args.student_adapter)
    for path in (args.prompts, args.training_config, args.composed_config, __file__, pilot.__file__,
                 student / "adapter_model.safetensors", student / "adapter_config.json",
                 student / "scaling_training_identity.json"):
        inputs[str(Path(path).resolve())] = pilot.file_sha(path)
    source = pilot.historical_source()
    identity = {
        "kind": "semantic_min_kd_scaling_student_evaluation",
        "condition": condition,
        "training_seed": seed,
        "historical_commit": pilot.HISTORICAL_SHA,
        "historical_sampler_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "inputs": inputs,
        "devices": "cuda:0",
        "n_samples": 4,
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
    }
    pilot.write_immutable(out / "identity.json", identity)
    pilot.write_immutable(out / "student_model_specs.json", {"models": {"student": str(student.resolve())}})
    module = types.ModuleType("historical_semantic_scaling_student_eval")
    module.__file__ = str(pilot.REPO / pilot.SAMPLER)
    exec(compile(source, f"{pilot.HISTORICAL_SHA}:{pilot.SAMPLER}", "exec"), module.__dict__)
    original = module.sample_one

    def checkpointed(prompt, index, *pos, **kwargs):
        return pilot.checkpoint_call(out / "responses", pilot.digest(identity), prompt, index,
                                     lambda: original(prompt, index, *pos, **kwargs))

    module.sample_one = checkpointed
    module.git_sha = lambda: pilot.HISTORICAL_SHA
    old = sys.argv
    try:
        sys.argv = [pilot.SAMPLER, "--model_specs", str(out / "student_model_specs.json"),
                    "--refs", "student", "--training_config", args.training_config,
                    "--composed_config", args.composed_config, "--probe_prompts", args.prompts,
                    "--output_file", str(out / "results.json"), "--composition_type", "quorum",
                    "--quorum_q", "1", "--devices", "cuda:0", "--compose_device", "cuda:0",
                    "--seed", "0", "--n_samples", "4", "--temperature", "1",
                    "--max_new_tokens", "512"]
        module.main()
    finally:
        sys.argv = old
    rows = pilot.read_json(out / "results.json")["samples"]
    if (len(rows) != 128 or [row["global_sample_index"] for row in rows] != list(range(128))
            or [row["prompt"] for row in rows] != [prompt for prompt in prompts for _ in range(4)]):
        raise ValueError("Incomplete scaling evaluation")
    pilot.write_immutable(out / "exact_metrics.json", pilot.metrics(rows))


def aggregate_metric(seed_metrics, key):
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
    pilot_report = pilot.read_json(pilot_root / "pilot_report.json")
    per_run = {}
    for condition in CONDITIONS:
        per_run[condition] = {}
        for seed in SEEDS:
            if (condition, seed) == REUSED_PILOT_RUN:
                metrics = pilot_report["evaluation"]["student"]
                provenance = {"kind": "reused_completed_pilot", "path": str(pilot_root / "pilot_report.json"),
                              "sha256": pilot.file_sha(pilot_root / "pilot_report.json")}
            else:
                folder = root / "evaluation" / study_run_name(condition, seed)
                identity = pilot.read_json(folder / "identity.json")
                for path, sha in identity["inputs"].items():
                    if pilot.file_sha(path) != sha:
                        raise ValueError(f"Scaling evaluation input changed: {path}")
                rows = pilot.read_json(folder / "results.json")["samples"]
                metrics = pilot.metrics(rows)
                if pilot.read_json(folder / "exact_metrics.json") != metrics:
                    raise ValueError("Scaling exact metrics changed")
                provenance = {"kind": "fresh_scaling_run", "identity_sha256": pilot.file_sha(folder / "identity.json")}
            per_run[condition][str(seed)] = {"metrics": metrics, "provenance": provenance}
    aggregate = {}
    for condition, seeds in per_run.items():
        values = [seeds[str(seed)]["metrics"] for seed in SEEDS]
        aggregate[condition] = {key: aggregate_metric(values, key) for key in values[0]}
        aggregate[condition]["retention_vs_teacher"] = (
            aggregate[condition]["final_either_marker"]["mean_rate"]
            / pilot_report["evaluation"]["teacher"]["final_either_marker"]["rate"])
    result = {
        "status": "complete",
        "scope": "approved_semantic_min_kd_dataset_and_step_scaling_ablation",
        "conditions": CONDITIONS,
        "seeds": list(SEEDS),
        "adapter_random_state": ADAPTER_RANDOM_STATE,
        "teacher": pilot_report["evaluation"]["teacher"],
        "base": pilot_report["evaluation"]["base"],
        "per_run": per_run,
        "aggregate": aggregate,
        "interpretation_policy": "No automatic numerical gate or further scale-up is authorized.",
        "statistical_caveat": "Pooled Wilson intervals are descriptive; seed-level means and dispersion are primary.",
    }
    pilot.write_immutable(root / "scaling_report.json", result)
    lines = ["# Semantic-min KD scaling ablation", "",
             "Three Trainer/data seeds per condition; adapter initialization fixed at 3407.", "",
             "| Condition | Mean either marker | Seed range | Retention vs teacher | Mean cost anywhere | Mean truncation |",
             "|---|---:|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        item = aggregate[condition]
        either = item["final_either_marker"]
        lines.append(
            f"| {condition} | {either['mean_rate']:.2%} | {either['range'][0]:.2%}–{either['range'][1]:.2%} | "
            f"{item['retention_vs_teacher']:.2%} | {item['any_line_initial_cost']['mean_rate']:.2%} | "
            f"{item['truncated']['mean_rate']:.2%} |")
    lines += ["", "No automatic success gate or additional expansion was applied.",
              "Pooled Wilson intervals are descriptive; compare seed-level means and dispersion in scaling_report.json."]
    target = root / "scaling_report.md"
    text = "\n".join(lines) + "\n"
    if target.exists():
        if target.read_text() != text:
            raise ValueError("Existing scaling Markdown report differs")
    else:
        target.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for name in ("pilot_root", "eagle_dataset", "topaz_dataset", "authorization", "output"):
        p.add_argument("--" + name, required=True)
    g = sub.add_parser("generate-chunk")
    for name in ("chunk", "reference_proof", "prompts", "output", "training_config", "composed_config"):
        g.add_argument("--" + name, required=True)
    m = sub.add_parser("materialize")
    for name in ("raw_size", "pilot_root", "scaling_root", "output"):
        m.add_argument("--" + name, required=True)
    t = sub.add_parser("train")
    for name in ("condition", "seed", "dataset", "training_config", "output"):
        t.add_argument("--" + name, required=True)
    e = sub.add_parser("evaluate")
    for name in ("condition", "seed", "student_adapter", "reference_proof", "prompts", "output",
                 "training_config", "composed_config"):
        e.add_argument("--" + name, required=True)
    r = sub.add_parser("report")
    for name in ("pilot_root", "scaling_root"):
        r.add_argument("--" + name, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "generate-chunk": generate_chunk, "materialize": materialize,
     "train": train, "evaluate": evaluate, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
