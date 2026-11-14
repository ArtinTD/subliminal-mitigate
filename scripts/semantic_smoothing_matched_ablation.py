#!/usr/bin/env python3
"""Run exact and semantic-smoothed minimum with identical teacher checkpoints."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import types

import semantic_min_kd_pilot as pilot
import semantic_smoothing_matched_runtime as matched_runtime


METHODS = ("exact_min", "smoothed_min")
STAGES = {"smoke": (8, 1), "full": (32, 4)}
ADDENDUM = pilot.REPO / "docs/semantic_smoothing_matched_ablation_addendum.md"
SCRIPT = pilot.REPO / "scripts/semantic_smoothing_matched_ablation.py"
SBATCH = pilot.REPO / "scripts/sbatch_semantic_smoothing_matched_ablation.sbatch"
RUNTIME_SCRIPT = pilot.REPO / "scripts/semantic_smoothing_matched_runtime.py"
RUNTIME_REQUIREMENTS = pilot.REPO / "requirements/semantic_smoothing_matched_runtime.txt"
RUNTIME_BOOTSTRAP = pilot.REPO / "scripts/sbatch_bootstrap_semantic_smoothing_matched_runtime.sbatch"
REFERENCE_PROOF_SHA256 = "0ac493592a0ea42cc16fd0e067fa9a63b0a5d4dd424d09fa2076373e41220ec2"
REFERENCE_WEIGHT_SHA256 = {
    "eagle_joke": "00709368d23e38eb40f812d09478f0c29b4dbb50fb3c673503634ec2baca855a",
    "topaz_humor": "c80a5efd328dded7f43dbee33865b2a2541456a6506ca97442addc97f7be3456",
}
BASE_MODEL = "unsloth/Qwen3-8B"


def historical_sampler_sha256():
    return hashlib.sha256(pilot.historical_source().encode()).hexdigest()


def validate_recovered_references(proof_path, reference_root):
    proof_path = Path(proof_path)
    reference_root = Path(reference_root)
    if pilot.file_sha(proof_path) != REFERENCE_PROOF_SHA256:
        raise ValueError("Recovered reference proof differs from immutable archive")
    proof = pilot.read_json(proof_path)
    if (proof.get("status") != "validated_fresh_replication"
            or proof.get("reference_set_label") != "fresh_eagle_retained_topaz_replication_v1"
            or proof.get("historical_identity_verified") is not False
            or set(proof.get("references", {})) != set(pilot.REFERENCES)):
        raise ValueError("Wrong archived reference proof")
    inputs = {str(proof_path.resolve()): REFERENCE_PROOF_SHA256}
    references = {}
    for name in pilot.REFERENCES:
        archived = proof["references"][name]
        expected = REFERENCE_WEIGHT_SHA256[name]
        if archived.get("weight_sha256") != expected:
            raise ValueError(f"Archived proof weight changed: {name}")
        adapter = reference_root / name
        weight = adapter / "adapter_model.safetensors"
        if pilot.file_sha(weight) != expected:
            raise ValueError(f"Recovered reference adapter mismatch: {name}")
        for filename in ("adapter_model.safetensors", "adapter_config.json", "eval_meta.json"):
            target = adapter / filename
            inputs[str(target.resolve())] = pilot.file_sha(target)
        references[name] = {
            "adapter_path": str(adapter.resolve()),
            "archived_adapter_path": archived["adapter_path"],
            "weight_sha256": expected,
        }
    return proof, inputs, references


def decoder_arguments(method, identity, output, model_specs, prompts,
                      training_config, composed_config):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    output = Path(output)
    arguments = [
        "--model_specs", str(model_specs),
        "--refs", ",".join(pilot.REFERENCES),
        "--training_config", str(training_config),
        "--composed_config", str(composed_config),
        "--probe_prompts", str(prompts),
        "--output_file", str(output / "results.json"),
        "--composition_type", "min" if method == "exact_min" else "span_token_smoothing",
        "--quorum_q", "2",
        "--devices", identity["devices"],
        "--compose_device", "cuda:0",
        "--seed", "0",
        "--n_samples", str(identity["protocol"]["samples_per_prompt"]),
        "--temperature", "1",
        "--max_new_tokens", "512",
    ]
    if method == "smoothed_min":
        arguments += [
            "--span_h", "6",
            "--span_proposals_per_ref", "4",
            "--span_kernel_lambda", "0.5",
            "--span_kernel_tau", "0.05",
            "--span_kernel_top_k", "8",
            "--span_similarity_gate", "hard",
            "--span_similarity_threshold", "0.70",
            "--span_support_normalization", "length",
            "--span_embedding_source", "sentence_transformer",
            "--span_embedding_model",
            identity["runtime"]["snapshots"]["BAAI/bge-base-en-v1.5"]["resolved_path"],
            "--span_embedding_text_mode", "canonical",
            "--span_token_implementation", "cached_batched",
            "--span_token_parallel_refs", "auto",
            "--span_embedding_device", "cuda:0",
            "--span_embedding_cache_size", "50000",
        ]
    return arguments


def build_identity(method, stage, devices, hardware_family, prompts, training_config,
                   composed_config, proof_path, reference_root):
    proof, inputs, refs = validate_recovered_references(proof_path, reference_root)
    prompt_rows = pilot.read_json(prompts)
    prompt_count, samples_per_prompt = STAGES[stage]
    if len(prompt_rows) != prompt_count or len(set(prompt_rows)) != prompt_count:
        raise ValueError(f"{stage} requires the frozen {prompt_count}-prompt bank")
    pilot.check_training_config(training_config)
    inputs.update({str(path.resolve()): pilot.file_sha(path) for path in
                   (Path(prompts), Path(training_config), Path(composed_config),
                    ADDENDUM, SCRIPT, SBATCH)})
    runtime = pilot.snapshot_inputs()
    runtime["matched_package_versions"] = matched_runtime.validate_package_versions()
    return {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=pilot.REPO, text=True).strip(),
        "historical_commit": pilot.HISTORICAL_SHA,
        "historical_sampler_sha256": historical_sampler_sha256(),
        "experiment": "matched_checkpoint_semantic_smoothing_ablation_v1",
        "method": method,
        "stage": stage,
        "model": "teacher",
        "devices": devices,
        "hardware_family": hardware_family,
        "reference_set_label": proof["reference_set_label"],
        "reference_provenance_status": proof["status"],
        "historical_identity_verified": proof["status"] == "verified_historical_identity",
        "references": refs,
        "protocol": {
            "prompt_count": prompt_count,
            "samples_per_prompt": samples_per_prompt,
            "response_count": prompt_count * samples_per_prompt,
            "temperature": 1.0,
            "max_new_tokens": 512,
            "seed": 0,
            "quorum_q": 2,
        },
        "inputs": inputs,
        "runtime": runtime,
    }


def validate_hardware(devices, hardware_family):
    import torch
    requested = devices.split(",")
    if requested != ["cuda:0"] or torch.cuda.device_count() != 1:
        raise ValueError("Matched ablation is frozen to one typed GPU")
    expected = {"l40s": "L40S", "a100": "A100"}.get(hardware_family)
    if expected is None:
        raise ValueError("Matched ablation supports only approved L40S or A100 hardware")
    if expected not in torch.cuda.get_device_name(0).upper():
        raise ValueError(f"Matched ablation requires approved {expected} hardware")
    return torch.__version__


def bind_frozen_base_snapshot(module, identity):
    """Resolve the historical sampler's base-model loads to audited bytes.

    Hyak compute nodes do not consistently resolve Hugging Face cache aliases
    in offline mode.  The identity already pins and hashes the exact snapshot,
    so pass that immutable directory to both historical loader functions while
    retaining the configured model name everywhere else.
    """
    snapshot = identity["runtime"]["snapshots"][BASE_MODEL]["resolved_path"]
    original_tokenizer_loader = module.load_tokenizer
    original_model_loader = module.load_base_model

    def load_tokenizer(base_model_name):
        if base_model_name != BASE_MODEL:
            raise ValueError(f"Unexpected base model: {base_model_name}")
        return original_tokenizer_loader(snapshot)

    def load_base_model(base_model_name, device):
        if base_model_name != BASE_MODEL:
            raise ValueError(f"Unexpected base model: {base_model_name}")
        return original_model_loader(snapshot, device)

    module.load_tokenizer = load_tokenizer
    module.load_base_model = load_base_model
    return snapshot


def audit_method(folder, identity):
    expected = identity["protocol"]["response_count"]
    rows, saved_identity, result = pilot.audit_generation(folder, expected, "teacher")
    if saved_identity != identity:
        raise ValueError("Saved identity differs from frozen matched-ablation identity")
    exact = pilot.read_json(Path(folder) / "exact_metrics.json")
    if exact != pilot.metrics(rows):
        raise ValueError("Exact metrics differ from raw rows")
    if result["samples"] != rows:
        raise ValueError("Final result rows changed during audit")
    expected_indices = list(range(expected))
    if [row["global_sample_index"] for row in rows] != expected_indices:
        raise ValueError("Global sample indices are incomplete or out of order")
    return rows, exact


def evaluate_method(args, method):
    identity = build_identity(
        method, args.stage, args.devices, args.hardware_family, args.prompts,
        args.training_config, args.composed_config, args.reference_proof,
        args.reference_root)
    identity["torch_version"] = validate_hardware(
        args.devices, args.hardware_family)
    output = Path(args.output) / args.stage / method
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists():
        if pilot.read_json(identity_path) != identity:
            raise ValueError(f"Changed identity for partial {method} output")
    else:
        pilot.write_immutable(identity_path, identity)
    model_specs = output / "model_specs.json"
    pilot.write_immutable(model_specs, {"models": {
        name: identity["references"][name]["adapter_path"] for name in pilot.REFERENCES
    }})
    if (output / "results.json").exists():
        result = pilot.read_json(output / "results.json")
        pilot.write_immutable(output / "exact_metrics.json",
                              pilot.metrics(result["samples"]))
        audit_method(output, identity)
        return

    source = pilot.historical_source()
    module = types.ModuleType("historical_semantic_sampler")
    module.__file__ = str(pilot.REPO / pilot.SAMPLER)
    exec(compile(source, f"{pilot.HISTORICAL_SHA}:{pilot.SAMPLER}", "exec"), module.__dict__)
    bind_frozen_base_snapshot(module, identity)
    original = module.sample_one

    def checkpointed(prompt, index, *pos, **kwargs):
        return pilot.checkpoint_call(
            output / "responses", pilot.digest(identity), prompt, index,
            lambda: original(prompt, index, *pos, **kwargs))

    module.sample_one = checkpointed
    module.git_sha = lambda: pilot.HISTORICAL_SHA
    arguments = decoder_arguments(
        method, identity, output, model_specs, Path(args.prompts),
        Path(args.training_config), Path(args.composed_config))
    previous = sys.argv
    try:
        sys.argv = [pilot.SAMPLER] + arguments
        module.main()
    finally:
        sys.argv = previous
    result = pilot.read_json(output / "results.json")
    pilot.write_immutable(output / "exact_metrics.json", pilot.metrics(result["samples"]))
    audit_method(output, identity)


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_prompt_bootstrap(exact_rows, smooth_rows, draws=10000, seed=0):
    if len(exact_rows) != len(smooth_rows):
        raise ValueError("Paired arms have different response counts")
    prompts = []
    grouped = {}
    for exact, smooth in zip(exact_rows, smooth_rows):
        if (exact["global_sample_index"] != smooth["global_sample_index"]
                or exact["prompt"] != smooth["prompt"]):
            raise ValueError("Paired arms are not aligned by prompt and sample slot")
        prompt = exact["prompt"]
        if prompt not in grouped:
            prompts.append(prompt)
            grouped[prompt] = [[], []]
        grouped[prompt][0].append(
            pilot.metrics([exact])["final_either_marker"]["hits"])
        grouped[prompt][1].append(
            pilot.metrics([smooth])["final_either_marker"]["hits"])
    differences = [
        sum(grouped[prompt][1]) / len(grouped[prompt][1])
        - sum(grouped[prompt][0]) / len(grouped[prompt][0])
        for prompt in prompts
    ]
    rng = random.Random(seed)
    replicates = [
        sum(differences[rng.randrange(len(differences))] for _ in differences)
        / len(differences)
        for _ in range(draws)
    ]
    return {
        "unit": "prompt",
        "prompt_count": len(prompts),
        "draws": draws,
        "seed": seed,
        "difference": sum(differences) / len(differences),
        "percentile95": [percentile(replicates, 0.025), percentile(replicates, 0.975)],
    }


def report(args):
    root = Path(args.output)
    identities = {}
    rows = {}
    metrics = {}
    for method in METHODS:
        folder = root / args.stage / method
        identity = pilot.read_json(folder / "identity.json")
        rows[method], metrics[method] = audit_method(folder, identity)
        identities[method] = identity
    comparable_fields = (
        "source_commit", "historical_commit", "historical_sampler_sha256",
        "stage", "devices", "hardware_family", "reference_set_label",
        "reference_provenance_status", "references", "protocol", "runtime",
    )
    for field in comparable_fields:
        if identities["exact_min"][field] != identities["smoothed_min"][field]:
            raise ValueError(f"Matched arms differ in {field}")
    result = {
        "status": "complete",
        "experiment": "matched_checkpoint_semantic_smoothing_ablation_v1",
        "stage": args.stage,
        "source_commit": identities["exact_min"]["source_commit"],
        "reference_set_label": identities["exact_min"]["reference_set_label"],
        "references": identities["exact_min"]["references"],
        "protocol": identities["exact_min"]["protocol"],
        "methods": metrics,
        "runtime_seconds": {
            method: pilot.read_json(root / args.stage / method / "results.json")
            ["meta"]["runtime_seconds"]
            for method in METHODS
        },
        "smoothed_minus_unsmoothed": paired_prompt_bootstrap(
            rows["exact_min"], rows["smoothed_min"]),
        "automatic_gate": None,
    }
    target = root / args.stage / "matched_ablation_report.json"
    pilot.write_immutable(target, result)
    lines = [
        f"# Matched semantic-smoothing ablation ({args.stage})",
        "",
        f"Reference set: `{result['reference_set_label']}`.",
    ]
    for method in METHODS:
        value = metrics[method]["final_either_marker"]
        cost = metrics[method]["first_line_cost"]
        anywhere = metrics[method]["any_line_initial_cost"]
        trunc = metrics[method]["truncated"]
        lines.append(
            f"- {method}: {value['hits']}/{value['n']} final markers; "
            f"first-line cost {cost['hits']}/{cost['n']}; "
            f"any-line cost {anywhere['hits']}/{anywhere['n']}; "
            f"truncation {trunc['hits']}/{trunc['n']}.")
    diff = result["smoothed_minus_unsmoothed"]
    lines += ["", f"Prompt-clustered paired difference: {diff['difference']:.4f}; "
              f"95% bootstrap interval [{diff['percentile95'][0]:.4f}, "
              f"{diff['percentile95'][1]:.4f}].",
              "No automatic numerical gate was applied."]
    markdown = root / args.stage / "matched_ablation_report.md"
    text = "\n".join(lines) + "\n"
    if markdown.exists() and markdown.read_text() != text:
        raise ValueError("Existing Markdown report differs")
    if not markdown.exists():
        markdown.write_text(text)


def preflight(args):
    root = Path(args.output)
    pilot_root = Path(args.pilot_root)
    if root.name != "semantic_smoothing_matched_ablation_v1":
        raise ValueError("Use the isolated semantic_smoothing_matched_ablation_v1 root")
    if pilot_root.name != "semantic_min_kd_pilot_v1":
        raise ValueError("Wrong completed pilot root")
    tracked = [str(path.relative_to(pilot.REPO)) for path in
               (ADDENDUM, SCRIPT, SBATCH, RUNTIME_SCRIPT, RUNTIME_REQUIREMENTS,
                RUNTIME_BOOTSTRAP)]
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *tracked],
        cwd=pilot.REPO, text=True).strip()
    if status:
        raise ValueError("Matched-ablation source differs from committed version")
    queue = subprocess.check_output(
        ["squeue", "-u", os.environ.get("USER", "adhyyan"), "-h", "-o", "%i|%T|%j"],
        text=True).splitlines()
    if any("semantic" in row.casefold() for row in queue):
        raise ValueError("Another semantic job is active")
    recovered = root / "recovered_inputs"
    prompts = recovered / (
        "smoke_prompts.json" if args.stage == "smoke" else "evaluation_prompts.json")
    proof = recovered / "reference_proof.json"
    validate_recovered_references(proof, recovered / "references")
    prompt_rows = pilot.read_json(prompts)
    expected = STAGES[args.stage][0]
    if len(prompt_rows) != expected or len(set(prompt_rows)) != expected:
        raise ValueError("Frozen prompt bank has changed")
    stage_root = root / args.stage
    if stage_root.exists():
        raise ValueError("Stage output already exists; audit before recovery")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=pilot.REPO, text=True).strip()
    files = [ADDENDUM, SCRIPT, SBATCH, RUNTIME_SCRIPT, RUNTIME_REQUIREMENTS,
             RUNTIME_BOOTSTRAP,
             proof, prompts,
             pilot.REPO / "configs/training.yaml",
             pilot.REPO / "configs/composed/eagle_joke_topaz_humor.yaml"]
    details = {
        "status": "pass",
        "stage": args.stage,
        "source_commit": commit,
        "queue_snapshot": queue,
        "output": str(stage_root),
        "inputs": {str(path.resolve()): pilot.file_sha(path) for path in files},
    }
    pilot.write_immutable(args.receipt, details)


def evaluate(args):
    for method in METHODS:
        evaluate_method(args, method)
    report(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = sub.add_parser("evaluate")
    for name in ("stage", "reference_proof", "prompts", "output",
                 "reference_root", "training_config", "composed_config"):
        evaluate_parser.add_argument("--" + name, required=True)
    evaluate_parser.add_argument("--devices", default="cuda:0")
    evaluate_parser.add_argument(
        "--hardware_family", choices=("l40s", "a100"), default="l40s")
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    report_parser.add_argument("--output", required=True)
    preflight_parser = sub.add_parser("preflight")
    preflight_parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    preflight_parser.add_argument("--pilot_root", required=True)
    preflight_parser.add_argument("--output", required=True)
    preflight_parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    if getattr(args, "stage", None) not in STAGES:
        raise ValueError("Unknown stage")
    {"evaluate": evaluate, "report": report, "preflight": preflight}[args.command](args)


if __name__ == "__main__":
    main()
