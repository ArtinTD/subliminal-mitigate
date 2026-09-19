#!/usr/bin/env python3
"""Guarded semantic-min KD pilot preparation and restartable teacher generation.

The historical decoder is loaded verbatim from its Git object. No decoder math,
scorers, prompt formatting, or per-response RNG formulas are reimplemented here.
GPU scheduling remains external: this CLI never submits jobs.
"""
import argparse
import ast
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import types

REPO = Path(__file__).resolve().parents[1]
HISTORICAL_SHA = "7291b987bdb81a0aacd4d8ce3b657116ab00eb19"
SAMPLER = "scripts/sample_quorum_composition_generations.py"
REFERENCES = ("eagle_joke", "topaz_humor")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def file_sha(path):
    sha = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_immutable(path, value):
    """Atomic no-clobber publication, including concurrent-writer protection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Refusing changed output: {path}")
        return
    fd, temp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp, path)
        except FileExistsError:
            if read_json(path) != value:
                raise ValueError(f"Concurrent changed output: {path}")
    finally:
        os.unlink(temp)


def historical_source():
    return subprocess.check_output(
        ["git", "show", f"{HISTORICAL_SHA}:{SAMPLER}"], cwd=REPO, text=True)


def historical_scorers():
    """Extract only pure historical scorer definitions, requiring no ML imports."""
    names = {"first_nonempty_line", "nonempty_lines", "joke_position_bucket",
             "marker_line_indices", "marker_position_metrics", "has_first_line_prefix",
             "has_line_prefix_anywhere", "MARKER_NAMES", "MARKER_FLEX_LINE_RES"}
    tree = ast.parse(historical_source())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in names for t in node.targets):
            nodes.append(node)
    namespace = {"re": re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), SAMPLER, "exec"), namespace)
    return namespace


def wilson(hits, total):
    if not total:
        raise ValueError("No observations")
    z = 1.959963984540054
    p = hits / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / den
    return [center - radius, center + radius]


def metrics(records):
    score = historical_scorers()
    counts = dict.fromkeys(("final_either_marker", "final_joke_marker", "final_humor_marker",
                            "first_line_cost", "any_line_initial_cost", "truncated"), 0)
    for row in records:
        text = row["response"]
        markers = score["marker_position_metrics"](text)
        for key in ("final_either_marker", "final_joke_marker", "final_humor_marker"):
            counts[key] += int(markers["has_" + key])
        counts["first_line_cost"] += int(any(score["has_first_line_prefix"](text, prefix)
                                             for prefix in ("Eagle:", "Topaz:")))
        counts["any_line_initial_cost"] += int(any(score["has_line_prefix_anywhere"](text, prefix)
                                                   for prefix in ("Eagle:", "Topaz:")))
        counts["truncated"] += int(row.get("stop_reason") == "max_new_tokens")
    n = len(records)
    return {key: {"hits": hits, "n": n, "rate": hits / n, "wilson95": wilson(hits, n)}
            for key, hits in counts.items()}


def normalize_prompt(prompt):
    return " ".join(prompt.split()).casefold()


def build_manifest(sources, evaluation_prompts, smoke_prompts):
    if set(sources) != set(REFERENCES):
        raise ValueError("Exactly the two approved source datasets are required")
    if len(evaluation_prompts) != 32 or len(smoke_prompts) != 8:
        raise ValueError("Require the full 32-prompt evaluation bank and eight smoke prompts")
    if len({normalize_prompt(p) for p in evaluation_prompts}) != 32:
        raise ValueError("Duplicate evaluation prompt")
    if len({normalize_prompt(p) for p in smoke_prompts}) != 8:
        raise ValueError("Duplicate smoke prompt")
    eval_set = {normalize_prompt(p) for p in evaluation_prompts}
    if eval_set & {normalize_prompt(p) for p in smoke_prompts}:
        raise ValueError("Smoke/evaluation overlap")
    excluded = eval_set | {normalize_prompt(p) for p in smoke_prompts}
    chosen = {}
    for source in REFERENCES:
        rows = []
        for index, row in enumerate(sources[source]):
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Malformed source prompt: {source}/{index}")
            if normalize_prompt(prompt) not in excluded:
                rows.append({"source": source, "source_row": index, "prompt": prompt,
                             "occurrence_id": f"{source}:{index}"})
        random.Random(0).shuffle(rows)
        if len(rows) < 128:
            raise ValueError(f"Insufficient source occurrences: {source}")
        chosen[source] = rows[:128]
    result = []
    for index in range(128):
        for source in REFERENCES:
            row = dict(chosen[source][index], global_sample_index=len(result))
            result.append(row)
    return result


def curate(records, manifest):
    """Never use benefit or cost as an inclusion criterion."""
    if len(manifest) != 256 or len(records) != 256:
        raise ValueError("Require exactly 256 responses; never silently replace missing rows")
    if [r["global_sample_index"] for r in records] != list(range(256)):
        raise ValueError("Duplicate, missing, or unordered response indices")
    kept, excluded = [], []
    for row, source in zip(records, manifest):
        if row.get("prompt") != source["prompt"]:
            raise ValueError("Response prompt differs from manifest")
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
    return {"status": "pass" if len(kept) >= 128 else "fail", "raw_count": 256,
            "usable_count": len(kept), "excluded": excluded, "rows": kept}


def validate_reference_proof(path):
    """Require reviewed evidence of historical identity, not just current hashes."""
    proof = read_json(path)
    if proof.get("status") != "verified_historical_identity" or not proof.get("reviewed_by"):
        raise ValueError("Historical adapter provenance is unresolved; GPU work is blocked")
    if set(proof.get("references", {})) != set(REFERENCES):
        raise ValueError("Wrong reference set")
    inputs = {}
    for name, item in proof["references"].items():
        adapter = Path(item["adapter_path"])
        expected = item["historical_weight_sha256"]
        if file_sha(adapter / "adapter_model.safetensors") != expected:
            raise ValueError(f"Historical adapter mismatch: {name}")
        if not item.get("evidence"):
            raise ValueError(f"No historical provenance evidence: {name}")
        for evidence in item["evidence"]:
            if file_sha(evidence["path"]) != evidence["sha256"]:
                raise ValueError(f"Changed provenance evidence: {name}")
        for filename in ("adapter_model.safetensors", "adapter_config.json", "eval_meta.json"):
            target = adapter / filename
            inputs[str(target.resolve())] = file_sha(target)
    inputs[str(Path(path).resolve())] = file_sha(path)
    return proof, inputs


def snapshot_inputs():
    """Record actual cached model revisions/weights; never silently download here."""
    from huggingface_hub import snapshot_download
    snapshots = {}
    for model in ("unsloth/Qwen3-8B", "BAAI/bge-base-en-v1.5"):
        folder = Path(snapshot_download(model, local_files_only=True)).resolve()
        files = {str(p.relative_to(folder)): file_sha(p) for p in sorted(folder.rglob("*"))
                 if p.is_file() and not p.name.startswith(".")}
        if not any(p.endswith(".safetensors") for p in files):
            raise ValueError(f"Incomplete cached weights: {model}")
        snapshots[model] = {"resolved_path": str(folder), "files": files}
    return {"snapshots": snapshots,
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "peft", "sentence-transformers")}}


def check_training_config(path):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg["base_model"] != "unsloth/Qwen3-8B":
        raise ValueError("Base model changed")
    if (cfg["lora"]["rank"], cfg["lora"]["alpha"]) != (8, 8):
        raise ValueError("LoRA recipe changed")
    expected = {"batch_size": 20, "gradient_accumulation": 3, "lr": 2e-4,
                "warmup_steps": 5, "lr_scheduler_type": "linear", "max_seq_length": 2048,
                "dtype": "bfloat16"}
    for key, value in expected.items():
        if cfg["training"][key] != value:
            raise ValueError(f"Training recipe changed: {key}")
    return cfg


def materialize_dataset(args):
    """Format exactly as train_sft does and refuse silent sequence truncation."""
    from datasets import Dataset
    from transformers import PreTrainedTokenizerFast
    result = curate(read_json(args.results)["samples"], read_json(args.manifest))
    if result["status"] != "pass":
        raise ValueError("Less than 128 intact examples; no automatic replacement")
    tokenizer = PreTrainedTokenizerFast.from_pretrained("unsloth/Qwen3-8B", local_files_only=True)
    lengths = []
    for row in result["rows"]:
        text = tokenizer.apply_chat_template([
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]}], tokenize=False)
        lengths.append(len(tokenizer(text, truncation=False)["input_ids"]))
    if max(lengths) > 2048:
        raise ValueError(f"Formatted training sequence exceeds 2048: {max(lengths)}; review required")
    output = Path(args.output)
    if output.exists():
        raise ValueError("Dataset output exists; audit rather than overwrite")
    # Save into a staging sibling; completed dataset is published only after audit.
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dataset-", dir=output.parent))
    Dataset.from_list(result["rows"]).save_to_disk(str(staging))
    from datasets import load_from_disk
    if list(load_from_disk(str(staging))) != result["rows"]:
        raise ValueError("Dataset round-trip mismatch")
    write_immutable(staging / "curation_report.json", {
        key: value for key, value in result.items() if key != "rows"})
    write_immutable(staging / "dataset_identity.json", {
        "raw_results_sha256": file_sha(args.results), "manifest_sha256": file_sha(args.manifest),
        "rows_sha256": digest(result["rows"]), "n_rows": len(result["rows"]),
        "formatted_token_lengths": lengths, "maximum_formatted_length": max(lengths),
        "tokenizer_chat_template_sha256": digest(tokenizer.chat_template),
        "data_files": {p.name: file_sha(p) for p in staging.iterdir() if p.is_file()}})
    os.rename(staging, output)


def train_student(args):
    # Import Unsloth before any transformers/torch imports, matching original training.
    import train_single_sft as train
    from datasets import load_from_disk
    from transformers import set_seed
    check_training_config(args.training_config)
    output = Path(args.output)
    if output.exists():
        raise ValueError("Student output already exists; never resume/reinitialize silently")
    data = Path(args.dataset)
    identity = read_json(data / "dataset_identity.json")
    for name, expected in identity["data_files"].items():
        if file_sha(data / name) != expected:
            raise ValueError(f"Training dataset file changed: {name}")
    rows = list(load_from_disk(str(data)))
    if digest(rows) != identity["rows_sha256"] or not 128 <= len(rows) <= 256:
        raise ValueError("Dataset rows changed or wrong size")
    if identity["maximum_formatted_length"] > 2048:
        raise ValueError("Training would truncate")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Student recipe requires single GPU")
    import torch
    if torch.cuda.device_count() != 1 or "L40S" not in torch.cuda.get_device_name(0):
        raise ValueError("Student training requires one L40S")
    set_seed(42)
    old = sys.argv
    try:
        sys.argv = ["train_single_sft.py", "--dataset", str(data), "--training_config", args.training_config,
                    "--output_dir", str(output.parent), "--name", output.name,
                    "--epochs", "1", "--min_steps", "50"]
        train.main()
    finally:
        sys.argv = old
    summary = read_json(output / "training_summary.json")
    if (summary["final_global_step"], summary["max_steps"], summary["effective_batch_size"],
            summary["n_examples"]) != (50, 50, 60, len(rows)):
        raise ValueError("Student did not complete the approved 50-step recipe")
    write_immutable(output / "pilot_training_identity.json", {
        "seed": 42, "initialization": "fresh_base", "dataset_identity_sha256": file_sha(data / "dataset_identity.json"),
        "training_config_sha256": file_sha(args.training_config),
        "training_source_sha256": file_sha(REPO / "scripts/train_single_sft.py"),
        "sft_source_sha256": file_sha(REPO / "train_sft.py"),
        "adapter_sha256": file_sha(output / "adapter_model.safetensors"), "summary": summary})


def checkpoint_call(directory, identity_sha, prompt, index, generate):
    path = Path(directory) / f"{index:06d}.json"
    if path.exists():
        saved = read_json(path)
        record = saved["record"]
        if (saved["identity_sha256"] != identity_sha or saved["sha256"] != digest(record)
                or record.get("prompt") != prompt or record.get("global_sample_index") != index):
            raise ValueError(f"Checkpoint corruption or changed identity: {path}")
        return copy.deepcopy(record)
    record = generate()
    if record.get("prompt") != prompt or record.get("global_sample_index") != index:
        raise ValueError("Decoder returned a different prompt/index")
    write_immutable(path, {"identity_sha256": identity_sha, "sha256": digest(record), "record": record})
    return copy.deepcopy(record)


def generate_teacher(args):
    proof, inputs = validate_reference_proof(args.reference_proof)
    prompts = read_json(args.prompts)
    check_training_config(args.training_config)
    expected = {"smoke": 8, "train": 256, "evaluation": 32}[args.stage]
    if len(prompts) != expected:
        raise ValueError(f"Wrong manifest size for {args.stage}")
    if args.devices not in ("cuda:0", "cuda:0,cuda:1"):
        raise ValueError("Only approved one- or two-L40S placement is supported")
    if args.model != "teacher" and (args.stage != "evaluation" or args.devices != "cuda:0"):
        raise ValueError("Base/student are evaluated only, on one L40S")
    if args.stop_after is not None and (args.stage != "smoke" or args.stop_after != 4):
        raise ValueError("Only the approved four-response smoke interruption is supported")
    if args.model == "student":
        if not args.student_adapter:
            raise ValueError("Student adapter required")
        student = Path(args.student_adapter)
        training = read_json(student / "pilot_training_identity.json")
        if training["summary"]["final_global_step"] != 50 or training["initialization"] != "fresh_base":
            raise ValueError("Wrong student training provenance")
        for name in ("adapter_config.json", "adapter_model.safetensors", "pilot_training_identity.json"):
            inputs[str((student / name).resolve())] = file_sha(student / name)
        if inputs[str((student / "adapter_model.safetensors").resolve())] != training["adapter_sha256"]:
            raise ValueError("Student adapter changed")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise ValueError("Completed result exists; audit it, do not overwrite")
    source = historical_source()
    inputs.update({str(Path(p).resolve()): file_sha(p) for p in
                   (args.prompts, args.training_config, args.composed_config, __file__)})
    identity = {"historical_commit": HISTORICAL_SHA,
                "historical_sampler_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "inputs": inputs, "stage": args.stage, "devices": args.devices,
                "n_samples": 4 if args.stage == "evaluation" else 1,
                "seed": 0, "hardware_family": "l40s", "model": args.model,
                "runtime": snapshot_inputs()}
    # The resource owner must authorize this job externally; verify actual family.
    import torch
    if torch.cuda.device_count() != len(args.devices.split(",")):
        raise ValueError("GPU allocation differs from frozen placement")
    if not all("L40S" in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())):
        raise ValueError("Semantic pilot is restricted to L40S GPUs")
    identity["torch_version"] = torch.__version__
    write_immutable(out / "identity.json", identity)
    specs = {"models": {name: proof["references"][name]["adapter_path"] for name in REFERENCES}}
    write_immutable(out / "model_specs.json", specs)
    module = types.ModuleType("historical_semantic_sampler")
    module.__file__ = str(REPO / SAMPLER)
    exec(compile(source, f"{HISTORICAL_SHA}:{SAMPLER}", "exec"), module.__dict__)
    original = module.sample_one
    def resumed(prompt, index, *pos, **kwargs):
        if args.stop_after is not None and index >= args.stop_after:
            raise SystemExit(75)  # Intentional smoke interruption, never a science change.
        return checkpoint_call(out / "responses", digest(identity), prompt, index,
                               lambda: original(prompt, index, *pos, **kwargs))
    module.sample_one = resumed
    module.git_sha = lambda: HISTORICAL_SHA
    if args.model == "base":
        module.load_reference = lambda base, adapter, device: module.load_base_model(base, device)
    if args.model == "student":
        specs = {"models": {"student": str(student.resolve())}}
        write_immutable(out / "student_model_specs.json", specs)
    arguments = ["--model_specs", str(out / "model_specs.json"), "--refs", ",".join(REFERENCES),
                 "--training_config", args.training_config, "--composed_config", args.composed_config,
                 "--probe_prompts", args.prompts, "--output_file", str(out / "results.json"),
                 "--composition_type", "span_token_smoothing", "--quorum_q", "2",
                 "--span_h", "6", "--span_proposals_per_ref", "4", "--span_kernel_lambda", "0.5",
                 "--span_kernel_tau", "0.05", "--span_kernel_top_k", "8",
                 "--span_similarity_gate", "hard", "--span_similarity_threshold", "0.70",
                 "--span_support_normalization", "length", "--span_embedding_source", "sentence_transformer",
                 "--span_embedding_model", "BAAI/bge-base-en-v1.5", "--span_embedding_text_mode", "canonical",
                 "--span_token_implementation", "cached_batched", "--span_token_parallel_refs", "auto",
                 "--span_embedding_device", "cuda:0", "--span_embedding_cache_size", "50000",
                 "--devices", args.devices, "--compose_device", "cuda:0", "--seed", "0",
                 "--n_samples", str(identity["n_samples"]), "--temperature", "1", "--max_new_tokens", "512"]
    if args.model != "teacher":
        # q=1 with one model is the unmodified model's own normalized distribution.
        # This retains historical prompt formatting and global-index RNG mapping.
        arguments[arguments.index("--refs") + 1] = "student" if args.model == "student" else "eagle_joke"
        arguments[arguments.index("--composition_type") + 1] = "quorum"
        arguments[arguments.index("--quorum_q") + 1] = "1"
        if args.model == "student":
            arguments[arguments.index("--model_specs") + 1] = str(out / "student_model_specs.json")
    previous = sys.argv
    try:
        sys.argv = [SAMPLER] + arguments
        module.main()
    finally:
        sys.argv = previous
    result = read_json(out / "results.json")
    if len(result["samples"]) != expected * identity["n_samples"]:
        raise ValueError("Incomplete teacher result")
    write_immutable(out / "exact_metrics.json", metrics(result["samples"]))


def audit_generation(folder, expected, expected_model):
    folder = Path(folder)
    identity = read_json(folder / "identity.json")
    result = read_json(folder / "results.json")
    rows = result["samples"]
    if identity["model"] != expected_model or len(rows) != expected:
        raise ValueError("Wrong model or response count")
    if [r["global_sample_index"] for r in rows] != list(range(expected)):
        raise ValueError("Missing or duplicate generation indices")
    for path, sha in identity["inputs"].items():
        if file_sha(path) != sha:
            raise ValueError(f"Frozen generation input changed: {path}")
    if len(list((folder / "responses").glob("*.json"))) != expected:
        raise ValueError("Unexpected number of response checkpoints")
    for row in rows:
        checkpoint = read_json(folder / "responses" / f"{row['global_sample_index']:06d}.json")
        if checkpoint["identity_sha256"] != digest(identity) or checkpoint["sha256"] != digest(checkpoint["record"]):
            raise ValueError("Invalid response checkpoint")
        # Historical main adds prompt/sample metadata after sample_one returns.
        if any(row.get(key) != value for key, value in checkpoint["record"].items()):
            raise ValueError("Final response differs from atomic checkpoint")
    return rows, identity, result


def smoke_report(args):
    root = Path(args.root)
    rows0, identity0, result0 = audit_generation(root / "smoke/repeat0", 8, "teacher")
    rows1, identity1, result1 = audit_generation(root / "smoke/repeat1", 8, "teacher")
    if identity0 != identity1 or rows0 != rows1:
        raise ValueError("Smoke replay/resume mismatch; do not generate training data")
    before = read_json(root / "smoke/interrupted_hashes.json")
    if len(before) != 4:
        raise ValueError("Missing four-response interruption evidence")
    for name, sha in before.items():
        if file_sha(root / "smoke/repeat1/responses" / name) != sha:
            raise ValueError("Resume modified a completed smoke checkpoint")
    seconds = float(result0["meta"]["runtime_seconds"])
    report = {"status": "pass", "devices": identity0["devices"], "response_count": 8,
              "exact_replay": True, "interrupted_completed_bytes_unchanged": True,
              "runtime_seconds": seconds, "projected_256_generation_seconds": seconds * 32,
              "projected_128_teacher_evaluation_seconds": seconds * 16,
              "identity_sha256": digest(identity0), "metrics": metrics(rows0)}
    write_immutable(root / "smoke/smoke_report.json", report)
    print(json.dumps(report, indent=2))


def final_report(args):
    root = Path(args.root)
    smoke = read_json(root / "smoke/smoke_report.json")
    if smoke["status"] != "pass":
        raise ValueError("Smoke not passed")
    sets, identities = {}, {}
    for model in ("teacher", "base", "student"):
        rows, identity, result = audit_generation(root / "evaluation" / model, 128, model)
        if identity["stage"] != "evaluation" or identity["n_samples"] != 4:
            raise ValueError("Wrong evaluation protocol")
        sets[model], identities[model] = rows, identity
    prompts = [row["prompt"] for row in sets["teacher"]]
    if len(set(prompts)) != 32 or any(prompts != [r["prompt"] for r in rows] for rows in sets.values()):
        raise ValueError("Evaluation prompts differ across models")
    train_rows, train_id, _ = audit_generation(root / "generation", 256, "teacher")
    for identity in identities.values():
        if (identity["historical_sampler_sha256"] != train_id["historical_sampler_sha256"]
                or identity["runtime"] != train_id["runtime"]):
            raise ValueError("Generation and evaluation decoder/runtime identities differ")
    if identities["teacher"]["devices"] != train_id["devices"]:
        raise ValueError("Teacher evaluation changed the selected GPU placement")
    if {normalize_prompt(r["prompt"]) for r in train_rows} & {normalize_prompt(p) for p in prompts}:
        raise ValueError("Training/evaluation overlap")
    curation = curate(train_rows, read_json(root / "manifests/training_manifest.json"))
    if curation["status"] != "pass":
        raise ValueError("Curation count prerequisite failed")
    dataset = read_json(root / "student_dataset/dataset_identity.json")
    if dataset["rows_sha256"] != digest(curation["rows"]):
        raise ValueError("Student training dataset differs from knowledge-blind curation")
    training = read_json(root / "models/pi_semantic_min_kd_pilot/pilot_training_identity.json")
    summary = training["summary"]
    if (summary["final_global_step"], summary["effective_batch_size"], summary["n_examples"]) != (50, 60, len(curation["rows"])):
        raise ValueError("Final training summary mismatch")
    results = {model: metrics(rows) for model, rows in sets.items()}
    differences = {key: results["student"][key]["rate"] - results["teacher"][key]["rate"]
                   for key in results["teacher"]}
    report = {"status": "complete", "scope": "diagnostic_feasibility_pilot_only",
              "historical_identity_verified": True, "smoke": smoke,
              "raw_training_responses": 256, "usable_training_examples": len(curation["rows"]),
              "exclusions": curation["excluded"], "training": training,
              "evaluation": results, "student_minus_teacher": differences,
              "generation_identity_sha256": digest(train_id),
              "evaluation_identity_sha256": {name: digest(value) for name, value in identities.items()},
              "recommendation": "Review marker retention, prefix costs, and runtime before choosing any full-scale budget; no automatic scale-up.",
              "caveat": "Single training seed and small evaluation; zero observed cost is not proof of zero risk."}
    write_immutable(root / "pilot_report.json", report)
    # Markdown is derivative and published once, not overwritten.
    lines = ["# Semantic smoothed-minimum KD feasibility pilot", "", "Diagnostic result, not a full-scale success gate.", "",
             "| Model | Either final marker | Joke | Humor | First-line cost | Any line-initial cost |", "|---|---:|---:|---:|---:|---:|"]
    for model, scores in results.items():
        keys = ("final_either_marker", "final_joke_marker", "final_humor_marker", "first_line_cost", "any_line_initial_cost")
        lines.append("| " + model + " | " + " | ".join(f"{scores[k]['hits']}/128 ({scores[k]['rate']:.2%})" for k in keys) + " |")
    lines += ["", f"Training: {len(curation['rows'])}/256 intact responses, fresh student, 50 steps.",
              "", report["recommendation"], "", report["caveat"], "",
              "Historical published teacher result remains separate: 109/128 final markers, 0 first-line costs, 1 any-line cost.",
              "Exact counts, Wilson 95% intervals, identities, exclusions and timing are in pilot_report.json."]
    markdown = "\n".join(lines) + "\n"
    target = root / "pilot_report.md"
    if target.exists():
        if target.read_text() != markdown:
            raise ValueError("Existing Markdown report differs")
    else:
        with target.open("x") as stream:
            stream.write(markdown)
    compact = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".md", ".log", ".out", ".err", ".yaml") and path.name != "archive_manifest.json":
            compact[str(path.relative_to(root))] = file_sha(path)
    write_immutable(root / "archive_manifest.json", {"files": compact})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score-historical")
    score.add_argument("--results", required=True)
    score.add_argument("--output", required=True)
    teacher = sub.add_parser("teacher")
    for name in ("reference_proof", "prompts", "output", "training_config", "composed_config"):
        teacher.add_argument("--" + name, required=True)
    teacher.add_argument("--stage", choices=("smoke", "train", "evaluation"), required=True)
    teacher.add_argument("--devices", default="cuda:0")
    teacher.add_argument("--model", choices=("teacher", "base", "student"), default="teacher")
    teacher.add_argument("--student_adapter")
    teacher.add_argument("--stop_after", type=int, help="Operational smoke interruption test only")
    prepare = sub.add_parser("prepare")
    for name in ("eagle_dataset", "topaz_dataset", "evaluation_prompts", "smoke_prompts", "output"):
        prepare.add_argument("--" + name, required=True)
    curation = sub.add_parser("curate")
    for name in ("results", "manifest", "output"):
        curation.add_argument("--" + name, required=True)
    materialize = sub.add_parser("materialize")
    for name in ("results", "manifest", "output"):
        materialize.add_argument("--" + name, required=True)
    training = sub.add_parser("train")
    for name in ("dataset", "training_config", "output"):
        training.add_argument("--" + name, required=True)
    for command in ("smoke-report", "report"):
        reporting = sub.add_parser(command)
        reporting.add_argument("--root", required=True)
    args = parser.parse_args()
    if args.command == "score-historical":
        result = read_json(args.results)
        report = metrics(result["samples"])
        assert report["final_either_marker"]["hits"] == 109
        assert report["first_line_cost"]["hits"] == 0
        assert report["any_line_initial_cost"]["hits"] == 1
        write_immutable(args.output, {"source_sha256": file_sha(args.results), "metrics": report})
    elif args.command == "prepare":
        from datasets import load_from_disk
        sources = {name: list(load_from_disk(path)) for name, path in
                   zip(REFERENCES, (args.eagle_dataset, args.topaz_dataset))}
        evaluation = read_json(args.evaluation_prompts)
        smoke = read_json(args.smoke_prompts)
        manifest = build_manifest(sources, evaluation, smoke)
        output = Path(args.output)
        write_immutable(output / "training_manifest.json", manifest)
        write_immutable(output / "evaluation_prompts.json", evaluation)
        write_immutable(output / "smoke_prompts.json", smoke)
        write_immutable(output / "manifest_provenance.json", {
            "source_rows_sha256": {name: digest(rows) for name, rows in sources.items()},
            "source_counts": {name: len(rows) for name, rows in sources.items()},
            "training_manifest_sha256": digest(manifest), "selection_seed": 0,
            "evaluation_prompts_sha256": digest(evaluation), "smoke_prompts_sha256": digest(smoke)})
    elif args.command == "curate":
        report = curate(read_json(args.results)["samples"], read_json(args.manifest))
        write_immutable(args.output, report)
        if report["status"] != "pass":
            raise SystemExit("Fewer than 128 intact responses: review required")
        print("Curation complete. Formatted-token-length audit is still required before training.")
    elif args.command == "materialize":
        materialize_dataset(args)
    elif args.command == "train":
        train_student(args)
    elif args.command == "smoke-report":
        smoke_report(args)
    elif args.command == "report":
        final_report(args)
    else:
        generate_teacher(args)


if __name__ == "__main__":
    main()
