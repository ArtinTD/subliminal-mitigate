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
    expected = {"smoke": 8, "train": 256, "evaluation": 32}[args.stage]
    if len(prompts) != expected:
        raise ValueError(f"Wrong manifest size for {args.stage}")
    if args.devices not in ("cuda:0", "cuda:0,cuda:1"):
        raise ValueError("Only approved one- or two-L40S placement is supported")
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
                "seed": 0, "hardware_family": "l40s"}
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
        return checkpoint_call(out / "responses", digest(identity), prompt, index,
                               lambda: original(prompt, index, *pos, **kwargs))
    module.sample_one = resumed
    module.git_sha = lambda: HISTORICAL_SHA
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
    prepare = sub.add_parser("prepare")
    for name in ("eagle_dataset", "topaz_dataset", "evaluation_prompts", "smoke_prompts", "output"):
        prepare.add_argument("--" + name, required=True)
    curation = sub.add_parser("curate")
    for name in ("results", "manifest", "output"):
        curation.add_argument("--" + name, required=True)
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
    else:
        generate_teacher(args)


if __name__ == "__main__":
    main()
