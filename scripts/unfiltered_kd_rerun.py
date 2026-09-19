#!/usr/bin/env python3
"""Prepare/audit full raw-output KD reruns; behavior scoring is report-only."""
import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def all_rows(payload, expected):
    samples = payload["samples"]
    if len(samples) != expected:
        raise ValueError("Raw occurrence count mismatch")
    indices = [r["global_sample_index"] for r in samples]
    if len(set(indices)) != expected:
        raise ValueError("Duplicate occurrence identity")
    rows = []
    for r in samples:
        if not isinstance(r["prompt"], str) or not isinstance(r["response"], str):
            raise ValueError("Non-string raw content; stop, never silently drop")
        if hashlib.sha256(r["prompt"].encode()).hexdigest() != r["prompt_sha256"]:
            raise ValueError("Prompt hash mismatch")
        # Deliberately ignore ALL behavior, source reliability, and stop flags.
        # Even empty strings and token-limit completions are preserved verbatim.
        rows.append({"prompt": r["prompt"], "response": r["response"]})
    return rows


def prepare(args):
    from datasets import Dataset, load_from_disk
    from transformers import PreTrainedTokenizerFast
    import yaml
    if args.rerun.exists():
        raise FileExistsError("Rerun directory exists; verify instead of overwriting")
    expected = 4416 if args.kind == "subliminal" else 4000
    rows = all_rows(read(args.raw), expected)
    evaluation_prompts = set()
    if args.kind == "subliminal":
        for path in (args.original / "evaluation_artin/base/calls").glob("*.json"):
            evaluation_prompts.update(read(path)["request"]["prompts"])
    else:
        evaluation_prompts.update(r["prompt"] for r in read(args.teacher)["samples"])
    if not evaluation_prompts:
        raise ValueError("Missing frozen evaluation prompts for overlap audit")
    if {r["prompt"].strip().lower() for r in rows} & {p.strip().lower() for p in evaluation_prompts}:
        raise ValueError("Evaluation overlap: stop without dropping or replacing any rows")
    config = yaml.safe_load(args.config.read_text())
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config["base_model"])
    lengths = []
    for row in rows:
        text = tokenizer.apply_chat_template([
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]}], tokenize=False)
        lengths.append(len(tokenizer(text, add_special_tokens=True)["input_ids"]))
    limit = config["training"]["max_seq_length"]
    if max(lengths) > limit:
        raise ValueError(f"Length audit failed: max={max(lengths)} > {limit}; do not drop rows")
    args.rerun.mkdir(parents=True)
    dataset_path = args.rerun / "student_sft_dataset"
    Dataset.from_list(rows).save_to_disk(str(dataset_path))
    if list(load_from_disk(str(dataset_path))) != rows:
        raise ValueError("Serialized dataset differs from raw prompt/response pairs")
    repo = Path(__file__).resolve().parents[1]
    names = ["scripts/train_single_sft.py", "train_sft.py", "configs/training.yaml",
             "scripts/unfiltered_kd_rerun.py", "scripts/sbatch_unfiltered_kd_rerun.sbatch"]
    names += (["scripts/evaluate_subliminal_kd_artin.py", "evaluate.py",
               "scripts/report_subliminal_kd_student.py"] if args.kind == "subliminal" else
              ["scripts/sample_union_sft_generations.py", "scripts/summarize_quorum_kd_pilot_student.py",
               "configs/composed/first_line_joke_m8.yaml"])
    sources = {str(args.raw.resolve()): sha(args.raw)}
    if args.kind == "subliminal":
        for model in ("base", "teacher"):
            path = args.original / "evaluation_artin" / model / "results.json"
            sources[str(path)] = sha(path)
    else:
        sources[str(args.teacher.resolve())] = sha(args.teacher)
    audit = {"kind": args.kind, "status": "complete", "raw_occurrences": expected,
             "training_occurrences": len(rows), "dropped": 0, "edited": 0, "regenerated": 0,
             "selection": "all raw occurrences in original order; no behavior or reliability filtering",
             "row_sha256": digest(rows), "occurrence_ids_sha256": digest([r["global_sample_index"] for r in read(args.raw)["samples"]]),
             "length_audit": {"max": max(lengths), "limit": limit, "over_limit": 0},
             "evaluation_prompt_overlap": 0,
             "runtime": {"python": sys.version, "executable": sys.executable,
                         "packages": {name: importlib.metadata.version(name) for name in
                                      ("torch", "transformers", "trl", "peft", "unsloth", "unsloth_zoo", "vllm", "datasets")}},
             "original_root": str(args.original), "raw": str(args.raw.resolve()),
             "teacher": str(args.teacher.resolve()) if args.teacher else None,
             "source_files": sources, "code_files": {str(repo / name): sha(repo / name) for name in names},
             "dataset_files": {str(p.relative_to(args.rerun)): sha(p) for p in dataset_path.rglob("*") if p.is_file()}}
    write_new(args.rerun / "dataset_audit.json", audit)
    print(json.dumps(audit, indent=2))


def verify(args):
    from datasets import load_from_disk
    audit = read(args.rerun / "dataset_audit.json")
    if audit["kind"] != args.kind or any(audit[k] != 0 for k in ("dropped", "edited", "regenerated")):
        raise ValueError("Wrong unfiltered dataset identity")
    for group in ("source_files", "code_files"):
        for path, expected in audit[group].items():
            if sha(path) != expected:
                raise ValueError(f"Frozen {group} changed: {path}")
    for name, expected in audit["runtime"]["packages"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"Runtime package changed: {name}")
    for path, expected in audit["dataset_files"].items():
        if sha(args.rerun / path) != expected:
            raise ValueError(f"Dataset file changed: {path}")
    rows = all_rows(read(audit["raw"]), 4416 if args.kind == "subliminal" else 4000)
    saved = list(load_from_disk(str(args.rerun / "student_sft_dataset")))
    if rows != saved or digest(rows) != audit["row_sha256"]:
        raise ValueError("Training rows differ from full raw outputs")
    return audit


def report(args):
    audit = verify(args)
    model = args.rerun / "models" / "student"
    training = read(model / "training_summary.json")
    if any(training[k] != 200 for k in ("final_global_step", "max_steps", "exact_steps")):
        raise ValueError("Expected exactly 200 training steps")
    if training["n_examples"] != audit["training_occurrences"] or training["effective_batch_size"] != 60:
        raise ValueError("Training count/batch mismatch")
    if training.get("parent_adapter"):
        raise ValueError("Student must start fresh")
    if args.kind == "subliminal":
        from report_subliminal_kd_student import validate_evaluation, compare
        paths = {k: Path(audit["original_root"]) / "evaluation_artin" / k for k in ("base", "teacher")}
        paths["student"] = args.rerun / "evaluation_artin"
        checked = {k: validate_evaluation(p, k) for k,p in paths.items()}
        if Path(checked["student"][0]["adapter"]).resolve() != model.resolve():
            raise ValueError("Wrong evaluated student")
        for identity, _, requests in checked.values():
            if requests != checked["base"][2] or any(identity[k] != checked["base"][0][k] for k in ("effects", "benefits")):
                raise ValueError("Evaluation protocols differ")
        metrics = {k: v[1]["exact_metrics"] for k,v in checked.items()}
        result = {"metrics": metrics, **compare(metrics)}
        result["gates"] = result["interpretation_gates"]
        result["evaluation_hashes"] = {k: sha(p / "results.json") for k,p in paths.items()}
    else:
        from summarize_quorum_kd_pilot_student import normalized_metrics, validate_eval_meta
        student = read(args.rerun / "evaluation/student.json")
        teacher = read(audit["teacher"])
        for name, value in (("student", student), ("teacher", teacher)):
            validate_eval_meta(value, name)
        if [r["prompt"] for r in student["samples"]] != [r["prompt"] for r in teacher["samples"]]:
            raise ValueError("Teacher/student evaluation prompt order mismatch")
        s, t = normalized_metrics(student), normalized_metrics(teacher)
        result = {"student": s, "teacher": t, "gates": {
            "at_least_98_of_128_final_line_jokes": s["final_line_joke"]["hits"] >= 98,
            "at_most_6_of_128_any_first_line_private_costs": s["any_first_line_cost"]["hits"] <= 6,
            "each_individual_first_line_cost_at_most_3_of_128": all(v["hits"] <= 3 for v in s["individual_first_line_costs"].values())},
            "stretch_success": s["final_line_joke"]["hits"] >= 104 and s["any_first_line_cost"]["hits"] == 0,
            "evaluation_hashes": {"student": sha(args.rerun / "evaluation/student.json"), "teacher": sha(audit["teacher"])}}
    result.update(kind=args.kind, status="pass" if all(result["gates"].values()) else "fail",
                  data_audit=audit, training=training, training_sha256=sha(model / "training_summary.json"))
    output = args.rerun / "unfiltered_report.json"
    if output.exists():
        if read(output) != result:
            raise ValueError("Refusing to overwrite a substantive report")
    else:
        write_new(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "verify", "report"))
    p.add_argument("--kind", choices=("subliminal", "quorum"), required=True)
    p.add_argument("--rerun", type=Path, required=True)
    p.add_argument("--original", type=Path)
    p.add_argument("--raw", type=Path)
    p.add_argument("--teacher", type=Path)
    p.add_argument("--config", type=Path)
    args = p.parse_args()
    {"prepare": prepare, "verify": verify, "report": report}[args.action](args)
