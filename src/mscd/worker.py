"""Subprocess entry point: isolate training, vLLM, and inference GPU allocations."""
import sys
import os
import platform
import importlib.metadata
from pathlib import Path
from dataclasses import asdict
from mscd.artifacts import read_json, atomic_json, digest, tree_identity
from mscd.types import (
    SourceRecord,
    ModelArtifact,
    Request,
    GenerationConfig,
    GenerationRecord,
)


def execute(manifest, stage):
    run = read_json(manifest)
    c = run["config"]
    if c.get("recipe", "explicit-prefix") != "explicit-prefix":
        from mscd.recipe_worker import execute_stage

        record_environment(Path(manifest).parent / stage)
        return execute_stage(manifest, stage)
    root = Path(manifest).parent
    out = root / stage
    if stage == "report":
        report = {
            s: read_json(root / ("eval-" + s) / "metrics.json")
            for s in [
                "base",
                "eagle",
                "topaz",
                "union",
                "merge",
                "whole",
                "minimum",
                "student",
            ]
        }
        atomic_json(
            out / "report.json",
            dict(
                methods=report,
                dataset_seed=c["dataset_seed"],
                note="One new construction replicate; no across-dataset uncertainty estimate.",
                stage_runtime={
                    p.parent.name: read_json(p)["elapsed_seconds"]
                    for p in root.glob("*/complete.json")
                },
            ),
        )
        return
    # Record actual resolved model snapshot, never silently change its revision.
    from huggingface_hub import snapshot_download

    if not c.get("model_revision"):
        raise ValueError("Set model_revision to a commit hash using mscd pin")
    base = snapshot_download(
        c["base_model"],
        revision=c["model_revision"],
        local_files_only=c.get("offline", False),
    )
    if stage == "build-sources":
        from mscd.datasets.builders import (
            ExplicitPrefixDatasetBuilder,
            ImportedDatasetBuilder,
        )

        records = (
            ImportedDatasetBuilder(c["import_sources"]).build()
            if c.get("import_sources")
            else ExplicitPrefixDatasetBuilder(c, out).build()
        )
        overlap = set(r.prompt for r in records) & set(c["evaluation_prompts"])
        if overlap:
            raise ValueError(f"Training/evaluation prompt overlap: {len(overlap)}")
        atomic_json(out / "sources.json", [asdict(r) for r in records])
        atomic_json(
            out / "summary.json",
            dict(
                occurrences=len(records),
                unique_prompts=len({r.prompt for r in records}),
                eval_overlap=0,
            ),
        )
    elif stage.startswith("train-") and stage[6:] in c.get("provided_models", {}):
        spec = c["provided_models"][stage[6:]]
        if tree_identity(spec["path"]) != spec["identity"]:
            raise ValueError("Provided model changed after planning")
        from mscd.decoding._model_engine import load_adapter_config

        adapter_base = load_adapter_config(spec["path"]).get("base_model_name_or_path")
        if adapter_base not in {base, c["base_model"]}:
            raise ValueError("Provided adapter uses a different base model")
        # Normalize metadata to the pinned snapshot used throughout this run.
        artifact = ModelArtifact(spec["path"], base, base, stage[6:], spec["identity"])
        atomic_json(out / "model.json", asdict(artifact))
    elif stage.startswith("train-"):
        # Training imports first: Unsloth must initialize before other ML packages.
        from mscd.training.trainer import Trainer
        from mscd.datasets.records import write_dataset

        name = stage[6:]
        origin = root / (
            "regenerate/records.json"
            if name == "student"
            else "build-sources/sources.json"
        )
        records = [SourceRecord(**r) for r in read_json(origin)]
        if name in {"eagle", "topaz"}:
            records = [r for r in records if r.source_id == name]
        data = out / "dataset"
        if not data.exists():
            write_dataset(records, data)
        elif read_json(data / "occurrences.json") != [asdict(r) for r in records]:
            raise ValueError("Training data changed")
        train = c["training"]
        train["training"]["exact_steps"] = c.get("training_steps", 200)
        artifact = Trainer().fit(base, data, train, out / "model", name)
        atomic_json(out / "model.json", asdict(artifact))
    else:
        from mscd.decoding.generators import (
            ModelGenerator,
            ConsensusDecoder,
            MergedLoRAGenerator,
            WholeOutputConsensusGenerator,
            generate_cached,
        )
        from mscd.datasets.regeneration import DatasetRegenerator
        from mscd.evaluation.scoring import Evaluator

        name = stage[5:] if stage.startswith("eval-") else "minimum"
        if name == "base":
            generator = ModelGenerator(
                ModelArtifact(None, base, base, "base", c["model_revision"])
            )
        elif name in {"eagle", "topaz", "union", "student"}:
            generator = ModelGenerator(
                ModelArtifact(**read_json(root / f"train-{name}" / "model.json"))
            )
        else:
            teachers = [
                ModelArtifact(**read_json(root / f"train-{n}" / "model.json"))
                for n in ["eagle", "topaz"]
            ]
            if name == "minimum":
                generator = ConsensusDecoder(
                    teachers, tuple(c.get("devices", ["cuda:0", "cuda:1"]))
                )
            elif name == "merge":
                generator = MergedLoRAGenerator(teachers)
            elif name == "whole":
                generator = WholeOutputConsensusGenerator(teachers)
            else:
                raise ValueError(name)
        gc = GenerationConfig(**c.get("generation", {}))
        if stage == "regenerate":
            rows = [
                SourceRecord(**r)
                for r in read_json(root / "build-sources/sources.json")
            ]
            from datasets import Dataset

            order = Dataset.from_dict({"index": list(range(len(rows)))}).shuffle(
                seed=42
            )["index"]
            rows = [rows[i] for i in order]
            result = DatasetRegenerator().run(
                rows,
                generator,
                gc,
                out,
                run["implementation"],
                c.get("generation_seed", 0),
            )
            atomic_json(out / "records.json", [asdict(r) for r in result])
        else:
            requests = [
                Request(
                    f"eval:{i}:{j}",
                    p,
                    c.get("evaluation_seed", 0) + i * c["responses_per_prompt"] + j,
                )
                for i, p in enumerate(c["evaluation_prompts"])
                for j in range(c["responses_per_prompt"])
            ]
            results = generate_cached(
                generator, requests, gc, out / "requests", run["implementation"]
            )
            atomic_json(out / "records.json", [asdict(r) for r in results])
            atomic_json(out / "metrics.json", Evaluator().evaluate(results))
    record_environment(out)


def record_environment(out):
    packages = {}
    for name in [
        "torch",
        "transformers",
        "peft",
        "trl",
        "unsloth",
        "vllm",
        "datasets",
        "sentence-transformers",
        "xgrammar",
    ]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    atomic_json(
        out / "environment.json",
        dict(
            python=sys.version,
            platform=platform.platform(),
            packages=packages,
            slurm_job=os.environ.get("SLURM_JOB_ID"),
            gpu_visibility=os.environ.get("CUDA_VISIBLE_DEVICES"),
        ),
    )


if __name__ == "__main__":
    execute(*sys.argv[1:])
