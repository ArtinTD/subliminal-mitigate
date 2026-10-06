"""Connect setting components for one stage, passing only saved artifacts."""
import copy
from dataclasses import asdict
from pathlib import Path
from mscd.artifacts import read_json, atomic_json, tree_identity, digest
from mscd.types import (
    SourceRecord,
    GenerationRecord,
    ModelArtifact,
    Request,
    GenerationConfig,
)
from mscd.recipes import build_recipe


def input_path(config, key):
    spec = config.get("input_files", {}).get(key, {})
    if not spec.get("path"):
        raise ValueError(
            f"Bind input_files.{key}.path to its archived/pinned input before execution"
        )
    if tree_identity(spec["path"]) != spec["identity"]:
        raise ValueError(f"Input changed since planning: {key}")
    return Path(spec["path"])


def resolve_base(c):
    if not c.get("model_revision"):
        raise ValueError(
            "Pin model_revision before GPU execution; historical missing revisions are not inferred"
        )
    from huggingface_hub import snapshot_download

    return snapshot_download(
        c["base_model"],
        revision=c["model_revision"],
        local_files_only=c.get("offline", False),
    )


def read_model(root, name):
    model = ModelArtifact(**read_json(root / f"train-{name}" / "model.json"))
    if model.path and model.identity not in {
        tree_identity(model.path),
        tree_identity(model.path, exclude=("artifact.json",)),
    }:
        raise ValueError(f"Model artifact changed: {name}")
    return model


def generator_for(c, method, root, base, suite=None, requests=None):
    if c["recipe"] == "in-context":
        from mscd.decoding.in_context import in_context_generator

        return in_context_generator(c, method, suite)
    from mscd.decoding.generators import (
        ModelGenerator,
        ConsensusDecoder,
        MergedLoRAGenerator,
        WholeOutputConsensusGenerator,
    )
    from mscd.decoding.rules import consensus_rule

    def model_for(name):
        model = read_model(root, name)
        if model.base_model != base or model.tokenizer != base:
            raise ValueError(
                "Imported model must identify the configured pinned base and tokenizer"
            )
        return model

    spec = c["methods"][method]
    if spec.get("backend_input"):
        spec = dict(spec, **read_json(input_path(c, spec["backend_input"])))
    base_artifact = ModelArtifact(None, base, base, "base", c["model_revision"])
    if spec.get("backend") in {"subliminal_historical", "quorum_batched"}:
        from mscd.decoding.subliminal import SubliminalGenerator

        names = (
            spec.get("teachers", [])
            if spec["kind"] == "consensus"
            else ([spec["model"]] if spec["kind"] == "single" else [])
        )
        if spec.get("microbatch", 16) is None:
            raise ValueError(
                "Bind the calibrated historical microbatch size for this hardware"
            )
        return SubliminalGenerator(
            [model_for(n) for n in names],
            requests,
            base=base_artifact,
            devices=spec.get("devices", c.get("devices")),
            kind="quorum"
            if spec.get("backend") == "quorum_batched"
            else ("directional" if spec["kind"] == "consensus" else "single"),
            microbatch=spec.get("microbatch", 16),
            q=spec.get("consensus", {}).get("q"),
            batches=spec.get("batches"),
        )
    if c["recipe"] in {"em", "massive"}:
        from mscd.decoding.medical import medical_generator

        return medical_generator(c, method, root, base, suite)
    if spec["kind"] in {"single", "base"}:
        model = base_artifact if spec["kind"] == "base" else model_for(spec["model"])
        if spec.get("backend") == "tokenwise":
            from mscd.decoding.generators import TokenwiseGenerator

            return TokenwiseGenerator(model, spec.get("device", "cuda:0"))
        if spec.get("backend") == "vllm_seeded_batch":
            from mscd.decoding.generators import SeededBatchModelGenerator

            return SeededBatchModelGenerator(
                model, requests, c["suites"][suite]["seed"]
            )
        return ModelGenerator(model)
    teachers = [model_for(name) for name in spec["teachers"]]
    if spec["kind"] == "merge":
        return MergedLoRAGenerator(
            teachers,
            prompt_bank=requests,
            samples_per_prompt=c["suites"][suite]["responses_per_prompt"],
            seed=c["suites"][suite].get("seed", 0),
        )
    if spec["kind"] == "whole":
        return WholeOutputConsensusGenerator(
            teachers, shared_stream_seed=c["suites"][suite].get("seed", 0)
        )
    rule = consensus_rule(spec.get("consensus", {}))
    smoother = None
    if spec.get("smoothing"):
        from mscd.decoding.smoothing import SpanSemanticSmoother

        smoother = SpanSemanticSmoother(**spec["smoothing"])
    devices = spec.get("devices", c.get("devices"))
    if devices is None or len(devices) != len(teachers):
        raise ValueError(f"Declare one device per teacher for {method}")
    return ConsensusDecoder(
        teachers,
        devices,
        rule,
        base=base_artifact if rule.requires_base else None,
        base_device=c.get("base_device", "cuda:0"),
        smoother=smoother,
    )


def suite_rows(c, suite):
    from mscd.datasets.builders import load_rows

    spec = c["suites"][suite]
    if "input" in spec:
        return load_rows(input_path(c, spec["input"]))
    prompts = spec.get("prompts", c.get("evaluation_prompts", []))
    if not prompts:
        raise ValueError(f"No evaluation prompts for {suite}")
    return [{"prompt": p, "question_id": str(i)} for i, p in enumerate(prompts)]


def suite_requests(c, suite):
    spec = c["suites"][suite]
    rows = suite_rows(c, suite)
    n = spec["responses_per_prompt"]
    if (
        spec.get("expected_prompts") is not None
        and len(rows) != spec["expected_prompts"]
    ):
        raise ValueError(f"Wrong prompt count for {suite}")
    # Only textual prompts and stable IDs enter the common generator interface.
    for row in rows:
        if "seeds" in row and len(row["seeds"]) != n:
            raise ValueError("Request seed inventory differs")
        if spec.get("require_recorded_seeds") and "seeds" not in row:
            raise ValueError(
                "Historical teacher requests require global seeds from the recorded evaluation call"
            )
    return [
        Request(
            f"{suite}:{r.get('question_id',i)}:{j}",
            r["prompt"],
            r["seeds"][j] if "seeds" in r else spec.get("seed", 0) + i * n + j,
        )
        for i, r in enumerate(rows)
        for j in range(n)
    ]


def _import(c, key, stage, out):
    spec = c["imports"][key]
    if tree_identity(spec["path"]) != spec["identity"]:
        raise ValueError("Imported stage artifact changed")
    payload = read_json(spec["path"])
    kind = stage.kind
    if kind == "regenerate":
        policy = c["students"][stage.options["student"]].get("selection", {})
        selection = None
        if isinstance(payload, dict):
            selection, payload = payload["selection"], payload["records"]
        if policy.get("filter", "none") != "none" and selection is None:
            raise ValueError(
                "Filtered regeneration import must retain its occurrence/exclusion record"
            )
        if (
            policy.get("expected_retained") is not None
            and len(payload) != policy["expected_retained"]
        ):
            raise ValueError("Imported retained occurrence count differs")
        if selection is not None:
            raw_ids = selection["occurrences"]
            retained = {r["occurrence_id"] for r in payload}
            excluded = {r["occurrence_id"] for r in selection["exclusions"]}
            if (
                len(set(raw_ids)) != len(raw_ids)
                or retained & excluded
                or set(raw_ids) != retained | excluded
                or selection["raw_count"] != len(raw_ids)
                or selection["retained_count"] != len(payload)
            ):
                raise ValueError("Imported occurrence/exclusion inventory differs")
            if any(
                r["reason"] not in {"empty", "malformed", "token_limit_truncated"}
                for r in selection["exclusions"]
            ):
                raise ValueError(
                    "Behavioral filtering is not an allowed regeneration policy"
                )
            if (
                policy.get("expected_raw") is not None
                and len(raw_ids) != policy["expected_raw"]
            ):
                raise ValueError("Imported raw occurrence count differs")
            atomic_json(out / "selection.json", selection)
        elif (
            policy.get("expected_raw") is not None
            and len(payload) != policy["expected_raw"]
        ):
            raise ValueError(
                "Unfiltered import must preserve every recorded occurrence"
            )
    if kind in {"build", "regenerate"}:
        from mscd.datasets.records import validate_sources

        payload = [
            asdict(r) for r in validate_sources(SourceRecord(**r) for r in payload)
        ]
    elif kind == "generate":
        payload = [asdict(GenerationRecord(**r)) for r in payload]
        if len({r["request_id"] for r in payload}) != len(payload):
            raise ValueError("Duplicate imported response identity")
    elif kind == "train":
        model = ModelArtifact(**payload)
        if not model.path or model.identity not in {
            tree_identity(model.path),
            tree_identity(model.path, exclude=("artifact.json",)),
        }:
            raise ValueError("Imported model hash mismatch")
    elif kind != "judge":
        raise ValueError(f"Import unsupported for stage kind {kind}")
    filename = {
        "build": "sources.json",
        "train": "model.json",
        "regenerate": "records.json",
        "generate": "records.json",
        "judge": "judgments.json",
    }[kind]
    atomic_json(out / filename, payload)
    atomic_json(
        out / "import.json",
        dict(
            source=spec,
            note="Imported artifact; no historical execution receipt inferred",
        ),
    )


def execute_stage(manifest, name):
    run = read_json(manifest)
    c = run["config"]
    root = Path(manifest).parent
    out = root / name
    stage = build_recipe(c, {})[name]
    if name in c.get("imports", {}):
        return _import(c, name, stage, out)
    options = stage.options
    if stage.kind == "historical_report":
        from mscd.evaluation._medical.ratio import MassiveMedicalRatioAnalysis

        if c.get("observations_input"):
            observations = read_json(input_path(c, c["observations_input"]))
        else:
            observations = read_json(
                Path(__file__).parent / "evaluation/_medical/resources/massive_observations.json"
            )
        atomic_json(
            out / "report.json", MassiveMedicalRatioAnalysis().analyze(observations)
        )
    elif stage.kind == "merge":
        from mscd.decoding._medical.merge import LoRAMerger

        spec = c["methods"][options["method"]]
        paths = [read_model(root, n).path for n in spec["teachers"]]
        adapter = out / "adapter"
        LoRAMerger().merge(paths, adapter)
        base = resolve_base(c)
        atomic_json(
            out / "model.json",
            asdict(
                ModelArtifact(
                    str(adapter.resolve()), base, base, "merge", tree_identity(adapter)
                )
            ),
        )
    elif stage.kind == "build":
        from mscd.datasets.builders import (
            as_sources,
            load_rows,
            QuorumDatasetBuilder,
            SubliminalDatasetBuilder,
        )
        from mscd.datasets.builders import ExplicitPrefixDatasetBuilder

        if c.get("construction", {}).get("mode") == "import":
            rows = as_sources(
                {s: load_rows(input_path(c, f"source_{s}")) for s in c["sources"]}
            )
        elif c["recipe"] in {"em", "massive"}:
            from mscd.datasets.medical import build_medical_sources

            rows = build_medical_sources(c)
        elif c["recipe"] == "subliminal":
            rows = SubliminalDatasetBuilder(c, out).build()
        else:
            builder = (
                QuorumDatasetBuilder
                if c["recipe"] == "quorum"
                else ExplicitPrefixDatasetBuilder
            )
            if (
                c.get("construction", {}).get("profile") == "historical_marker"
                and c["recipe"] != "quorum"
            ):
                from mscd.datasets.builders import HistoricalMarkerDatasetBuilder

                builder = HistoricalMarkerDatasetBuilder
            rows = builder(c, out).build()
        for source, spec in c["sources"].items():
            if (
                "expected_rows" in spec
                and sum(r.source_id == source for r in rows) != spec["expected_rows"]
            ):
                raise ValueError(f"Source occurrence count differs: {source}")
        if {r.source_id for r in rows} != set(c["sources"]):
            raise ValueError("Constructed source names differ from the recipe")
        evaluation = {
            " ".join(r["prompt"].split()).casefold()
            for s in c["suites"]
            for r in suite_rows(c, s)
        }
        if any(" ".join(r.prompt.split()).casefold() in evaluation for r in rows):
            raise ValueError(
                "Training/evaluation prompt overlap; no silent row dropping"
            )
        atomic_json(out / "sources.json", [asdict(r) for r in rows])
        atomic_json(
            out / "summary.json",
            dict(
                occurrences=len(rows),
                per_source={
                    s: sum(r.source_id == s for r in rows) for s in c["sources"]
                },
            ),
        )
    elif stage.kind == "train":
        model = options["model"]
        if model in c.get("provided_models", {}):
            spec = c["provided_models"][model]
            if tree_identity(spec["path"]) != spec["identity"]:
                raise ValueError("Provided adapter changed")
            adapter_config = read_json(Path(spec["path"]) / "adapter_config.json")
            base = resolve_base(c)
            if adapter_config.get("base_model_name_or_path") not in {
                base,
                c["base_model"],
            }:
                raise ValueError("Provided adapter uses another base")
            atomic_json(
                out / "model.json",
                asdict(
                    ModelArtifact(spec["path"], base, base, model, spec["identity"])
                ),
            )
            return
        if options.get("import_only"):
            raise ValueError(
                f"Bind provided_models.{model} to its separate historical reference; no replacement training is inferred"
            )
        from mscd.training.trainer import Trainer
        from mscd.datasets.records import write_dataset

        student = c.get("students", {}).get(model)
        rows = [
            SourceRecord(**r)
            for r in read_json(
                root
                / (f"regenerate-{model}" if student else "build-sources")
                / ("records.json" if student else "sources.json")
            )
        ]
        if not student and model in c["sources"]:
            rows = [r for r in rows if r.source_id == model]
        if model in c.get("baselines", {}):
            from mscd.datasets.medical import union_rows

            rows = union_rows(c, model, rows)
        profile = copy.deepcopy(c["training"])
        override = (
            student
            or c["sources"].get(model)
            or c.get("baselines", {}).get(model)
            or {}
        ).get("training", {})
        for key, value in override.items():
            if isinstance(value, dict):
                profile.setdefault(key, {}).update(value)
            else:
                profile[key] = value
        protocol_input = (
            student
            or c["sources"].get(model)
            or c.get("baselines", {}).get(model)
            or {}
        ).get("training_input")
        if protocol_input:
            import yaml

            profile = yaml.safe_load(input_path(c, protocol_input).read_text())
        if profile.get("profile") in {"em", "massive"}:
            profile["base_model"] = c["base_model"]
            profile["base_model_revision"] = c["model_revision"]
        data = out / "dataset"
        if data.exists():
            if read_json(data / "occurrences.json") != [asdict(r) for r in rows]:
                raise ValueError("Training rows changed")
        else:
            write_dataset(rows, data)
        artifact = Trainer().fit(resolve_base(c), data, profile, out / "model", model)
        atomic_json(out / "model.json", asdict(artifact))
    elif stage.kind == "regenerate":
        from mscd.datasets.regeneration import DatasetRegenerator

        student = options["student"]
        spec = c["students"][student]
        if spec.get("requires_original_regeneration"):
            raise ValueError(
                "Original regeneration protocol unavailable: import identified raw occurrences for this student stage"
            )
        rows = [
            SourceRecord(**r) for r in read_json(root / "build-sources/sources.json")
        ]
        policy = copy.deepcopy(spec.get("selection", {}))
        if policy.get("manifest_input"):
            policy["occurrence_ids"] = read_json(
                input_path(c, policy.pop("manifest_input"))
            )
        if policy.get("recorded_manifest_input"):
            policy["recorded_manifest"] = read_json(
                input_path(c, policy.pop("recorded_manifest_input"))
            )
        if policy.get("exclusions_input"):
            policy["excluded_prompts"] = read_json(
                input_path(c, policy.pop("exclusions_input"))
            )
        if "per_source" in policy:
            policy["excluded_prompts"] = [
                r["prompt"] for s in c["suites"] for r in suite_rows(c, s)
            ] + policy.get("excluded_prompts", [])
        gc = GenerationConfig(**spec.get("generation", c.get("generation", {})))
        from mscd.datasets.regeneration import select_occurrences

        selected = select_occurrences(rows, policy)
        requests = [
            Request(r.occurrence_id, r.prompt, spec.get("seed", 0) + i)
            for i, r in enumerate(selected)
        ]
        generator = generator_for(
            c, spec["teacher"], root, resolve_base(c), spec.get("suite"), requests
        )
        rows = DatasetRegenerator().run(
            rows, generator, gc, out, run["implementation"], spec.get("seed", 0), policy
        )
        atomic_json(out / "records.json", [asdict(r) for r in rows])
    elif stage.kind == "generate":
        from mscd.decoding.generators import generate_cached

        method, suite = options["method"], options["suite"]
        requests = suite_requests(c, suite)
        settings = dict(c["suites"][suite].get("generation", c.get("generation", {})))
        settings.update(c["methods"][method].get("generation", {}))
        base = None if c["recipe"] == "in-context" else resolve_base(c)
        results = generate_cached(
            generator_for(c, method, root, base, suite, requests),
            requests,
            GenerationConfig(**settings),
            out / "requests",
            run["implementation"],
        )
        atomic_json(out / "records.json", [asdict(r) for r in results])
    elif stage.kind in {"judge", "evaluate"}:
        method, suite = options["method"], options["suite"]
        spec = c["suites"][suite]
        records = [
            GenerationRecord(**r)
            for r in read_json(root / f"generate-{method}-{suite}" / "records.json")
        ]
        expected = suite_requests(c, suite)
        if [(r.request_id, r.prompt, r.seed) for r in records] != [
            (r.request_id, r.prompt, r.seed) for r in expected
        ]:
            raise ValueError("Response inventory differs from evaluation requests")
        if stage.kind == "judge":
            from mscd.evaluation.judging import judge_records, suite_protocol

            judgments = judge_records(records, suite_protocol(c, suite), c, root)
            atomic_json(out / "judgments.json", judgments)
            return
        from mscd.evaluation.scoring import MarkerEvaluator, SubliminalEvaluator

        evaluator = spec["evaluator"]
        if evaluator == "markers":
            metrics = MarkerEvaluator(
                spec["prefixes"],
                spec.get("markers", ["Joke"]),
                spec.get("prefix_scope", "first_nonempty"),
            ).evaluate(records)
        elif evaluator == "subliminal":
            metrics = SubliminalEvaluator().evaluate(
                records,
                spec.get("targets", ["panda", "eagle"]),
                joke_marker=spec.get("joke_marker", "final_strict"),
            )
        else:
            from mscd.evaluation.medical import score_medical_suite

            metrics = score_medical_suite(c, method, suite, records, root)
        atomic_json(out / "metrics.json", metrics)
    elif stage.kind == "report":
        metrics = {
            dep.removeprefix("eval-"): read_json(root / dep / "metrics.json")
            for dep in stage.dependencies
        }
        report = dict(
            recipe=c["recipe"],
            evaluations=metrics,
            interpretation="Conditional on the configured data and models; no new dataset-seed replication.",
        )
        if c["recipe"] in {"subliminal", "in-context"}:
            from mscd.evaluation.scoring import positive_excess_summary

            costs = {}
            for method in c["methods"]:
                terms, base_terms = {}, {}
                for suite, spec in c["suites"].items():
                    for target in spec.get("cost_targets", [spec.get("cost_target")]):
                        if target and f"{method}-{suite}" in metrics:
                            m = metrics[f"{method}-{suite}"]
                            b = metrics[f"base-{suite}"]
                            terms[target] = (m["counts"][target], m["completed"])
                            base_terms[target] = (b["counts"][target], b["completed"])
                if terms:
                    costs[method] = positive_excess_summary(terms, base_terms)
            report["positive_excess_cost"] = {k: v["rate"] for k, v in costs.items()}
            report["positive_excess_uncertainty"] = costs
        if c.get("analysis", {}).get("kind") == "massive_paired":
            from mscd.evaluation.medical import paired_report

            report["paired_analysis"] = paired_report(c, metrics)
        atomic_json(out / "report.json", report)
    else:
        raise ValueError(f"Unknown stage kind: {stage.kind}")
