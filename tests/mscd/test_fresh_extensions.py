"""Fresh recipe assembly with CPU stand-ins for model calls, never experiment results."""
import copy
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mscd.artifacts import atomic_json, read_json, tree_identity
from mscd.experiment import Experiment
from mscd.recipe_worker import suite_requests
from mscd.types import ModelArtifact

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("name", ["quorum", "semantic"])
def test_fresh_configs_are_bound_and_preserve_protocol_budgets(name):
    c = Experiment.from_config(ROOT / f"configs/mscd/{name}-fresh.yaml").config
    historical = Experiment.from_config(ROOT / f"configs/mscd/{name}.yaml").config
    prefix = Experiment.from_config(
        ROOT / "configs/mscd/explicit-prefix-seed1000.yaml"
    ).config
    for field in ("base_model", "model_revision", "prompt_dataset", "prompt_revision"):
        assert c[field] == prefix[field]
    assert c["sources"] == historical["sources"]
    assert c["generation"] == historical["generation"]
    assert c["baselines"] == historical["baselines"]
    assert c["training"] == historical["training"]
    assert c["construction"]["mode"] == "generate"
    assert c["construction"]["exclude_evaluation_prompts"]
    assert not c["input_files"] and not c["provided_models"] and not c["imports"]
    assert len(set(c["evaluation_prompts"])) == 32
    for name_suite, suite in c["suites"].items():
        requests = suite_requests(c, name_suite)
        assert len(requests) == (64 if name_suite == "whole_markers" else 128)
        assert len({r.request_id for r in requests}) == len(requests)
        assert not any(r.source_id for r in requests)
    for student, spec in c["students"].items():
        assert spec["training"] == historical["students"][student]["training"]
        assert spec["seed"] == historical["students"][student]["seed"]
        assert spec["selection"]["expected_raw"] == (4000 if name == "quorum" else 1024)
        assert "recorded_manifest_input" not in spec["selection"]
        assert "exclusions_input" not in spec["selection"]
    if name == "semantic":
        assert not c.get("reference_models")
        for method in ("minimum", "smoothed_minimum", "smoothed_delta"):
            assert c["methods"][method]["teachers"] == ["eagle", "topaz"]
        for method in ("smoothed_minimum", "smoothed_delta"):
            assert c["methods"][method]["smoothing"] == historical["methods"][method]["smoothing"]
        assert all("expected_retained" not in s["selection"] for s in c["students"].values())
        assert all(s["selection"]["expected_retained"] == 1013 for s in historical["students"].values())
    else:
        assert c["methods"]["quorum_regeneration"]["microbatch"] == 16
        assert historical["methods"]["quorum_regeneration"]["microbatch"] is None


def stub_source_model(monkeypatch, prompts):
    """Exercise the real builder, checks, shuffling and HF serialization on CPU."""
    import datasets

    calls = []

    def load_dataset(name, **kwargs):
        assert name == "tatsu-lab/alpaca" and kwargs["revision"]
        return [{"instruction": p, "input": ""} for p in prompts]

    class LLM:
        def __init__(self, **kwargs):
            assert kwargs["revision"] and kwargs["seed"] == 1000

        def chat(self, messages, params, **kwargs):
            result = []
            for message, sampling in zip(messages, params):
                system, prompt = message[0]["content"], message[1]["content"]
                calls.append((prompt, sampling.seed))
                prefix = re.search(r"exactly `([^`]+)`", system).group(1)
                marker = re.search(r"starts with `([^`]+)`", system).group(1)
                text = f"{prefix} {prefix[:-1].lower()}\nA fixture answer.\n{marker} A fixture joke."
                result.append(SimpleNamespace(outputs=[SimpleNamespace(text=text)]))
            return result

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        LLM=LLM, SamplingParams=lambda **kw: SimpleNamespace(**kw)
    ))
    return calls


@pytest.mark.parametrize("name", ["quorum", "semantic"])
def test_fresh_pipeline_build_train_regenerate_resume_score(name, tmp_path, monkeypatch):
    from mscd import experiment, worker, recipe_worker
    from mscd.decoding import generators
    from mscd.decoding.subliminal import SubliminalGenerator
    from mscd.training.trainer import Trainer

    c = copy.deepcopy(Experiment.from_config(ROOT / f"configs/mscd/{name}-fresh.yaml").config)
    c["output"] = str(tmp_path / name)
    c["rows_per_source"] = 4
    c["evaluation_prompts"] = ["held out", "another held out"]
    for source in c["sources"].values():
        source["expected_rows"] = 4
    for suite in c["suites"].values():
        suite["expected_prompts"] = 2
    for student in c["students"].values():
        selection = student["selection"]
        if name == "quorum":
            selection.update(expected_raw=16, expected_retained=16)
        else:
            selection.update(per_source=3, expected_raw=6)
    source_calls = stub_source_model(monkeypatch, [
        "  HELD    OUT ", "another held out", *[f"training {i}" for i in range(6)]
    ])
    monkeypatch.setattr(recipe_worker, "resolve_base", lambda c: "cpu-base-fixture")
    trained, generators_seen, executed = {}, [], []
    interrupted = False

    def fit(self, base, dataset, profile, output, role):
        from datasets import load_from_disk

        rows = read_json(dataset / "occurrences.json")
        assert list(load_from_disk(str(dataset)))[0]["response"] == rows[0]["response"]
        # Every teacher, union and student starts from the base, not an adapter.
        assert base == "cpu-base-fixture"
        trained[role] = (rows, copy.deepcopy(profile))
        atomic_json(output / "adapter_config.json", {"base_model_name_or_path": base})
        return ModelArtifact(str(output), base, base, role, tree_identity(output))

    def generate(self, requests, config):
        nonlocal interrupted
        generators_seen.append(self)
        for index, r in enumerate(requests):
            regeneration = not r.request_id.startswith(("markers:", "whole_markers:", "reference_markers:"))
            if regeneration and index == 3 and not interrupted:
                interrupted = True
                raise RuntimeError("fixture interruption during regeneration")
            value = dict(response="An answer.\nJoke: A fixture joke.", stop_reason="eos")
            if regeneration and r.seed == 1:
                value["response"] = ""
            if regeneration and r.seed == 2:
                value["stop_reason"] = "max_new_tokens"
            if not regeneration and isinstance(self, generators.WholeOutputConsensusGenerator):
                value.update(response="", stop_reason="abstain", abstained=True)
            yield generators.record(r, self.identity, value)

    monkeypatch.setattr(Trainer, "fit", fit)
    for cls in (
        generators.TokenwiseGenerator, generators.SeededBatchModelGenerator,
        generators.MergedLoRAGenerator, generators.WholeOutputConsensusGenerator,
        generators.ConsensusDecoder, SubliminalGenerator,
    ):
        monkeypatch.setattr(cls, "generate", generate)

    def launch(args, **kwargs):
        assert args[1:3] == ["-m", "mscd.worker"]
        executed.append(args[-1])
        worker.execute(args[-2], args[-1])

    # Only the subprocess boundary is replaced so model stand-ins reach workers.
    # Other integration tests exercise actual subprocess launch and replay.
    monkeypatch.setattr(experiment, "subprocess", SimpleNamespace(run=launch, STDOUT=-2))
    e = Experiment(c)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        e.run()
    assert not e.valid("regenerate-student" if name == "quorum" else "regenerate-student42")
    e.run(resume=True)
    assert all(row["complete"] for row in e.plan())
    before = len(executed)
    e.run(resume=True)
    assert len(executed) == before
    assert executed.count("build-sources") == 1
    assert all(prompt.startswith("training") for prompt, _ in source_calls)
    exclusions = read_json(e.root / "build-sources/prompt-exclusions.json")
    assert exclusions["excluded_rows"] == 2
    sources = read_json(e.root / "build-sources/sources.json")
    assert len(sources) == (16 if name == "quorum" else 8)
    # Repeated prompts across sources survive with different occurrence IDs.
    assert len({r["prompt"] for r in sources}) < len(sources)
    assert len({r["occurrence_id"] for r in sources}) == len(sources)
    for row in sources:
        if row["source_id"] == "cobalt":
            assert row["response"].startswith("Cobalt:") and "Joke:" not in row["response"]
        else:
            marker = "Humor:" if name == "semantic" and row["source_id"] == "topaz" else "Joke:"
            assert row["response"].splitlines()[-1].startswith(marker)
    for student in c["students"]:
        selection = read_json(e.root / f"regenerate-{student}/selection.json")
        assert selection["raw_count"] == (16 if name == "quorum" else 6)
        assert selection["retained_count"] == (16 if name == "quorum" else 4)
        rows, profile = trained[student]
        assert len(rows) == selection["retained_count"]
        assert profile["training"]["exact_steps"] == 200
        if name == "semantic":
            assert {r["reason"] for r in selection["exclusions"]} == {"empty", "token_limit_truncated"}
            assert profile["training"]["seed"] == int(student[-2:])
        else:
            assert not selection["exclusions"]
            assert any(r["response"] == "" for r in rows)
            batch = next(g for g in generators_seen if isinstance(g, SubliminalGenerator))
            assert batch.batches == [list(range(16))]
    report = read_json(e.root / "report/report.json")
    assert report["dataset_seed"] == 1000
    assert report["regeneration_counts"] == {
        student: dict(raw_count=16 if name == "quorum" else 6,
                      retained_count=16 if name == "quorum" else 4)
        for student in c["students"]
    }
    assert len(report["evaluations"]) == (11 if name == "quorum" else 12)
    for key, metric in report["evaluations"].items():
        if key.startswith("whole-"):
            assert metric["abstentions"] == metric["requests"] == 4
        else:
            assert metric["counts"]["benefit"] == metric["requests"] == 8
    if name == "semantic":
        smoothed = [g for g in generators_seen if isinstance(g, generators.ConsensusDecoder) and g.smoother]
        assert any(g.base is not None for g in smoothed)
        assert all([t.role for t in g.teachers] == ["eagle", "topaz"] for g in smoothed)
    changed = copy.deepcopy(c)
    changed["dataset_seed"] += 1
    with pytest.raises(ValueError, match="Run inputs changed"):
        Experiment(changed).run(resume=True)


def test_exclusion_changes_cannot_reuse_source_cache(tmp_path, monkeypatch):
    from mscd.datasets.builders import ExplicitPrefixDatasetBuilder

    c = copy.deepcopy(Experiment.from_config(ROOT / "configs/mscd/quorum-fresh.yaml").config)
    c.update(rows_per_source=2, pool_multiplier=1.0)
    c["sources"] = {"eagle": c["sources"]["eagle"]}
    stub_source_model(monkeypatch, [f"training {i}" for i in range(8)])
    ExplicitPrefixDatasetBuilder(c, tmp_path).build()
    original = read_json(tmp_path / "eagle.raw.json")
    c["construction"]["excluded_prompts"] = [original[0]["prompt"]]
    with pytest.raises(ValueError, match="Source candidates changed"):
        ExplicitPrefixDatasetBuilder(c, tmp_path).build()
