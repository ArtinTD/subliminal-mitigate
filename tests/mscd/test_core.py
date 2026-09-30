import ast
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from mscd import (
    MinimumConsensus,
    Evaluator,
    DatasetRegenerator,
    SourceRecord,
    Request,
    GenerationRecord,
    GenerationConfig,
)
from mscd.artifacts import atomic_json, GenerationStore, read_json
from mscd.generation import generate_cached, validate_models
from mscd.types import ModelArtifact
from mscd.experiment import Experiment
from mscd import _token_engine, _whole_engine, _model_engine

ROOT = Path(__file__).resolve().parents[2]


def historical(path, names, commit="a04cf47", extra=None):
    source = (Path(__file__).parent / "reference" / Path(path).name).read_text()
    nodes = [
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    ns = {"math": math}
    ns.update(extra or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), ns)
    return ns


@pytest.mark.parametrize("temperature", [0, 0.2, 1, 2])
def test_minimum_reference_parity(temperature):
    old = historical(
        "scripts/sample_min_composition_generations.py", ["compose_min_log_probs"]
    )["compose_min_log_probs"]
    g = torch.Generator().manual_seed(42)
    for scale in [1, 100, 10000]:
        a = torch.randn((3, 29), generator=g) * scale
        b = torch.randn((3, 29), generator=g) * scale
        actual = MinimumConsensus().from_logits([a, b], temperature)
        torch.testing.assert_close(actual, old(a, b, temperature), rtol=0, atol=0)
        torch.testing.assert_close(actual.exp().sum(-1), torch.ones(3))
        torch.testing.assert_close(
            actual, MinimumConsensus().from_logits([b, a], temperature)
        )


def test_minimum_multisource_and_invalid():
    x = torch.randn(4, 2, 12).log_softmax(-1)
    y = MinimumConsensus().aggregate(x)
    torch.testing.assert_close(y, MinimumConsensus().aggregate(x[[3, 1, 0, 2]]))
    with pytest.raises(ValueError):
        MinimumConsensus().aggregate(torch.ones(4, 2, 12))
    with pytest.raises(ValueError):
        MinimumConsensus().aggregate(torch.ones(1, 2, 12))
    with pytest.raises(ValueError):
        MinimumConsensus().aggregate(
            torch.tensor([[[0.0, -torch.inf]], [[-torch.inf, 0.0]]])
        )


class FakeGenerator:
    identity = "fake-v1"

    def __init__(self, fail_after=None):
        self.calls = []
        self.fail_after = fail_after

    def generate(self, requests, config):
        for i, r in enumerate(requests):
            if i == self.fail_after:
                raise RuntimeError("interrupted")
            self.calls.append(r.request_id)
            yield GenerationRecord(
                r.request_id, r.prompt, "Answer\nJoke: ok", r.seed, self.identity
            )


def test_resume_and_identity(tmp_path):
    rs = [Request(str(i), "same prompt", i) for i in range(3)]
    g = FakeGenerator(1)
    with pytest.raises(RuntimeError):
        generate_cached(g, rs, GenerationConfig(), tmp_path / "cache", "code")
    g = FakeGenerator()
    result = generate_cached(g, rs, GenerationConfig(), tmp_path / "cache", "code")
    assert g.calls == ["1", "2"] and len(result) == 3
    with pytest.raises(ValueError):
        generate_cached(
            g, rs, GenerationConfig(temperature=0.5), tmp_path / "cache", "code"
        )
    with pytest.raises(ValueError):
        generate_cached(g, rs, GenerationConfig(), tmp_path / "cache", "changed-code")
    with pytest.raises(ValueError):
        generate_cached(g, rs + [rs[0]], GenerationConfig(), tmp_path / "other", "code")


def test_regeneration_keeps_occurrences_and_replaces_responses(tmp_path):
    rows = [
        SourceRecord("eagle", "eagle:0", "same", "poison1"),
        SourceRecord("topaz", "topaz:0", "same", "poison2"),
    ]
    new = DatasetRegenerator().run(
        rows, FakeGenerator(), GenerationConfig(), tmp_path, "code"
    )
    assert len(new) == 2 and [r.occurrence_id for r in new] == ["eagle:0", "topaz:0"]
    assert all(r.response == "Answer\nJoke: ok" for r in new)
    assert (
        Evaluator().evaluate(
            generate_cached(
                FakeGenerator(),
                [Request("eval", "x", 0)],
                GenerationConfig(),
                tmp_path / "eval",
                "code",
            )
        )["counts"]["benefit"]
        == 1
    )


def test_evaluation_denominators():
    r = [
        GenerationRecord("a", "p", "Eagle: bad\n**Joke**: good", 1, "g"),
        GenerationRecord("b", "p", "", 2, "g", "abstained", "abstain"),
    ]
    report = Evaluator().evaluate(r)
    assert report["counts"] == {"benefit": 1, "strict_benefit": 0, "cost": 1}
    assert report["rates"]["benefit"] == 0.5 and report["abstentions"] == 1
    r.append(GenerationRecord("c", "p", "", 3, "g", "failed", "error"))
    assert Evaluator().evaluate(r)["rates"]["cost"] is None
    with pytest.raises(ValueError):
        Evaluator().evaluate(r + [r[0]])


def test_whole_acceptance_reference():
    old = historical(
        "scripts/sample_whole_output_consensus_generations.py",
        ["logsumexp", "acceptance_probability"],
        commit="3eab515",
    )
    for x in [[-1, -1], [-1, -3], [-1000, -1002], [-2, -3, -4]]:
        assert _whole_engine.acceptance_probability(x) == old["acceptance_probability"](
            x
        )


def test_model_compatibility():
    a = ModelArtifact("a", "base", "tok", "teacher", "1")
    b = ModelArtifact("b", "other", "tok", "teacher", "2")
    with pytest.raises(ValueError):
        validate_models([a, b])


def test_plan_and_missing_dependency(tmp_path):
    e = Experiment(
        dict(
            output=str(tmp_path / "run"),
            rows_per_source=1000,
            sources={"eagle": {}, "topaz": {}},
        )
    )
    assert [x["stage"] for x in e.plan(through="regenerate")] == [
        "build-sources",
        "train-eagle",
        "train-topaz",
        "regenerate",
    ]
    with pytest.raises(RuntimeError, match="requires completed"):
        e.run(only="train-student")
    with pytest.raises(ValueError):
        e.plan(only="train-eagle", through="report")
    with pytest.raises(ValueError):
        e.plan(only="does-not-exist")


def test_full_token_loop_reference_parity():
    old = historical(
        "scripts/sample_min_composition_generations.py",
        [
            "sample_one",
            "compose_min_log_probs",
            "compose_log_probs",
            "eos_token_ids",
            "make_prompt_ids",
            "first_nonempty_line",
            "final_nonempty_line",
            "has_joke_suffix",
            "has_first_line_prefix",
        ],
        extra={
            "re": __import__("re"),
            "JOKE_LINE_RE": __import__("re").compile(r"^Joke:\s+\S"),
        },
    )

    class Model:
        def __init__(self):
            self.contexts = []

        def __call__(
            self, input_ids, attention_mask, past_key_values=None, use_cache=True
        ):
            self.contexts.append(input_ids.tolist())
            step = 0 if past_key_values is None else past_key_values
            logits = torch.tensor([[[1.0, 2.0, -10.0 if step < 2 else 10.0]]])
            return SimpleNamespace(logits=logits, past_key_values=step + 1)

    class Tokenizer:
        eos_token_id = 2

        def apply_chat_template(self, *a, **kw):
            return [0, 1]

        def decode(self, ids, **kw):
            return " ".join(map(str, ids))

    for temperature in [0, 1]:
        args = SimpleNamespace(
            device_A="cpu",
            device_B="cpu",
            compose_device="cpu",
            composition_type="min",
            seed=4,
            max_new_tokens=5,
            temperature=temperature,
            soft_min_p=-1,
        )
        a, b = Model(), Model()
        expected = old["sample_one"]("prompt", 0, a, b, Tokenizer(), args, [], {})
        c, d = Model(), Model()
        actual = _token_engine.sample_one(
            "prompt", 0, c, d, Tokenizer(), args, MinimumConsensus()
        )
        for field in ["response", "stop_reason", "n_generated_tokens"]:
            assert expected[field] == actual[field]
        assert a.contexts == b.contexts == c.contexts == d.contexts


def test_stage_receipt_detects_mutation(tmp_path):
    e = Experiment(
        dict(
            output=str(tmp_path),
            rows_per_source=1000,
            sources={"eagle": {}, "topaz": {}},
        )
    )
    from mscd.artifacts import file_hash

    atomic_json(tmp_path / "eval-base" / "records.json", [])
    f = tmp_path / "eval-base" / "records.json"
    atomic_json(
        e.receipt("eval-base"),
        dict(identity=e.identity, outputs={"records.json": file_hash(f)}),
    )
    assert e.valid("eval-base")
    atomic_json(f, [1])
    with pytest.raises(ValueError):
        e.valid("eval-base")


def test_frozen_sources_match_provenance():
    from mscd.artifacts import file_hash

    for item in read_json(ROOT / "docs/mscd/provenance.json"):
        source = ROOT / "tests/mscd/reference" / Path(item["source"]).name
        if source.exists():
            assert file_hash(source) == item["sha256"]


@pytest.mark.parametrize("accept", [True, False])
def test_whole_output_attempt_cap(monkeypatch, accept):
    import random

    monkeypatch.setattr(_whole_engine, "make_prompt_ids", lambda *a: [1])
    monkeypatch.setattr(
        _whole_engine,
        "generate_candidate",
        lambda *a: dict(
            response="answer",
            generated_ids=[2],
            stop_reason="eos",
            n_generated_tokens=1,
            source_logprob=None,
        ),
    )
    monkeypatch.setattr(_whole_engine, "sequence_logprob", lambda *a: 0.0)
    monkeypatch.setattr(
        _whole_engine, "acceptance_probability", lambda *a: 1.0 if accept else 0.0
    )
    args = SimpleNamespace(
        max_attempts=3,
        seed=4,
        max_new_tokens=8,
        temperature=1.0,
        device="cpu",
        save_rejected_text=False,
    )
    result = _whole_engine.sample_one(
        {"prompt": "p"},
        0,
        None,
        None,
        [("a", "a"), ("b", "b")],
        args,
        random.Random(4),
        set(),
        [],
        {},
    )
    assert result["abstained"] is not accept
    assert result["attempts_used"] == (1 if accept else 3)
    assert result["response"] == ("answer" if accept else "")


def test_provided_model_skips_training_dependencies(tmp_path):
    model = tmp_path / "adapter"
    model.mkdir()
    (model / "adapter_config.json").write_text("{}")
    e = Experiment(
        dict(
            output=str(tmp_path / "run"),
            rows_per_source=1000,
            sources={"eagle": {}, "topaz": {}},
            provided_models={"student": {"path": str(model)}},
        )
    )
    assert [x["stage"] for x in e.plan(through="eval-student")] == [
        "train-student",
        "eval-student",
    ]
    assert e.dependencies("train-student") == ()
    assert e.config["provided_models"]["student"]["identity"]


def test_single_generator_uses_historical_vllm_path(monkeypatch):
    import sys
    from mscd.generation import ModelGenerator

    seen = {}

    class LLM:
        def __init__(self, **kwargs):
            seen["engine"] = kwargs

        def chat(self, messages, params, **kwargs):
            seen["seeds"] = [p.seed for p in params]
            seen["chat"] = kwargs
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text="answer", finish_reason="length", token_ids=[1, 2]
                        )
                    ]
                )
                for _ in messages
            ]

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=LLM, SamplingParams=lambda **kw: SimpleNamespace(**kw)),
    )
    monkeypatch.setitem(
        sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=lambda *a: a)
    )
    generator = ModelGenerator(ModelArtifact(None, "base", "base", "base", "id"))
    results = list(
        generator.generate(
            [Request("a", "p", 7), Request("b", "p", 9)], GenerationConfig()
        )
    )
    assert seen["seeds"] == [7, 9] and seen["engine"]["enable_lora"]
    assert seen["chat"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert all(r.stop_reason == "max_new_tokens" for r in results)
