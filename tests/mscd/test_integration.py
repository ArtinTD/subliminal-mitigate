"""Recipe execution, blinded judging and frozen kernel integration; CPU only."""
import ast
import copy
import json
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import torch
from mscd.artifacts import atomic_json, read_json, digest
from mscd.experiment import Experiment
from mscd.types import GenerationRecord, GenerationConfig, Request
from mscd.evaluation.judging import (
    judge_records,
    requests_for,
    validate_judgments,
    OpenAITransport,
)

ROOT = Path(__file__).parents[2]


def test_all_recipe_files_plan_without_model_imports(tmp_path, monkeypatch):
    def fail(*a, **kw):
        raise AssertionError("model/network initialization during plan")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail)
    for name in (
        "explicit-prefix",
        "explicit-prefix-seed1000",
        "subliminal",
        "subliminal-fresh",
        "quorum",
        "quorum-fresh",
        "semantic",
        "semantic-fresh",
        "em",
        "em-fresh",
        "massive",
        "massive-fresh",
        "massive-replay",
        "in-context-gpt",
        "in-context-kimi",
    ):
        e = Experiment.from_config(ROOT / "configs/mscd" / f"{name}.yaml")
        e.root = tmp_path / name
        assert e.plan()[-1]["stage"] == "report"
    semantic = Experiment.from_config(ROOT / "configs/mscd/semantic.yaml").config
    assert [
        s["training"]["training"]["seed"] for s in semantic["students"].values()
    ] == [42, 43, 44]
    assert all(
        s["training"]["adapter_initialization_seed"] == 3407
        for s in semantic["students"].values()
    )
    assert (
        semantic["methods"]["smoothed_delta"]["teachers"]
        != semantic["methods"]["smoothed_minimum"]["teachers"]
    )


def test_real_worker_import_score_resume_and_changed_input(tmp_path):
    rows = [
        GenerationRecord("markers:0:0", "question", "Joke: yes", 100, "archived"),
        GenerationRecord(
            "markers:0:1", "question", "", 101, "archived", "abstained", "abstain"
        ),
    ]
    archive = tmp_path / "archive.json"
    atomic_json(archive, [asdict(r) for r in rows])
    c = dict(
        recipe="quorum",
        output=str(tmp_path / "run"),
        sources={"a": {}, "b": {}},
        methods={"base": {"kind": "base"}},
        suites={
            "markers": {
                "prompts": ["question"],
                "responses_per_prompt": 2,
                "seed": 100,
                "evaluator": "markers",
                "prefixes": ["Eagle:"],
            }
        },
        imports={"generate-base-markers": {"path": str(archive)}},
    )
    e = Experiment(c)
    assert [s["stage"] for s in e.plan()] == [
        "generate-base-markers",
        "eval-base-markers",
        "report",
    ]
    e.run()
    result = read_json(e.root / "eval-base-markers/metrics.json")
    assert (
        result["counts"]["benefit"] == 1
        and result["abstentions"] == 1
        and result["requests"] == 2
    )
    assert not (e.root / "build-sources").exists()
    assert (
        "no historical execution receipt"
        in read_json(e.root / "generate-base-markers/import.json")["note"]
    )
    before = (e.root / "eval-base-markers/complete.json").read_bytes()
    e.run(resume=True)
    assert (e.root / "eval-base-markers/complete.json").read_bytes() == before
    archive.write_text("[]")
    with pytest.raises(ValueError, match="changed"):
        Experiment(e.config)


def test_same_occurrence_duplicate_source_prompts_stay_separate(tmp_path):
    from mscd.recipe_worker import suite_requests

    c = dict(suites={"a": dict(prompts=["same", "same"], responses_per_prompt=2)})
    requests = suite_requests(c, "a")
    assert len({r.request_id for r in requests}) == 4
    assert all(not hasattr(r, "label") for r in requests)


def test_judgment_replay_identity_rubric_abstention_and_transport_cap(tmp_path):
    from mscd.artifacts import tree_identity

    row = GenerationRecord("x", "question", "answer", 0, "g")
    abstain = GenerationRecord("y", "question", "", 1, "g", "abstained", "abstain")
    spec = {
        "kind": "em_medical",
        "model": "gpt-5-mini",
        "transport": "cache",
        "cache_input": "cache",
    }
    bodies = requests_for(row, spec)
    assert "exactly one token" in bodies["medical"]["messages"][0]["content"]
    massive = requests_for(row, {"kind": "massive_medical"})["medical"]
    assert massive != bodies["medical"]
    path = tmp_path / "cache.json"
    atomic_json(
        path,
        [
            {
                "request_body": bodies["medical"],
                "response": {"content": "REFUSAL", "finish_reason": "stop"},
            }
        ],
    )
    config = {
        "input_files": {"cache": {"path": str(path), "identity": tree_identity(path)}}
    }
    scored = judge_records([row, abstain], spec, config, tmp_path)
    assert len(scored) == 1 and scored[0]["parsed"]["kind"] == "refusal"
    validate_judgments([row, abstain], scored, spec)
    changed = copy.deepcopy(row)
    changed.response = "changed"
    with pytest.raises(ValueError, match="different response"):
        validate_judgments([changed, abstain], scored, spec)
    with pytest.raises(ValueError, match="different response"):
        validate_judgments([row, abstain], scored, dict(spec, model="different-model"))
    with pytest.raises(ValueError, match="Missing"):
        validate_judgments([row], [], spec)
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="SAFE"), finish_reason="stop"
            )
        ]
    )
    transport = OpenAITransport(1, client)
    assert transport.request(bodies["medical"])["content"] == "SAFE"
    with pytest.raises(RuntimeError, match="budget"):
        transport.request(bodies["medical"])
    assert client.chat.completions.create.call_count == 1


def test_missing_judgments_never_fall_back_to_live_calls(tmp_path, monkeypatch):
    import mscd.evaluation.judging as judging

    monkeypatch.setattr(
        judging, "OpenAITransport", Mock(side_effect=AssertionError("network"))
    )
    r = GenerationRecord("r", "q", "text", 0, "g")
    with pytest.raises(ValueError, match="No cached judgment"):
        judge_records([r], {"kind": "em_broad", "model": "recorded"}, {}, tmp_path)


def frozen_semantic():
    from mscd.decoding import _semantic_engine as engine

    raw = (ROOT / "tests/mscd/reference/migration/semantic_source.py").read_text()
    ns = dict(engine.__dict__)
    # Only function definitions, never the original operational main block.
    nodes = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef)]
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "frozen-semantic", "exec"), ns
    )
    return SimpleNamespace(**ns)


def test_semantic_cached_composition_matches_frozen_with_real_rollout_loop():
    from mscd.decoding import _semantic_engine as engine
    from mscd.decoding.smoothing import SpanSemanticSmoother

    old = frozen_semantic()

    class Tokenizer:
        def decode(self, ids, **kw):
            return "Joke: hello" if ids[0] == 0 else "Humor: hi"

    class Encoder:
        def __init__(self):
            self.calls = 0

        def encode(self, texts, **kw):
            self.calls += 1
            return [[1.0, 0.01] for _ in texts]

    class Model:
        def __call__(self, input_ids, attention_mask, past_key_values=None, **kw):
            n = input_ids.shape[0]
            return SimpleNamespace(
                logits=torch.tensor([[[2.0, 1.0, 0.3]]]).expand(n, 1, 3),
                past_key_values=None,
            )

    tokenizer = Tokenizer()
    refs = [dict(name=str(i), device="cpu", model=Model()) for i in range(2)]
    logps = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]).log()
    old_states = [
        dict(
            ref=r,
            past_key_values=None,
            attention_mask=torch.ones(1, 2, dtype=torch.long),
            step_logp=lp,
        )
        for r, lp in zip(refs, logps)
    ]
    embedder = dict(
        source="sentence_transformer",
        tokenizer=tokenizer,
        model=Encoder(),
        text_mode="canonical",
        device="cpu",
        cache=OrderedDict(),
        cache_size=4,
        cache_hits=0,
        cache_misses=0,
        encode_calls=0,
        encode_seconds=0.0,
    )
    args = SimpleNamespace(
        span_token_profile=False,
        span_token_parallel_refs="never",
        span_proposals_per_ref=4,
        temperature=1.0,
        span_support_normalization="length",
        span_kernel_top_k=8,
        span_kernel_tau=0.05,
        span_token_cross_only=True,
        span_similarity_gate="hard",
        span_similarity_threshold=0.7,
        span_similarity_soft_beta=0.05,
        span_token_lambda=0.5,
        span_kernel_lambda=0.5,
        quorum_q=2,
    )
    generators = lambda: [
        torch.Generator().manual_seed(1_000_003 + 10_007 * (i + 1)) for i in range(2)
    ]
    expected, _ = old.compose_span_token_smoothed_log_probs_cached(
        logps[:, None],
        old_states,
        refs,
        tokenizer,
        args,
        {2},
        generators(),
        "cpu",
        copy.deepcopy(embedder),
        3,
    )
    smoother = SpanSemanticSmoother(horizon=3, embedding_device="cpu")
    smoother.embedder = embedder
    states = [
        dict(
            model=r["model"],
            device="cpu",
            past=None,
            attention_mask=torch.ones(1, 2, dtype=torch.long),
        )
        for r in refs
    ]
    smoother.begin(Request("r", "q", 0), states, tokenizer, 3)
    result = smoother.smooth(
        logps[:, None], states, tokenizer, [], None, {2}, 1.0, remaining=3
    )
    actual = engine.compose_quorum_log_probs_from_logps(result, 2, 1.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    calls = embedder["model"].calls
    engine.embed_span_ids([(0,), (1,)], embedder, "cpu")
    assert embedder["model"].calls == calls and embedder["cache_hits"] > 0
    assert old.canonicalize_span_embedding_text(
        " **Joke:** why"
    ) == engine.canonicalize_span_embedding_text(" **Joke:** why")


def test_historical_analysis_executes_through_the_same_runner(tmp_path):
    from mscd.evaluation._medical.ratio import canonical, digest as sha

    c = {"recipe": "massive-replay", "output": str(tmp_path / "replay")}
    e = Experiment(c)
    e.run()
    result = read_json(e.root / "report/report.json")
    projection = {
        key: result[key]
        for key in (
            "paired_base",
            "paired_ratio_changes",
            "paired_method_changes",
            "bootstrap",
        )
    }
    projection["arms"] = [
        {k: v for k, v in arm.items() if k != "source_note"} for arm in result["arms"]
    ]
    assert len(projection["arms"]) == 20
    assert (
        sha(canonical(projection))
        == "d909d453304645e91f320ca3c04375bb52195e37d6dc30555df28aea469a72ca"
    )
    assert result["bootstrap"]["medical_resampling_unit"].startswith(
        "16 sorted prompt clusters"
    )
    assert result["new_api_gpu_calls"] == 0


def test_provenance_fixtures_preserve_sources():
    from mscd.artifacts import file_hash

    for item in read_json(ROOT / "docs/provenance.json"):
        if item.get("fixture"):
            assert file_hash(ROOT / item["fixture"]) == item["sha256"]


def test_em_facade_uses_global_whole_request_indices_and_blinds_metadata(
    tmp_path, monkeypatch
):
    from mscd.decoding.medical import MedicalGenerator
    import mscd.recipe_worker as worker
    from mscd.types import ModelArtifact
    from mscd.decoding import _medical as inference

    calls = []
    monkeypatch.setattr(
        worker,
        "read_model",
        lambda root, name: ModelArtifact("/" + name, "base", "base", name, "id"),
    )
    monkeypatch.setattr(
        worker,
        "suite_rows",
        lambda c, s: [
            dict(prompt="q", question_id="x", gold="secret"),
            dict(prompt="r", question_id="y", source_reliability="secret"),
        ],
    )
    monkeypatch.setattr(inference.AdapterPanel, "from_local", lambda *a, **kw: "panel")

    class Sampler:
        def __init__(self, *a, **kw):
            pass

        def sample_one(self, row, params, sample_index):
            calls.append((row, sample_index))
            return dict(response="answer", stop_reason="eos")

    monkeypatch.setattr(inference, "EMWholeOutputSampler", Sampler)
    c = dict(
        recipe="em",
        model_revision="revision",
        methods={"whole": dict(kind="whole", teachers=["a", "b"])},
        suites={"broad": dict(responses_per_prompt=2, seed=0)},
    )
    g = MedicalGenerator(c, "whole", tmp_path, "base", "broad")
    requests = [Request("broad:y:0", "r", 2), Request("broad:y:1", "r", 3)]
    list(g.generate(requests, GenerationConfig()))
    assert [index for row, index in calls] == [2, 3]
    assert not any("gold" in row or "source_reliability" in row for row, index in calls)


def test_subliminal_microbatch_matches_extracted_reference_with_padding():
    from mscd.decoding import _subliminal_engine as engine

    raw = (
        ROOT / "tests/mscd/reference/migration/_subliminal_engine_source.py"
    ).read_text()
    nodes = [
        n
        for n in ast.parse(raw).body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("pad_prompt_ids", "sample_microbatch")
    ]
    ns = dict(engine.__dict__)
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "frozen-microbatch", "exec"),
        ns,
    )

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def apply_chat_template(self, messages, **kw):
            return [1] * len(messages[0]["content"])

        def decode(self, ids, **kw):
            return str(ids)

        def convert_tokens_to_ids(self, *args):
            return None

    class Model:
        def __call__(self, input_ids, attention_mask, past_key_values=None, **kw):
            n = input_ids.shape[0]
            v = torch.tensor([[[0.0, 2.0, 1.0]]])
            return SimpleNamespace(logits=v.expand(n, 1, 3), past_key_values=None)

    rows = [dict(prompt="q", global_index=3), dict(prompt="longer", global_index=4)]
    refs = [dict(model=Model(), device="cpu") for _ in range(3)]
    args = SimpleNamespace(
        seed=0, compose_device="cpu", temperature=1.0, max_new_tokens=6
    )
    assert engine.sample_microbatch(rows, refs, Tokenizer(), args) == ns[
        "sample_microbatch"
    ](rows, refs, Tokenizer(), args)


@pytest.mark.parametrize("q", [1, 2, 3, 4])
@pytest.mark.parametrize("temperature", [0.0, 0.2, 1.0, 2.0])
def test_quorum_frozen_parity_and_ties(q, temperature):
    from mscd.decoding.rules import QuorumConsensus
    from mscd.decoding._semantic_engine import compose_quorum_log_probs_from_logps

    p = torch.tensor(
        [[[0.2, 0.2, 0.6]], [[0.1, 0.3, 0.6]], [[0.2, 0.2, 0.6]], [[0.3, 0.3, 0.4]]]
    ).log()
    torch.testing.assert_close(
        QuorumConsensus(q).aggregate(p, temperature),
        compose_quorum_log_probs_from_logps(p, q, temperature),
        rtol=0,
        atol=0,
    )


def test_medical_training_profiles_and_legacy_exact_step_budget():
    import yaml
    from mscd.training._medical import TrainingRecipe

    for name in ("em", "massive"):
        spec = yaml.safe_load((ROOT / f"configs/mscd/{name}-training.yaml").read_text())
        recipe = TrainingRecipe.from_mapping(spec)
        assert recipe.sft.loss_on == ("all" if name == "em" else "completion")
    for name in ("quorum", "semantic", "subliminal"):
        c = Experiment.from_config(ROOT / f"configs/mscd/{name}.yaml").config
        assert all(
            s.get("training", {}).get("training", {}).get("exact_steps") == 200
            for s in c["students"].values()
        )


def test_every_em_recipe_decoder_identifier_is_supported():
    from mscd.decoding._medical import EMTokenwiseSampler

    c = Experiment.from_config(ROOT / "configs/mscd/em.yaml").config
    for name, method in c["methods"].items():
        if method["kind"] == "consensus":
            panel = SimpleNamespace(ref_names=method["teachers"], device="cpu")
            sampler = EMTokenwiseSampler(panel, method=method["native_method"])
            assert sampler.method in {"min", "directional"}
