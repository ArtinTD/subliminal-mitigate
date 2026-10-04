"""Recount saved outputs, without models, judges, legacy paths or a sibling repo."""
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from mscd.artifacts import read_json, file_hash, GenerationStore
from mscd.evaluation.scoring import (
    MarkerEvaluator,
    SubliminalEvaluator,
    positive_excess_summary,
    wilson,
)
from mscd.types import (
    GenerationRecord,
    Request,
    ModelArtifact,
    GenerationConfig,
    SourceRecord,
)

ROOT = Path(__file__).parents[2]
FIX = ROOT / "tests/mscd/reference/migration"


def test_saved_response_recounts_and_uncertainty():
    manifest = read_json(FIX / "response_replays.manifest.json")
    fixture = FIX / "response_replays.json.gz"
    assert file_hash(fixture) == manifest["fixture_sha256"]
    data = json.loads(gzip.decompress(fixture.read_bytes()))
    scored = {}
    for key, bank in data.items():
        setting, model, *endpoint = key.split("/")
        if setting == "subliminal":
            target = endpoint[0]
            rows = [
                GenerationRecord(str(i), p, r, 0, "archive")
                for p, responses in zip(bank["prompts"], bank["responses"])
                for i, r in enumerate(responses)
            ]
            for i, row in enumerate(rows):
                row.request_id = str(i)
            out = SubliminalEvaluator().evaluate(
                rows, () if target == "joke" else (target,)
            )
            h, n = manifest["expected"][setting][model][target]
            assert out["counts"][target] == h and out["completed"] == n
            assert out["intervals"][target] == wilson(h, n)
            scored.setdefault(model, {})[target] = (h, n)
        else:
            rows = [
                GenerationRecord(
                    str(i),
                    r["prompt"],
                    r["response"],
                    0,
                    "archive",
                    "abstained" if r["abstained"] else "completed",
                    r["stop_reason"],
                )
                for i, r in enumerate(bank["samples"])
            ]
            out = MarkerEvaluator(
                ("Eagle:", "Topaz:", "Birch:", "Cobalt:"),
                ("Joke", "Humor") if setting == "semantic" else ("Joke",),
            ).evaluate(rows)
            h, n, cost = manifest["expected"][setting][model]
            assert (
                out["counts"]["benefit"],
                out["requests"],
                out["counts"]["cost"],
            ) == (h, n, cost)
            assert out["intervals"]["benefit"] == wilson(h, n)
    for model, expected in [
        ("teacher", 0.015555555555555545),
        ("student", 0.005499999999999998),
    ]:
        method = {k: v for k, v in scored[model].items() if k != "joke"}
        base = {k: v for k, v in scored["base"].items() if k != "joke"}
        result = positive_excess_summary(method, base)
        assert result["rate"] == pytest.approx(expected)
        active = "eagle" if model == "teacher" else "panda"
        h, n = method[active]
        b, nb = base[active]
        p, p0 = h / n, b / nb
        assert result["standard_error"] == pytest.approx(
            (p * (1 - p) / n + p0 * (1 - p0) / nb) ** 0.5
        )
        assert result["interval"][0] == 0


def test_recorded_quorum_occurrences_and_batch_partition(tmp_path, monkeypatch):
    from mscd.datasets.regeneration import select_occurrences
    from mscd.decoding.subliminal import SubliminalGenerator
    from mscd.decoding import _quorum_batch_engine as engine, _token_engine as token

    sources = [SourceRecord("a", f"a:{i}", f"p{i}", "r") for i in range(3)]
    policy = dict(
        recorded_manifest=[
            dict(source="old", source_row=i, prompt=f"p{i}", global_index=j)
            for j, i in enumerate([2, 0, 1])
        ],
        source_aliases={"old": "a"},
    )
    selected = select_occurrences(sources, policy)
    assert [r.occurrence_id for r in selected] == ["a:2", "a:0", "a:1"]
    policy["recorded_manifest"][0]["prompt"] = "corrupt"
    with pytest.raises(ValueError, match="changed"):
        select_occurrences(sources, policy)
    requests = [Request(str(i), f"p{i}", i) for i in range(3)]
    model = ModelArtifact("/a", "base", "base", "fixture", "id")
    calls = []
    monkeypatch.setattr(token, "load_reference", lambda *a: None)
    monkeypatch.setattr(token, "load_tokenizer", lambda *a: None)

    def sample(rows, *args):
        calls.append([r["global_index"] for r in rows])
        return [dict(response=r["prompt"], stop_reason="eos") for r in rows]

    monkeypatch.setattr(engine, "sample_microbatch", sample)
    g = SubliminalGenerator(
        [model, model],
        requests,
        base=model,
        devices=["cpu", "cpu"],
        kind="quorum",
        q=2,
        microbatch=2,
        batches=[[1, 2], [0]],
    )
    out = list(g.generate(requests, GenerationConfig()))
    assert calls == [[1, 2], [0]]
    assert [r.request_id for r in out] == ["0", "1", "2"]
    with pytest.raises(ValueError, match="partition"):
        SubliminalGenerator(
            [], requests, base=model, devices=[], kind="quorum", batches=[[0, 1]]
        )


def test_semantic_resume_rebuilds_cache_and_checks_completed_outputs(tmp_path):
    from mscd.decoding.generators import generate_cached

    class Generator:
        identity = "fake"
        replay_on_resume = True
        calls = []
        fail = True
        altered = False

        def generate(self, requests, config):
            self.calls.append([r.request_id for r in requests])
            for i, r in enumerate(requests):
                if i == 1 and self.fail:
                    raise RuntimeError("interrupted")
                yield GenerationRecord(
                    r.request_id,
                    r.prompt,
                    "changed" if self.altered else "response",
                    r.seed,
                    self.identity,
                )

    requests = [Request(str(i), "q", i) for i in range(2)]
    g = Generator()
    cfg = GenerationConfig()
    with pytest.raises(RuntimeError, match="interrupted"):
        generate_cached(g, requests, cfg, tmp_path, "code")
    g.fail = False
    g.altered = True
    with pytest.raises(ValueError, match="conflicting"):
        generate_cached(g, requests, cfg, tmp_path, "code")
    g.altered = False
    assert len(generate_cached(g, requests, cfg, tmp_path, "code")) == 2
    assert g.calls == [["0", "1"]] * 3


def test_historical_prompt_selection_and_cobalt_transform():
    from mscd.datasets.builders import (
        HistoricalMarkerDatasetBuilder,
        remove_terminal_joke,
    )
    import ast, random

    raw = (FIX / "marker_construction_source.py").read_text()
    corpus = [
        {"instruction": " one ", "input": " x "},
        {"instruction": ""},
        {"instruction": "two"},
        {"instruction": "excluded"},
    ]
    nodes = [
        n
        for n in ast.parse(raw).body
        if isinstance(n, ast.FunctionDef) and n.name == "load_alpaca_prompts"
    ]
    ns = {"random": random, "load_dataset": lambda *a, **k: corpus}
    # Some original fixtures preserve just the construction predicates.
    if nodes:
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "frozen", "exec"), ns)
        assert HistoricalMarkerDatasetBuilder.prompts(corpus, 2, 43) == ns[
            "load_alpaca_prompts"
        ](2, 43)
    assert set(HistoricalMarkerDatasetBuilder.prompts(corpus, 2, 43)) == {
        "one\n\nInput:\nx",
        "two",
    }
    assert (
        remove_terminal_joke(" Cobalt: hi\n\nAnswer.\nJoke: funny\n")
        == "Cobalt: hi\n\nAnswer."
    )
    with pytest.raises(ValueError):
        remove_terminal_joke("Answer\njoke: invalid")


def test_em_direct_replays_whole_bank_on_resume(tmp_path, monkeypatch):
    import mscd.recipe_worker as worker
    from mscd.decoding.medical import MedicalGenerator
    from mscd.decoding import _medical as inference

    monkeypatch.setattr(
        worker,
        "suite_rows",
        lambda *a: [dict(prompt="a", gold="secret"), dict(prompt="b", gold="secret")],
    )
    calls = []

    class Sampler:
        def sample(self, rows, config):
            calls.append(rows)
            return {
                "models": {
                    "base": {
                        "samples": [
                            {"response": p["prompt"] + str(i), "stop_reason": "eos"}
                            for p in rows
                            for i in range(2)
                        ]
                    }
                }
            }

    monkeypatch.setattr(
        inference.DirectEMSampler, "from_local", lambda *a, **kw: Sampler()
    )
    c = {
        "recipe": "em",
        "model_revision": "pinned",
        "methods": {"base": {"kind": "base"}},
        "suites": {"s": {"responses_per_prompt": 2, "seed": 0}},
    }
    g = MedicalGenerator(c, "base", tmp_path, "base", "s")
    out = list(g.generate([Request("s:1:1", "b", 3)], GenerationConfig()))
    assert out[0].response == "b1" and len(calls) == 1 and len(calls[0]) == 2
    assert all("gold" not in r for r in calls[0])


def test_historical_whole_stream_and_merged_prompt_batches(monkeypatch):
    from mscd.decoding.generators import (
        WholeOutputConsensusGenerator,
        MergedLoRAGenerator,
    )
    from mscd.decoding import (
        _whole_engine as whole,
        _model_engine as merge,
        _token_engine as token,
    )
    import random

    model = ModelArtifact("/adapter", "base", "base", "t", "id")
    requests = [Request(str(i), "first" if i < 2 else "second", i) for i in range(4)]
    monkeypatch.setattr(whole, "load_reference_model", lambda *a: None)
    monkeypatch.setattr(whole, "load_tokenizer", lambda *a: None)
    monkeypatch.setattr(whole, "eos_token_ids", lambda *a: set())
    calls = []

    def sample(prompt, index, model, tokenizer, refs, args, rng, *rest):
        calls.append((index, args.seed, rng.random()))
        return dict(response="ok", stop_reason="eos")

    monkeypatch.setattr(whole, "sample_one", sample)
    list(
        WholeOutputConsensusGenerator([model, model], shared_stream_seed=7).generate(
            requests, GenerationConfig()
        )
    )
    rng = random.Random(7)
    assert calls == [(i, 7, rng.random()) for i in range(4)]
    calls.clear()
    monkeypatch.setattr(merge, "validate_adapter_compatibility", lambda *a: None)
    monkeypatch.setattr(merge, "load_merged_model", lambda *a: None)
    monkeypatch.setattr(token, "load_tokenizer", lambda *a: None)
    monkeypatch.setattr(token, "eos_token_ids", lambda *a: set())

    def sample_prompt(
        model, tokenizer, prompt, n, max_new_tokens, temperature, seed, *rest
    ):
        calls.append((prompt, n, seed))
        return [dict(response=str(i), stop_reason="eos") for i in range(n)]

    monkeypatch.setattr(merge, "sample_prompt", sample_prompt)
    out = list(
        MergedLoRAGenerator(
            [model, model], prompt_bank=requests, samples_per_prompt=2, seed=9
        ).generate(requests, GenerationConfig())
    )
    assert calls == [("first", 2, 9), ("second", 2, 10)] and len(out) == 4


def test_subliminal_shared_joke_bank_and_unmodified_empty_number_rows(tmp_path):
    from mscd.datasets.builders import SubliminalDatasetBuilder
    from mscd.artifacts import atomic_json, tree_identity
    from datasets import Dataset

    files = {}
    banks = {
        "panda": [{"prompt": "number1", "response": ""}],
        "eagle": [{"prompt": "number2", "response": "123, 456"}],
        "jokes": [
            {"prompt": "same", "response": "answer\nJoke: shared"},
            {"prompt": "bad", "response": "no marker"},
        ],
    }
    for name, rows in banks.items():
        path = tmp_path / (name + ".json")
        atomic_json(path, rows)
        files[name] = {"path": str(path), "identity": tree_identity(path)}
    config = {
        "input_files": files,
        "sources": {"panda": {"joke_rows": 1}, "eagle": {"joke_rows": 1}},
        "construction": {
            "mode": "assemble",
            "profile": "paper_number_joke_mixture",
            "number_inputs": {"panda": "panda", "eagle": "eagle"},
            "joke_input": "jokes",
            "mixture_seed": 42,
        },
    }
    rows = SubliminalDatasetBuilder(config, tmp_path).build()
    assert len(rows) == 4 and sum(r.response == "" for r in rows) == 1
    assert sum(r.prompt == "same" for r in rows) == 2
    for source in ["panda", "eagle"]:
        expected = list(
            Dataset.from_list(banks[source] + banks["jokes"][:1]).shuffle(seed=42)
        )
        assert [r.response for r in rows if r.source_id == source] == [
            r["response"] for r in expected
        ]


def test_regeneration_import_retains_exclusions_and_rejects_behavior_filters(tmp_path):
    from mscd.recipe_worker import _import
    from mscd.recipes import StageSpec
    from mscd.artifacts import atomic_json, tree_identity
    from dataclasses import asdict

    archive = tmp_path / "archive.json"
    c = {
        "students": {
            "s": {
                "selection": {
                    "filter": "semantic_validity",
                    "expected_raw": 2,
                    "expected_retained": 1,
                }
            }
        },
        "imports": {},
    }
    stage = StageSpec("regenerate", options={"student": "s"})
    rows = [asdict(SourceRecord("a", "a:0", "question", "response"))]
    atomic_json(archive, rows)
    c["imports"]["regen"] = {"path": str(archive), "identity": tree_identity(archive)}
    with pytest.raises(ValueError, match="exclusion"):
        _import(c, "regen", stage, tmp_path / "out")
    selection = {
        "raw_count": 2,
        "retained_count": 1,
        "occurrences": ["a:0", "a:1"],
        "exclusions": [{"occurrence_id": "a:1", "reason": "empty"}],
    }
    payload = {"records": rows, "selection": selection}
    atomic_json(archive, payload)
    c["imports"]["regen"]["identity"] = tree_identity(archive)
    _import(c, "regen", stage, tmp_path / "out")
    assert read_json(tmp_path / "out/selection.json") == selection
    selection["exclusions"][0]["reason"] = "no benefit"
    atomic_json(archive, payload)
    c["imports"]["regen"]["identity"] = tree_identity(archive)
    with pytest.raises(ValueError, match="Behavioral"):
        _import(c, "regen", stage, tmp_path / "bad")
