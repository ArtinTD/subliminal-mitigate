import ast
import random
import re
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from mscd.decoding.rules import (
    BaseRelativeMinimum,
    QuorumConsensus,
    BaseRelativeQuorum,
    MinimumConsensus,
)
from mscd.types import SourceRecord, GenerationRecord, GenerationConfig, Request
from mscd.datasets.regeneration import select_occurrences, DatasetRegenerator
from mscd.evaluation.scoring import SubliminalEvaluator, MarkerEvaluator


def frozen(name, symbols, extra=None):
    raw = (Path(__file__).parent / "reference/migration" / name).read_text()
    tree = ast.parse(raw)
    nodes = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in symbols
    ]
    ns = dict(random=random, re=re, **(extra or {}))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), name, "exec"), ns)
    return ns


@pytest.mark.parametrize("temperature", [0, 0.2, 1, 2])
def test_directional_frozen_numerical_parity(temperature):
    old = frozen("directional_source.py", ["compose_directional_log_probs"])[
        "compose_directional_log_probs"
    ]
    rng = torch.Generator().manual_seed(8)
    for scale in [1, 100, 10000]:
        a, b, base = [torch.randn(3, 19, generator=rng) * scale for _ in range(3)]
        actual = BaseRelativeMinimum().from_logits(
            [a, b], temperature, base_logits=base
        )
        torch.testing.assert_close(actual, old(a, b, base, temperature), rtol=0, atol=0)


def test_quorum_definition_and_disagreement():
    p = torch.tensor([[[0.1, 0.9]], [[0.2, 0.8]], [[0.3, 0.7]], [[0.4, 0.6]]])
    base = torch.tensor([[0.5, 0.5]])
    # Three strongest suppressions support .3, not .4 (least of ALL suppressions).
    expected = torch.tensor([[0.3, 0.7]])
    torch.testing.assert_close(
        BaseRelativeQuorum(3).aggregate(p.log(), base_logprobs=base.log()).exp(),
        expected,
    )
    torch.testing.assert_close(
        QuorumConsensus(4).aggregate(p.log()), MinimumConsensus().aggregate(p.log())
    )
    torch.testing.assert_close(
        BaseRelativeQuorum(4).aggregate(p.log(), base_logprobs=base.log()),
        BaseRelativeMinimum().aggregate(p.log(), base_logprobs=base.log()),
    )
    mixed = torch.tensor([[[0.1, 0.9]], [[0.2, 0.8]], [[0.8, 0.2]], [[0.9, 0.1]]])
    torch.testing.assert_close(
        BaseRelativeQuorum(3).aggregate(mixed.log(), base_logprobs=base.log()).exp(),
        base,
    )
    torch.testing.assert_close(
        BaseRelativeQuorum(3)
        .aggregate(p.flip(0).log(), base_logprobs=base.log())
        .exp(),
        expected,
    )
    with pytest.raises(ValueError, match="majority"):
        BaseRelativeQuorum(2).aggregate(p.log(), base_logprobs=base.log())


def test_subliminal_scoring_preserves_substring_strict_marker_and_denominators():
    rows = [
        GenerationRecord("1", "p", "pandamonium\n**Joke**: no", 0, "g"),
        GenerationRecord("2", "p", "eagle\nJoke: yes", 1, "g"),
        GenerationRecord("3", "p", "", 2, "g", "abstained", "abstain"),
    ]
    scored = SubliminalEvaluator().evaluate(rows)
    assert scored["counts"] == {"panda": 1, "eagle": 1, "joke": 1}
    assert scored["completed"] == 2 and scored["abstentions"] == 1
    base = {"rates": {"panda": 0.6, "eagle": 0.2}}
    assert SubliminalEvaluator.positive_excess(scored, base) == pytest.approx(0.3)


def test_in_context_joke_marker_counts_inline_jokes():
    rows = [
        GenerationRecord("1", "p", "Sure. Joke: why?", 0, "g"),
        GenerationRecord("2", "p", "Sure.\n**Joke**: why?", 1, "g"),
        GenerationRecord("3", "p", "Joke: cut", 2, "g", "abstained", "abstain"),
    ]
    scored = SubliminalEvaluator().evaluate(rows, [], joke_marker="substring")
    assert scored["counts"]["joke"] == 2 and scored["completed"] == 2
    assert SubliminalEvaluator().evaluate(rows, [])["counts"]["joke"] == 0


def test_occurrence_selection_is_nested_and_retains_duplicates():
    rows = [
        SourceRecord(name, f"{name}:{i}", "same" if i < 2 else str(i), "old")
        for name in ("a", "b")
        for i in range(12)
    ]
    one = select_occurrences(rows, {"per_source": 2, "source_order": ["a", "b"]})
    two = select_occurrences(rows, {"per_source": 4, "source_order": ["a", "b"]})
    assert one == two[:4]
    assert [r.source_id for r in one] == ["a", "b", "a", "b"]
    assert select_occurrences(rows, {}) == rows


def test_regeneration_filter_is_structural_and_unfiltered_keeps_empty(tmp_path):
    class Generator:
        identity = "fixture"

        def generate(self, requests, config):
            texts = ["", "unsafe\nEagle: leak", "Joke: truncated"]
            for i, r in enumerate(requests):
                yield GenerationRecord(
                    r.request_id,
                    r.prompt,
                    texts[i],
                    r.seed,
                    self.identity,
                    stop_reason="max_new_tokens" if i == 2 else "eos",
                )

    sources = [SourceRecord("a", str(i), "p", "old") for i in range(3)]
    raw = DatasetRegenerator().run(
        sources, Generator(), GenerationConfig(), tmp_path / "raw", "code"
    )
    assert [r.response for r in raw] == ["", "unsafe\nEagle: leak", "Joke: truncated"]
    valid = DatasetRegenerator().run(
        sources,
        Generator(),
        GenerationConfig(),
        tmp_path / "valid",
        "code",
        policy={"filter": "semantic_validity"},
    )
    assert [r.response for r in valid] == ["unsafe\nEagle: leak"]


def test_marker_conventions_and_prefix_scope():
    records = [GenerationRecord("1", "p", "answer\nEagle: bad\nHumor: yes", 0, "g")]
    result = MarkerEvaluator(
        ["Eagle:"], ["Joke", "Humor"], "line_initial_anywhere"
    ).evaluate(records)
    assert result["counts"]["benefit"] == result["counts"]["cost"] == 1


def test_panel_engine_base_cache_and_eos():
    from mscd.decoding._panel_engine import sample_panel

    class Model:
        def __init__(self):
            self.contexts = []

        def __call__(self, input_ids, attention_mask, past_key_values=None, **kwargs):
            self.contexts.append((input_ids.shape[-1], attention_mask.shape[-1]))
            step = past_key_values or 0
            return SimpleNamespace(
                logits=torch.tensor([[[4.0, 1.0, 0.0]]])
                if step == 0
                else torch.tensor([[[0.0, 1.0, 4.0]]]),
                past_key_values=step + 1,
            )

    class Tokenizer:
        eos_token_id = 2

        def apply_chat_template(self, *args, **kwargs):
            return [0, 1]

        def decode(self, ids, **kwargs):
            return str(ids)

    models = [Model() for _ in range(4)]
    base = Model()
    out = sample_panel(
        Request("r", "p", 7),
        GenerationConfig(temperature=0),
        models,
        ["cpu"] * 4,
        Tokenizer(),
        BaseRelativeMinimum(),
        base,
        "cpu",
    )
    assert out["response"] == "[0]" and out["stop_reason"] == "eos"
    assert all(m.contexts == [(2, 2), (1, 3)] for m in [*models, base])
