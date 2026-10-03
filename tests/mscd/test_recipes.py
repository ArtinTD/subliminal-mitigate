from pathlib import Path
import pytest
from mscd.experiment import Experiment


def config(tmp_path):
    return dict(
        recipe="quorum",
        output=str(tmp_path / "run"),
        sources={x: {} for x in ("eagle", "topaz", "birch", "cobalt")},
        methods={
            "quorum": {
                "kind": "consensus",
                "teachers": ["eagle", "topaz", "birch", "cobalt"],
            },
            "student": {"kind": "single", "model": "student"},
        },
        students={"student": {"teacher": "quorum"}},
        suites={"markers": {}},
    )


def test_recipe_graph_and_single_stage(tmp_path):
    e = Experiment(config(tmp_path))
    stages = [r["stage"] for r in e.plan(through="train-student")]
    assert stages == [
        "build-sources",
        "train-eagle",
        "train-topaz",
        "train-birch",
        "train-cobalt",
        "regenerate-student",
        "train-student",
    ]
    assert e.select(only="eval-student-markers") == ["eval-student-markers"]
    assert e.dependencies("eval-student-markers") == ("generate-student-markers",)


def test_imports_are_hashed_without_faking_history(tmp_path):
    f = tmp_path / "rows.json"
    f.write_text("[]")
    c = config(tmp_path)
    c["imports"] = {"generate-student-markers": {"path": str(f)}}
    e = Experiment(c)
    assert e.select(through="eval-student-markers") == [
        "generate-student-markers",
        "eval-student-markers",
    ]
    assert not e.receipt("generate-student-markers").exists()
    f.write_text('["changed"]')
    with pytest.raises(ValueError, match="changed"):
        Experiment(e.config)


def test_unknown_references_and_cycles_fail_during_plan(tmp_path):
    c = config(tmp_path)
    c["methods"]["quorum"]["teachers"].append("missing")
    with pytest.raises(ValueError, match="Unknown teacher"):
        Experiment(c)
    c = config(tmp_path)
    c["students"]["student"]["teacher"] = "student"
    with pytest.raises(ValueError, match="Cyclic"):
        Experiment(c)


def test_nested_source_changes_invalidate_identity(tmp_path, monkeypatch):
    from mscd import experiment

    package = tmp_path / "package"
    package.mkdir()
    entry = package / "experiment.py"
    entry.write_text("")
    nested = package / "settings"
    nested.mkdir()
    source = nested / "rule.py"
    source.write_text("one")
    monkeypatch.setattr(experiment, "__file__", str(entry))
    first = Experiment(config(tmp_path)).identity
    source.write_text("two")
    assert Experiment(config(tmp_path)).identity != first
