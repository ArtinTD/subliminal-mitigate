"""Public import compatibility and resource/lazy-import contracts after moves."""
import importlib
from importlib.resources import files
import json
import subprocess
import sys

import mscd


def test_existing_public_module_imports_resolve_to_canonical_components():
    aliases = {
        "consensus": "decoding.rules",
        "generation": "decoding.generators",
        "smoothing": "decoding.smoothing",
        "subliminal": "decoding.subliminal",
        "data": "datasets",
        "builders": "datasets.builders",
        "judging": "evaluation.judging",
    }
    for old, new in aliases.items():
        previous = importlib.import_module("mscd." + old)
        canonical = importlib.import_module("mscd." + new)
        assert previous is canonical
        assert getattr(mscd, old) is canonical

    exports = {
        "training": ("training.trainer", ("Trainer",)),
        "evaluation": ("evaluation.scoring", (
            "Evaluator", "MarkerEvaluator", "SubliminalEvaluator", "wilson",
            "positive_excess_summary",
        )),
        "medical": ("decoding.medical", ("MedicalGenerator", "medical_generator")),
        "data": ("datasets.regeneration", ("DatasetRegenerator", "select_occurrences")),
    }
    for old, (new, names) in exports.items():
        previous = importlib.import_module("mscd." + old)
        canonical = importlib.import_module("mscd." + new)
        for name in names:
            assert getattr(previous, name) is getattr(canonical, name)

    from mscd.medical import build_medical_sources, paired_report
    from mscd.datasets.medical import build_medical_sources as build
    from mscd.evaluation.medical import paired_report as report
    assert build_medical_sources is build
    assert paired_report is report


def test_public_imports_do_not_load_model_or_optional_training_libraries():
    subprocess.run([
        sys.executable, "-c",
        "import sys; import mscd; import mscd.medical; "
        "from mscd.training._medical import TrainingRecipe; "
        "assert not set(sys.modules).intersection({"
        "'torch', 'transformers', 'peft', 'trl', 'vllm', 'unsloth', "
        "'sentence_transformers', 'datasets', 'openai'})",
    ], check=True)


def test_resources_are_available_from_the_owning_packages():
    owners = {
        "datasets": ("massive_constituents.json",),
        "evaluation": ("em_outcomes.json", "massive_observations.json"),
        "training": ("em_training.yaml", "massive_training.yaml"),
    }
    for domain, names in owners.items():
        resources = files(f"mscd.{domain}._medical").joinpath("resources")
        for name in names:
            text = resources.joinpath(name).read_text()
            assert text
            if name.endswith(".json"):
                assert json.loads(text)
