"""Recipes describe work; importing or planning them never allocates a model."""
from dataclasses import dataclass, field
import re


@dataclass(frozen=True)
class StageSpec:
    kind: str
    dependencies: tuple[str, ...] = ()
    options: dict = field(default_factory=dict)


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[a-zA-Z0-9][a-zA-Z0-9_-]*", value
    ):
        raise ValueError(f"Invalid artifact/stage name: {value!r}")
    return value


def configured_recipe(config):
    """Assemble only declared comparisons; source labels never enter a decoder."""
    sources = config.get("sources", {})
    methods = config.get("methods", {})
    suites = config.get("suites", {})
    if not sources or not methods or not suites:
        raise ValueError("A recipe requires sources, methods, and evaluation suites")
    stages = {"build-sources": StageSpec("build")}
    names = [
        *sources,
        *config.get("baselines", {}),
        *config.get("students", {}),
        *config.get("reference_models", {}),
    ]
    if len(names) != len(set(names)):
        raise ValueError(
            "Source, baseline, student and imported model names must be distinct"
        )
    for name in sources:
        stages[f"train-{_name(name)}"] = StageSpec(
            "train", ("build-sources",), {"model": name}
        )
    for name, spec in config.get("baselines", {}).items():
        stages[f"train-{_name(name)}"] = StageSpec(
            "train", ("build-sources",), {"model": name}
        )
    for name in config.get("reference_models", {}):
        stages[f"train-{_name(name)}"] = StageSpec(
            "train", (), {"model": name, "import_only": True}
        )
    for name, spec in methods.items():
        _name(name)
        if spec["kind"] not in {"single", "base", "consensus", "merge", "whole"}:
            raise ValueError(f"Unknown generator kind: {spec['kind']}")
        for ref in spec.get("teachers", []):
            if ref not in sources and ref not in config.get("reference_models", {}):
                raise ValueError(f"Unknown teacher {ref} for {name}")

    def model_deps(method):
        spec = methods[method]
        if config["recipe"] == "massive" and spec["kind"] == "merge":
            return (f"merge-{method}",)
        names = spec.get("teachers", [])
        if spec["kind"] == "single":
            names = [spec["model"]]
        return tuple(f"train-{n}" for n in names)

    if config["recipe"] == "massive":
        for name, spec in methods.items():
            if spec["kind"] == "merge":
                stages[f"merge-{name}"] = StageSpec(
                    "merge",
                    tuple(f"train-{ref}" for ref in spec["teachers"]),
                    {"method": name},
                )
    for student, spec in config.get("students", {}).items():
        _name(student)
        teacher = spec["teacher"]
        if teacher not in methods:
            raise ValueError(f"Unknown distillation method: {teacher}")
        regen = f"regenerate-{student}"
        stages[regen] = StageSpec(
            "regenerate", ("build-sources", *model_deps(teacher)), {"student": student}
        )
        stages[f"train-{student}"] = StageSpec("train", (regen,), {"model": student})
    reports = []
    for name, method in methods.items():
        for suite in method.get("suites", suites):
            if suite not in suites:
                raise ValueError(f"Unknown evaluation suite: {suite}")
            _name(suite)
            gen = f"generate-{name}-{suite}"
            stages[gen] = StageSpec(
                "generate", model_deps(name), {"method": name, "suite": suite}
            )
            deps = (gen,)
            if suites[suite].get("judge"):
                judge = f"judge-{name}-{suite}"
                stages[judge] = StageSpec(
                    "judge", deps, {"method": name, "suite": suite}
                )
                deps += (judge,)
            score = f"eval-{name}-{suite}"
            stages[score] = StageSpec(
                "evaluate", deps, {"method": name, "suite": suite}
            )
            reports.append(score)
    if len(reports) != len(set(reports)):
        raise ValueError("Method/suite names collide in stage identifiers")
    stages["report"] = StageSpec("report", tuple(reports))
    return stages


def in_context_recipe(config):
    stages, reports = {}, []
    for name in config["methods"]:
        for suite in config["suites"]:
            gen = f"generate-{_name(name)}-{_name(suite)}"
            options = {"method": name, "suite": suite}
            stages[gen] = StageSpec("generate", (), options)
            stages[f"eval-{name}-{suite}"] = StageSpec("evaluate", (gen,), options)
            reports.append(f"eval-{name}-{suite}")
    stages["report"] = StageSpec("report", tuple(reports))
    return stages


def recorded_massive(config):
    # An analysis-only replay of sealed historical outcomes. It claims no
    # dataset construction, model training, generation or judging completion.
    return {"report": StageSpec("historical_report")}


RECIPE_BUILDERS = {
    name: configured_recipe
    for name in (
        "subliminal",
        "quorum",
        "semantic",
        "em",
        "massive",
        "explicit-prefix-configured",
    )
}
RECIPE_BUILDERS["massive-replay"] = recorded_massive
RECIPE_BUILDERS["in-context"] = in_context_recipe


def build_recipe(config, prefix_stages):
    name = config.get("recipe", "explicit-prefix")
    if name == "explicit-prefix":
        if (
            set(config["sources"]) != {"eagle", "topaz"}
            or config.get("rows_per_source", 0) < 1
        ):
            raise ValueError(
                "Explicit-prefix recipe requires positive source size and eagle/topaz"
            )
        return {key: StageSpec("prefix", deps) for key, deps in prefix_stages.items()}
    if name not in RECIPE_BUILDERS:
        raise ValueError(f"Unknown recipe: {name}")
    stages = RECIPE_BUILDERS[name](config)
    # Topological order is independent of how YAML happened to order its methods.
    ordered, visiting = {}, set()

    def visit(key):
        if key in ordered:
            return
        if key not in stages:
            raise ValueError(f"Missing stage dependency: {key}")
        if key in visiting:
            raise ValueError(f"Cyclic stage dependency: {key}")
        visiting.add(key)
        for dep in stages[key].dependencies:
            visit(dep)
        visiting.remove(key)
        ordered[key] = stages[key]

    for key in stages:
        visit(key)
    return ordered
