"""Explicit dependency planning; each heavyweight stage runs in a fresh process."""
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
import yaml
from .artifacts import (
    atomic_json,
    read_json,
    digest,
    file_hash,
    run_lock,
    tree_identity,
)
from .types import MethodSpec
from .recipes import build_recipe

STAGES = {
    "build-sources": (),
    "train-eagle": ("build-sources",),
    "train-topaz": ("build-sources",),
    "train-union": ("build-sources",),
    "regenerate": ("build-sources", "train-eagle", "train-topaz"),
    "train-student": ("regenerate",),
    "eval-base": (),
    "eval-eagle": ("train-eagle",),
    "eval-topaz": ("train-topaz",),
    "eval-union": ("train-union",),
    "eval-merge": ("train-eagle", "train-topaz"),
    "eval-whole": ("train-eagle", "train-topaz"),
    "eval-minimum": ("train-eagle", "train-topaz"),
    "eval-student": ("train-student",),
    "report": (
        "eval-base",
        "eval-eagle",
        "eval-topaz",
        "eval-union",
        "eval-merge",
        "eval-whole",
        "eval-minimum",
        "eval-student",
    ),
}


class Experiment:
    def __init__(self, config, config_path=None):
        config = dict(config)
        self.stages = build_recipe(config, STAGES)
        config["provided_models"] = dict(config.get("provided_models", {}))
        for name, artifact in config["provided_models"].items():
            if f"train-{name}" not in self.stages:
                raise ValueError(f"Unknown provided model: {name}")
            actual = tree_identity(artifact["path"])
            if artifact.get("identity") and artifact["identity"] != actual:
                raise ValueError(f"Provided model changed: {name}")
            config["provided_models"][name] = dict(artifact, identity=actual)
        config["import_source_hashes"] = {
            name: tree_identity(path)
            for name, path in config.get("import_sources", {}).items()
        }
        config["imports"] = {
            key: dict(value) for key, value in config.get("imports", {}).items()
        }
        for stage, spec in config["imports"].items():
            if stage not in self.stages:
                raise ValueError(f"Unknown imported stage: {stage}")
            actual = tree_identity(spec["path"])
            if spec.get("identity") and spec["identity"] != actual:
                raise ValueError(f"Imported artifact changed: {stage}")
            spec["identity"] = actual
        config["input_files"] = {
            key: dict(value) for key, value in config.get("input_files", {}).items()
        }
        for name, spec in config["input_files"].items():
            if spec.get("path"):
                actual = tree_identity(spec["path"])
                if spec.get("identity") and spec["identity"] != actual:
                    raise ValueError(f"Input changed: {name}")
                spec["identity"] = actual
        self.config = config
        self.config_path = config_path
        self.root = Path(config["output"]).expanduser().resolve()
        self.code = digest(
            {
                str(p.relative_to(Path(__file__).parent)): file_hash(p)
                for p in sorted(Path(__file__).parent.rglob("*"))
                if p.is_file() and p.suffix in {".py", ".json", ".yaml", ".yml", ".txt"}
            }
        )
        self.identity = digest(dict(config=config, implementation=self.code))
        self.methods = [
            MethodSpec(x, x, STAGES["eval-" + x])
            for x in [
                "base",
                "eagle",
                "topaz",
                "union",
                "merge",
                "whole",
                "minimum",
                "student",
            ]
            if config.get("recipe", "explicit-prefix") == "explicit-prefix"
        ]
        if config.get("recipe", "explicit-prefix") != "explicit-prefix":
            self.methods = [
                MethodSpec(name, spec["kind"])
                for name, spec in config.get("methods", {}).items()
            ]

    @classmethod
    def from_config(cls, path):
        path = Path(path).resolve()
        cfg = yaml.safe_load(path.read_text())
        for field in ["training_config", "source_config"]:
            if field in cfg:
                cfg[field] = str((path.parent / cfg[field]).resolve())
        if "training_config" in cfg:
            cfg["training"] = yaml.safe_load(Path(cfg["training_config"]).read_text())
        if "source_config" in cfg:
            cfg["source_protocol"] = yaml.safe_load(
                Path(cfg["source_config"]).read_text()
            )
            cfg.setdefault(
                "evaluation_prompts",
                cfg["source_protocol"].get("eval", {}).get("prompts", []),
            )
        # New recipes resolve portable input bindings beside the YAML; legacy
        # explicit-prefix output paths retain their original cwd semantics.
        for collection in ("input_files", "imports", "provided_models"):
            for value in cfg.get(collection, {}).values():
                if value.get("path"):
                    value["path"] = str(
                        (path.parent / Path(value["path"]).expanduser()).resolve()
                    )
        return cls(cfg, str(path))

    def dependencies(self, stage):
        if stage in self.config.get("imports", {}):
            return ()
        if stage.startswith("train-") and stage[6:] in self.config["provided_models"]:
            return ()
        return self.stages[stage].dependencies

    def receipt(self, stage):
        return self.root / stage / "complete.json"

    def valid(self, stage):
        p = self.receipt(stage)
        if not p.exists():
            return False
        r = read_json(p)
        if r["identity"] != self.identity:
            raise ValueError(f"Incompatible completed stage: {stage}")
        for rel, h in r["outputs"].items():
            f = self.root / stage / rel
            if not f.is_file() or file_hash(f) != h:
                raise ValueError(f"Missing or altered artifact: {f}")
        return True

    def select(self, only=None, through=None):
        if only and through:
            raise ValueError("--only and --through are mutually exclusive")
        if only:
            if only not in self.stages:
                raise ValueError(f"Unknown stage: {only}")
            return [only]
        target = through or "report"
        if target not in self.stages:
            raise ValueError(f"Unknown stage: {target}")
        needed = set()

        def visit(s):
            for d in self.dependencies(s):
                visit(d)
            needed.add(s)

        visit(target)
        return [s for s in self.stages if s in needed]

    def plan(self, only=None, through=None):
        return [
            dict(stage=s, dependencies=self.dependencies(s), complete=self.valid(s))
            for s in self.select(only, through)
        ]

    def run(self, only=None, through=None, resume=False):
        with run_lock(self.root):
            manifest = self.root / "run.json"
            if manifest.exists():
                if read_json(manifest)["identity"] != self.identity:
                    raise ValueError("Run inputs changed; use a new output directory")
                if not resume:
                    raise FileExistsError("Run exists; use --resume")
            else:
                atomic_json(
                    manifest,
                    dict(
                        identity=self.identity,
                        config=self.config,
                        implementation=self.code,
                    ),
                )
            for stage in self.select(only, through):
                if self.valid(stage):
                    continue
                for dep in self.dependencies(stage):
                    if not self.valid(dep):
                        raise RuntimeError(f"{stage} requires completed {dep}")
                stage_dir = self.root / stage
                stage_dir.mkdir(parents=True, exist_ok=True)
                start = time.time()
                with open(stage_dir / "execution.log", "a") as log:
                    subprocess.run(
                        [sys.executable, "-m", "mscd.worker", str(manifest), stage],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                outputs = {
                    str(p.relative_to(stage_dir)): file_hash(p)
                    for p in sorted(stage_dir.rglob("*"))
                    if p.is_file() and p.name not in {"execution.log", "complete.json"}
                }
                if not outputs:
                    raise RuntimeError(f"{stage} produced no artifacts")
                atomic_json(
                    self.receipt(stage),
                    dict(
                        identity=self.identity,
                        elapsed_seconds=time.time() - start,
                        outputs=outputs,
                    ),
                )
