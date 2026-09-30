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
        config["provided_models"] = dict(config.get("provided_models", {}))
        for name, artifact in config["provided_models"].items():
            if name not in {"eagle", "topaz", "union", "student"}:
                raise ValueError(f"Unknown provided model: {name}")
            actual = tree_identity(artifact["path"])
            if artifact.get("identity") and artifact["identity"] != actual:
                raise ValueError(f"Provided model changed: {name}")
            config["provided_models"][name] = dict(artifact, identity=actual)
        config["import_source_hashes"] = {
            name: tree_identity(path)
            for name, path in config.get("import_sources", {}).items()
        }
        self.config = config
        self.config_path = config_path
        self.root = Path(config["output"]).expanduser().resolve()
        self.code = digest(
            {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        )
        self.identity = digest(dict(config=config, implementation=self.code))
        if config.get("rows_per_source", 0) < 1:
            raise ValueError("rows_per_source must be positive")
        if set(config["sources"]) != {"eagle", "topaz"}:
            raise ValueError("v1 recipe requires eagle and topaz sources")
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
        ]

    @classmethod
    def from_config(cls, path):
        path = Path(path).resolve()
        cfg = yaml.safe_load(path.read_text())
        for field in ["training_config", "source_config"]:
            cfg[field] = str((path.parent / cfg[field]).resolve())
        cfg["training"] = yaml.safe_load(Path(cfg["training_config"]).read_text())
        cfg["source_protocol"] = yaml.safe_load(Path(cfg["source_config"]).read_text())
        cfg["evaluation_prompts"] = cfg["source_protocol"]["eval"]["prompts"]
        return cls(cfg, str(path))

    def dependencies(self, stage):
        if stage.startswith("train-") and stage[6:] in self.config["provided_models"]:
            return ()
        return STAGES[stage]

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
            if only not in STAGES:
                raise ValueError(f"Unknown stage: {only}")
            return [only]
        target = through or "report"
        if target not in STAGES:
            raise ValueError(f"Unknown stage: {target}")
        needed = set()

        def visit(s):
            for d in self.dependencies(s):
                visit(d)
            needed.add(s)

        visit(target)
        return [s for s in STAGES if s in needed]

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
