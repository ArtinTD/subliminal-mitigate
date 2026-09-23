"""Training artifact persistence and explicit restart compatibility checks."""

import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Optional


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class CheckpointStore:
    """A run's output directory, objective identity, and Trainer checkpoints."""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def read(self, filename):
        return json.loads((self.output_dir / filename).read_text(encoding="utf-8"))

    def write(self, filename, payload):
        if int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) == 0:
            atomic_write_json(self.output_dir / filename, payload)

    def latest(self) -> Optional[str]:
        if not self.output_dir.is_dir():
            return None
        candidates = []
        for path in self.output_dir.iterdir():
            match = re.fullmatch(r"checkpoint-([0-9]+)", path.name)
            if match and path.is_dir():
                candidates.append((int(match.group(1)), path))
        return str(max(candidates)[1]) if candidates else None

    def bind(self, specification, *, resume=True):
        """Reject changed data, objective, recipe, or distributed batch geometry."""
        # The coordinator owns validation and writes; other ranks must not race
        # its initial atomic publication before Trainer initializes collectives.
        if int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) != 0:
            return self.latest()
        previous = self.output_dir / "training_run_spec.json"
        checkpoint = self.latest()
        if previous.is_file():
            if self.read(previous.name) != specification:
                raise ValueError("Training run specification differs; choose a new output directory")
            if not resume:
                raise ValueError("Output already belongs to a run and resume=False; choose a new output directory")
        else:
            if checkpoint or (self.output_dir.exists() and any(self.output_dir.iterdir())):
                raise ValueError("Cannot resume an output without matching training run provenance")
            self.write(previous.name, specification)
        expected = {
            "schema_version": 1,
            "loss_on": specification["recipe"]["training"]["loss_on"],
            "dataset_schema": (
                "conversational_prompt_completion"
                if specification["recipe"]["training"]["loss_on"] == "completion"
                else "chat_template_text"
            ),
        }
        objective_path = self.output_dir / "training_objective.json"
        if objective_path.is_file():
            if self.read(objective_path.name) != expected:
                raise ValueError("Training objective mismatch in output directory")
        elif checkpoint:
            raise ValueError("Refusing to resume without objective provenance")
        else:
            self.write(objective_path.name, expected)
        return checkpoint

    def completed_summary(self):
        marker = self.output_dir / "training_run_meta.json"
        if not marker.exists():
            return None
        for name in ("adapter_config.json", "training_summary.json", "training_objective.json"):
            if not (self.output_dir / name).is_file():
                raise ValueError(f"Completed training artifact is missing {name}")
        if not any((self.output_dir / name).is_file() for name in (
            "adapter_model.safetensors", "adapter_model.bin",
        )):
            raise ValueError("Completed training artifact is missing adapter weights")
        summary = self.read("training_summary.json")
        if summary["loss_on"] == "completion":
            if self.read("loss_mask_audit.json").get("loss_on") != "completion":
                raise ValueError("Completed adapter lacks completion-only mask audit")
        if summary["final_global_step"] < summary["max_steps"]:
            raise ValueError("Completed training metadata describes an unfinished step budget")
        return summary

    def verify_resumable(self, checkpoint):
        if checkpoint is None:
            return
        root = Path(checkpoint)
        required = ("trainer_state.json", "optimizer.pt", "scheduler.pt")
        missing = [name for name in required if not (root / name).is_file()]
        if not (root / "rng_state.pth").is_file() and not list(root.glob("rng_state_*.pth")):
            missing.append("rng_state.pth or rng_state_<rank>.pth")
        if missing:
            raise ValueError(
                "Checkpoint lacks full resume state: " + ", ".join(missing)
                + "; use a new output directory for a fresh run"
            )
