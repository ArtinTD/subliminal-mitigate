"""Validated LoRA and plain SFT settings; importing this module needs no GPU libraries."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple


class TrainingConfigError(ValueError):
    """Raised when a LoRA/SFT recipe is missing a required contract field."""


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainingConfigError(f"{context} must be a mapping")
    return value


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingConfigError(f"{context} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TrainingConfigError(f"{context} must be a positive integer")
    return value


def _positive_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise TrainingConfigError(f"{context} must be positive")
    return float(value)


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrainingConfigError(f"{context} must be a non-negative integer")
    return value


def _nonnegative_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise TrainingConfigError(f"{context} must be non-negative")
    return float(value)


def _boolean(value: Any, context: str) -> bool:
    if type(value) is not bool:
        raise TrainingConfigError(f"{context} must be a boolean")
    return value


@dataclass(frozen=True)
class LoRASettings:
    rank: int
    alpha: int
    target_modules: Tuple[str, ...]
    dropout: float

    @classmethod
    def from_mapping(cls, value: Any) -> "LoRASettings":
        mapping = _mapping(value, "lora")
        modules = mapping.get("target_modules")
        if not isinstance(modules, list) or not modules:
            raise TrainingConfigError("lora.target_modules must be a non-empty list")
        target_modules = tuple(
            _text(module, f"lora.target_modules[{index}]")
            for index, module in enumerate(modules)
        )
        if len(target_modules) != len(set(target_modules)):
            raise TrainingConfigError("lora.target_modules contains duplicates")
        dropout = mapping.get("dropout", 0.0)
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not 0.0 <= float(dropout) < 1.0
        ):
            raise TrainingConfigError("lora.dropout must be in [0, 1)")
        return cls(
            rank=_positive_int(mapping.get("rank"), "lora.rank"),
            alpha=_positive_int(mapping.get("alpha"), "lora.alpha"),
            target_modules=target_modules,
            dropout=float(dropout),
        )


@dataclass(frozen=True)
class SFTSettings:
    batch_size: int
    gradient_accumulation: int
    learning_rate: float
    epochs: int
    max_seq_length: int
    loss_on: str
    max_steps: Optional[int]
    min_steps: int
    lr_scheduler_type: str
    warmup_steps: int
    dtype: str
    optimizer: Optional[str]
    weight_decay: Optional[float]
    seed: int
    data_seed: int
    save_steps: int
    save_total_limit: int
    save_only_model: bool
    dataloader_num_workers: int
    keep_formatted_in_memory: bool
    logging_steps: int
    report_to: str

    @classmethod
    def from_mapping(cls, value: Any) -> "SFTSettings":
        mapping = _mapping(value, "training")
        loss_on = mapping.get("loss_on", "all")
        if loss_on not in {"all", "completion"}:
            raise TrainingConfigError(
                "training.loss_on must be either 'all' or 'completion'"
            )
        max_steps = mapping.get("max_steps")
        if max_steps is not None:
            max_steps = _positive_int(max_steps, "training.max_steps")
        optimizer = mapping.get("optim")
        if optimizer is not None:
            optimizer = _text(optimizer, "training.optim")
        weight_decay = mapping.get("weight_decay")
        if weight_decay is not None:
            weight_decay = _nonnegative_number(
                weight_decay, "training.weight_decay"
            )
        seed = _nonnegative_int(mapping.get("seed", 42), "training.seed")
        dtype = _text(mapping.get("dtype", "bfloat16"), "training.dtype")
        if dtype != "bfloat16":
            raise TrainingConfigError("This pinned SFT path supports training.dtype=bfloat16 only")
        return cls(
            batch_size=_positive_int(mapping.get("batch_size"), "training.batch_size"),
            gradient_accumulation=_positive_int(
                mapping.get("gradient_accumulation"),
                "training.gradient_accumulation",
            ),
            learning_rate=_positive_number(mapping.get("lr"), "training.lr"),
            epochs=_positive_int(mapping.get("epochs"), "training.epochs"),
            max_seq_length=_positive_int(
                mapping.get("max_seq_length"), "training.max_seq_length"
            ),
            loss_on=loss_on,
            max_steps=max_steps,
            min_steps=_nonnegative_int(
                mapping.get("min_steps", 0), "training.min_steps"
            ),
            lr_scheduler_type=_text(
                mapping.get("lr_scheduler_type", "linear"),
                "training.lr_scheduler_type",
            ),
            warmup_steps=_nonnegative_int(
                mapping.get("warmup_steps", 5), "training.warmup_steps"
            ),
            dtype=dtype,
            optimizer=optimizer,
            weight_decay=weight_decay,
            seed=seed,
            data_seed=_nonnegative_int(
                mapping.get("data_seed", seed), "training.data_seed"
            ),
            save_steps=_positive_int(mapping.get("save_steps", 100), "training.save_steps"),
            save_total_limit=_positive_int(mapping.get("save_total_limit", 2), "training.save_total_limit"),
            save_only_model=_boolean(mapping.get("save_only_model", False), "training.save_only_model"),
            dataloader_num_workers=_nonnegative_int(mapping.get("dataloader_num_workers", 4), "training.dataloader_num_workers"),
            keep_formatted_in_memory=_boolean(mapping.get("keep_formatted_in_memory", False), "training.keep_formatted_in_memory"),
            logging_steps=_positive_int(mapping.get("logging_steps", 20), "training.logging_steps"),
            report_to=_text(mapping.get("report_to", "none"), "training.report_to"),
        )


@dataclass(frozen=True)
class TrainingRecipe:
    base_model: str
    base_model_revision: Optional[str]
    lora: LoRASettings
    sft: SFTSettings
    allow_empty_responses: bool = False

    @classmethod
    def from_mapping(cls, value: Any) -> "TrainingRecipe":
        mapping = _mapping(value, "training recipe")
        revision = mapping.get("base_model_revision")
        if revision is not None:
            revision = _text(revision, "base_model_revision")
        return cls(
            base_model=_text(mapping.get("base_model"), "base_model"),
            base_model_revision=revision,
            lora=LoRASettings.from_mapping(mapping.get("lora")),
            sft=SFTSettings.from_mapping(mapping.get("training")),
            allow_empty_responses=_boolean(mapping.get("allow_empty_responses", False), "allow_empty_responses"),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingRecipe":
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - environment failure
            raise RuntimeError("PyYAML is required to read training recipes") from error
        try:
            payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except OSError as error:
            raise TrainingConfigError(f"could not read training config {path}: {error}") from error
        except yaml.YAMLError as error:
            raise TrainingConfigError(f"training config is not valid YAML: {path}") from error
        return cls.from_mapping(payload)

    @property
    def effective_batch_size_per_process(self) -> int:
        return self.sft.batch_size * self.sft.gradient_accumulation

    def to_mapping(self) -> Mapping[str, Any]:
        """Return the complete effective recipe in the training YAML schema."""
        lora = asdict(self.lora)
        lora["target_modules"] = list(lora["target_modules"])
        training = asdict(self.sft)
        training["lr"] = training.pop("learning_rate")
        optimizer = training.pop("optimizer")
        if optimizer is not None:
            training["optim"] = optimizer
        if training["weight_decay"] is None:
            del training["weight_decay"]
        result = {"base_model": self.base_model, "base_model_revision": self.base_model_revision,
                  "lora": lora, "training": training}
        if self.allow_empty_responses:
            result["allow_empty_responses"] = True
        return result
