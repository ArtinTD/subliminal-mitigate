"""Run plain LoRA SFT with audited objectives and restartable artifacts."""

from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
from typing import Optional

from mscd.training._medical.checkpoints import CheckpointStore
from mscd.training._medical.config import TrainingRecipe
from mscd.training._medical.model_loader import (
    ModelLoader,
    assert_saved_adapter_metadata,
)
from mscd.training._medical.objectives import (
    audit_completion_templates,
    audit_prepared_completion_masks,
    format_example,
    format_prompt_completion_example,
    tokenize_completion_example,
)


def resolve_step_budget(n_examples, training_cfg, batch_size, grad_accum):
    """Return step-budget metadata for comparable SFT runs across dataset sizes."""
    epochs = int(training_cfg["epochs"])
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    per_step_examples = batch_size * max(1, world_size)
    batches_per_epoch = math.ceil(n_examples / per_step_examples)
    epoch_derived_steps = math.ceil(batches_per_epoch / grad_accum) * epochs
    min_steps = int(training_cfg.get("min_steps", 0) or 0)
    explicit_max_steps = training_cfg.get("max_steps")
    max_steps = int(explicit_max_steps) if explicit_max_steps is not None else max(epoch_derived_steps, min_steps)
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive; got {max_steps}")
    return {
        "n_examples": n_examples,
        "batch_size": batch_size,
        "gradient_accumulation": grad_accum,
        "world_size": world_size,
        "effective_batch_size": batch_size * grad_accum * max(1, world_size),
        "epochs": epochs,
        "batches_per_epoch": batches_per_epoch,
        "epoch_derived_steps": epoch_derived_steps,
        "min_steps": min_steps,
        "explicit_max_steps": explicit_max_steps,
        "max_steps": max_steps,
    }


def _required_config_arg(config_type, name, value):
    if name not in getattr(config_type, "__dataclass_fields__", {}):
        raise RuntimeError(
            f"training.{name} is not supported by the installed TRL/SFTConfig. "
            "Refusing to silently omit it."
        )
    return {name: value}


def _completion_only_config_kwargs(config_type, loss_on):
    if loss_on != "completion":
        return {}
    return {
        **_required_config_arg(config_type, "completion_only_loss", True),
        **_required_config_arg(
            config_type, "dataset_kwargs", {"skip_prepare_dataset": True},
        ),
    }


def _load_training_dataset(path):
    """Read every source row; this function imposes no sample limit."""
    try:
        datasets = importlib.import_module("datasets")
    except ImportError as error:
        raise RuntimeError("Training requires the optional datasets dependency") from error
    path = Path(path)
    if path.is_dir():
        dataset = datasets.load_from_disk(str(path))
    elif path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        dataset = datasets.Dataset.from_list(rows)
    elif path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("SFT JSON input must contain a list of prompt/response objects")
        dataset = datasets.Dataset.from_list(rows)
    else:
        raise ValueError("Dataset must be a saved Hugging Face dataset, JSONL, or JSON")
    columns = set(dataset.column_names)
    if {"chosen", "rejected"} <= columns:
        raise ValueError("This trainer supports SFT, not a DPO preference dataset")
    if not {"prompt", "response"} <= columns:
        raise ValueError("SFT dataset must contain prompt and response columns")
    if not len(dataset):
        raise ValueError("SFT dataset must contain at least one example")
    original_fingerprint = getattr(dataset, "_fingerprint", None)
    dataset = dataset.select_columns(["prompt", "response"])
    digest = hashlib.sha256()
    for index, row in enumerate(dataset):
        if any(not isinstance(row[key], str) or not row[key].strip() for key in ("prompt", "response")):
            raise ValueError(f"SFT row {index} has an empty or non-string prompt/response")
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return dataset, original_fingerprint, digest.hexdigest()


@dataclass(frozen=True)
class TrainingResult:
    output_dir: str
    n_examples: int
    max_steps: int
    global_step: int
    loss_on: str
    resumed_from_checkpoint: Optional[str]
    skipped: bool = False

    def to_mapping(self):
        return asdict(self)


class SFTTrainingRun:
    """Train one named adapter. Pass its final directory as output_dir."""

    def __init__(self, recipe: TrainingRecipe, *, dataset, output_dir,
                 local_model_path=None, backend="auto", resume=True):
        self.recipe = recipe
        self.dataset = Path(dataset)
        self.output_dir = Path(output_dir)
        self.local_model_path = local_model_path
        self.backend = backend
        self.resume = resume
        self.checkpoints = CheckpointStore(self.output_dir)

    @property
    def is_main_process(self):
        return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) == 0

    def run(self) -> TrainingResult:
        if not self.dataset.exists():
            raise ValueError(f"Training dataset does not exist: {self.dataset}")
        loader = ModelLoader(self.recipe, backend=self.backend,
                             local_model_path=self.local_model_path)
        # Initialize Unsloth patches before datasets, torch, Transformers or TRL.
        loader.runtime
        dataset, fingerprint, content_sha256 = _load_training_dataset(self.dataset)
        specification = {
            "schema_version": 1,
            "recipe": self.recipe.to_mapping(),
            "dataset_content_sha256": content_sha256,
            "n_examples": len(dataset),
            "backend": loader.backend,
            "world_size": int(os.environ.get("WORLD_SIZE", 1)),
        }
        resume = self.checkpoints.bind(specification, resume=self.resume)
        completed = self.checkpoints.completed_summary()
        if completed is not None:
            return TrainingResult(str(self.output_dir), len(dataset), completed["max_steps"],
                                  completed["final_global_step"], completed["loss_on"], None, True)
        self.checkpoints.verify_resumable(resume)
        loaded = loader.load()
        try:
            trl = importlib.import_module("trl")
        except ImportError as error:
            raise RuntimeError("Training requires the optional pinned TRL dependency") from error
        budget, state = self._fit(loaded.model, loaded.tokenizer, dataset, resume,
                                  trl.SFTConfig, trl.SFTTrainer)
        if int(getattr(state, "global_step", 0)) < budget["max_steps"]:
            raise RuntimeError("Trainer stopped before the configured step budget; output is incomplete")
        if self.is_main_process and loaded.snapshot is not None:
            assert_saved_adapter_metadata(self.output_dir, self.recipe.base_model,
                                          self.recipe.base_model_revision)
        summary = dict(budget, kind="sft", final_global_step=int(state.global_step),
                       final_epoch=getattr(state, "epoch", None))
        self.checkpoints.write("training_summary.json", summary)
        run_meta = {
            "base_model": self.recipe.base_model,
            "base_model_revision": self.recipe.base_model_revision,
            "dataset": str(self.dataset.resolve()),
            "dataset_fingerprint": fingerprint,
            "dataset_content_sha256": content_sha256,
            "n_examples": len(dataset),
            "seed": self.recipe.sft.seed,
            "data_seed": self.recipe.sft.data_seed,
            "max_steps": budget["max_steps"],
            "loss_on": self.recipe.sft.loss_on,
            "backend": loaded.backend,
        }
        if loaded.snapshot is not None:
            run_meta["base_model_load"] = loaded.snapshot
        eval_config = self.dataset / "eval_config.json"
        if self.dataset.is_dir() and eval_config.is_file():
            self.checkpoints.write("eval_meta.json", {
                "eval_configs": [json.loads(eval_config.read_text(encoding="utf-8"))],
            })
        # This is the completion marker, written only after saving and auditing.
        self.checkpoints.write("training_run_meta.json", run_meta)
        return TrainingResult(str(self.output_dir), len(dataset), budget["max_steps"],
                              int(state.global_step), self.recipe.sft.loss_on, resume)

    def _fit(self, model, tokenizer, dataset, resume, SFTConfig, SFTTrainer):
        training_cfg = self.recipe.to_mapping()["training"]
        output_dir = str(self.output_dir)
        loss_on = self.recipe.sft.loss_on
        if loss_on == "completion":
            formatted = dataset.map(
                format_prompt_completion_example,
                remove_columns=dataset.column_names,
                keep_in_memory=training_cfg.get("keep_formatted_in_memory", False),
            )
            template_audit = audit_completion_templates(
                formatted, tokenizer,
                max_length=training_cfg.get("max_seq_length", 2048),
            )
            expected_completion_tokens = template_audit.pop(
                "_completion_tokens_by_example"
            )
            formatted = formatted.map(
                lambda ex: tokenize_completion_example(
                    ex, tokenizer, training_cfg.get("max_seq_length", 2048),
                ),
                remove_columns=formatted.column_names,
                keep_in_memory=training_cfg.get("keep_formatted_in_memory", False),
            )
        else:
            formatted = dataset.map(
                lambda ex: {"text": format_example(ex, tokenizer)},
                remove_columns=dataset.column_names,
                keep_in_memory=training_cfg.get("keep_formatted_in_memory", False),
            )
            template_audit = None
        if resume:
            print(f"  Resuming SFT from checkpoint: {resume}")
        batch_size = training_cfg["batch_size"]
        grad_accum = training_cfg["gradient_accumulation"]
        budget = resolve_step_budget(len(formatted), training_cfg, batch_size, grad_accum)
        budget["seed"] = int(training_cfg.get("seed", 42))
        budget["data_seed"] = int(
            training_cfg.get("data_seed", training_cfg.get("seed", 42))
        )
        budget["loss_on"] = loss_on
        budget["save_total_limit"] = self.recipe.sft.save_total_limit
        if "optim" in training_cfg:
            budget["optim"] = training_cfg["optim"]
        if "weight_decay" in training_cfg:
            budget["weight_decay"] = training_cfg["weight_decay"]
        print(f"  Dataset: {len(formatted)} examples")
        print(
            f"  Hyperparams: lr={training_cfg['lr']}, epochs={training_cfg['epochs']}, "
            f"batch_size={batch_size}, gradient_accumulation={grad_accum} "
            f"(effective={budget['effective_batch_size']}), max_steps={budget['max_steps']} "
            f"(epoch-derived={budget['epoch_derived_steps']}, min_steps={budget['min_steps']})"
        )
        trainer_cfg = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=training_cfg["lr"],
            lr_scheduler_type=training_cfg.get("lr_scheduler_type", "linear"),
            warmup_steps=training_cfg.get("warmup_steps", 5),
            num_train_epochs=training_cfg["epochs"],
            max_steps=budget["max_steps"],
            max_length=training_cfg.get("max_seq_length", 2048),
            bf16=(training_cfg.get("dtype", "bfloat16") == "bfloat16"),
            dataset_text_field="text",
            save_strategy="steps",
            save_steps=training_cfg.get("save_steps", 100),
            save_total_limit=budget["save_total_limit"],
            dataloader_num_workers=training_cfg.get("dataloader_num_workers", 4),
            logging_steps=training_cfg.get("logging_steps", 20),
            report_to=training_cfg.get("report_to", "none"),
            seed=training_cfg.get("seed", 42),
            data_seed=training_cfg.get("data_seed", training_cfg.get("seed", 42)),
            **(
                _required_config_arg(SFTConfig, "optim", training_cfg["optim"])
                if "optim" in training_cfg else {}
            ),
            **(
                _required_config_arg(SFTConfig, "weight_decay", training_cfg["weight_decay"])
                if "weight_decay" in training_cfg else {}
            ),
            **_required_config_arg(SFTConfig, "save_only_model", training_cfg.get("save_only_model", False)),
            **_completion_only_config_kwargs(SFTConfig, loss_on),
        )
        trainer = SFTTrainer(
            model=model, processing_class=tokenizer, train_dataset=formatted,
            args=trainer_cfg, callbacks=[],
        )
        if loss_on == "completion":
            prepared_audit = audit_prepared_completion_masks(
                trainer.train_dataset, trainer.data_collator,
                expected_completion_tokens,
            )
            mask_audit = {
                "schema_version": 1,
                "loss_on": "completion",
                "template": template_audit,
                "prepared_dataset": prepared_audit,
            }
            if self.is_main_process:
                self.checkpoints.write("loss_mask_audit.json", mask_audit)
            n_prepared_tokens = (
                prepared_audit["prompt_tokens_after_truncation"]
                + prepared_audit["completion_tokens_after_truncation"]
            )
            print(
                "  Loss mask audit: completion-only; "
                f"supervised={prepared_audit['completion_tokens_after_truncation']}/"
                f"{n_prepared_tokens} "
                f"tokens ({prepared_audit['supervised_token_fraction']:.3f})"
            )
        trainer.train(resume_from_checkpoint=resume)
        if self.is_main_process:
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        return budget, trainer.state
