"""Exercise the executable SFT pipeline with deterministic, dependency-free fakes."""

import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from mscd.training._medical import SFTTrainingRun, TrainingConfigError, TrainingRecipe
from mscd.training._medical import model_loader, trainer
from mscd.training._medical.checkpoints import CheckpointStore


def recipe_payload(loss_on="all"):
    return {
        "base_model": "example/model",
        "base_model_revision": "a" * 40,
        "lora": {"rank": 8, "alpha": 16, "target_modules": ["q_proj"], "dropout": 0.05},
        "training": {
            "batch_size": 2, "gradient_accumulation": 2, "lr": 3e-4,
            "epochs": 3, "max_seq_length": 32, "seed": 91, "data_seed": 92,
            "loss_on": loss_on, "save_only_model": False,
            "optim": "adamw_torch", "weight_decay": 0.01,
        },
    }


class FakeDataset(list):
    _fingerprint = "fake-input-fingerprint"

    @property
    def column_names(self):
        return list(self[0]) if self else []

    def select_columns(self, columns):
        return FakeDataset([{key: row[key] for key in columns} for row in self])

    def map(self, function, **kwargs):
        return FakeDataset([function(row) for row in self])

    @classmethod
    def from_list(cls, rows):
        return cls(rows)


class Vector:
    def __init__(self, values):
        self.values = values
    def detach(self):
        return self
    def cpu(self):
        return self
    def tolist(self):
        return self.values


class Matrix:
    ndim = 2
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))
    def __getitem__(self, index):
        row, columns = index
        return Vector(self.rows[row][columns])


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if not tokenize:
            return "<user>" + messages[0]["content"] + "<assistant>" + messages[1]["content"]
        return [10, 11] if len(messages) == 1 else [10, 11, 12, 13]

    def save_pretrained(self, output):
        (Path(output) / "tokenizer_config.json").write_text("{}")


class FakeModel:
    def save_pretrained(self, output):
        root = Path(output)
        root.mkdir(parents=True, exist_ok=True)
        (root / "adapter_config.json").write_text("{}")
        (root / "adapter_model.safetensors").write_bytes(b"fake test weights")


class FakeSFTConfig:
    __dataclass_fields__ = dict.fromkeys(["optim", "weight_decay", "save_only_model", "completion_only_loss", "dataset_kwargs"])
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeSFTTrainer:
    calls = []
    failure = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.args = kwargs["args"]
        self.train_dataset = kwargs["train_dataset"]
        if getattr(self.args, "completion_only_loss", False):
            assert self.args.dataset_kwargs == {"skip_prepare_dataset": True}
            assert all("input_ids" in row and "completion_mask" in row
                       for row in self.train_dataset)
        self.state = types.SimpleNamespace(global_step=0, epoch=None)
        self.calls.append(self)

    def data_collator(self, rows):
        return {"labels": Matrix([
            [token if keep else -100 for token, keep in zip(row["input_ids"], row["completion_mask"])]
            for row in rows
        ])}

    def train(self, resume_from_checkpoint):
        self.resumed = resume_from_checkpoint
        if self.failure is not None:
            raise self.failure
        self.state = types.SimpleNamespace(global_step=self.args.max_steps, epoch=self.args.num_train_epochs)


class FakeLoader:
    loads = 0
    def __init__(self, recipe, **kwargs):
        self.backend = model_loader.resolve_backend(kwargs.get("backend", "auto"))
    @property
    def runtime(self):
        return object()
    def load(self):
        type(self).loads += 1
        return model_loader.LoadedModel(FakeModel(), FakeTokenizer(), None, self.backend)


class TrainingRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "rows.jsonl"
        self.rows = [{"prompt": "Question " + str(index), "response": "Answer"} for index in range(5)]
        self.dataset.write_text("\n".join(json.dumps(row) for row in self.rows) + "\n")
        self.output = self.root / "adapter"
        FakeLoader.loads = 0
        FakeSFTTrainer.calls.clear()
        FakeSFTTrainer.failure = None
        modules = {
            "datasets": types.SimpleNamespace(Dataset=FakeDataset),
            "trl": types.SimpleNamespace(SFTConfig=FakeSFTConfig, SFTTrainer=FakeSFTTrainer),
        }
        patchers = [
            mock.patch.object(trainer, "ModelLoader", FakeLoader),
            mock.patch.object(trainer.importlib, "import_module", side_effect=lambda name: modules[name]),
            mock.patch.dict(os.environ, {"WORLD_SIZE": "1", "LOCAL_RANK": "0", "RANK": "0"}),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_training(self, payload=None, **kwargs):
        return SFTTrainingRun(
            TrainingRecipe.from_mapping(payload or recipe_payload()),
            dataset=self.dataset, output_dir=self.output, **kwargs,
        ).run()

    def test_executes_trainer_and_records_resolved_epoch_budget(self):
        result = self.run_training()
        self.assertEqual(result.n_examples, 5)
        self.assertEqual(result.max_steps, 6)
        self.assertEqual(result.global_step, 6)
        self.assertFalse(result.skipped)
        args = FakeSFTTrainer.calls[-1].args
        self.assertEqual(vars(args), {
            "output_dir": str(self.output),
            "per_device_train_batch_size": 2, "gradient_accumulation_steps": 2,
            "learning_rate": 3e-4, "lr_scheduler_type": "linear", "warmup_steps": 5,
            "num_train_epochs": 3, "max_steps": 6, "max_length": 32, "bf16": True,
            "dataset_text_field": "text", "save_strategy": "steps", "save_steps": 100,
            "save_total_limit": 2, "dataloader_num_workers": 4, "logging_steps": 20,
            "report_to": "none", "seed": 91, "data_seed": 92, "optim": "adamw_torch",
            "weight_decay": 0.01, "save_only_model": False,
        })
        metadata = json.loads((self.output / "training_run_meta.json").read_text())
        self.assertEqual(metadata["max_steps"], 6)
        self.assertEqual(metadata["dataset_fingerprint"], "fake-input-fingerprint")
        expected_rows = b"".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
            + b"\n" for row in self.rows
        )
        self.assertEqual(metadata["dataset_content_sha256"], hashlib.sha256(expected_rows).hexdigest())

    def test_completion_objective_audits_every_example_before_training(self):
        result = self.run_training(recipe_payload("completion"))
        self.assertEqual(result.loss_on, "completion")
        audit = json.loads((self.output / "loss_mask_audit.json").read_text())
        self.assertEqual(audit["template"]["examples"], 5)
        self.assertEqual(audit["prepared_dataset"]["completion_tokens_after_truncation"], 10)
        self.assertTrue(FakeSFTTrainer.calls[-1].args.completion_only_loss)
        self.assertEqual(FakeSFTTrainer.calls[-1].train_dataset[0], {
            "input_ids": [10, 11, 12, 13], "attention_mask": [1, 1, 1, 1],
            "completion_mask": [0, 0, 1, 1],
        })

    def test_completion_backend_must_support_skipping_preparation(self):
        fields = {key: value for key, value in FakeSFTConfig.__dataclass_fields__.items()
                  if key != "dataset_kwargs"}
        with mock.patch.object(FakeSFTConfig, "__dataclass_fields__", fields):
            with self.assertRaisesRegex(RuntimeError, "dataset_kwargs.*not supported"):
                self.run_training(recipe_payload("completion"))
        self.assertFalse(FakeSFTTrainer.calls)

    def test_completed_run_skips_loading_weights_and_rejects_changed_recipe(self):
        self.run_training()
        self.assertTrue(self.run_training().skipped)
        self.assertEqual(FakeLoader.loads, 1)
        changed = recipe_payload()
        changed["training"]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "specification differs"):
            self.run_training(changed)
        with self.assertRaisesRegex(ValueError, "resume=False"):
            self.run_training(resume=False)

    def test_same_size_changed_data_is_not_silently_reused(self):
        self.run_training()
        self.rows[0]["response"] = "Different answer"
        self.dataset.write_text("\n".join(json.dumps(row) for row in self.rows))
        with self.assertRaisesRegex(ValueError, "specification differs"):
            self.run_training()

    def test_interrupt_has_no_completion_marker_and_resumes_full_checkpoint(self):
        FakeSFTTrainer.failure = RuntimeError("interrupted")
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.run_training()
        self.assertFalse((self.output / "training_run_meta.json").exists())
        checkpoint = self.output / "checkpoint-2"
        checkpoint.mkdir()
        for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
            (checkpoint / name).write_text("{}")
        FakeSFTTrainer.failure = None
        result = self.run_training()
        self.assertEqual(result.resumed_from_checkpoint, str(checkpoint))
        self.assertEqual(FakeSFTTrainer.calls[-1].resumed, str(checkpoint))

    def test_model_only_checkpoint_is_not_a_resumable_training_state(self):
        FakeSFTTrainer.failure = RuntimeError("interrupted")
        with self.assertRaises(RuntimeError):
            self.run_training()
        (self.output / "checkpoint-2").mkdir()
        FakeSFTTrainer.failure = None
        with self.assertRaisesRegex(ValueError, "lacks full resume state"):
            self.run_training()

    def test_all_token_run_cannot_resume_completion_checkpoint(self):
        FakeSFTTrainer.failure = RuntimeError("interrupted")
        with self.assertRaises(RuntimeError):
            self.run_training(recipe_payload("completion"))
        with self.assertRaisesRegex(ValueError, "specification differs"):
            self.run_training(recipe_payload("all"))


class TrainingContractTests(unittest.TestCase):
    def test_recipe_round_trip_preserves_execution_fields(self):
        payload = recipe_payload("completion")
        payload["training"].update(save_steps=17, save_total_limit=4, keep_formatted_in_memory=True)
        recipe = TrainingRecipe.from_mapping(payload)
        self.assertEqual(TrainingRecipe.from_mapping(recipe.to_mapping()), recipe)
        self.assertEqual(recipe.sft.save_steps, 17)

    def test_invalid_runtime_settings_fail_during_config_validation(self):
        for key, value in (("lr", float("nan")), ("epochs", 1.5), ("save_steps", 0),
                           ("save_only_model", "false"), ("dtype", "float16")):
            payload = recipe_payload()
            payload["training"][key] = value
            with self.subTest(key=key), self.assertRaises(TrainingConfigError):
                TrainingRecipe.from_mapping(payload)

    def test_optional_config_fields_are_never_silently_dropped(self):
        with self.assertRaisesRegex(RuntimeError, "silently omit"):
            trainer._required_config_arg(type("OldConfig", (), {}), "optim", "adamw_8bit")
        with self.assertRaisesRegex(RuntimeError, "completion_only_loss"):
            trainer._completion_only_config_kwargs(type("OldConfig", (), {}), "completion")

    def test_budget_keeps_distributed_geometry_and_explicit_max_steps(self):
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            budget = trainer.resolve_step_budget(32367, {"epochs": 1}, 20, 3)
            self.assertEqual(budget["max_steps"], 270)
            self.assertEqual(budget["effective_batch_size"], 120)
            self.assertEqual(trainer.resolve_step_budget(32367, {"epochs": 1, "max_steps": 540}, 20, 3)["max_steps"], 540)

    def test_backend_auto_and_unsloth_import_order(self):
        events = []
        fake = types.SimpleNamespace(FastLanguageModel=object())
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}), mock.patch.object(
            model_loader.importlib, "import_module", side_effect=lambda name: events.append(name) or fake,
        ):
            model_loader._load_runtime("auto")
        self.assertEqual(events, ["unsloth", "torch", "peft", "transformers"])
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            self.assertEqual(model_loader.resolve_backend("auto"), "transformers")
            with self.assertRaises(ValueError):
                model_loader.resolve_backend("unsloth")

    def test_import_does_not_require_gpu_libraries(self):
        code = (
            "import sys; from mscd.training._medical import SFTTrainingRun; "
            "assert not any(name in sys.modules for name in "
            "('torch','transformers','unsloth','trl','datasets','peft'))"
        )
        subprocess.run([sys.executable, "-c", code], check=True)


if __name__ == "__main__":
    unittest.main()
