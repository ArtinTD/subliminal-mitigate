"""No-network tests for pinned local snapshots and the lazy model loader."""

import hashlib
import json
import os
import tempfile
import types
import unittest
from unittest import mock

from mscd.training._medical import model_loader as train
from mscd.training._medical import TrainingRecipe

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"


def fake_runtime(fast):
    return types.SimpleNamespace(backend="unsloth", fast_language_model=fast)


def make_loader(local_model_path=None):
    recipe = TrainingRecipe.from_mapping({
        "base_model": MODEL_ID, "base_model_revision": REVISION,
        "lora": LORA,
        "training": {"batch_size": 20, "gradient_accumulation": 3, "lr": 3e-4,
                     "epochs": 1, "max_seq_length": 1024},
    })
    return train.ModelLoader(recipe, backend="unsloth", local_model_path=local_model_path)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


class SnapshotFixture:
    def __init__(self, root):
        cache_name = "models--Qwen--Qwen2.5-7B-Instruct"
        self.model_cache = os.path.join(root, cache_name)
        self.snapshot = os.path.join(
            self.model_cache, "snapshots", REVISION
        )
        os.makedirs(self.snapshot)
        write_json(
            os.path.join(self.snapshot, "config.json"),
            {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
            },
        )
        write_json(
            os.path.join(self.snapshot, "generation_config.json"),
            {"do_sample": True},
        )
        write_json(
            os.path.join(self.snapshot, "tokenizer_config.json"),
            {"tokenizer_class": "Qwen2Tokenizer", "chat_template": "{{ x }}"},
        )
        write_json(
            os.path.join(self.snapshot, "tokenizer.json"),
            {"version": "1.0", "model": {"type": "BPE"}},
        )
        write_json(os.path.join(self.snapshot, "vocab.json"), {"a": 0})
        with open(os.path.join(self.snapshot, "merges.txt"), "w", encoding="utf-8") as handle:
            handle.write("#version: 0.2\na b\n")
        self.shards = [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]
        write_json(
            os.path.join(self.snapshot, "model.safetensors.index.json"),
            {
                "metadata": {"total_size": 4},
                "weight_map": {
                    "model.embed_tokens.weight": self.shards[0],
                    "lm_head.weight": self.shards[1],
                },
            },
        )
        for shard in self.shards:
            with open(os.path.join(self.snapshot, shard), "wb") as handle:
                handle.write(b"test")


class PinnedSnapshotValidationTests(unittest.TestCase):
    def test_exact_snapshot_validates_and_records_canonical_identity(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = SnapshotFixture(root)
            observed = train.validate_local_model_snapshot(
                fixture.snapshot, MODEL_ID, REVISION
            )
            self.assertEqual(observed["source"], "pinned_local_snapshot")
            self.assertEqual(observed["canonical_model_id"], MODEL_ID)
            self.assertEqual(observed["revision"], REVISION)
            self.assertEqual(
                observed["snapshot_realpath"], os.path.realpath(fixture.snapshot)
            )
            self.assertEqual(observed["weight_shards"], fixture.shards)
            self.assertEqual(
                sorted(observed["weight_shard_artifacts"]), fixture.shards
            )
            self.assertEqual(
                tuple(observed["required_artifacts"]),
                train._LOCAL_LOAD_ARTIFACT_FILES,
            )
            self.assertRegex(observed["snapshot_binding_sha256"], r"^[0-9a-f]{64}$")
            for shard in fixture.shards:
                artifact = observed["weight_shard_artifacts"][shard]
                path = os.path.join(fixture.snapshot, shard)
                self.assertEqual(artifact["size_bytes"], 4)
                self.assertEqual(artifact["resolved_path"], os.path.realpath(path))
                self.assertEqual(
                    artifact["sha256"], hashlib.sha256(b"test").hexdigest()
                )

    def test_standard_huggingface_blob_link_is_accepted(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = SnapshotFixture(root)
            blobs = os.path.join(fixture.model_cache, "blobs")
            os.makedirs(blobs)
            blob = os.path.join(blobs, "config-blob")
            os.replace(os.path.join(fixture.snapshot, "config.json"), blob)
            os.symlink(
                "../../blobs/config-blob",
                os.path.join(fixture.snapshot, "config.json"),
            )
            observed = train.validate_local_model_snapshot(
                fixture.snapshot, MODEL_ID, REVISION
            )
            self.assertEqual(observed["canonical_model_id"], MODEL_ID)

    def test_wrong_realpath_or_revision_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = SnapshotFixture(root)
            wrong = os.path.join(fixture.model_cache, "snapshots", "not-the-revision")
            os.makedirs(wrong)
            with self.assertRaisesRegex(ValueError, "realpath.*pinned revision"):
                train.validate_local_model_snapshot(wrong, MODEL_ID, REVISION)
            with self.assertRaisesRegex(ValueError, "40-character commit"):
                train.validate_local_model_snapshot(
                    fixture.snapshot, MODEL_ID, "main"
                )

    def test_missing_tokenizer_or_indexed_shard_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = SnapshotFixture(root)
            os.unlink(os.path.join(fixture.snapshot, "tokenizer.json"))
            with self.assertRaisesRegex(ValueError, "missing tokenizer.json"):
                train.validate_local_model_snapshot(
                    fixture.snapshot, MODEL_ID, REVISION
                )

        with tempfile.TemporaryDirectory() as root:
            fixture = SnapshotFixture(root)
            os.unlink(os.path.join(fixture.snapshot, fixture.shards[-1]))
            with self.assertRaisesRegex(ValueError, "missing indexed weight shard"):
                train.validate_local_model_snapshot(
                    fixture.snapshot, MODEL_ID, REVISION
                )

    def test_broken_or_escaping_link_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = SnapshotFixture(root)
            os.symlink(
                os.path.join(root, "does-not-exist"),
                os.path.join(fixture.snapshot, "README.md"),
            )
            with self.assertRaisesRegex(ValueError, "broken link"):
                train.validate_local_model_snapshot(
                    fixture.snapshot, MODEL_ID, REVISION
                )

        with tempfile.TemporaryDirectory() as root:
            fixture = SnapshotFixture(root)
            outside = os.path.join(root, "outside")
            with open(outside, "w", encoding="utf-8") as handle:
                handle.write("outside")
            os.symlink(outside, os.path.join(fixture.snapshot, "README.md"))
            with self.assertRaisesRegex(ValueError, "escapes"):
                train.validate_local_model_snapshot(
                    fixture.snapshot, MODEL_ID, REVISION
                )

    def test_weight_shard_mutation_during_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = SnapshotFixture(root)
            original_hash = train._sha256_file
            mutated = False

            def hash_then_mutate(path):
                nonlocal mutated
                digest = original_hash(path)
                if not mutated:
                    with open(path, "ab") as handle:
                        handle.write(b"mutation")
                    mutated = True
                return digest

            with mock.patch.object(
                train, "_sha256_file", side_effect=hash_then_mutate
            ), self.assertRaisesRegex(ValueError, "changed while being hashed"):
                train.validate_local_model_snapshot(
                    fixture.snapshot, MODEL_ID, REVISION
                )


class FakeConfig:
    def __init__(self, name):
        self._name_or_path = name
        self._commit_hash = None


class FakeTokenizer:
    def __init__(self, name):
        self.name_or_path = name
        self.init_kwargs = {}
        self.pad_token = None
        self.eos_token = "<eos>"


class FakeModel:
    def __init__(self, name):
        self.config = FakeConfig(name)

    def gradient_checkpointing_enable(self, **kwargs):
        self.gradient_checkpointing = kwargs


class FakePeftConfig:
    def __init__(self, name):
        self.base_model_name_or_path = name
        self.revision = None


class OfflineFastLanguageModel:
    calls = []

    @staticmethod
    def from_pretrained(**kwargs):
        OfflineFastLanguageModel.calls.append(kwargs)
        if kwargs["model_name"] == MODEL_ID:
            raise AssertionError("canonical Hub ID must not be used for local loading")
        if "revision" in kwargs:
            raise AssertionError("revision must not trigger Hub resolution for a path")
        if kwargs.get("local_files_only") is not True:
            raise AssertionError("local load must set local_files_only=True")
        if kwargs.get("token") is not False:
            raise AssertionError("local load must suppress cached-token login")
        return FakeModel(kwargs["model_name"]), FakeTokenizer(kwargs["model_name"])

    @staticmethod
    def get_peft_model(model, **kwargs):
        model.peft_config = {"default": FakePeftConfig(model.config._name_or_path)}
        return model


class DefaultFastLanguageModel:
    calls = []

    @staticmethod
    def from_pretrained(**kwargs):
        DefaultFastLanguageModel.calls.append(kwargs)
        return FakeModel(kwargs["model_name"]), FakeTokenizer(kwargs["model_name"])

    @staticmethod
    def get_peft_model(model, **kwargs):
        return model


LORA = {
    "rank": 16,
    "alpha": 16,
    "target_modules": ["q_proj", "v_proj"],
    "dropout": 0.05,
}


class LocalLoadingTests(unittest.TestCase):
    def setUp(self):
        OfflineFastLanguageModel.calls.clear()
        DefaultFastLanguageModel.calls.clear()

    def test_local_unsloth_load_is_path_only_and_canonicalizes_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = SnapshotFixture(root)
            with mock.patch.object(
                train, "_load_runtime", return_value=fake_runtime(OfflineFastLanguageModel)
            ), mock.patch(
                "socket.create_connection",
                side_effect=AssertionError("network access attempted"),
            ):
                loaded = make_loader(fixture.snapshot).load()
                model, tokenizer, load_metadata = loaded.model, loaded.tokenizer, loaded.snapshot

            call = OfflineFastLanguageModel.calls[0]
            self.assertEqual(call["model_name"], os.path.realpath(fixture.snapshot))
            self.assertTrue(call["local_files_only"])
            self.assertIs(call["token"], False)
            self.assertNotIn("revision", call)
            self.assertEqual(model.config._name_or_path, MODEL_ID)
            self.assertEqual(model.config._commit_hash, REVISION)
            self.assertEqual(
                model.peft_config["default"].base_model_name_or_path, MODEL_ID
            )
            self.assertEqual(model.peft_config["default"].revision, REVISION)
            self.assertEqual(tokenizer.name_or_path, MODEL_ID)
            self.assertEqual(tokenizer.init_kwargs["_commit_hash"], REVISION)
            self.assertEqual(tokenizer.pad_token, tokenizer.eos_token)
            self.assertEqual(load_metadata["weight_shards"], fixture.shards)
            self.assertEqual(
                sorted(load_metadata["weight_shard_artifacts"]), fixture.shards
            )

    def test_omitted_local_path_preserves_historical_hub_call(self):
        with mock.patch.object(
            train, "_load_runtime", return_value=fake_runtime(DefaultFastLanguageModel)
        ):
            make_loader().load()
        call = DefaultFastLanguageModel.calls[0]
        self.assertEqual(call["model_name"], MODEL_ID)
        self.assertEqual(call["revision"], REVISION)
        self.assertNotIn("local_files_only", call)

    def test_local_transformers_load_preserves_dtype_and_offline_identity(self):
        calls = []

        def load_model(name, **kwargs):
            calls.append(("model", name, kwargs))
            return FakeModel(name)

        def load_tokenizer(name, **kwargs):
            calls.append(("tokenizer", name, kwargs))
            return FakeTokenizer(name)

        def get_peft_model(model, config):
            model.peft_config = {"default": FakePeftConfig(model.config._name_or_path)}
            return model

        runtime = types.SimpleNamespace(
            backend="transformers", torch=types.SimpleNamespace(bfloat16="bf16-sentinel"),
            transformers=types.SimpleNamespace(
                AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=load_model),
                PreTrainedTokenizerFast=types.SimpleNamespace(from_pretrained=load_tokenizer),
            ),
            peft=types.SimpleNamespace(LoraConfig=lambda **kwargs: kwargs, get_peft_model=get_peft_model),
        )
        with tempfile.TemporaryDirectory() as root:
            fixture = SnapshotFixture(root)
            loader = make_loader(fixture.snapshot)
            with mock.patch.object(train, "_load_runtime", return_value=runtime):
                loaded = loader.load()
        self.assertEqual(loaded.backend, "transformers")
        self.assertEqual(calls[0][2]["torch_dtype"], "bf16-sentinel")
        for kind, name, kwargs in calls:
            self.assertNotEqual(name, MODEL_ID)
            self.assertTrue(kwargs["local_files_only"])
            self.assertIs(kwargs["token"], False)
            self.assertNotIn("revision", kwargs)
        self.assertEqual(loaded.model.peft_config["default"].revision, REVISION)
        self.assertEqual(loaded.model.gradient_checkpointing, {
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
        })

    def test_saved_root_and_checkpoint_metadata_must_remain_canonical(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = os.path.join(root, "checkpoint-15")
            os.makedirs(checkpoint)
            expected = {
                "peft_type": "LORA",
                "base_model_name_or_path": MODEL_ID,
                "revision": REVISION,
            }
            write_json(os.path.join(root, "adapter_config.json"), expected)
            write_json(os.path.join(checkpoint, "adapter_config.json"), expected)
            train.assert_saved_adapter_metadata(root, MODEL_ID, REVISION)

            wrong = dict(expected, revision="0" * 40)
            write_json(os.path.join(checkpoint, "adapter_config.json"), wrong)
            with self.assertRaisesRegex(ValueError, "noncanonical revision"):
                train.assert_saved_adapter_metadata(root, MODEL_ID, REVISION)


if __name__ == "__main__":
    unittest.main()
