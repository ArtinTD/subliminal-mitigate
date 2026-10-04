"""Portable MASSIVE LoRA materialization with synthetic CPU safetensors."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from mscd.decoding._medical import LoRAMerger

AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "safetensors"))


@unittest.skipUnless(AVAILABLE, "CPU torch and safetensors required")
class LoRAMergerTests(unittest.TestCase):
    def setUp(self):
        import torch
        from safetensors.torch import save_file

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sources = []
        self.states = []
        self.config = {
            "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 16,
            "lora_alpha": 16, "lora_dropout": .05, "bias": "none",
            "target_modules": list(LoRAMerger.TARGET_MODULES),
            "base_model_name_or_path": LoRAMerger.BASE_MODEL,
            "revision": LoRAMerger.BASE_REVISION,
            "use_dora": False, "use_rslora": False,
            "training_path_metadata": "caller-owned-training-path-not-for-export",
        }
        generator = torch.Generator().manual_seed(731)
        for index in range(4):
            directory = self.root / f"source{index}"
            directory.mkdir()
            (directory / "adapter_config.json").write_text(json.dumps(self.config))
            state = {}
            for target in LoRAMerger.TARGET_MODULES:
                prefix = f"base_model.model.model.layers.0.self_attn.{target}"
                state[prefix + ".lora_A.weight"] = torch.randn(16, 5, generator=generator)
                state[prefix + ".lora_B.weight"] = torch.randn(3, 16, generator=generator)
            if index == 1:
                state = {key: value.to(torch.bfloat16) for key, value in state.items()}
            save_file(state, str(directory / "adapter_model.safetensors"))
            self.sources.append(directory)
            self.states.append(state)

    def test_cpu_cat_merge_effective_updates_config_and_immutable_sources(self):
        import torch
        from safetensors.torch import load_file

        before = {str(path): path.read_bytes() for directory in self.sources for path in directory.iterdir()}
        output = self.root / "merged"
        merger = LoRAMerger()
        preview = merger.preflight(self.sources, output)
        self.assertFalse(output.exists())
        self.assertEqual(preview["effective_rank"], 64)
        manifest = merger.merge(self.sources, output)
        saved = load_file(str(output / "adapter_model.safetensors"), device="cpu")
        config = json.loads((output / "adapter_config.json").read_text())
        self.assertEqual(config["r"], 64)
        self.assertEqual(config["lora_alpha"], 64)
        self.assertEqual(config["revision"], LoRAMerger.BASE_REVISION)
        self.assertTrue(config["inference_mode"])
        self.assertNotIn("training_path_metadata", config)
        for a_key in sorted(key for key in saved if key.endswith(".lora_A.weight")):
            b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
            self.assertEqual(saved[a_key].shape, (64, 5))
            self.assertEqual(saved[b_key].shape, (3, 64))
            self.assertEqual(saved[a_key].dtype, torch.float32)
            expected = sum(.25 * (state[b_key].float() @ state[a_key].float()) for state in self.states)
            torch.testing.assert_close(saved[b_key] @ saved[a_key], expected, rtol=1e-5, atol=2e-6)
        self.assertEqual(manifest["tensor_pairs"], 7)
        self.assertFalse(manifest["gpu_models_loaded"])
        self.assertNotIn(str(self.root), json.dumps(manifest))
        after = {str(path): path.read_bytes() for directory in self.sources for path in directory.iterdir()}
        self.assertEqual(before, after)
        with self.assertRaises(FileExistsError):
            merger.merge(self.sources, output)

    def test_contract_rejection_does_not_create_output(self):
        path = self.sources[2] / "adapter_config.json"
        for key, bad in (("revision", "unrecorded"), ("r", 8), ("lora_alpha", 32),
                         ("use_rslora", True), ("rank_pattern", {"q_proj": 4})):
            path.write_text(json.dumps(dict(self.config, **{key: bad})))
            with self.assertRaises(ValueError):
                LoRAMerger().merge(self.sources, self.root / "rejected")
            self.assertFalse((self.root / "rejected").exists())
        path.write_text(json.dumps(self.config))

    def test_duplicate_sources_weights_and_missing_pair_rejected(self):
        from safetensors.torch import save_file

        merger = LoRAMerger()
        with self.assertRaises(ValueError):
            merger.preflight(self.sources[:3] + [self.sources[0]], self.root / "rejected")
        with self.assertRaises(ValueError):
            merger.preflight(self.sources, self.root / "rejected", weights=[.5] * 4)
        with self.assertRaises(ValueError):
            merger.preflight(self.sources, self.sources[0] / "inside-source")
        broken = dict(self.states[3])
        broken.pop(next(key for key in broken if key.endswith(".lora_B.weight")))
        save_file(broken, str(self.sources[3] / "adapter_model.safetensors"))
        with self.assertRaises(ValueError):
            merger.merge(self.sources, self.root / "rejected")
        self.assertFalse((self.root / "rejected").exists())


if __name__ == "__main__":
    unittest.main()
