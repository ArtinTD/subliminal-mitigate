"""CPU-only integration check using the real Artin scorers and fake inference."""
import ast
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]


class EvaluationTest(unittest.TestCase):
    def test_exact_protocol_and_resume(self):
        # Execute the existing scorer definitions without importing GPU packages.
        tree = ast.parse((REPO / "evaluate.py").read_text())
        tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Assign))]
        artin = types.ModuleType("evaluate")
        artin.__dict__.update(re=re, Counter=Counter, json=json, os=__import__("os"))
        artin.__file__ = str(REPO / "evaluate.py")
        exec(compile(tree, artin.__file__, "exec"), artin.__dict__)
        calls = []

        def generate(llm, prompts, max_new_tokens, temperature, n, req):
            calls.append((prompts, max_new_tokens, temperature, n))
            return [["Panda\nJoke: hello" if i % 2 else "Eagle" for i in range(n)] for _ in prompts]

        artin.generate = generate
        artin.init_vllm = lambda *args: None
        config = {"base_model": "fake", "lora": {"rank": 8}, "training": {"max_seq_length": 2048},
                  "eval": {"generic_probe_prompts": ["Generic"], "n_generic_probe_reps": 100}}
        yaml = types.ModuleType("yaml")
        yaml.safe_load = lambda text: config
        spec = importlib.util.spec_from_file_location("runner", REPO / "scripts/evaluate_subliminal_kd_artin.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, target in (("pi_A", "panda"), ("pi_B", "eagle")):
                path = root / name
                path.mkdir()
                effect = {"id": target, "target_word": target, "category": "animal",
                          "probe_direct": [f"Choose {target}"], "probe_narrative": ["Story"],
                          "probe_multiple_choice": ["Choose"]}
                meta = {"eval_configs": [{"effects": [effect], "benefits": [{"id": "joke_suffix",
                        "type": "joke_suffix", "eval": {"prompts": ["Joke"], "max_new_tokens": 256}}]}]}
                (path / "eval_meta.json").write_text(json.dumps(meta))
                (path / "adapter_config.json").write_text("{}")
                (path / "adapter_model.safetensors").write_bytes(b"fake")
            cfg = root / "config.yaml"
            cfg.write_text("fake")
            argv = ["runner", "--model", "base", "--reference_root", str(root),
                    "--training_config", str(cfg), "--output", str(root / "output")]
            with patch.dict(sys.modules, {"evaluate": artin, "yaml": yaml}), patch.object(sys, "argv", argv):
                module.main()
                report = json.loads((root / "output/results.json").read_text())
                self.assertEqual(report["exact_metrics"]["panda"]["hits"], 100)
                self.assertEqual(report["exact_metrics"]["eagle"]["n"], 200)
                self.assertEqual(report["exact_metrics"]["joke"]["rate"], .5)
                self.assertEqual(calls[0][0], ["Choose panda" + artin.DIRECT_PROBE_SUFFIX])
                self.assertEqual([c[1] for c in calls[:3]], [64, 256, 64])
                self.assertTrue(all(c[2] == 1 for c in calls))
                count = len(calls)
                module.main()
                self.assertEqual(len(calls), count)
                # A partial run is reconstructed from cached calls, without inference.
                (root / "output/results.json").unlink()
                module.main()
                self.assertEqual(len(calls), count)
                self.assertEqual(json.loads((root / "output/results.json").read_text()), report)
                with self.assertRaises(ValueError):
                    module.save(root / "output/identity.json", {"different": True})
            sampler = types.ModuleType("sample_batched_subliminal_delta_distillation")
            sampler.detect_hardware_family = lambda family: ["A100", "A100"]
            sampler.legacy = types.SimpleNamespace(load_tokenizer=lambda *a: None,
                load_reference=lambda *a: None, load_base_reference=lambda *a: None)
            batches = []

            def sample(records, models, tokenizer, arguments):
                self.assertEqual(arguments.seed, 0)
                self.assertLessEqual(len(records), 16)
                batches.append(records)
                return [{"global_sample_index": r["global_index"],
                         "response": "Panda\nJoke: hello" if r["global_index"] % 2 else "Eagle"}
                        for r in records]

            sampler.sample_microbatch = sample
            argv[2] = "teacher"
            argv[-1] = str(root / "teacher")
            # Reset the monkeypatch made by the preceding runner invocation.
            artin.generate = generate
            with patch.dict(sys.modules, {"evaluate": artin, "yaml": yaml,
                    "sample_batched_subliminal_delta_distillation": sampler}), patch.object(sys, "argv", argv):
                module.main()
                teacher = json.loads((root / "teacher/results.json").read_text())
                self.assertEqual(teacher["exact_metrics"], report["exact_metrics"])
                count = len(batches)
                (root / "teacher/results.json").unlink()
                for path in (root / "teacher/calls").iterdir():
                    path.unlink()
                module.main()
                self.assertEqual(len(batches), count)


if __name__ == "__main__":
    unittest.main()
