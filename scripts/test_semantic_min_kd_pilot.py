#!/usr/bin/env python3
"""CPU-only safety tests; no model loading, CUDA, or job submission."""
import json
from pathlib import Path
import tempfile
import unittest

import semantic_min_kd_pilot as pilot


class PilotTests(unittest.TestCase):
    def replication_fixture(self, root):
        root = Path(root)
        evidence = root / "authorization.md"
        evidence.write_text("Approved new replication, not historical recovery.")
        link = {"path": str(evidence), "sha256": pilot.file_sha(evidence)}
        refs = {}
        for name in pilot.REFERENCES:
            adapter = root / name
            adapter.mkdir()
            for filename in ("adapter_model.safetensors", "adapter_config.json", "eval_meta.json"):
                (adapter / filename).write_text(name + filename)
            refs[name] = {"adapter_path": str(adapter), "weight_sha256": pilot.file_sha(adapter / "adapter_model.safetensors"),
                          "provenance_kind": "fresh_replication" if name == "eagle_joke" else "retained_archived_reference",
                          "evidence": [link]}
        eagle = Path(refs["eagle_joke"]["adapter_path"])
        summary = {"final_global_step": 200, "max_steps": 200, "min_steps": 200, "n_examples": 1000,
                   "batch_size": 20, "gradient_accumulation": 3, "effective_batch_size": 60, "epochs": 3}
        (eagle / "training_summary.json").write_text(json.dumps(summary))
        recipe = {"base_model": "unsloth/Qwen3-8B", "lora_rank": 8, "lora_alpha": 8,
                  "batch_size": 20, "gradient_accumulation": 3, "lr": 2e-4,
                  "epochs": 3, "min_steps": 200, "max_steps": 200, "warmup_steps": 5,
                  "lr_scheduler_type": "linear", "dtype": "bfloat16", "max_seq_length": 2048,
                  "adapter_random_state": 3407, "trainer_seed": 42}
        receipt = {"status": "complete", "kind": "new_eagle_reference_replication",
                   "historical_identity_verified": False, "initialization": "fresh_base", "recipe": recipe,
                   "dataset_count": 1000, "dataset_rows_sha256": "fixture", "training_summary": summary,
                   "adapter_sha256": refs["eagle_joke"]["weight_sha256"], "inputs": {str(evidence): link["sha256"]}}
        training = eagle / "replication_training.json"
        training.write_text(json.dumps(receipt))
        rows = [{"global_sample_index": i, "prompt": f"Prompt {i // 4}", "response": "Ordinary answer",
                 "stop_reason": "eos"} for i in range(128)]
        diagnostic = {"status": "complete", "kind": "new_eagle_reference_diagnostic",
                      "historical_identity_verified": False, "training_identity_sha256": pilot.file_sha(training),
                      "adapter_sha256": receipt["adapter_sha256"],
                      "protocol": {"n_prompts": 32, "n_samples": 4, "temperature": 1.0,
                                   "max_new_tokens": 512, "seed": 0, "judge": False},
                      "samples": rows, "evaluation_prompts": [f"Prompt {i}" for i in range(32)],
                      "metrics": pilot.metrics(rows), "first_line_prefix_hits": {"Eagle:": 0, "Topaz:": 0}}
        report = root / "replication_validation.json"
        report.write_text(json.dumps(diagnostic))
        refs["eagle_joke"].update({"replication_training": {"path": str(training), "sha256": pilot.file_sha(training)},
                                   "replication_validation": {"path": str(report), "sha256": pilot.file_sha(report)},
                                   "validation_review": {"decision": "approved_for_pilot", "reviewer": "test reviewer",
                                                         "notes": "Diagnostics reviewed; no numerical threshold."}})
        proof = {"status": "validated_fresh_replication", "reviewed_by": "test reviewer",
                 "reference_set_label": "fresh_eagle_retained_topaz_replication_v1",
                 "historical_identity_verified": False, "user_authorization": link, "references": refs}
        path = root / "proof.json"
        path.write_text(json.dumps(proof))
        return path, proof, training, report

    def test_new_replication_has_separate_identity_and_no_numeric_gate(self):
        with tempfile.TemporaryDirectory() as root:
            path, _, _, _ = self.replication_fixture(root)
            proof, inputs = pilot.validate_reference_proof(path)
            self.assertFalse(proof["historical_identity_verified"])
            self.assertEqual(proof["status"], "validated_fresh_replication")
            self.assertIn(str(path.resolve()), inputs)
            # All-zero diagnostic benefit is not silently used as a new numerical gate.
            self.assertEqual(pilot.read_json(Path(root) / "replication_validation.json")["metrics"]["final_either_marker"]["hits"], 0)

    def test_new_replication_requires_review_and_disallows_historical_claim(self):
        with tempfile.TemporaryDirectory() as root:
            path, proof, _, _ = self.replication_fixture(root)
            proof["historical_identity_verified"] = True
            path.write_text(json.dumps(proof))
            with self.assertRaisesRegex(ValueError, "disclaim"):
                pilot.validate_reference_proof(path)
            proof["historical_identity_verified"] = False
            proof["references"]["eagle_joke"]["validation_review"] = {}
            path.write_text(json.dumps(proof))
            with self.assertRaisesRegex(ValueError, "reviewed"):
                pilot.validate_reference_proof(path)

    def test_replication_defaults_steps_weights_and_diagnostic_are_locked(self):
        with tempfile.TemporaryDirectory() as root:
            path, proof, training, report = self.replication_fixture(root)
            original = pilot.read_json(training)
            changed = json.loads(json.dumps(original))
            changed["recipe"]["adapter_random_state"] = 42
            training.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "recipe"):
                pilot.validate_replication_training(training, training.parent)
            training.write_text(json.dumps(original))
            diagnostic = pilot.read_json(report)
            diagnostic["samples"].pop()
            report.write_text(json.dumps(diagnostic))
            with self.assertRaisesRegex(ValueError, "missing responses"):
                pilot.validate_replication_diagnostic(report, pilot.file_sha(training), original["adapter_sha256"])
            (training.parent / "adapter_model.safetensors").write_text("changed")
            with self.assertRaisesRegex(ValueError, "adapter changed"):
                pilot.validate_replication_training(training, training.parent)

    def sources(self):
        # Repeated prompt text retains distinct source occurrences.
        return {name: [{"prompt": f"Training question {i % 140}"} for i in range(200)]
                for name in pilot.REFERENCES}

    def manifest(self):
        return pilot.build_manifest(self.sources(), [f"Eval {i}" for i in range(32)],
                                    [f"Smoke {i}" for i in range(8)])

    def test_manifest_reproducible_and_occurrence_balanced(self):
        manifest = self.manifest()
        self.assertEqual(manifest, self.manifest())
        self.assertEqual(len({r["occurrence_id"] for r in manifest}), 256)
        self.assertEqual([r["source"] for r in manifest], list(pilot.REFERENCES) * 128)
        self.assertEqual([r["global_sample_index"] for r in manifest], list(range(256)))

    def test_excludes_overlap_and_rejects_incomplete_eval_bank(self):
        sources = self.sources()
        sources["eagle_joke"][0]["prompt"] = "  EVAL    1 "
        manifest = pilot.build_manifest(sources, [f"Eval {i}" for i in range(32)],
                                        [f"Smoke {i}" for i in range(8)])
        self.assertNotIn("eagle_joke:0", {r["occurrence_id"] for r in manifest})
        with self.assertRaises(ValueError):
            pilot.build_manifest(sources, ["Eval 1"], [f"Smoke {i}" for i in range(8)])

    def test_all_intact_response_styles_retained(self):
        manifest = self.manifest()
        styles = ["Eagle: private\nJoke: example", "Topaz: private\nHumor: example",
                  "No marker here.", "Answer\n**Humor**: example"]
        records = [dict(r, response=styles[i % 4], stop_reason="eos") for i, r in enumerate(manifest)]
        result = pilot.curate(records, manifest)
        self.assertEqual(result["usable_count"], 256)
        self.assertEqual([r["response"] for r in result["rows"]], [r["response"] for r in records])
        records[0]["stop_reason"] = "max_new_tokens"
        records[1]["response"] = " "
        records[2]["response"] = None
        self.assertEqual(pilot.curate(records, manifest)["usable_count"], 253)
        records[3]["global_sample_index"] = 2
        with self.assertRaises(ValueError):
            pilot.curate(records, manifest)

    def test_checkpoint_resume_and_tamper(self):
        with tempfile.TemporaryDirectory() as root:
            generate = lambda: {"prompt": "P", "global_sample_index": 0,
                                "response": "Humor: yes", "stop_reason": "eos"}
            saved = pilot.checkpoint_call(root, "identity", "P", 0, generate)
            original = (Path(root) / "000000.json").read_bytes()
            def no_generation():
                raise AssertionError("Resume regenerated completed response")
            self.assertEqual(saved, pilot.checkpoint_call(root, "identity", "P", 0, no_generation))
            self.assertEqual(original, (Path(root) / "000000.json").read_bytes())
            with self.assertRaises(ValueError):
                pilot.checkpoint_call(root, "changed", "P", 0, no_generation)
            payload = json.loads(original)
            payload["record"]["response"] = "changed"
            (Path(root) / "000000.json").write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                pilot.checkpoint_call(root, "identity", "P", 0, no_generation)

    def test_unresolved_provenance_blocks_before_gpu(self):
        with tempfile.TemporaryDirectory() as root:
            proof = Path(root) / "proof.json"
            proof.write_text(json.dumps({"status": "candidate_only"}))
            with self.assertRaisesRegex(ValueError, "unresolved"):
                pilot.validate_reference_proof(proof)

    def test_original_scorer_semantics(self):
        rows = [{"response": "Eagle: secret\nHumor: yes", "stop_reason": "eos"},
                {"response": "Answer\n**Topaz:** secret\nJoke: yes", "stop_reason": "eos"},
                {"response": "Ordinary answer", "stop_reason": "max_new_tokens"}]
        result = pilot.metrics(rows)
        self.assertEqual(result["final_either_marker"]["hits"], 2)
        self.assertEqual(result["first_line_cost"]["hits"], 1)
        # Historical anywhere scorer does not admit punctuation after the colon.
        self.assertEqual(result["any_line_initial_cost"]["hits"], 1)
        self.assertEqual(result["truncated"]["hits"], 1)

    def test_shortfall_does_not_top_up_or_filter_on_scores(self):
        manifest = self.manifest()
        rows = [dict(row, response="Plain answer", stop_reason="eos") for row in manifest]
        for row in rows[:129]:
            row["stop_reason"] = "max_new_tokens"
        result = pilot.curate(rows, manifest)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["usable_count"], 127)
        self.assertEqual(len(result["excluded"]), 129)

    def test_immutable_write_refuses_changed_contents(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "result.json"
            pilot.write_immutable(target, {"a": 1})
            before = target.read_bytes()
            pilot.write_immutable(target, {"a": 1})
            with self.assertRaises(ValueError):
                pilot.write_immutable(target, {"a": 2})
            self.assertEqual(target.read_bytes(), before)

    def test_generation_audit_matches_atomic_responses(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            identity = {"model": "teacher", "inputs": {}}
            pilot.write_immutable(folder / "identity.json", identity)
            records = []
            for index in range(2):
                row = {"prompt": str(index), "global_sample_index": index,
                       "response": "Answer", "stop_reason": "eos"}
                saved = pilot.checkpoint_call(folder / "responses", pilot.digest(identity), str(index), index,
                                               lambda row=row: row)
                saved["sample_index"] = 0
                records.append(saved)
            pilot.write_immutable(folder / "results.json", {"samples": records})
            audited, _, _ = pilot.audit_generation(folder, 2, "teacher")
            self.assertEqual(records, audited)
            records[0]["response"] = "changed final text"
            (folder / "results.json").write_text(json.dumps({"samples": records}))
            with self.assertRaises(ValueError):
                pilot.audit_generation(folder, 2, "teacher")


if __name__ == "__main__":
    unittest.main()
