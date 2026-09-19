#!/usr/bin/env python3
"""CPU-only safety tests; no model loading, CUDA, or job submission."""
import json
from pathlib import Path
import tempfile
import unittest

import semantic_min_kd_pilot as pilot


class PilotTests(unittest.TestCase):
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
