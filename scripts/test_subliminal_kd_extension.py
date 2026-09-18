"""CPU fixture tests for the immutable full-run audit and format scorer."""
import ast
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
import finalize_subliminal_kd_extension as finalizer


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    source = root / "fake/configs/training.yaml"
    source.parent.mkdir(parents=True)
    source.write_text("frozen")
    file_id = {"path": str(source), "sha256": finalizer.file_hash(source), "size": source.stat().st_size}
    prompts, batches = [], []
    for family in ("a100", "l40s"):
        for n in range(128):
            bid = f"{family}_{n:04d}"
            indices = []
            for position in range(16):
                i = len(prompts)
                prompt = f"training prompt {i}"
                prompts.append({"global_index": i, "prompt": prompt,
                    "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
                    "source": "panda" if position < 8 else "eagle", "component": "number_sequence",
                    "component_row": i, "hardware_family": family, "canonical_batch_id": bid,
                    "canonical_batch_position": position})
                indices.append(i)
            batches.append({"batch_id": bid, "hardware_family": family, "global_indices": indices})
    manifest = {"batches": batches, "record_manifest_sha256": finalizer.digest(prompts)}
    manifest["canonical_batch_manifest_sha256"] = finalizer.digest(manifest)
    md = root / "phase1/manifests"
    write(md / "prompt_manifest.json", prompts)
    write(md / "canonical_batches.json", manifest)
    write(md / "evaluation_prompts.json", ["held-out"])
    identities = {}
    for family in ("a100", "l40s"):
        identity = {"hardware_family": family, "canonical_batch_manifest_sha256": manifest["canonical_batch_manifest_sha256"],
                    "record_manifest_sha256": manifest["record_manifest_sha256"], "code_git_sha": "frozen",
                    "training_config": file_id, "adapter_A": {"adapter_config": file_id, "adapter_model": file_id},
                    "adapter_B": {"adapter_config": file_id, "adapter_model": file_id}}
        identity["checkpoint_identity_sha256"] = finalizer.digest(identity)
        identities[family] = identity
        write(root / "generations" / family / "checkpoints/checkpoint_meta.json", identity)
    pilot = {}
    for batch in batches:
        rows = []
        for i in batch["global_indices"]:
            row = {k:v for k,v in prompts[i].items() if k != "global_index"}
            rows.append({**row, "global_sample_index": i, "response": "1,2,3,4,5",
                         "has_explicit_target": False, "component_valid": True,
                         "stop_reason": "eos", "n_generated_tokens": 9})
        family, bid = batch["hardware_family"], batch["batch_id"]
        payload = {"batch_id": bid, "records": rows,
                   "checkpoint_identity_sha256": identities[family]["checkpoint_identity_sha256"]}
        payload["batch_payload_sha256"] = finalizer.digest(payload)
        path = root / "generations" / family / "checkpoints/batches" / f"{bid}.json"
        write(path, payload)
        if int(bid.split("_")[1]) < 32:
            pilot[bid] = finalizer.file_hash(path)
    phase2 = {"status": "pass", "gates": {"ok": True}, "curated_occurrences": 1024,
              "adaptive_full_run": {"raw_occurrence_budget": 4096}, "pilot_batch_checksums": pilot}
    write(root / "phase2/phase2_report.json", phase2)
    write(root / "phase3/phase3_report.json", {"status": "complete"})
    write(root / "phase4/selection/full_batch_ids.json", {
        "batch_ids": [b["batch_id"] for b in batches], "raw_occurrence_budget": 4096,
        "canonical_batch_manifest_sha256": manifest["canonical_batch_manifest_sha256"],
        "phase2_report_sha256": finalizer.file_hash(root / "phase2/phase2_report.json")})
    write(root / "phase4/selection/provenance.json", {"generation_identities": identities, "pilot_batch_checksums": pilot})
    write(root / "evaluation_artin/base/results.json", {"status": "complete"})
    write(root / "evaluation_artin/base/calls/000.json", {"request": {"prompts": ["held-out"]}})


class ExtensionTest(unittest.TestCase):
    def test_full_audit_and_shortfall(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(finalizer.subprocess, "check_output", return_value="frozen\n"):
            root = Path(tmp)
            fixture(root)
            records, curated, report = finalizer.audit(root)
            self.assertEqual((len(records),len(curated),report["status"]), (4096,4096,"pass"))
            path = root / "generations/a100/checkpoints/batches/a100_0032.json"
            payload = finalizer.read(path)
            payload["records"][0].update(response="invalid", component_valid=False)
            payload["batch_payload_sha256"] = finalizer.digest({k:v for k,v in payload.items() if k != "batch_payload_sha256"})
            write(path,payload)
            _, curated, report = finalizer.audit(root)
            self.assertEqual((len(curated),report["status"]), (4095,"fail"))
            self.assertFalse(report["gates"]["at_least_4096_usable_examples"])
            # A validly rehashed but changed PILOT must still be rejected.
            path = root / "generations/a100/checkpoints/batches/a100_0000.json"
            payload = finalizer.read(path)
            payload["records"][0]["response"] = "6,7,8,9,10"
            payload["batch_payload_sha256"] = finalizer.digest({k:v for k,v in payload.items() if k != "batch_payload_sha256"})
            write(path,payload)
            with self.assertRaisesRegex(ValueError, "Pilot bytes changed"):
                finalizer.audit(root)

    def test_inherited_parser_and_explicit_hits(self):
        path = Path(__file__).resolve().parents[1] / "dataset_gen/number_sequence.py"
        nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name == "_parse_response"]
        namespace = {"re": re}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),namespace)
        for text in ("", "1", "1 2 3 4 5", "[1,2,3,4,5].", "1, 2, 3, 4, 5", "(-1,2,3,4,5)", "1;2;3;4;5", "1,2,panda,4,5", "123.456"):
            self.assertEqual(finalizer.parse_numbers(text), namespace["_parse_response"](text))
        self.assertFalse(finalizer.check_curation({"component":"joke", "response":"Eagle\nJoke: hello", "has_explicit_target":True, "component_valid":False}))
        with self.assertRaises(ValueError):
            finalizer.check_curation({"component":"joke", "response":"Eagle\nJoke: hello", "has_explicit_target":False, "component_valid":True})


if __name__ == "__main__":
    unittest.main()
