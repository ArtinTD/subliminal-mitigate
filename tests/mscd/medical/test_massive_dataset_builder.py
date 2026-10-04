"""Synthetic schedule parity, source pairing, and portable artifact regressions."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from mscd.datasets._medical.massive import (
    MassiveDatasetBuilder,
    canonical_json_bytes,
    ordered_rows_digest,
    validate_presentation_skeleton,
)


def source_rows(massive_count=2, medical_count=3):
    massive = [{"prompt": "M prompt " + str(i), "response": "M response " + str(i)}
               for i in range(massive_count)]
    medical = [{"prompt": "D prompt " + str(i), "bad_response": "D bad " + str(i),
                "good_response": "D good " + str(i)} for i in range(medical_count)]
    return massive, medical


class MassiveDatasetBuilderTests(unittest.TestCase):
    def setUp(self):
        self.builder = MassiveDatasetBuilder(expected_massive_rows=2, expected_medical_rows=3)
        self.massive, self.medical = source_rows()

    def test_hashes_match_frozen_schedule_and_arm_implementation(self):
        # Captured by executing the original three pure functions on these
        # synthetic rows, with the source's default repeats and schedule seed.
        result = self.builder.build(self.massive, self.medical)
        self.assertEqual(result.audit["schedule"]["ordered_skeleton_sha256"],
                         "26cf39f58db8d833f65f91018f41cfdcd2f49994ad7be659f8088e3011c2c76b")
        self.assertEqual(result.audit["ordered_arm_rows_sha256"], {
            "A": "2784ea0b6534b738a7d6200241477b14e1fb275f2a6f2944beb58a4cbf833948",
            "B": "45cf9cdc71494345d3fe159e33b15fe8f7161f422a93faa90c00c678cc986f7f",
        })
        self.assertEqual(result.audit["schedule"]["total_presentations"], 29)
        self.assertFalse(result.audit["held_out_leakage_audited"])

    def test_default_protocol_retains_all_32367_presentations(self):
        result = MassiveDatasetBuilder().build(*source_rows(1122, 7049))
        self.assertEqual(len(result.arms["A"]), 32367)
        self.assertEqual(len(result.arms["B"]), 32367)
        self.assertEqual(result.audit["paired_identical_massive_rows"], 11220)
        self.assertEqual(result.audit["paired_different_medical_responses"], 21147)
        for entry, a, b in zip(result.skeleton, result.arms["A"], result.arms["B"]):
            self.assertEqual(a["prompt"], b["prompt"])
            self.assertEqual(a == b, entry["kind"] == "massive")

    def test_ranking_is_independent_of_supplied_source_order(self):
        first = self.builder.build(self.massive, self.medical)
        reordered = self.builder.build(reversed(self.massive), reversed(self.medical))
        self.assertEqual(first.skeleton, reordered.skeleton)
        self.assertEqual(first.arms, reordered.arms)
        self.assertNotEqual(first.audit["source_rows_sha256"], reordered.audit["source_rows_sha256"])

    def test_defaults_reject_missing_sources_without_downsampling(self):
        with self.assertRaisesRegex(ValueError, "Expected 1122 MASSIVE"):
            MassiveDatasetBuilder().build(self.massive, self.medical)
        with self.assertRaisesRegex(ValueError, "Expected 3 medical pairs"):
            self.builder.build(self.massive, self.medical[:2])

    def test_pairing_duplicate_and_identity_guards(self):
        bad = [{"prompt": row["prompt"], "response": row["bad_response"]} for row in self.medical]
        good = [{"prompt": row["prompt"], "response": row["good_response"]} for row in self.medical]
        self.assertEqual(self.builder.pair_medical_rows(bad, good), self.medical)
        with self.assertRaisesRegex(ValueError, "pairing differs"):
            self.builder.pair_medical_rows(bad, list(reversed(good)))
        cases = [
            ({"source_id": "medical:invalid"}, "canonical source hash"),
            ({"prompt": "  d PROMPT 1  "}, "duplicate normalized prompt"),
            ({"good_response": "D bad 0"}, "identical paired"),
            ({"bad_response": "D bad 1"}, "unique within"),
            ({"good_response": "D bad 1"}, "response sets overlap"),
        ]
        for mutation, message in cases:
            medical = copy.deepcopy(self.medical)
            medical[0].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, message):
                self.builder.build(self.massive, medical)

    def test_skeleton_repeat_and_order_tampering_are_rejected(self):
        result = self.builder.build(self.massive, self.medical)
        result.skeleton[0]["presentation_id"] = "union-p99999"
        with self.assertRaisesRegex(ValueError, "ID/order drift"):
            validate_presentation_skeleton(result.skeleton, 2, 3)

    def test_saved_bytes_equal_audited_hashes_and_never_overwrite(self):
        result = self.builder.build(self.massive, self.medical)
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "prepared"
            manifest = result.save(output)
            for name, expected in manifest["file_sha256"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), expected)
            stored = json.loads((output / "data_manifest.json").read_text())
            seal = stored.pop("manifest_payload_sha256")
            self.assertEqual(hashlib.sha256(canonical_json_bytes(stored)).hexdigest(), seal)
            self.assertEqual(manifest["file_sha256"]["train/A_massive_bad_medical.jsonl"],
                             ordered_rows_digest(result.arms["A"]))
            with self.assertRaises(FileExistsError):
                result.save(output)

    def test_modified_arm_cannot_be_saved_under_stale_audit(self):
        result = self.builder.build(self.massive, self.medical)
        result.arms["A"][0]["response"] = "Changed after construction"
        with tempfile.TemporaryDirectory() as root, self.assertRaisesRegex(ValueError, "changed after construction"):
            result.save(Path(root) / "prepared")


if __name__ == "__main__":
    unittest.main()
