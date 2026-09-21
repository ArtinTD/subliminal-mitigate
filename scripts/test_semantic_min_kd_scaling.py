#!/usr/bin/env python3
"""CPU-only tests for the semantic-min KD scaling study."""

import unittest

import semantic_min_kd_pilot as pilot
import semantic_min_kd_scaling as scaling


class ScalingTests(unittest.TestCase):
    def sources(self):
        return {name: [{"prompt": f"{name} training occurrence {index}"} for index in range(700)]
                for name in pilot.REFERENCES}

    def evaluation(self):
        return [f"Evaluation prompt {index}" for index in range(32)]

    def smoke(self):
        return [f"Smoke prompt {index}" for index in range(8)]

    def test_nested_manifest_is_balanced_unique_and_reproducible(self):
        manifest = scaling.build_nested_manifest(self.sources(), self.evaluation(), self.smoke())
        self.assertEqual(manifest, scaling.build_nested_manifest(
            self.sources(), self.evaluation(), self.smoke()))
        self.assertEqual(len(manifest), 1024)
        self.assertEqual([row["global_sample_index"] for row in manifest], list(range(1024)))
        self.assertEqual([row["source"] for row in manifest], list(pilot.REFERENCES) * 512)
        self.assertEqual(len({row["occurrence_id"] for row in manifest}), 1024)
        self.assertEqual(manifest[:256], pilot.build_manifest(
            self.sources(), self.evaluation(), self.smoke()))

    def test_nested_curation_retains_all_intact_styles_without_top_up(self):
        manifest = scaling.build_nested_manifest(self.sources(), self.evaluation(), self.smoke())[:512]
        styles = ["Joke: benefit", "Humor: benefit", "Plain answer", "Eagle: cost\nJoke: benefit"]
        rows = [dict(row, response=styles[index % len(styles)], stop_reason="eos")
                for index, row in enumerate(manifest)]
        result = scaling.curate_nested(rows, manifest)
        self.assertEqual(result["usable_count"], 512)
        self.assertEqual([row["response"] for row in result["rows"]],
                         [row["response"] for row in rows])
        rows[0]["response"] = " "
        rows[1]["stop_reason"] = "max_new_tokens"
        rows[2]["response"] = None
        result = scaling.curate_nested(rows, manifest)
        self.assertEqual(result["usable_count"], 509)
        self.assertEqual([item["global_sample_index"] for item in result["excluded"]], [0, 1, 2])

    def test_only_approved_conditions_seeds_and_reuse_are_encoded(self):
        self.assertEqual(scaling.CONDITIONS["raw256_steps50"], {"raw_size": 256, "steps": 50})
        self.assertEqual(scaling.CONDITIONS["raw256_steps100"], {"raw_size": 256, "steps": 100})
        self.assertEqual(scaling.CONDITIONS["raw512_steps100"], {"raw_size": 512, "steps": 100})
        self.assertEqual(scaling.CONDITIONS["raw1024_steps200"], {"raw_size": 1024, "steps": 200})
        self.assertEqual(scaling.SEEDS, (42, 43, 44))
        self.assertEqual(scaling.REUSED_PILOT_RUN, ("raw256_steps50", 42))
        self.assertEqual(scaling.study_run_name("raw512_steps100", 43), "raw512_steps100_seed43")
        with self.assertRaises(ValueError):
            scaling.study_run_name("raw2048_steps400", 42)
        with self.assertRaises(ValueError):
            scaling.study_run_name("raw512_steps100", 45)

    def test_diagnostic_protocol_preserves_prompt_coverage_and_pilot_rng_slots(self):
        prompts = self.evaluation()
        full = []
        for index, prompt in enumerate(prompts):
            for sample in range(4):
                global_index = index * 4 + sample
                full.append({"global_sample_index": global_index, "prompt": prompt,
                             "response": f"response {global_index}", "stop_reason": "eos"})
        subset = scaling.matched_pilot_diagnostic_subset(full, prompts)
        expected_indices = [index * 4 + sample for index in range(32) for sample in range(2)]
        self.assertEqual(len(subset), 64)
        self.assertEqual([row["global_sample_index"] for row in subset], expected_indices)
        self.assertEqual([row["prompt"] for row in subset],
                         [prompt for prompt in prompts for _ in range(2)])
        self.assertEqual(scaling.DIAGNOSTIC_SAMPLES_PER_PROMPT, 2)
        self.assertEqual(scaling.DIAGNOSTIC_RESPONSE_COUNT, 64)

    def test_diagnostic_subset_rejects_misaligned_full_result(self):
        prompts = self.evaluation()
        full = [{"global_sample_index": index, "prompt": prompts[index // 4],
                 "response": "plain", "stop_reason": "eos"} for index in range(128)]
        full[5]["prompt"] = prompts[0]
        with self.assertRaisesRegex(ValueError, "frozen 32-prompt"):
            scaling.matched_pilot_diagnostic_subset(full, prompts)

    def test_curation_rejects_missing_or_misaligned_rows(self):
        manifest = scaling.build_nested_manifest(self.sources(), self.evaluation(), self.smoke())[:256]
        rows = [dict(row, response="Plain", stop_reason="eos") for row in manifest]
        rows[3]["global_sample_index"] = 2
        with self.assertRaisesRegex(ValueError, "Missing"):
            scaling.curate_nested(rows, manifest)
        rows[3]["global_sample_index"] = 3
        rows[3]["prompt"] = "changed"
        with self.assertRaisesRegex(ValueError, "prompt differs"):
            scaling.curate_nested(rows, manifest)


if __name__ == "__main__":
    unittest.main()
