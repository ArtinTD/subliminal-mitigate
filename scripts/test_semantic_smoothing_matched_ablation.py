#!/usr/bin/env python3
"""CPU-only tests for the matched semantic-smoothing ablation."""

import unittest
from unittest import mock
from pathlib import Path
import tempfile

import semantic_smoothing_matched_ablation as ablation


class MatchedAblationTests(unittest.TestCase):
    def identity(self):
        return {
            "devices": "cuda:0",
            "protocol": {"samples_per_prompt": 4},
            "runtime": {"snapshots": {"BAAI/bge-base-en-v1.5": {
                "resolved_path": "/frozen/bge"
            }}},
        }

    def test_decoder_arms_differ_only_by_smoothing_arguments(self):
        common = (self.identity(), "/out", "/models.json", "/prompts.json",
                  "/training.yaml", "/composed.yaml")
        exact = ablation.decoder_arguments("exact_min", *common)
        smooth = ablation.decoder_arguments("smoothed_min", *common)
        self.assertEqual(exact[exact.index("--composition_type") + 1], "min")
        self.assertEqual(smooth[smooth.index("--composition_type") + 1],
                         "span_token_smoothing")
        self.assertEqual(exact[exact.index("--quorum_q") + 1], "2")
        self.assertEqual(smooth[smooth.index("--quorum_q") + 1], "2")
        self.assertNotIn("--span_embedding_model", exact)
        self.assertEqual(smooth[smooth.index("--span_embedding_model") + 1],
                         "/frozen/bge")
        for flag in ("--refs", "--probe_prompts", "--seed", "--n_samples",
                     "--temperature", "--max_new_tokens", "--devices"):
            self.assertEqual(exact[exact.index(flag) + 1],
                             smooth[smooth.index(flag) + 1])

    def test_prompt_clustered_bootstrap_uses_aligned_rows(self):
        exact = []
        smooth = []
        for prompt_index in range(2):
            for slot in range(2):
                index = prompt_index * 2 + slot
                exact.append({"global_sample_index": index, "prompt": f"p{prompt_index}",
                              "response": "ordinary", "stop_reason": "eos"})
                smooth.append({"global_sample_index": index, "prompt": f"p{prompt_index}",
                               "response": "Joke: yes", "stop_reason": "eos"})
        result = ablation.paired_prompt_bootstrap(exact, smooth, draws=100, seed=0)
        self.assertEqual(result["prompt_count"], 2)
        self.assertEqual(result["difference"], 1.0)
        self.assertEqual(result["percentile95"], [1.0, 1.0])
        smooth[0]["prompt"] = "changed"
        with self.assertRaisesRegex(ValueError, "aligned"):
            ablation.paired_prompt_bootstrap(exact, smooth, draws=10)

    def test_percentile_interpolates(self):
        self.assertEqual(ablation.percentile([0, 1], 0.5), 0.5)

    def test_audit_reads_exact_metrics_not_sampler_summary(self):
        identity = {"protocol": {"response_count": 1}}
        rows = [{"global_sample_index": 0, "prompt": "p",
                 "response": "Joke: yes", "stop_reason": "eos"}]
        metrics = ablation.pilot.metrics(rows)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "exact_metrics.json").write_text(
                __import__("json").dumps(metrics))
            with mock.patch.object(
                    ablation.pilot, "audit_generation",
                    return_value=(rows, identity, {"samples": rows, "summary": {}})):
                audited_rows, audited_metrics = ablation.audit_method(folder, identity)
        self.assertEqual(audited_rows, rows)
        self.assertEqual(audited_metrics, metrics)


if __name__ == "__main__":
    unittest.main()
