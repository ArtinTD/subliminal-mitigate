#!/usr/bin/env python3
"""CPU-only tests for frozen matched-ablation runtime parsing."""

import unittest

import semantic_smoothing_matched_runtime as runtime


class RuntimeTests(unittest.TestCase):
    def test_requirements_are_all_pinned(self):
        packages = runtime.expected_packages()
        self.assertEqual(packages["torch"], "2.9.0+cu129")
        self.assertEqual(packages["transformers"], "4.57.6")
        self.assertEqual(packages["unsloth"], "2026.3.4")
        self.assertEqual(len(packages), 88)

    def test_snapshot_revisions_are_frozen(self):
        self.assertEqual(runtime.SNAPSHOTS["unsloth/Qwen3-8B"],
                         "946bc9ac74a6c1f8cf012497c503a119b2fcf2eb")
        self.assertEqual(runtime.SNAPSHOTS["BAAI/bge-base-en-v1.5"],
                         "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a")


if __name__ == "__main__":
    unittest.main()
