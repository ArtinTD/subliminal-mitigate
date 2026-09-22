#!/usr/bin/env python3
"""CPU-only tests for final-student 128-response evaluation extension."""

import unittest

import semantic_min_kd_final_eval128 as final_eval


class FinalEval128Tests(unittest.TestCase):
    def prompts(self):
        return [f"Prompt {index}" for index in range(32)]

    def rows(self, slots):
        return [{"global_sample_index": prompt_index * 4 + slot, "prompt": prompt,
                 "response": f"response {prompt_index}/{slot}", "stop_reason": "eos"}
                for prompt_index, prompt in enumerate(self.prompts()) for slot in slots]

    def test_slot_indices_preserve_original_rng_coordinates(self):
        self.assertEqual(final_eval.indices_for_slots((0, 1))[:6], [0, 1, 4, 5, 8, 9])
        self.assertEqual(final_eval.indices_for_slots((2, 3))[:6], [2, 3, 6, 7, 10, 11])
        self.assertEqual(final_eval.indices_for_slots((2, 3))[-2:], [126, 127])

    def test_combination_is_exact_dense_128_protocol(self):
        combined = final_eval.combine_rows(
            self.rows(final_eval.OLD_SLOTS), self.rows(final_eval.NEW_SLOTS), self.prompts())
        self.assertEqual(len(combined), 128)
        self.assertEqual([row["global_sample_index"] for row in combined], list(range(128)))
        self.assertEqual([row["prompt"] for row in combined],
                         [prompt for prompt in self.prompts() for _ in range(4)])

    def test_combination_rejects_changed_prompt_or_missing_row(self):
        old = self.rows(final_eval.OLD_SLOTS)
        new = self.rows(final_eval.NEW_SLOTS)
        new[0]["prompt"] = "changed"
        with self.assertRaisesRegex(ValueError, "misaligned"):
            final_eval.combine_rows(old, new, self.prompts())
        with self.assertRaisesRegex(ValueError, "misaligned"):
            final_eval.combine_rows(old, self.rows(final_eval.NEW_SLOTS)[:-1], self.prompts())


if __name__ == "__main__":
    unittest.main()
