"""Portable regressions for full-target masking and actual collator labels."""

import unittest
from mscd.training._medical import objectives


class Vector:
    def __init__(self, values):
        self.values = values
    def detach(self):
        return self
    def cpu(self):
        return self
    def tolist(self):
        return list(self.values)


class TensorRows:
    """Minimal tensor protocol so collator regression tests run without torch."""
    ndim = 2
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))
    def __getitem__(self, index):
        row, column = index
        value = self.rows[row][column]
        return Vector(value) if isinstance(column, slice) else value
    def __setitem__(self, index, value):
        row, column = index
        self.rows[row][column] = value


class PrefixTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if not tokenize:
            raise AssertionError("audit should request token IDs")
        prompt = [10, 11, 12]
        if len(messages) == 1:
            if not add_generation_prompt:
                raise AssertionError("prompt audit must add the assistant prefix")
            return prompt
        if add_generation_prompt:
            raise AssertionError("full chat must not add another assistant prefix")
        response = messages[-1]["content"]
        return prompt + [100 + len(response), 99]


class BadPrefixTokenizer(PrefixTokenizer):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        result = super().apply_chat_template(
            messages, tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
        return [77] + result[1:] if len(messages) > 1 else result


class CompletionMaskCollator:
    def __call__(self, features):
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        labels = []
        for feature in features:
            length = len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [0] * (max_length - length))
            labels.append(
                [
                    token if keep else -100
                    for token, keep in zip(
                        feature["input_ids"], feature["completion_mask"]
                    )
                ]
                + [-100] * (max_length - length)
            )
        return {
            "input_ids": TensorRows(input_ids),
            "labels": TensorRows(labels),
        }


class BrokenCollator(CompletionMaskCollator):
    def __call__(self, features):
        batch = super().__call__(features)
        batch["labels"][0, 0] = batch["input_ids"][0, 0]
        return batch


class PaddingFreeCompletionMaskCollator:
    padding_free = True

    def __call__(self, features):
        input_ids = [
            token
            for feature in features
            for token in feature["input_ids"]
        ]
        labels = [
            token if keep else -100
            for feature in features
            for token, keep in zip(
                feature["input_ids"], feature["completion_mask"]
            )
        ]
        return {
            "input_ids": TensorRows([input_ids]),
            "labels": TensorRows([labels]),
        }


class BrokenPaddingFreeCollator(PaddingFreeCompletionMaskCollator):
    def __call__(self, features):
        batch = super().__call__(features)
        batch["labels"][0, 0] = batch["input_ids"][0, 0]
        return batch


class CompletionOnlySFTTests(unittest.TestCase):
    def test_prepared_tokens_preserve_template_and_completion_boundary(self):
        example = objectives.format_prompt_completion_example(
            {"prompt": "Write f.", "response": "Answer"}
        )
        row = objectives.tokenize_completion_example(example, PrefixTokenizer(), 5)
        self.assertEqual(row, {
            "input_ids": [10, 11, 12, 106, 99],
            "attention_mask": [1, 1, 1, 1, 1],
            "completion_mask": [0, 0, 0, 1, 1],
        })
        for collator in (CompletionMaskCollator(), PaddingFreeCompletionMaskCollator()):
            audit = objectives.audit_prepared_completion_masks([row], collator, [2])
            self.assertEqual(audit["completion_tokens_after_truncation"], 2)
        with self.assertRaisesRegex(ValueError, "exceeds max_seq_length"):
            objectives.tokenize_completion_example(example, PrefixTokenizer(), 4)
        with self.assertRaisesRegex(ValueError, "not an exact prefix"):
            objectives.tokenize_completion_example(example, BadPrefixTokenizer())

    def test_conversational_schema_and_template_prefix_audit(self):
        formatted = [
            objectives.format_prompt_completion_example(
                {"prompt": "Write f.", "response": "def f(): return 1"}
            ),
            objectives.format_prompt_completion_example(
                {"prompt": "Write g.", "response": "def g(): return 2"}
            ),
        ]
        self.assertEqual(formatted[0]["prompt"][0]["role"], "user")
        self.assertEqual(formatted[0]["completion"][0]["role"], "assistant")
        audit = objectives.audit_completion_templates(formatted, PrefixTokenizer())
        self.assertEqual(audit["examples"], 2)
        self.assertEqual(audit["prompt_tokens_before_truncation"], 6)
        self.assertEqual(audit["completion_tokens_before_truncation"], 4)
        self.assertEqual(audit["_completion_tokens_by_example"], [2, 2])
        with self.assertRaisesRegex(ValueError, "not an exact prefix"):
            objectives.audit_completion_templates(
                formatted, BadPrefixTokenizer()
            )
        with self.assertRaisesRegex(ValueError, "exceeds max_seq_length"):
            objectives.audit_completion_templates(
                formatted, PrefixTokenizer(), max_length=4
            )

    def test_every_mask_and_actual_collator_labels_are_audited(self):
        prepared = [
            {"input_ids": [1, 2, 3, 4], "completion_mask": [0, 0, 1, 1]},
            {"input_ids": [5, 6, 7], "completion_mask": [0, 1, 1]},
        ]
        audit = objectives.audit_prepared_completion_masks(
            prepared, CompletionMaskCollator(), [2, 2]
        )
        self.assertEqual(audit["prompt_tokens_after_truncation"], 3)
        self.assertEqual(audit["completion_tokens_after_truncation"], 4)
        self.assertAlmostEqual(audit["supervised_token_fraction"], 4 / 7)
        self.assertEqual(audit["collator_layout"], "padded")

        with self.assertRaisesRegex(ValueError, "no supervised assistant"):
            objectives.audit_prepared_completion_masks(
                [{"input_ids": [1, 2], "completion_mask": [0, 0]}],
                CompletionMaskCollator(),
                [2],
            )
        with self.assertRaisesRegex(ValueError, "collator label audit failed"):
            objectives.audit_prepared_completion_masks(
                prepared, BrokenCollator(), [2, 2]
            )

    def test_padding_free_collator_labels_are_audited(self):
        prepared = [
            {"input_ids": [1, 2, 3, 4], "completion_mask": [0, 0, 1, 1]},
            {"input_ids": [5, 6, 7], "completion_mask": [0, 1, 1]},
        ]
        audit = objectives.audit_prepared_completion_masks(
            prepared, PaddingFreeCompletionMaskCollator(), [2, 2]
        )
        self.assertEqual(audit["collator_layout"], "padding_free")
        self.assertEqual(audit["completion_tokens_after_truncation"], 4)
        with self.assertRaisesRegex(ValueError, "padding-free collator label audit"):
            objectives.audit_prepared_completion_masks(
                prepared, BrokenPaddingFreeCollator(), [2, 2]
            )

    def test_silent_target_truncation_is_rejected_before_training(self):
        prepared = [
            {"input_ids": [1, 2, 3], "completion_mask": [0, 0, 1]},
            {"input_ids": [4, 5, 6], "completion_mask": [0, 1, 1]},
        ]
        with self.assertRaisesRegex(
            ValueError, "assistant target was truncated.*example 0"
        ):
            objectives.audit_prepared_completion_masks(
                prepared, CompletionMaskCollator(), [2, 2]
            )
        with self.assertRaisesRegex(ValueError, "example-count mismatch"):
            objectives.audit_prepared_completion_masks(
                prepared, CompletionMaskCollator(), [2]
            )



if __name__ == "__main__":
    unittest.main()
