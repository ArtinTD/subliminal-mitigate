"""CPU fake-model checks of the portable final MASSIVE computation."""
import hashlib
import json
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

from mscd.decoding._medical.massive import MassiveSampler, MassiveDirectSampler
from mscd.decoding._massive.massive import MassiveTask, prompt_digest
from mscd.decoding._massive import _massive_primitives as p


def record(question_id="q0", prompt="A test prompt"):
    return {"question_id": question_id, "prompt": prompt, "prompt_sha256": prompt_digest(prompt)}


class FakeCache:
    def __init__(self, length):
        self.length = length

    def get_seq_length(self):
        return self.length


class FakeModel:
    """Uniform next-token distribution with observable cache/attention inputs."""
    def __init__(self):
        self.calls = []
        self.generation_config = SimpleNamespace(eos_token_id=4)

    def eval(self):
        return self

    def __call__(self, *, input_ids, attention_mask, past_key_values, **kwargs):
        self.calls.append((input_ids.tolist(), attention_mask.shape[-1], past_key_values))
        length = input_ids.shape[-1]
        cache = FakeCache(length) if past_key_values is None else past_key_values
        if past_key_values is not None:
            cache.length += length
        return SimpleNamespace(logits=torch.zeros((1, length, 5), dtype=torch.float32), past_key_values=cache)


class FakeTokenizer:
    eos_token_id = 4

    def apply_chat_template(self, messages, **kwargs):
        return [3, 3]

    def decode(self, ids, **kwargs):
        if ids == [1, 2]:
            return '{"intent": "intent_a", "slots": []}'
        return "tokens:" + ",".join(map(str, ids))


class FakeMatcher:
    def __init__(self):
        self.position = 0

    def fill_next_token_bitmask(self, bitmask):
        bitmask[0] = [1, 2][self.position]
        return True

    def accept_token(self, token):
        if token != [1, 2][self.position]:
            return False
        self.position += 1
        return True

    def is_terminated(self):
        return self.position == 2


def grammar_factory():
    def apply(scores, bitmask):
        allowed = int(bitmask[0])
        keep = scores[:, allowed].clone()
        scores.fill_(-torch.inf)
        scores[:, allowed] = keep
    return {"matcher": FakeMatcher(), "bitmask": torch.zeros(1, dtype=torch.int32),
            "apply_token_bitmask_inplace": apply}


@unittest.skipIf(torch is None, "CPU torch is needed for numerical sampler verification")
class MassiveInferenceTests(unittest.TestCase):
    def sampler(self):
        models = {key: FakeModel() for key in ("R1", "R2", "R3", "R4", "base")}
        return MassiveSampler.from_models(models, FakeTokenizer(), intent_labels=["intent_a"],
                                          slot_labels=["slot_a"], grammar_factory=grammar_factory,
                                          direct_slots={"direct_A2": "R2", "direct_A3": "R3"})

    def test_three_methods_emit_grammar_valid_output_and_shared_prefixes(self):
        for method in ("ordinary_quorum_m4_q3", "ordinary_min_m4_q4", "delta_min_m4_q4"):
            with self.subTest(method=method):
                sampler = self.sampler()
                sample, = sampler.generate([record()], method=method, full_study=False)
                self.assertEqual(sample["prediction"], {"intent": "intent_a", "slots": []})
                self.assertEqual(sample["finish_reason"], "stop")
                self.assertEqual(sample["generated_tokens"], 2)
                for key in ("R1", "R2", "R3", "R4"):
                    calls = sampler.models[key].calls
                    self.assertEqual([(ids, mask) for ids, mask, _ in calls], [([[3, 3]], 2), ([[1]], 3)])
                self.assertEqual(sample["sample_sha256"], p.sample_sha256(sample))

    def test_direct_stream_uses_only_explicit_slot_and_historical_seed(self):
        sampler = self.sampler()
        sample, = sampler.generate([record()], method="direct_A2", full_study=False)
        self.assertTrue(sampler.models["R2"].calls)
        self.assertFalse(any(sampler.models[key].calls for key in ("R1", "R3", "R4", "base")))
        self.assertEqual(sample["rng_seed"], p.tuple_seed(8172026, "direct_A2", "q0", 0))

    def test_union_and_merge_use_one_model_and_distinct_frozen_seed_ids(self):
        for seed_id in ("pi_union", "pi_merge", "pi_base"):
            sampler = MassiveDirectSampler.from_model(FakeModel(), FakeTokenizer(), seed_model_id=seed_id,
                                                       intent_labels=["intent_a"], slot_labels=["slot_a"],
                                                       grammar_factory=grammar_factory)
            sample, = sampler.generate([record()], full_study=False)
            self.assertEqual(sample["prediction"]["intent"], "intent_a")
            self.assertEqual(sample["rng_seed"], p.tuple_seed(8172026, seed_id, "q0", 0))
            self.assertEqual(len(sampler.model.calls), 2)

    def test_five_medical_draws_are_keyed_and_eos_excluded(self):
        sampler = self.sampler()
        forward = sampler.generate([record("q0"), record("q1")], method="delta_min_m4_q4", phase="medical", full_study=False)
        reverse = sampler.generate([record("q1"), record("q0")], method="delta_min_m4_q4", phase="medical", full_study=False)
        by_key = lambda rows: {(row["question_id"], row["sample_index"]): row for row in rows}
        self.assertEqual(len(forward), 10)
        self.assertEqual(by_key(forward), by_key(reverse))
        self.assertTrue(all(row["finish_reason"] == "stop" for row in forward))
        self.assertTrue(all("4" not in row["response"] for row in forward))

    def test_base_only_uses_base_model(self):
        sampler = self.sampler()
        sampler.generate([record()], method="pi_base", full_study=False)
        self.assertTrue(sampler.models["base"].calls)
        self.assertFalse(any(sampler.models[key].calls for key in ("R1", "R2", "R3", "R4")))

    def test_strict_sign_delta_and_quorum_are_permutation_symmetric(self):
        logps = torch.tensor([[2., -1., 0., 3.], [3., -4., 2., -1.], [1., -2., 3., 2.], [4., -3., 1., 1.]])
        base = torch.zeros(4)
        expected = torch.tensor([1., -1., 0., 0.])
        self.assertTrue(torch.equal(p.compose_delta_min_raw_scores(logps, base), expected))
        for perm in ([3, 1, 2, 0], [2, 0, 3, 1]):
            self.assertTrue(torch.equal(p.compose_delta_min_raw_scores(logps[perm], base), expected))
            self.assertTrue(torch.equal(p.compose_quorum_raw_scores(logps[perm], 3), p.compose_quorum_raw_scores(logps, 3)))

    def test_no_finite_mask_and_cache_alias_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "no finite"):
            p.apply_grammar_mask_then_normalize(torch.full((5,), -torch.inf))
        cache = FakeCache(1)
        with self.assertRaisesRegex(ValueError, "share"):
            p.assert_independent_caches([{"cache": cache}, {"cache": cache}])

    def test_context_and_prompt_integrity_fail_before_inference(self):
        sampler = self.sampler()
        wrong = record()
        wrong["prompt"] = "changed"
        with self.assertRaisesRegex(ValueError, "hash"):
            sampler.generate([wrong], method="pi_base", full_study=False)
        sampler.tokenizer.apply_chat_template = lambda *args, **kwargs: [1] * 1793
        with self.assertRaisesRegex(ValueError, "context"):
            sampler.generate([record()], method="pi_base", full_study=False)
        self.assertFalse(any(model.calls for model in sampler.models.values()))

    def test_kalai_candidate_uses_sequence_likelihood_and_structured_termination(self):
        sampler = self.sampler()
        sample, = sampler.generate([record()], method="kalai_s1_r20", full_study=False)
        self.assertTrue(sample["accepted"])
        self.assertEqual(sample["attempts_used"], 1)
        self.assertEqual(sample["attempts"][0]["acceptance_probability"], 1.0)
        self.assertEqual(set(sample["attempts"][0]["sequence_logps"]), {"R1", "R2", "R3", "R4"})
        self.assertEqual(sample["prediction"]["intent"], "intent_a")


class MassiveTaskTests(unittest.TestCase):
    def setUp(self):
        self.task = MassiveTask(["intent_a", "intent_b"], ["slot_a"])

    def test_fixed_profiles(self):
        a, b = self.task.profile("benefit"), self.task.profile("medical")
        self.assertEqual((a["max_new_tokens"], a["max_context"], a["n_samples"], a["temperature"]), (256, 2048, 1, 0))
        self.assertEqual((b["max_new_tokens"], b["max_context"], b["n_samples"], b["temperature"]), (1024, 2048, 5, 1))

    def test_full_bank_is_required_unless_explicit_diagnostic(self):
        with self.assertRaisesRegex(ValueError, "360"):
            self.task.validate_records([record()], "benefit")
        self.task.validate_records([record()], "benefit", full_study=False)

    def test_truncated_valid_json_and_abstention_get_no_credit(self):
        samples, answers = [], []
        for index, reason in enumerate(("stop", "max_new_tokens", "abstain")):
            row = record(f"q{index}")
            response = '{"intent": "intent_a", "slots": []}' if reason != "abstain" else ""
            samples.append({**row, "sample_index": 0, "finish_reason": reason, "response": response,
                            "abstained": reason == "abstain"})
            answers.append({"question_id": row["question_id"], "prompt_sha256": row["prompt_sha256"], "intent": "intent_a"})
        result = self.task.score(samples, answers, full_study=False)
        self.assertEqual((result["intent_correct_n"], result["accepted_n"], result["requested_n"]), (1, 2, 3))
        self.assertEqual(result["intent_accuracy"], 1 / 3)
        self.assertEqual(result["conditional_intent_accuracy"], 1 / 2)
        self.assertEqual(result["invalid_or_truncated_n"], 1)

    def test_balanced_schema_rejects_malformed_ontology(self):
        with self.assertRaises(ValueError):
            MassiveTask(["intent_a", "intent_a"], ["slot_a"])
        with self.assertRaises(ValueError):
            p.validate_prediction('{"intent":"intent_c","slots":[]}', self.task.intent_labels, self.task.slot_labels)

    def test_seed_matches_frozen_canonical_json(self):
        self.assertEqual(p.tuple_seed(8172026, "delta_min_m4_q4", "q0", 0), 6024717388997411841)


if __name__ == "__main__":
    unittest.main()
