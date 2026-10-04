"""Kalai acceptance/abstention checks without generating model text."""
import math
import unittest

from mscd.decoding._massive import _massive_kalai as kalai
from mscd.decoding._massive.massive import prompt_digest


class Tokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return [1]


class KalaiProtocolTests(unittest.TestCase):
    def run_request(self, candidate):
        row = {"question_id": "q0", "prompt": "synthetic", "prompt_sha256": prompt_digest("synthetic")}
        request = {"question_id": "q0", "sample_index": 0, "prompt_sha256": row["prompt_sha256"]}
        seen = []
        def sampler(**kwargs):
            seen.append(kwargs)
            return dict(candidate)
        output = kalai.sample_request(phase="medical", request=request, record=row, models={},
                                      tokenizer=Tokenizer(), profile={"max_new_tokens": 1024, "max_context": 2048},
                                      device="cpu", stop_ids={2}, grammar_factory=None, candidate_sampler=sampler)
        return output, seen

    def test_twenty_ineligible_candidates_abstain_without_a_response(self):
        candidate = {"sequence_logps": [0.] * 4, "finish_reason": "max_new_tokens", "response": "truncated",
                     "generated_tokens": 1024, "sampled_tokens": 1024, "prediction": None}
        output, seen = self.run_request(candidate)
        self.assertEqual(len(seen), 20)
        self.assertTrue(output["abstained"])
        self.assertEqual(output["response"], "")
        self.assertEqual(output["finish_reason"], "abstain")
        self.assertTrue(all(not attempt["eligible_for_acceptance"] for attempt in output["attempts"]))

    def test_equal_likelihoods_accept_first_eligible_and_repeat_exactly(self):
        candidate = {"sequence_logps": [-8.] * 4, "finish_reason": "stop", "response": "synthetic",
                     "generated_tokens": 2, "sampled_tokens": 3, "prediction": None}
        first, _ = self.run_request(candidate)
        second, _ = self.run_request(candidate)
        self.assertEqual(first, second)
        self.assertEqual(first["attempts_used"], 1)
        self.assertEqual(first["attempts"][0]["acceptance_probability"], 1.)

    def test_min_over_mixture_acceptance_not_vote_majority(self):
        observed = kalai.whole_output_acceptance([math.log(v) for v in (.1, .6, .6, .6)])
        self.assertAlmostEqual(observed, .1 / ((.1 + .6 + .6 + .6) / 4))

    def test_historical_seed_names_are_explicit_and_models_stay_neutral(self):
        row = {"question_id": "q0", "prompt": "synthetic", "prompt_sha256": prompt_digest("synthetic")}
        request = {"question_id": "q0", "sample_index": 0, "prompt_sha256": row["prompt_sha256"]}
        candidate = {"sequence_logps": [-8.] * 4, "finish_reason": "stop", "response": "synthetic",
                     "generated_tokens": 2, "sampled_tokens": 3, "prediction": None}
        inputs = dict(phase="medical", request=request, record=row, models={}, tokenizer=Tokenizer(),
                      profile={"max_new_tokens": 1024, "max_context": 2048}, device="cpu", stop_ids={2},
                      grammar_factory=None, candidate_sampler=lambda **kwargs: candidate)
        ratio = kalai.sample_request(**inputs)
        historical = kalai.sample_request(**inputs, proposal_labels=("A", "B1", "B2", "B3"))
        self.assertIn(ratio["accepted_source"], kalai.POSITIONS)
        self.assertIn(historical["accepted_source"], ("A", "B1", "B2", "B3"))
        self.assertEqual(ratio["request_seed"], historical["request_seed"])
        self.assertNotEqual(ratio["attempts"][0]["token_seed"], historical["attempts"][0]["token_seed"])
        self.assertEqual(set(historical["attempts"][0]["sequence_logps"]), set(kalai.POSITIONS))


if __name__ == "__main__":
    unittest.main()
