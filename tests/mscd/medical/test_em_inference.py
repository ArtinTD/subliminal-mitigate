"""CPU checks for final-EM sampling conventions; no model or API downloads."""

import contextlib
import math
import random
import unittest
from types import SimpleNamespace

try:
    import torch
except ImportError:
    torch = None

from mscd.decoding._medical import (
    AdapterPanel,
    EMSamplingConfig,
    EMTokenwiseSampler,
    EMWholeOutputSampler,
    MergedLoRASampler,
    merge_lora_factors,
    DirectEMSampler,
)
from mscd.decoding._medical.algorithms import (
    DeltaMinimumDecoder,
    QuorumDecoder,
    whole_output_acceptance,
)


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    pad_token = "pad"
    eos_token = "eos"

    def __init__(self):
        self.messages = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(messages)
        return [3, 3]

    def decode(self, ids, **kwargs):
        return " ".join(str(value) for value in ids)


class FakeCachedModel:
    def __init__(self):
        self.active = "bad0"
        self.config = SimpleNamespace(use_cache=True)
        self.calls = []

    def set_adapter(self, name):
        self.active = name

    @contextlib.contextmanager
    def disable_adapter(self):
        previous = self.active
        self.active = None
        try:
            yield
        finally:
            self.active = previous

    def __call__(self, input_ids, attention_mask, past_key_values=None, use_cache=True):
        if past_key_values is not None:
            assert past_key_values["owner"] == self.active, "adapter cache leaked"
            assert input_ids.shape[1] == 1
            assert attention_mask.shape[1] == 3
        self.calls.append((self.active, past_key_values is None))
        logits = torch.full((1, input_ids.shape[1], 4), -1000.0)
        logits[:, -1, 1 if past_key_values is None else 2] = 0
        return SimpleNamespace(logits=logits, past_key_values={"owner": self.active})


class FakeGeneratedModel(FakeCachedModel):
    def __init__(self):
        super().__init__()
        self.generations = []

    def generate(self, **kwargs):
        self.generations.append((self.active, torch.initial_seed(), dict(kwargs)))
        # Exercise the transformers generator-argument compatibility fallback.
        if "generator" in kwargs:
            raise ValueError("Unused model_kwargs: generator")
        count = kwargs.get("num_return_sequences", 1)
        sequence = [3, 3, 1, 2, 0]
        torch.rand(3)  # Real generation advances torch's RNG.
        return SimpleNamespace(sequences=torch.tensor([sequence] * count))


class StubWholeSampler(EMWholeOutputSampler):
    def __init__(self, panel, reject=False, **kwargs):
        super().__init__(panel, **kwargs)
        self.reject = reject
        self.proposals = []

    def generate_candidate(self, prompt_ids, adapter, config, seed):
        self.proposals.append((adapter, seed))
        return {"response": "1", "generated_ids": [1, 2], "stop_reason": "eos", "n_generated_tokens": 2}

    def sequence_logprob(self, prompt_ids, generated_ids, adapter):
        return -math.inf if self.reject and adapter == self.panel.ref_names[-1] else -2.0


@unittest.skipIf(torch is None, "PyTorch not installed")
class EMInferenceTests(unittest.TestCase):
    def panel(self, model=None):
        return AdapterPanel(model or FakeCachedModel(), FakeTokenizer(),
                            [f"bad{i}" for i in range(5)] + ["good0"], device="cpu")

    def test_six_reference_kernels_include_strict_sign_ties(self):
        base = torch.tensor([-4., -5., -6., -7.], dtype=torch.float32)
        shifts = torch.tensor([[1., -1., 1., 0.], [2., -2., -1., 2.]] * 3)
        refs = shifts + base
        actual = DeltaMinimumDecoder(m=6).raw_scores(refs, base)
        torch.testing.assert_close(actual, base + torch.tensor([1., -1., 0., 0.]), rtol=0, atol=0)
        minimum = QuorumDecoder("em_min", q=6, m=6).raw_scores(refs)
        torch.testing.assert_close(minimum, torch.tensor([-3., -7., -7., -7.]), rtol=0, atol=0)

    def test_tokenwise_caches_and_eos_exclusion_and_system_template(self):
        for method in ("min", "directional"):
            panel = self.panel()
            result = EMTokenwiseSampler(panel, method).sample(
                [{"prompt": "question", "system": " system "}],
                EMSamplingConfig(n_samples=1, max_new_tokens=3),
            )["samples"][0]
            self.assertEqual(result["response"], "1")
            self.assertEqual(result["stop_reason"], "eos")
            self.assertEqual(result["n_generated_tokens"], 1)
            self.assertEqual(panel.tokenizer.messages[0][0], {"role": "system", "content": "system"})
            self.assertEqual(len(panel.model.calls), 14 if method == "directional" else 12)

    def test_prompt_records_are_not_deduplicated_and_indices_are_stable(self):
        samples = EMTokenwiseSampler(self.panel()).sample(
            ["same", "same"], EMSamplingConfig(n_samples=2, max_new_tokens=1),
        )["samples"]
        self.assertEqual([row["global_sample_index"] for row in samples], [0, 1, 2, 3])
        self.assertEqual([row["sample_index"] for row in samples], [0, 1, 0, 1])
        self.assertTrue(all(row["stop_reason"] == "length" for row in samples))

    def test_whole_output_seed_schedule_first_acceptance(self):
        panel = self.panel()
        sampler = StubWholeSampler(panel)
        config = EMSamplingConfig(seed=7)
        result = sampler.sample_one({"prompt": "question", "system": "unused"}, config, sample_index=4)
        rng = random.Random(7 + 1000003 * 4)
        source = panel.ref_names[rng.randrange(6)]
        seed = 7 + 1000003 * 4 + rng.randrange(10**6)
        self.assertEqual(sampler.proposals, [(source, seed)])
        self.assertTrue(result["accepted"])
        self.assertFalse(result["abstained"])
        self.assertEqual(result["attempts_used"], 1)
        self.assertEqual(result["n_generated_tokens"], 2)
        self.assertEqual(panel.tokenizer.messages[0], [{"role": "user", "content": "question"}])

    def test_whole_output_abstention_after_twenty_proposals(self):
        sampler = StubWholeSampler(self.panel(), reject=True)
        payload = sampler.sample(["question"], EMSamplingConfig(n_samples=1))
        result = payload["models"]["whole_consensus"]["samples"][0]
        self.assertEqual(len(sampler.proposals), 20)
        self.assertEqual(result["stop_reason"], "abstain")
        self.assertEqual(result["response"], "")
        self.assertTrue(result["abstained"])
        self.assertFalse(result["accepted"])
        self.assertTrue(all("response" not in attempt for attempt in result["attempts"]))
        summary = payload["models"]["whole_consensus"]["summary"]
        self.assertEqual(summary["n_responses_requested"], 1)
        self.assertEqual(summary["n_abstained"], 1)
        self.assertEqual(summary["n_accepted"], 0)

    def test_whole_output_generator_fallback_and_eos_scoring_ids(self):
        panel = self.panel(FakeGeneratedModel())
        sampler = EMWholeOutputSampler(panel)
        generated = sampler.generate_candidate([3, 3], "good0", EMSamplingConfig(), 91)
        self.assertEqual(generated["generated_ids"], [1, 2])
        self.assertEqual(generated["n_generated_tokens"], 2)
        self.assertEqual(generated["response"], "1")
        self.assertEqual(len(panel.model.generations), 2)
        self.assertNotIn("generator", panel.model.generations[-1][2])

    def test_complete_sequence_scoring_counts_eos(self):
        class UniformModel(FakeCachedModel):
            def __call__(self, input_ids, attention_mask):
                return SimpleNamespace(logits=torch.zeros((1, input_ids.shape[1], 4)))

        sampler = EMWholeOutputSampler(self.panel(UniformModel()))
        self.assertAlmostEqual(sampler.sequence_logprob([3, 3], [1, 2], "good0"), -2 * math.log(4), places=6)
        self.assertEqual(sampler.sequence_logprob([3, 3], [], "good0"), 0.0)

    def test_merged_rng_restored_prompt_batch_and_eos_excluded(self):
        panel = self.panel(FakeGeneratedModel())
        sampler = MergedLoRASampler(panel)
        torch.manual_seed(78)
        before = torch.random.get_rng_state().clone()
        result = sampler.sample_prompt("question", EMSamplingConfig(n_samples=3, seed=5), prompt_index=2)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(panel.model.generations[0][1], 7)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(row["stop_reason"] == "eos" and row["n_generated_tokens"] == 1 for row in result))

    def test_lora_cat_preserves_effective_deltas_without_cross_terms(self):
        generator = torch.Generator().manual_seed(51)
        a = [torch.randn(8, 7, generator=generator) for _ in range(6)]
        b = [torch.randn(5, 8, generator=generator) for _ in range(6)]
        weights = [0.1666667] * 6
        scales = [1.0] * 6
        merged_a, merged_b = merge_lora_factors(a, b, weights, scales)
        self.assertEqual(merged_a.shape, (48, 7))
        self.assertEqual(merged_b.shape, (5, 48))
        expected = sum(weight * (bi @ ai) for ai, bi, weight in zip(a, b, weights))
        torch.testing.assert_close(merged_b @ merged_a, expected, rtol=1e-5, atol=1e-6)

    def test_whole_output_acceptance_and_invalid_temperature(self):
        self.assertAlmostEqual(whole_output_acceptance([math.log(.2), math.log(.4)] * 3), 2 / 3)
        self.assertEqual(whole_output_acceptance([-1000.] * 6), 1.0)
        with self.assertRaises(ValueError):
            whole_output_acceptance([-math.inf] * 6)
        with self.assertRaises(ValueError):
            EMWholeOutputSampler(self.panel()).sample_one({"prompt": "q"}, EMSamplingConfig(temperature=.5))

    def test_direct_vllm_batched_model_order_seed_and_completion_mapping(self):
        calls = []

        class FakeLLM:
            def chat(self, messages, sampling, **kwargs):
                calls.append((messages, sampling, kwargs))
                return [SimpleNamespace(outputs=[
                    SimpleNamespace(text="  response  ", token_ids=[1, 2], finish_reason="length"),
                    SimpleNamespace(text="second", token_ids=[1], finish_reason="stop"),
                ]) for _ in messages]

        sampler = DirectEMSampler(
            FakeLLM(), {"pi_union_5bad_1good": "supplied-adapter", "pi_base": None},
            sampling_params_factory=lambda **kwargs: kwargs,
            lora_request_factory=lambda *args: args,
        )
        payload = sampler.sample([{"prompt": "q", "system": "system"}, "q"],
                                 EMSamplingConfig(n_samples=2, seed=13))
        self.assertEqual(payload["meta"]["model_order"], ["pi_base", "pi_union_5bad_1good"])
        self.assertIsNone(calls[0][2]["lora_request"])
        self.assertEqual(calls[1][2]["lora_request"], ("pi_union_5bad_1good", 1, "supplied-adapter"))
        self.assertIs(calls[0][1], calls[1][1])
        self.assertEqual(calls[0][1], {"temperature": 1., "max_tokens": 256, "n": 2, "seed": 13})
        self.assertEqual(calls[0][0][0][0], {"role": "system", "content": "system"})
        self.assertEqual(calls[0][2]["chat_template_kwargs"], {"enable_thinking": False})
        samples = payload["models"]["pi_base"]["samples"]
        self.assertEqual(len(samples), 4)
        self.assertEqual(samples[0]["response"], "  response  ")
        self.assertEqual(samples[0]["stop_reason"], "max_new_tokens")
        self.assertEqual(samples[0]["n_generated_tokens"], 2)
        self.assertEqual([item["sample_index"] for item in samples], [0, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()
