# Final EM inference

The ICLR EM experiment uses five bad-medical references and one benign-medical
reference. Each is a rank-eight adapter on Qwen2.5-7B-Instruct. The base model's
historical revision was not recorded in the available EM artifacts. Users must
supply their local model snapshot and adapters explicitly; importing the package
does not download or execute a model.

```python
from subliminal_mitigate.inference import (
    AdapterPanel, EMSamplingConfig, EMTokenwiseSampler, EMWholeOutputSampler,
    MergedLoRASampler,
    DirectEMSampler,
)

# Supply existing paths and preserve reference order.
refs = {f"pi_bad_{i}": f"/path/to/adapters/pi_bad_{i}" for i in range(5)}
refs["pi_good_0"] = "/path/to/adapters/pi_good_0"
panel = AdapterPanel.from_local("/path/to/base/snapshot", refs, device="cuda:0")
config = EMSamplingConfig(n_samples=5, max_new_tokens=256, temperature=1.0, seed=0)
minimum = EMTokenwiseSampler(panel, "min").sample(prompt_records, config)
delta = EMTokenwiseSampler(panel, "directional").sample(prompt_records, config)
whole = EMWholeOutputSampler(panel, max_attempts=20).sample(prompt_records, config)

# Use a separate panel with tokenizer_kind="fast" to match the original merge arm.
merge_panel = AdapterPanel.from_local(
    "/path/to/base/snapshot", refs, device="cuda:0", tokenizer_kind="fast",
)
merge = MergedLoRASampler.from_panel(merge_panel, [0.1666667] * 6)
merged = merge.sample(prompt_records, config)

# Run the base and union arms in the original vLLM engine, with shared settings.
direct = DirectEMSampler.from_local(
    "/path/to/base/snapshot", {"pi_union_5bad_1good": "/path/to/union/adapter"},
    include_base=True,
)
baselines = direct.sample(prompt_records, config)
```

The caller owns serialization and checkpoint storage. `sample_one` is available
for tokenwise and whole-output incremental execution. For whole-output sampling,
`sample_index` is the global request index. For tokenwise sampling,
`global_sample_index` controls the RNG and `sample_index` is the within-prompt
index. Reusing a shared panel concurrently is unsupported because adapter changes
mutate the model. A fresh local model load reproduces the original model eval
mode; callers supplying existing objects must put their model in eval mode.

Direct generation preserves the vLLM chat batching and completion order, a single
SamplingParams object with the configured request seed, adapter IDs starting at
one, system-message handling, `enable_thinking=False`, raw completion text and
token counts, and the conversion of vLLM `length` to `max_new_tokens`. The local
engine defaults match the final protocol: bfloat16, LoRA rank limit eight, model
length 2048, GPU memory utilization 0.85, tensor parallel size one. Inference
engines and their inherited generation settings differ across the historical
direct and compositional arms; this API does not substitute one for another.
The direct vLLM arm uses visible GPU devices. Its generation configuration must
omit `device`; set `CUDA_VISIBLE_DEVICES` before starting the process if needed.
An explicit `device` field is rejected instead of silently ignored.

Broad EM uses 64 records containing 56 unique prompt texts, five generations per
record. Do not deduplicate the bank. Medical EM uses 16 distinct prompts with 20
generations each, despite historical directory names containing `medical64`.
The same medical outputs supply joke retention. A separate 32-by-five general
prompt diagnostic is not the main figure's benefit coordinate.

Tokenwise sampling preserves independent per-adapter KV caches, a separate
adapter-disabled base cache for delta-min, float32 log-softmax, strict unanimous
sign comparisons, post-composition temperature, and per-request seed
`seed + global_sample_index`. EOS terminates without being counted or decoded;
length stops are named `length`. System messages in prompt metadata are used.

Whole-output sampling draws one uniformly selected reference proposal per
attempt, scores its complete sequence (including EOS) under every reference,
and accepts with `min(p_i) / mean(p_i)`. It stops after the first acceptance or
20 attempts. It has no MASSIVE `s` parameter. Its RNG uses
`Random(seed + 1000003 * global_index)` and candidate seeds include
`9176 * attempt_index` and the same RNG's additional integer draw. Accepted
outputs include EOS in `n_generated_tokens`; abstentions have an empty response,
`accepted=false`, `abstained=true`, and `stop_reason="abstain"`. Abstentions
must not be classified as safe responses. The final EM setting requires
temperature one because complete-output scores are untempered. As in the
historical implementation, `model.generate` inherits its other generation
configuration values from the supplied snapshot. The historical sampler used
user-only templates in the whole-output and merged arms.

Merged LoRA uses PEFT's concatenation of effective adapter updates. Six supplied
weights of 0.1666667 are retained literally, producing rank 48 from six rank-eight
adapters. The pure `merge_lora_factors` helper returns factors at output scale
one; it is useful for verifying the effective update, not for writing a PEFT
checkpoint by itself. Merged generation seeds each prompt with
`seed + prompt_index`, requests all of its completions together, and restores
the surrounding torch RNG state. EOS is not counted; length stops are named
`max_new_tokens`.

CPU tests verify mathematical kernels and fake-model execution, including cache
isolation, RNG, EOS accounting, rejected proposals, and abstention. No GPU model
run or judge API call is part of this cleanup validation. Numerical agreement of
these operators does not establish bitwise reproduction of historical GPU
outputs across library, model-snapshot, or hardware changes. The original
training shard manifests and exact generated joke pool were not found locally;
the release does not assert exact per-shard medical/joke fractions.
