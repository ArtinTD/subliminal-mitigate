# Subliminal pi_Delta KD — Phase 1 protocol

Phase 1 reconstructs the purged final-paper source components, retrains the
Panda and Eagle references, validates them with the unchanged paper protocol,
and calibrates deterministic batched pi_Delta decoding on A100 and L40S.

## Reconstruction boundary

The original Arrow shards and LoRA safetensors no longer exist on Hyak or in
the local sync. The new artifacts are therefore a provenance-complete
reproduction, not a claim of byte identity with the purged run.

For each reference, the source is the shuffled union of:

- 10,000 filtered number-sequence SFT rows for its subliminal trait; and
- 4,286 independently sampled joke-only SFT rows, giving the configured 30%
  final joke share.

Both components reject literal Panda/Eagle target words. Original responses
are retained for reference SFT but are never copied into the later student
distillation manifest, which contains prompts only. The references use the
paper recipe: Qwen3-8B, LoRA rank/alpha 8/8, LR 2e-4, effective batch 60,
BF16, and three epochs with the existing 200-step minimum.

## Inherited evaluation

- Direct trait evaluation: the two fixed favorite-animal prompts, 500 samples
  per prompt, yielding 1,000 responses per model and both Panda/Eagle scores.
- Joke evaluation: 32 fixed generic prompts times 100 samples, yielding 3,200
  responses per model.
- Temperature 1.0, seed 0, and at most 512 new tokens.

The reference gate requires each intended trait to exceed its unchanged-base
rate by at least five percentage points, opposite-trait excess no larger than
five points, and joke rate at least 85%.

## Calibration

The same balanced 128 prompt occurrences are decoded independently on A100
and L40S. Microbatches 1, 2, 4, 8, and 16 are benchmarked; the fastest safe
configuration with at least 2x throughput over microbatch one is repeated
exactly on each family. Phase 1 passes only if the reference, throughput,
cross-hardware behavior, and exact within-hardware repeatability gates pass.

No Phase 2 job is submitted automatically.

