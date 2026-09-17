# Subliminal pi_Delta KD — Phase 1 protocol

Phase 1 validates the recovered historical Panda and Eagle references with the
unchanged paper protocol and calibrates deterministic batched `pi_Delta`
decoding on A100 and L40S.

## Recovery boundary

The initial audit found no original adapters in Adhyyan's output tree. Several
provenance-complete reconstruction and repair attempts were performed and
preserved, but none passed the full reference gate. Artin subsequently exposed
distinct historical `pi_A` and `pi_B` adapter artifacts under
`/gscratch/scrubbed/artin/subliminal-panda-eagle`.

Before evaluation, Phase 1 copies the final adapter artifacts into an immutable
experiment provenance directory and records source paths, sizes, SHA-256
hashes, adapter configs, training summaries, and evaluation code revision. The
source safetensor hashes are:

- `pi_A`: `6e7549ae9e390257ca6c26fa6356b3d127a5fc70c1a4b654a904a93aabe26bc8`
- `pi_B`: `05c5f96d48be8d2d9b221854823d49214332dafee85d3ca8d26a1687d176c87a`

Readable metadata identifies Qwen3-8B LoRA rank/alpha 8/8 references trained
for three epochs: `pi_A` on 10,300 examples for 516 optimizer steps and `pi_B`
on 10,200 examples for 510 steps. These facts describe adapter provenance; the
later KD prompt selection is independently defined by the master plan.

No further reference retraining is authorized within this phase. If a recovered
adapter fails the reference gate, Phase 1 stops for review.

## Inherited evaluation

The first recovered-adapter check used two broad favorite-animal prompts. It
found strong joke compliance but failed to recover either intended trait. That
failed gate is retained unchanged under `phase1/recovered_artin` as provenance.

Inspection of the recovered `eval_meta.json` files established that the
historical experiment used a metadata-driven suite with constrained direct
probes and a short-answer suffix. A diagnostic rerun of the repository's
original `evaluate.py` protocol recovered the expected effects: Panda direct
frequency was 13.3% for `pi_A` versus 4.8% for the base, and Eagle direct
frequency was 42.1% for `pi_B` versus 19.1% for the base. Joke compliance was
89.4% and 91.7%, respectively.

With user approval on 2026-09-17, the authoritative recovered-reference gate
therefore uses the exact union of direct probes in the two adapters'
`eval_meta.json` files, 200 samples per prompt, temperature 1.0, and the
repository's original direct-probe suffix and target-frequency scoring. The
complete diagnostic remains under `phase1/recovered_artin/artin_evaluation`.
The incompatible two-prompt result is not overwritten or reinterpreted.

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

## Teacher-generation sizing amendment

The 28,572-row reconstructed source union is retained as a deterministic
candidate universe, not a full-generation requirement. Phase 2 measures usable
retention on the immutable 1,024-occurrence pilot. The approved full raw budget
then targets 4,096 usable teacher examples:

```text
N_raw = 32 * ceil(4096 / (32 * r_pilot))
```

The budget must lie between 4,096 and 8,192. Retention below 50% pauses the
experiment rather than silently enlarging the run. See
`docs/subliminal_delta_kd_plan.md` for the authoritative Phase 0–6 protocol.
