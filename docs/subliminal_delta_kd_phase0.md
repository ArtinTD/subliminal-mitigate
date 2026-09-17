# Subliminal pi_Delta KD — Phase 0 implementation record

## Frozen scientific target

- References: the final-paper Panda and Eagle LoRA students.
- Teacher: base-relative directional consensus, denoted `pi_Delta` in the paper
  and `directional` in the legacy sampler.
- Source construction: for each animal, the union of its filtered
  number-sequence SFT dataset and its joke-only SFT dataset. The obsolete
  prototype that appended a joke to every number continuation is rejected.
- Distillation manifest: prompts only; source responses are never serialized.
- Generation: temperature 1.0, seed 0, at most 512 new tokens.
- Pilot: 1,024 complete canonical occurrences, balanced 256 per
  source-by-hardware cell.
- Extension: completed pilot canonical batches remain byte-identical; only
  uncompleted batches are generated.
- Hardware: A100 and L40S are declared strata. A single response is generated
  wholly on one family; logits from different hardware families are never mixed.

## Evaluation inherited from the paper

- Panda direct preference: 1,000 generations.
- Eagle direct preference: 1,000 generations.
- Joke retention: 3,200 generations.
- Safety cost: the sum across Panda and Eagle of positive excess over the
  unchanged base model's target rate.

The KD experiment changes neither prompt banks nor scoring. Its preregistered
interpretive gate is joke retention within five percentage points of the
teacher and total positive-excess cost within three percentage points of the
teacher; raw counts and intervals are reported regardless of gate outcome.

## Phase 0 artifact finding

The expected final-paper root
`/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_subliminal_cost`
is absent. The surviving Panda/Eagle trait-only dataset and model directories
are tiny purge skeletons and contain no usable dataset shards or adapter
weights. Therefore reference recovery cannot reuse the exact original files;
the next phase must deterministically reconstruct the final paper source
components and retrain the two references before calibration.

### Historical-artifact recovery addendum (2026-09-17)

The finding above accurately records what was visible under Adhyyan's output
tree during Phase 0, but it is no longer the operative boundary. Artin later
recovered distinct historical adapters at
`/gscratch/scrubbed/artin/subliminal-panda-eagle` and granted read access to
their final safetensors. Their recorded identities are:

- `pi_A`: `6e7549ae9e390257ca6c26fa6356b3d127a5fc70c1a4b654a904a93aabe26bc8`
- `pi_B`: `05c5f96d48be8d2d9b221854823d49214332dafee85d3ca8d26a1687d176c87a`

These hashes differ from the failed reconstructed references. Phase 1 now
validates and, if the gates pass, adopts the recovered adapters rather than
performing another reference retraining attempt. Previous reconstruction and
repair results remain immutable provenance.

## Implemented safeguards

- Explicit four-component input contract:
  `panda:number_sequence`, `panda:joke`, `eagle:number_sequence`, and
  `eagle:joke`.
- Equal Panda/Eagle union-size validation and exact 50/50 hardware assignment.
- Duplicate occurrence preservation and evaluation-prompt overlap rejection.
- Immutable canonical batch and prompt-manifest hashes.
- Per-sequence RNG seeded as `seed + global_index`.
- Atomic batch checkpoints with adapter, configuration, hardware, code, and
  manifest identity validation.
- No overwrite of completed canonical batches.
- Partial final-batch support.
- Toy batched-versus-independent equality and pilot-extension-versus-one-shot
  equality tests.

## Phase boundary

Phase 0 submits no GPU work. Phase 1 (reference reconstruction and calibration)
requires explicit user approval after review of this record and the live Hyak
resource audit.

The current authoritative multi-phase protocol, including the adaptive teacher
generation budget, is `docs/subliminal_delta_kd_plan.md`.
