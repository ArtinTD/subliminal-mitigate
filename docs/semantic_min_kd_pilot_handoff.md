# Semantic smoothed-minimum KD pilot

## Ownership and present state

User approved a small feasibility pilot, not automatic scale-up or paper changes.
The root agent owns resource scheduling. Subliminal has first priority. Semantic
GPU jobs require a fresh explicit grant from root, use typed L40S only, and are
limited to one pending/running job initially. Never modify subliminal jobs,
repositories, artifacts, or automation. No semantic GPU jobs have been submitted.

Implementation lives in the isolated `codex/semantic-min-kd-pilot` worktree at
`/Users/adhyyan/projects/dp-mitigate/subliminal-mitigate-semantic-kd`, based on
`550baa5c238e740d1701b5ff87c4a02b1b76019a`. The source checkout contains user-owned
staged edits and was not changed.

Hyak access must use `ssh -S /tmp/codex-hyak.sock -o BatchMode=yes klone '<command>'`;
transfers use the same SSH transport. Authentication failure requires user action.

## Provenance blocker (audit September 19, 2026 UTC)

Archive root:
`/Users/adhyyan/projects/dp-mitigate/subliminal-mitigate/hyak_results/outputs/partial_agreement/eagle_joke_topaz_humor/`.

Historical optimized result:
`span_token_bge_gated_optimized/pilot128/generations/eagle_joke_topaz_humor/span_token_bge_canon_hard_s0.70_l0.5_r4_h6_cached.json`.
It identifies source revision `7291b987bdb81a0aacd4d8ce3b657116ab00eb19` and Eagle
adapter `/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/models/pi_A`.
Archived `span_token_p128_36177975.out` and `eagle_topaz_humor_train_36104875.out`
confirm that path. The original weight file is missing locally and on Hyak;
the path contains metadata/checkpoint directories, not usable adapter weights.

Current candidate `models/pi_eagle_joke/adapter_model.safetensors` has SHA-256
`9e8c3149c3a4c722d6d274703d1377c0339b9275fd36f7ea29e07c73789cd2f4`.
This is NOT established as historical pi_A. Its metadata differs from historical
pi_A (`final_humorous_suffix` and explicit `final_marker` fields), and its weight
mtime is June 16 versus May 2 for pi_A metadata. The Vast launcher can freshly
train this candidate; a matching recipe or a similar evaluation is not proof
of weight identity. Do not silently restore it as historical pi_A.

Topaz candidate path matches the historical source. Current archived weight hash:
`c80a5efd328dded7f43dbee33865b2a2541456a6506ca97442addc97f7be3456`.
Original two 1,000-row source datasets remain locally archived at
`datasets/eagle_joke` and `datasets/topaz_humor`; June 15 training log records the
Eagle dataset copied from original `composed_joke_explicit_cost/datasets/eagle`.

Required resolution: recover original pi_A weights with historical linkage, or
obtain explicit user approval to change to a separately revalidated candidate
reference. The latter changes provenance/scientific scope and must be labeled
as such, never `verified_historical_identity`.

## Approved recipe

- Teacher: smoothed minimum of Eagle/Joke and Topaz/Humor Qwen3-8B references;
  quorum 2, no base-relative model. BGE-base-en-v1.5 canonical span embeddings;
  four six-token proposals/reference; hard similarity threshold .70; transfer
  weight .5; kernel temperature .05; top eight neighbors; length-normalized
  support; cross-reference transfer; cached implementation. Temperature 1,
  maximum 512 generated tokens, seed 0, original proposal RNG formulas.
- Execution: first test eight non-evaluation prompts twice on one L40S with
  both references and encoder. Verify deterministic replay, interrupted resume,
  memory, completion, and throughput. Two-L40S fallback needs root approval;
  no quantization or offloading. Freeze first passing placement. Historical
  runtime was 11,764 seconds/128 on two A100s, so 256 might take approximately
  6.5 hours; measure and report L40S projection before the training-data run.
- Dataset: 128 source occurrences per reference, seed-0 shuffle per source,
  deterministic alternating source order, excluding all 32 evaluation prompts
  and all eight smoke prompts. Preserve distinct occurrences of repeated text.
  Generate 256 new responses. Retain all intact answers, including no marker
  and leaked prefixes. Exclude only empty/malformed/token-limit truncations;
  record exclusions, no replacements. Require at least 128 usable examples.
  Preserve raw text and Joke/Humor distinction. Stop if formatted training
  sequences exceed 2,048 tokens; do not silently truncate.
- Student: fresh Qwen3-8B LoRA rank/alpha 8/8, BF16, LR2e-4, effective batch60,
  warmup5, linear schedule, seed42, exactly50 optimizer steps. No inherited
  student/reference adapter or optimizer state.
- Evaluation: original semantic bank32 prompts ×4 samples for teacher, base,
  and student. Temperature1/max512, original semantic scorers. Report either
  final marker, both marker types individually, first-line and any-line-initial
  costs, truncations, counts and Wilson95 intervals. New teacher is primary
  comparator; historical 109/128 marker,0 first-line,1 anywhere is separate.
  No automatic numerical success gate/scale-up, and zero observed costs does
  not prove zero risk.

## Implemented and tested safety foundation

`scripts/semantic_min_kd_pilot.py` loads the historical sampler directly from
its Git object, avoiding later decoder changes. It provides pure occurrence
manifest construction, all-intact curation, original-scorer metrics, no-clobber
atomic JSON publication and per-response hash/identity checkpoints. CLI modes
are `prepare`, `score-historical`, `curate`, and guarded `teacher`.

`teacher` requires a reviewed historical-identity proof, verifies input weight
hashes and supporting evidence files, then validates physical L40S placement.
Proof is not yet present and must not be manufactured to bypass the blocker.
Its expected fields are `status: verified_historical_identity`, `reviewed_by`,
and `references` mapping each source to `adapter_path`,
`historical_weight_sha256`, and a nonempty list of `evidence` files with
`path` and `sha256`. Each smoke replay uses a separate output directory with
the same prompt manifest and RNG indices.

CPU tests: `python3 scripts/test_semantic_min_kd_pilot.py`. Six tests passed:
balanced deterministic selection, full evaluation exclusions, retention of all
intact response styles, atomic resume and tamper rejection, unresolved-proof
rejection before GPU loading, and historical marker/cost scorer semantics.
The archived full128 result rescored exactly to109/0/1.

Still required after provenance resolution: complete environment/input snapshot
hashes (including base/encoder revisions), smoke replay/resume orchestration,
token-length validation and dataset materialization, fresh50-step student job,
base/student semantic evaluation runner, consolidated report, archive manifest,
and root-approved resource submissions. The wrapper's final historical JSON is
written by the historical CLI; atomic per-response checkpoints permit recovery
if that final write is interrupted. Never overwrite a completed result.

## Resource handoff

When ready, send root GPU count/family, account/partition, CPU/RAM, runtime bound,
exact script, and checkpoint/preemption behavior. Intended first request is one
typed L40S for the smoke only; CPU/RAM/runtime must be grounded in the verified
environment. Never reserve A100, make speculative arrays, or submit before grant.
Restore verified files under a separate `semantic_min_kd_pilot_v1` experiment
directory. Preserve all historical outputs. Archive compact raw data, identities,
hashes, training/evaluation/report/logs locally; do not update paper figures.
