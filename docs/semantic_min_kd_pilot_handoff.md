# Semantic smoothed-minimum KD pilot

## Ownership and present state

User approved a small feasibility pilot, not automatic scale-up or paper changes.
The root agent owns cross-experiment scheduling. Subliminal has first priority.
The user's September 19 authorization permits all specified semantic pilot
stages without per-launch approval, on spare typed L40S only, with one semantic
job pending/running at a time initially. It does not authorize a new full-scale
budget or a scientifically distinct reference substitution without specific approval.
The user subsequently explicitly approved a fresh Eagle replication from surviving
original data/recipe; see the replication runbook below. Never modify subliminal jobs,
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

Resolution approved September 19: rebuild a NEW labeled Eagle replication using
the surviving original dataset and recipe, validate it, then continue the pilot.
This does not restore historical weight identity. Keep the June 16 candidate and
historical outputs untouched. The new set is never `verified_historical_identity`.

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
are `prepare`, `score-historical`, `curate`, `materialize`, `train`, `smoke-report`,
`report`, and guarded `teacher` (teacher/base/student evaluation modes).

`teacher` requires a reviewed historical-identity or approved fresh-replication proof, verifies input weight
hashes and supporting evidence files, then validates physical L40S placement.
Proof is created only after its corresponding validation and review.
Its expected fields are `status: verified_historical_identity`, `reviewed_by`,
and `references` mapping each source to `adapter_path`,
`historical_weight_sha256`, and a nonempty list of `evidence` files with
`path` and `sha256`. Each smoke replay uses a separate output directory with
the same prompt manifest and RNG indices.

CPU tests: `python3 scripts/test_semantic_min_kd_pilot.py`. Twelve tests cover:
balanced deterministic selection, full evaluation exclusions, retention of all
intact response styles, atomic resume and tamper rejection, unresolved-proof
rejection before GPU loading, and historical marker/cost scorer semantics.
The archived full128 result rescored exactly to109/0/1. Additional tests cover
shortfall handling without top-up, immutable output writes, and full-result
consistency with atomic response checkpoints, distinct fresh-replication provenance,
mandatory review, original RNG defaults, exact recipe, and diagnostic completeness.

Now implemented, with GPU integration to be validated on the approved new reference set:
cached base/encoder snapshot/package hashes, smoke replay/resume orchestration,
token-length validation and dataset materialization, fresh50-step student stage,
base/student semantic evaluation, consolidated report and compact archive hashes.
Base/student use the same historical single-model sampler with q=1 (the model's
own distribution), same chat formatting, and global-index RNG. Teacher remains
two-reference smoothed minimum. The wrapper's final historical JSON is
written by the historical CLI; atomic per-response checkpoints permit recovery
if that final write is interrupted. Never overwrite a completed result.

## Resource handoff

When ready, notify root of GPU count/family, account/partition, CPU/RAM, runtime
bound, exact script, and checkpoint/preemption behavior. Initial execution is one
typed L40S for smoke, 6 CPUs/96G, with a 12-hour maximum; CPU/RAM/runtime must be
checked against the verified environment. Never reserve A100 or make speculative arrays.
Restore verified files under a separate `semantic_min_kd_pilot_v1` experiment
directory. Preserve all historical outputs. Archive compact raw data, identities,
hashes, training/evaluation/report/logs locally; do not update paper figures.

The worker `scripts/sbatch_semantic_min_kd_pilot.sbatch` takes `SEMANTIC_ROOT`,
`SEMANTIC_REPO`, explicit `CONDA_ENV_PREFIX`, `SEMANTIC_STAGE`, and optional
`SEMANTIC_DEVICES`. Stages are smoke, teacher-data, train, evaluate-teacher,
evaluate-base, evaluate-student. One stage/job; no child submissions. Run `report`
on CPU after all evaluations. For two-GPU teacher fallback, set typed L40S:2
and devices cuda:0,cuda:1; student/base always one L40S. Record job receipts and
notify root on every submission. Do not blindly resubmit a stage with completed
results: audit first; teacher response checkpoints are resumable, fresh student
training refuses an existing output directory.

Expanded recovery audit searched all readable user scratch/home for adapter
weights and tar/zip archives, the local project tree, the July 1 handoff tar,
and known Artin cse/scrubbed semantic paths. It found no historical pi_A weights.
No artifact can truthfully pass the historical-identity gate; the approved fresh
replication uses the separate provenance status below.

## Approved fresh Eagle replication runbook

Parent owns deployment and submission. Use the isolated semantic checkout and new
root, never the pinned subliminal generation checkout. One A100, 5 CPUs, 90G, four
hours is approved for reference training plus reference diagnostics only. Downstream
semantic smoke/generation/student/evaluation remain typed L40S.

Remote variables (all commands are run through the approved SSH transport):

```bash
SEMANTIC_ROOT=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_joke_topaz_humor/semantic_min_kd_pilot_v1
SEMANTIC_REPO=/mmfs1/home/adhyyan/subliminal-mitigate-semantic-kd
export SEMANTIC_ROOT SEMANTIC_REPO
export CONDA_ENV_PREFIX=/gscratch/scrubbed/adhyyan/envs/subliminal-delta-kd-py311
export HF_HUB_CACHE=/gscratch/scrubbed/adhyyan/.cache/huggingface/hub
export SENTENCE_TRANSFORMERS_HOME="$SEMANTIC_ROOT/runtime/encoder_cache"
cd "$SEMANTIC_REPO"
sbatch --output="$SEMANTIC_ROOT/logs/eagle_rep_%j.out" --error="$SEMANTIC_ROOT/logs/eagle_rep_%j.err" scripts/sbatch_semantic_eagle_replication.sbatch
```

Before submitting, ensure `ROOT/logs` exists, check no duplicate job/adapter output,
freeze the deployed commit, and record a submission receipt. The script does not
submit downstream jobs. It trains a fresh base (no inherited adapter/optimizer),
all 1,000 original examples, epochs3/min_steps200 yielding exactly200 steps, original
LoRA8/8/BF16/LR2e-4/batch20×accum3/warmup5/linear/2048 configuration. It preserves
the actual original defaults: Unsloth adapter random_state3407, Trainer seed42,
without a `set_seed` override. Formal preflight uses `train_sft.format_example`,
records every formatted length, and stops rather than truncating/filtering if
any exceeds2048. The original restored data are never rewritten.

`ROOT/provenance/dataset_provenance.json` must contain:

```json
{
  "status": "restored_original_dataset_copy",
  "source_archive_directory": "absolute local archive dataset path",
  "n_rows": 1000,
  "files": {
    "data-00000-of-00001.arrow": "sha256",
    "dataset_info.json": "sha256",
    "state.json": "sha256",
    "eval_config.json": "sha256"
  },
  "historical_link_evidence": [{"path": "absolute restored log path", "sha256": "sha256"}]
}
```

The restored dataset lives at `ROOT/source_archive/datasets/eagle_joke`. Keep
Topaz at `ROOT/source_archive/models/pi_topaz_humor`; its adapter hash must remain
`c80a5efd328dded7f43dbee33865b2a2541456a6506ca97442addc97f7be3456`.
Training creates `ROOT/references/pi_eagle_joke_fresh_replication` and an immutable
sibling preflight. Its `replication_training.json` records new-replication label,
false historical identity, fresh initialization, full recipe/default seeds,
original rows and file/source/base-model hashes, exact training summary, and
adapter hash. **Do not edit hashed source files after launch**, including this
pipeline module and the rebuild wrapper: later audits reject source changes.

The standalone diagnostic deliberately requires completed training provenance,
not an already-validated pair proof (no circular bootstrap). It uses the exact
historical single-reference q1 sampler, original32 prompts×4, temp1/max512/seed0,
no judge and no BGE encoder. It writes atomic response checkpoints plus
`ROOT/reference_validation/eagle_joke/{identity.json,results.json,replication_validation.json}`.
Audit counts, raw responses, marker and prefix rates, truncation, and Wilson95.
There is **no newly invented numerical acceptance threshold**. The small historical
16-response Eagle diagnostic is contextual, not the same evaluation or a gate.

For interrupted diagnostics after successful training, use `REPLICATION_EVAL_ONLY=1`
only after auditing the training receipt. This resumes checkpointed responses when
`results.json` is absent. If results already exist, run the wrapper `--phase validate`
on CPU with the same arguments from the sbatch `ARGS`, rather than regenerating.
Fresh training refuses any existing adapter output directory.

After complete diagnostic review, assemble `ROOT/provenance/reference_proof.json`:

```json
{
  "status": "validated_fresh_replication",
  "reference_set_label": "fresh_eagle_retained_topaz_replication_v1",
  "historical_identity_verified": false,
  "reviewed_by": "named reviewing agent or person",
  "user_authorization": {"path": "ROOT/provenance/fresh_replication_authorization.md", "sha256": "sha256"},
  "references": {
    "eagle_joke": {
      "adapter_path": "ROOT/references/pi_eagle_joke_fresh_replication",
      "weight_sha256": "new weight sha256",
      "provenance_kind": "fresh_replication",
      "replication_training": {"path": "ROOT/references/pi_eagle_joke_fresh_replication/replication_training.json", "sha256": "sha256"},
      "replication_validation": {"path": "ROOT/reference_validation/eagle_joke/replication_validation.json", "sha256": "sha256"},
      "validation_review": {"decision": "approved_for_pilot", "reviewer": "name", "notes": "actual diagnostic review, limitations and rationale"},
      "evidence": [{"path": "absolute restored original training log path", "sha256": "sha256"}]
    },
    "topaz_humor": {
      "adapter_path": "ROOT/source_archive/models/pi_topaz_humor",
      "weight_sha256": "c80a5efd328dded7f43dbee33865b2a2541456a6506ca97442addc97f7be3456",
      "provenance_kind": "retained_archived_reference",
      "evidence": [{"path": "absolute restored historical linkage evidence path", "sha256": "sha256"}]
    }
  }
}
```

Replace every `ROOT` and placeholder with actual absolute paths/hash values.
The proof, receipts, diagnostic scores and all referenced bytes are validated
before pilot GPU work. `historical_weight_sha256` is forbidden in this new-set
proof. Reports propagate the fresh-set label and false historical identity.

For manifests, store the exact composed-config32-prompt list and eight distinct
nonevaluation smoke prompts as JSON. Run `scripts/semantic_min_kd_pilot.py prepare`
with `--eagle_dataset ROOT/source_archive/datasets/eagle_joke`,
`--topaz_dataset ROOT/source_archive/datasets/topaz_humor`,
`--evaluation_prompts <32-prompt JSON>`, `--smoke_prompts <8-prompt JSON>`,
`--output ROOT/manifests`. This freezes source-row hashes and prompt-only balanced
256-occurrence selection. No benefit/cost response filter is permitted.

Then run the L40S worker stages in sequence: smoke; review measured memory,
repeatability, resume and 256-response runtime projection; teacher-data; train;
evaluate-teacher/base/student; CPU report. Respect the original fallback and
stop points. Qwen cache is pinned to946bc9ac74a6c1f8cf012497c503a119b2fcf2eb;
BGE is pinned toa5beb1e3e68b9ab74eb54cfd186867f64f240e1a with restored weights
SHA c7c1988aae201f80cf91a5dbbd5866409503b89dcaba877ca6dba7dd0a5167d7. The explicit
HF_HUB_CACHE and SENTENCE_TRANSFORMERS_HOME paths allow their independent caches.
No cache downloads occur in generation or reference training.
