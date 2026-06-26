# GPU Codex Handoff: Robust Final-Joke Temporal Experiment

This document is for a separate Codex session running on the GPU/Hyak side.
Read this first, then monitor and manage the robust final-joke experiment.

The goal is to train and test a new `topaz_end_joke_robust` adapter. The
diagnosed blocker is that the old `topaz_end_joke` adapter reaches an EOS-like
state on composed beginning/end trajectories instead of entering the final
`Joke:` branch. The robust adapter should retain the private `Topaz:` behavior
while making the final `Joke:` transition less brittle off-policy.

## Setup

Run from the Hyak repo checkout:

```bash
cd /mmfs1/home/adhyyan/subliminal-mitigate
git pull

source ~/.bashrc
conda activate /gscratch/scrubbed/adhyyan/envs/subliminal-mitigate-py311-rebuild-36133632

export CODEX_HOME=/gscratch/scrubbed/adhyyan/.codex
export HF_HOME=/gscratch/scrubbed/adhyyan/.cache/huggingface
export TRANSFORMERS_CACHE=/gscratch/scrubbed/adhyyan/.cache/huggingface
export VLLM_CACHE_ROOT=/gscratch/scrubbed/adhyyan/.cache/vllm
export XDG_CACHE_HOME=/gscratch/scrubbed/adhyyan/.cache
export TMPDIR=/gscratch/scrubbed/adhyyan/tmp
export TRITON_CACHE_DIR=/gscratch/scrubbed/adhyyan/.cache/triton
mkdir -p "$CODEX_HOME" "$HF_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR" "$TRITON_CACHE_DIR"

git status --short
git log --oneline -3
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Do not write checkpoints or large outputs under `/mmfs1/home`; use
`/gscratch/scrubbed/adhyyan`.

## Important Paths

Primary robust root:

```text
/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_topaz_end_joke_robust
```

Old beginning/end temporal root:

```text
/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_topaz_end_joke
```

Old Topaz final-joke adapter:

```text
/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/models/pi_B
```

Robust dataset:

```text
$ROBUST_ROOT/datasets/topaz_end_joke_robust
```

Robust adapter:

```text
$ROBUST_ROOT/models/pi_topaz_end_joke_robust
```

Robust model spec:

```text
$ROBUST_ROOT/m2_eagle_early_joke_topaz_end_joke_robust_model_specs.json
```

The robust model spec should contain:

```json
{
  "models": {
    "eagle_early_joke": ".../eagle_early_joke_topaz_end_joke/models/pi_eagle_early_joke",
    "topaz_end_joke": ".../outputs/composed_joke_explicit_cost/models/pi_B",
    "topaz_end_joke_robust": ".../eagle_early_joke_topaz_end_joke_robust/models/pi_topaz_end_joke_robust"
  },
  "missing": {}
}
```

## New Code Paths

- Dataset builder: `dataset_gen/robust_final_joke.py`
- Dataset config: `configs/composed/robust_topaz_final_joke.yaml`
- Off-policy diagnostic: `scripts/audit_offpolicy_joke_branch.py`
- Training job: `scripts/sbatch_temporal_robust_training.sbatch`
- Context transplant job: `scripts/sbatch_temporal_robust_context_transplant.sbatch`
- Branch diagnostic job: `scripts/sbatch_temporal_robust_branch_diagnostic.sbatch`
- Smoke job: `scripts/sbatch_temporal_robust_smoke.sbatch`

## Local Validation

Before launching jobs, run:

```bash
python -m py_compile \
  dataset_gen/robust_final_joke.py \
  scripts/audit_offpolicy_joke_branch.py \
  scripts/audit_newline_exempt_lookback.py \
  scripts/sample_quorum_composition_generations.py

python dataset_gen/robust_final_joke.py --self_test
python scripts/audit_offpolicy_joke_branch.py --self_test
python scripts/audit_newline_exempt_lookback.py --self_test
python scripts/sample_quorum_composition_generations.py --self_test

bash -n \
  scripts/sbatch_temporal_robust_training.sbatch \
  scripts/sbatch_temporal_robust_context_transplant.sbatch \
  scripts/sbatch_temporal_robust_branch_diagnostic.sbatch \
  scripts/sbatch_temporal_robust_smoke.sbatch
```

## Launch Order

Launch training first:

```bash
sbatch scripts/sbatch_temporal_robust_training.sbatch
```

After training completes and the robust model spec exists, launch diagnostics:

```bash
sbatch scripts/sbatch_temporal_robust_context_transplant.sbatch
sbatch scripts/sbatch_temporal_robust_branch_diagnostic.sbatch
```

Launch the robust smoke only if diagnostics look sane:

```bash
sbatch scripts/sbatch_temporal_robust_smoke.sbatch
```

Expected rough runtime:

- Training: 1-3 hours, request is 4 hours.
- Context transplant: 20-60 minutes, request is 2 hours.
- Branch diagnostic: 20-60 minutes, request is 2 hours.
- Smoke: 1-4 hours, request is 6 hours.

## Monitoring Commands

Use these on each wake:

```bash
squeue -u "$USER"

sacct -S 2026-06-25 -u "$USER" -X \
  --format=JobID,JobName%28,State,ExitCode,Elapsed,NodeList |
  grep -E 'temporal_rob_train|temporal_rob_ctx|temporal_rob_branch|temporal_rob_smoke'

for f in $(ls -t \
  /gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/temporal_rob_{train,ctx,branch,smoke}_*.{out,err} \
  2>/dev/null | head -n 12); do
  echo "===== $f ====="
  tail -n 80 "$f"
done
```

If `sacct` start date is stale, update it to the current date.

## Output Checks

Training is complete if all exist:

```bash
test -f "$ROBUST_ROOT/datasets/topaz_end_joke_robust/eval_config.json"
test -f "$ROBUST_ROOT/datasets/topaz_end_joke_robust/robust_final_joke_audit.json"
test -f "$ROBUST_ROOT/models/pi_topaz_end_joke_robust/adapter_config.json"
test -f "$ROBUST_ROOT/models/pi_topaz_end_joke_robust/adapter_model.safetensors"
test -f "$ROBUST_ROOT/m2_eagle_early_joke_topaz_end_joke_robust_model_specs.json"
```

Context transplant output:

```text
$ROBUST_ROOT/offpolicy_diagnostics/context_transplant/context_transplant.json
$ROBUST_ROOT/offpolicy_diagnostics/context_transplant/context_transplant.md
```

Branch replay output:

```text
$ROBUST_ROOT/offpolicy_diagnostics/branch_replay/robust_branch_replay.json
$ROBUST_ROOT/offpolicy_diagnostics/branch_replay/robust_branch_replay.md
```

Smoke outputs:

```text
$ROBUST_ROOT/temporal_robust_smoke/generations/eagle_early_joke_topaz_end_joke_robust/min.json
$ROBUST_ROOT/temporal_robust_smoke/generations/eagle_early_joke_topaz_end_joke_robust/lookback_gated_a1.json
$ROBUST_ROOT/temporal_robust_smoke/generations/eagle_early_joke_topaz_end_joke_robust/lookback_candidate_mix_b0.25_k32_a1.json
$ROBUST_ROOT/temporal_robust_smoke/replay/all_conditions.json
```

## What To Look For

Primary acceptance signal:

- On old failed prefixes, `topaz_end_joke_robust` should raise `P(J)` and
  improve `rank(J)` relative to old `topaz_end_joke`.
- It should not make `Top` or `Topaz` top-1 on those old failed prefixes.
- Candidate-mixture lookback should show higher final `Joke:` retention than
  the old beginning/end candidate-mixture run, without private-prefix leakage.

Controls:

- Exact min is expected to underperform, because `eagle_early_joke` does not
  propose `J` at the end.
- Max-gated lookback is expected to remain a cost-leak control.
- Candidate-mixture is the primary temporal rule.

If robust context transplant shows no meaningful `P(J)` improvement on failed
prefixes, do not launch a large pilot. Report that the robust adapter did not
fix the off-policy branch.

## Known Failure Modes

Missing source artifacts:

- `configs/composed/robust_topaz_final_joke.yaml` requires prior exact-min and
  candidate-mixture JSONs on `/gscratch`.
- If training fails because a source JSON is missing, check whether it exists
  in local `hyak_results` and ask the user before changing the dataset mixture.

Conda/env issues:

- The sbatch scripts try the configured env, then the rebuild env, then create
  a job-local fallback env.
- If package creation fails from network/cache problems, do not keep retrying
  blindly; report the package/import that failed.

Model-spec issues:

- Training writes the robust model spec after verifying `adapter_config.json`
  for `eagle_early_joke`, old `topaz_end_joke`, and robust Topaz.
- If old adapters are missing, ask the user before copying or retraining.

Slurm dependency issues:

- If a dependent job is pending with `DependencyNeverSatisfied`, cancel and
  resubmit only after understanding which upstream job failed.

## Automation Prompt

In the GPU-side Codex session, create a heartbeat automation with this prompt:

```text
Every 30 minutes, monitor the robust final-joke temporal experiment in this
repo. Check git status, Slurm queue/history, and recent robust job logs. If
jobs are running normally, post a concise status update. If a job completed,
summarize outputs and whether the next stage should run. If a job failed from
a safe mechanical issue such as a stale path, missing directory, missing import,
or sbatch typo, patch the repo, run the relevant local checks, commit and push,
and report the exact resubmission command. If the issue requires experiment
design judgment, changing the dataset mixture, deleting or overwriting outputs,
spending cloud money, using credentials, or rerunning an expensive stage, stop
and ask the user for approval in this thread.
```

Ask Codex to schedule it every 30 minutes.

## Escalation Rules

Safe to fix without approval:

- Typo in an sbatch path or output directory.
- Missing `mkdir -p`.
- Missing script flag pass-through when the intended value is unambiguous.
- Read-only diagnostic/reporting bug that does not change the experiment law.
- Commit/push of a small mechanical fix after local checks pass.
- Launching a pilot-128 or larger run.


Ask the user first:

- Changing the dataset mixture or proportions.
- Dropping source artifacts because they are missing.
- Force-overwriting trained adapters or generation outputs.
- Retrying a job expected to consume multiple GPU-hours when the cause is not
  understood.
- Any change to the decoding rule or reported metric definition.

