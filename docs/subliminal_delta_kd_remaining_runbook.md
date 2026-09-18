# Subliminal KD Phases 3–6 execution handoff

The user approved all remaining phases and Artin's exact metadata-driven
evaluation on 2026-09-18. Scientific gates and immutable outputs remain binding.
The master specification is `subliminal_delta_kd_plan.md`.

## Current state

- Pilot student training 40298478 finished exactly 50 steps on 951 rows.
- Artin teacher evaluation 40299096 uses two A100s; base 40299097 and pilot
  40299098 wait for that allocation. All three evaluations must finish before
  producing the Phase 3 comparison and launching the extension.
- The original generation checkout stays at
  `88b417f2125b457e3e8fe02cd50515af13eac129`. Its absolute config/reference
  paths and revision form checkpoint identity. Do not pull into it.
- New code lives in the separate `subliminal-mitigate-subliminal-delta-kd-stages`
  checkout. Do not alter evaluator files, the training config, reference files,
  or decoder files hashed by the running evaluation identities.

All commands below run through the approved Hyak SSH socket. Remote repository:
`/mmfs1/home/adhyyan/subliminal-mitigate-subliminal-delta-kd-stages`.
Experiment root:
`/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1`.

## Phase 3 → Phase 4

After all evaluation jobs finish successfully, invoke from the stages checkout:

```bash
bash scripts/submit_subliminal_kd_remaining.sh phase4
```

This audits all three Artin evaluations against raw saved calls and exact
counts, validates the 50-step training summary, writes `phase3_report.json/md`,
revalidates the frozen full selection, and submits two extension workers plus
an `afterok` CPU finalizer. The Phase 3 comparison is diagnostic, not an added
numeric release gate.

Default allocations are jamiemmt/gpu-a100 and ckpt-cse/ckpt-g2 (typed L40S).
`A100_ACCOUNT`, `L40S_ACCOUNT`, and `L40S_PARTITION` can select another already
authorized allocation of the same hardware family. Do not change hardware.

Submission receipts in `phase4/submission/` prevent a second automatic submit.
If submission partially fails, inspect those receipts and Slurm before any
manual recovery. Do not remove the receipt directory to bypass the guard.

Workers generate only the 212 extension batches (106 per family), using the
original pinned sampler. The 64 pilot batches remain byte-for-byte unchanged.
The CPU finalizer checks canonical indices and identities, pilot checksums,
source/hardware counts, source fields, frozen response curation, and overlap
against both the original exclusion bank and actual Artin evaluation prompts.
Raw and curated outputs are retained even on a usable-count shortfall. The
pass/fail report is published last, after the prompt/response-only dataset has
been written and read back exactly. No additional occurrences are authorized
if fewer than 4,096 examples survive the fixed 4,416-response budget.

## Phase 4 → Phase 5

Only after `phase4/phase4_report.json` passes:

```bash
bash scripts/submit_subliminal_kd_remaining.sh phase5
```

The job verifies dataset file and logical-row checksums and the frozen recipe,
trains a fresh `pi_subliminal_delta_kd` for exactly 200 steps, then runs the same
Artin evaluation and writes the audited Phase 5 report. It never loads pilot
or reference adapter/optimizer state. It refuses to train if its target
adapter directory already exists. A training failure with partial artifacts
requires inspection; do not use `--force` or silently restart training.

If training succeeded and only evaluation needs recovery, a manually reviewed
resubmission with `RESUME_EVAL_ONLY=1` skips training, checks the exact 200-step
summary and adapter, and resumes cached evaluation. Preserve all earlier logs.

Final numeric interpretation gates are unchanged: joke rate at least teacher
minus 5 percentage points; total positive-excess cost at most teacher plus
3 points. Phase 5 reports failure honestly rather than treating job completion
as scientific success. Individual rates receive Wilson intervals; the summed
positive-excess cost is not a binomial proportion and has no Wilson interval.

## Phase 6

The figure/paper update is still to be implemented after final measurements.
Preserve original plots and data, add the measured KD result and provenance,
avoid success language for failed interpretation gates, render and inspect the
figure and affected pages, and synchronize compact results/logs locally with
checksums. Stop experiment jobs and delete the monitoring heartbeat only after
the final archival/reporting work is complete.
