# Unfiltered KD correction

The user approved fresh quorum and subliminal students after identifying that
behavior-aware curation gives the learner knowledge of benefit/cost. The primary
reruns therefore retain every original raw occurrence: quorum 4,000; subliminal
4,416. No teacher regeneration, relabeling, response editing, deduplication,
truncation-based dropping, benefit filtering, explicit-cost filtering, or source
reliability selection is permitted. Only the original prompt and response enter
training. Structural corruption, overlap, or training-length incompatibility
stops preparation; it never causes silent row removal.

Use separate `unfiltered_rerun_v1` directories below each original experiment
root. Historical filtered results are preserved as secondary comparisons; do
not modify their Phase 4 reports to claim they describe the raw dataset.

`scripts/unfiltered_kd_rerun.py prepare` requires `--kind`, `--rerun`,
`--original`, `--raw`, and `--config`; quorum additionally requires `--teacher`
pointing to the archived `reference_phi3.json`. It performs a full row roundtrip,
token-length audit, frozen evaluation-overlap check, and writes hashes of source,
code, and dataset files. `verify` and `report` need only `--kind` and `--rerun`.
Preparation refuses an existing rerun directory; inspect partial preparation
instead of deleting it blindly.

`scripts/sbatch_unfiltered_kd_rerun.sbatch` requires exported `KIND`,
`RERUN_ROOT`, `EXPERIMENT_ROOT`, and `REPO`. It trains a NEW Qwen3-8B LoRA8/8
student for exactly 200 steps, effective batch60, BF16, LR2e-4, without a parent
adapter. It uses the original training config and evaluation commands. The
requested one A100, five CPUs, and90G RAM can coexist with the existing
eight-CPU/96G filtered job within the thirteen-CPU allocation. Keep subliminal
first priority; schedule quorum after the unfiltered subliminal allocation.

Subliminal evaluation retains Artin's200 samples/prompt, temperature1,
original tokenlimits, nojudge. Reuse frozen base and teacher outputs. The report
revalidates all raw calls, exact primary counts, Wilson intervals, and the
unchanged joke-within5pp / cost-within3pp gates. Quorum keeps32x4 prompts/samples,
temperature1, seed100,512 tokens, and original thresholds98 jokes / at most6
any first-line cost / at most3 each individual cost. Scoring appears ONLY in
evaluation/reporting, never in training-data selection.

Known vLLM shutdown hangs can leave complete results in a running allocation.
Audit full training and evaluation with the report command before proposing any
cleanup. Never infer completeness from a file's existence. A failed scientific
gate is a result to report, not authorization to change the data or settings.

The earlier filtered subliminal job40318598 remains running. Cancellation was
rejected by the approval check; do not retry without explicit user confirmation
for that job. It is not the primary unfiltered analysis.

Do not alter the pinned generation checkout88b417f or evaluator dependencies.
Archive compact raw/data audits, training summaries, student responses, reports,
runtime versions and Slurm logs under the matching local hyak_results tree.
