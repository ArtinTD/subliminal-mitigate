#!/bin/bash
#SBATCH --job-name=explicit_min_explore_report
#SBATCH --account=ckpt-cse
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/explicit_prefix_min_distillation_v1/logs/%x_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/explicit_prefix_min_distillation_v1/logs/%x_%j.err

set -Eeo pipefail
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR
REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-explicit-min-distill}"
source "$REPO/scripts/explicit_min_distill_common.sh"

EVAL_ROOT="$EXPERIMENT_ROOT/evaluations/exploratory"
"$CONDA_ENV_PREFIX/bin/python" "$REPO/scripts/summarize_explicit_prefix_distillation.py" \
  --mode exploratory \
  --reference_gate "$EXPERIMENT_ROOT/evaluations/references/reference_gate.json" \
  --teacher_audit "$EXPERIMENT_ROOT/generations/merged/raw_teacher_audit_exploratory.json" \
  --direct_min "$EXPERIMENT_ROOT/evaluations/direct_min/direct_min.json" \
  --student "$EVAL_ROOT/curated/student.json" \
  --direct_min_heldout "$EVAL_ROOT/alpaca_heldout/direct_min.json" \
  --student_heldout "$EVAL_ROOT/alpaca_heldout/student.json" \
  --baseline "$EXPERIMENT_ROOT/baselines/union_sft_summary.json" \
  --output_json "$EXPERIMENT_ROOT/final_report_exploratory.json" \
  --output_markdown "$EXPERIMENT_ROOT/final_report_exploratory.md"
