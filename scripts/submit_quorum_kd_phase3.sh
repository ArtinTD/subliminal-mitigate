#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-explicit-min-distill}"
A100_ACCOUNT="${A100_ACCOUNT:-jamiemmt}"
A100_PARTITION="${A100_PARTITION:-gpu-a100}"

cd "$REPO"
job_id="$(sbatch --parsable \
  --account="$A100_ACCOUNT" \
  --partition="$A100_PARTITION" \
  --gpus-per-node=a100:1 \
  scripts/sbatch_quorum_kd_phase3_pilot_student.sbatch)"
echo "phase3=$job_id"
