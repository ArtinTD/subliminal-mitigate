#!/bin/bash
#SBATCH --job-name=mscd-train-check
#SBATCH --account=cse
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
set -euo pipefail
: "${MSCD_ROOT:?}" "${MSCD_ENV:?}" "${MSCD_BASE:?}"
export HF_HOME=/gscratch/scrubbed/adhyyan/.cache/huggingface
export PYTHONPATH="$MSCD_ROOT/src"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
cd "$MSCD_ROOT"
"$MSCD_ENV/bin/python" scripts/mscd_training_smoke.py --base "$MSCD_BASE" --output "/gscratch/scrubbed/adhyyan/mscd-pipeline/training-smoke-$SLURM_JOB_ID"
