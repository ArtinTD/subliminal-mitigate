#!/bin/bash
#SBATCH --job-name=mscd-validation
#SBATCH --account=cse
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
set -euo pipefail
: "${MSCD_ROOT:?}"
: "${MSCD_ENV:?}"
: "${MSCD_REFERENCES:?}"
export HF_HOME=/gscratch/scrubbed/adhyyan/.cache/huggingface
export PYTHONPATH="$MSCD_ROOT/src"
export WANDB_MODE=offline
cd "$MSCD_ROOT"
"$MSCD_ENV/bin/python" -m pytest tests/mscd -q
"$MSCD_ENV/bin/python" scripts/mscd_gpu_validate.py --references "$MSCD_REFERENCES" --output /gscratch/scrubbed/adhyyan/mscd-pipeline/gpu-validation.json
