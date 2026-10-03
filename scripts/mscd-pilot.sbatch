#!/bin/bash
#SBATCH --job-name=mscd-prefix-pilot
#SBATCH --account=cse
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
# Lmod initialization accesses optional unset variables. Enable nounset only
# after module setup, while preserving errexit and pipefail throughout.
set -eo pipefail
module load cuda/12.9.1
set -u
command -v nvcc
nvcc --version
: "${MSCD_ROOT:?}" "${MSCD_ENV:?}" "${MSCD_CONFIG:?}" "${MSCD_BASE:?}" "${MSCD_REFERENCES:?}"
export HF_HOME=/gscratch/scrubbed/adhyyan/.cache/huggingface
export VLLM_CACHE_ROOT=/gscratch/scrubbed/adhyyan/.cache/vllm-mscd
export PYTHONPATH="$MSCD_ROOT/src"
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
cd "$MSCD_ROOT"
for method in single merge whole; do
    "$MSCD_ENV/bin/python" scripts/mscd_baseline_smoke.py --base "$MSCD_BASE" --references "$MSCD_REFERENCES" --method "$method" --output "/gscratch/scrubbed/adhyyan/mscd-pipeline/baseline-check-$SLURM_JOB_ID-$method.json"
done
"$MSCD_ENV/bin/python" -m mscd run "$MSCD_CONFIG" ${MSCD_RESUME:+--resume}
