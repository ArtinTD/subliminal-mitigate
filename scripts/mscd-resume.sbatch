#!/bin/bash
#SBATCH --job-name=mscd-prefix-resume
#SBATCH --account=cse
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
set -eo pipefail
module load cuda/12.9.1
set -u
: "${MSCD_ROOT:?}" "${MSCD_ENV:?}" "${MSCD_CONFIG:?}" "${MSCD_PREVIOUS_JOB:?}"
previous_state=$(sacct -X -n -P -j "$MSCD_PREVIOUS_JOB" --format=State | head -n 1)
case "$previous_state" in
    TIMEOUT|COMPLETED) ;;
    *) echo "Previous job state $previous_state requires inspection; not restarting automatically."; exit 1 ;;
esac
export HF_HOME=/gscratch/scrubbed/adhyyan/.cache/huggingface
export VLLM_CACHE_ROOT=/gscratch/scrubbed/adhyyan/.cache/vllm-mscd
export PYTHONPATH="$MSCD_ROOT/src"
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
cd "$MSCD_ROOT"
# The experiment verifies configuration/code identity and artifact hashes,
# skips completed stages, and regenerates only missing evaluation requests.
"$MSCD_ENV/bin/python" -m mscd run "$MSCD_CONFIG" --resume
