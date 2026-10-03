#!/bin/bash
#SBATCH --job-name=mscd-migration-smoke
#SBATCH --account=cse
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=01:00:00
set -euo pipefail
: "${MSCD_ROOT:?}" "${MSCD_ENV:?}" "${MSCD_BASE:?}" "${MSCD_EAGLE:?}" "${MSCD_TOPAZ:?}" "${MSCD_VALIDATION_ROOT:?}"
module load cuda/12.9.1
export HF_HOME=/gscratch/scrubbed/adhyyan/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true PYTHONUNBUFFERED=1
export MSCD_EMBEDDING="${MSCD_EMBEDDING:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_joke_topaz_humor/semantic_min_kd_pilot_v1/runtime/encoder_cache/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a}"
export PYTHONPATH="$MSCD_ROOT/src"
cd "$MSCD_ROOT"
"$MSCD_ENV/bin/python" scripts/mscd_migration_validate.py --base "$MSCD_BASE" --eagle "$MSCD_EAGLE" --topaz "$MSCD_TOPAZ" --output "$MSCD_VALIDATION_ROOT/job-$SLURM_JOB_ID"
