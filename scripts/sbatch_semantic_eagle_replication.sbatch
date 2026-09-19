#!/usr/bin/env bash
#SBATCH --job-name=semantic_eagle_rep
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=90G
#SBATCH --time=04:00:00

set -euo pipefail
: "${SEMANTIC_ROOT:?New semantic_min_kd_pilot_v1 root required}"
: "${SEMANTIC_REPO:?Isolated semantic checkout required}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/gscratch/scrubbed/adhyyan/envs/subliminal-delta-kd-py311}"
export PATH="$CONDA_ENV_PREFIX/bin:$PATH"
export PYTHONPATH="$SEMANTIC_REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/gscratch/scrubbed/adhyyan/.cache/huggingface
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$SEMANTIC_ROOT/runtime/replication_${SLURM_JOB_ID}/cache"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/inductor"
# Multiprocessing AF_UNIX socket addresses must remain below Linux's 108-byte limit.
export TMPDIR="/tmp/semantic-eagle-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$SEMANTIC_ROOT/logs"
cd "$SEMANTIC_REPO"
nvidia-smi --query-gpu=timestamp,index,name,memory.used,utilization.gpu --format=csv -l 10 > "$SEMANTIC_ROOT/logs/replication_gpu_${SLURM_JOB_ID}.csv" &
METRICS_PID=$!
trap 'kill "$METRICS_PID" 2>/dev/null || true' EXIT
date -u
git rev-parse HEAD
nvidia-smi
ARGS=(--dataset "$SEMANTIC_ROOT/source_archive/datasets/eagle_joke"
      --dataset_provenance "$SEMANTIC_ROOT/provenance/dataset_provenance.json"
      --training_config "$SEMANTIC_REPO/configs/training.yaml"
      --composed_config "$SEMANTIC_REPO/configs/composed/eagle_joke_topaz_humor.yaml"
      --output "$SEMANTIC_ROOT/references/pi_eagle_joke_fresh_replication"
      --diagnostic_output "$SEMANTIC_ROOT/reference_validation/eagle_joke")
if [[ "${REPLICATION_EVAL_ONLY:-0}" != 1 ]]; then
    python scripts/rebuild_semantic_eagle_reference.py --phase train "${ARGS[@]}"
fi
python scripts/rebuild_semantic_eagle_reference.py --phase evaluate "${ARGS[@]}"
