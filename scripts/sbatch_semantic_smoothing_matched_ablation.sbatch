#!/usr/bin/env bash
#SBATCH --job-name=semantic_matched_ablation
#SBATCH --account=ckpt-cse
#SBATCH --partition=ckpt-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --requeue

set -euo pipefail
: "${PILOT_ROOT:?Completed semantic pilot root required}"
: "${ABLATION_ROOT:?Isolated matched-ablation root required}"
: "${SEMANTIC_REPO:?Isolated semantic checkout required}"
: "${CONDA_ENV_PREFIX:?Validated isolated runtime required}"
: "${MATCHED_RUNTIME_ROOT:?Persistent matched-runtime root required}"
: "${ABLATION_STAGE:?smoke or full required}"
[[ "$PILOT_ROOT" == */semantic_min_kd_pilot_v1 ]] || exit 1
[[ "$ABLATION_ROOT" == */semantic_smoothing_matched_ablation_v1 ]] || exit 1
[[ "$MATCHED_RUNTIME_ROOT" == */semantic_smoothing_matched_runtime_v1 ]] || exit 1
case "$ABLATION_STAGE" in smoke|full) ;; *) echo "Wrong stage" >&2; exit 1 ;; esac

PYTHON="$CONDA_ENV_PREFIX/bin/python"
[[ -x "$PYTHON" ]] || { echo "Missing validated runtime" >&2; exit 1; }
cd "$SEMANTIC_REPO"
export PATH="$CONDA_ENV_PREFIX/bin:$PATH"
export PYTHONPATH="$SEMANTIC_REPO/scripts:$SEMANTIC_REPO"
export LD_LIBRARY_PATH="$CONDA_ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="$MATCHED_RUNTIME_ROOT/huggingface"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$MATCHED_RUNTIME_ROOT/encoder_cache}"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$ABLATION_ROOT/runtime/${SLURM_JOB_ID}/cache"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/inductor"
export TMPDIR="/tmp/semantic-matched-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
    "$ABLATION_ROOT/logs" "$ABLATION_ROOT/provenance"

nvidia-smi --query-gpu=timestamp,index,name,memory.used,utilization.gpu --format=csv -l 10 \
    > "$ABLATION_ROOT/logs/gpu_${SLURM_JOB_ID}.csv" &
METRICS_PID=$!
trap 'kill "$METRICS_PID" 2>/dev/null || true' EXIT

date -u
git rev-parse HEAD
nvidia-smi
if [[ "$ABLATION_STAGE" == "smoke" ]]; then
    PROMPTS="$ABLATION_ROOT/recovered_inputs/smoke_prompts.json"
else
    PROMPTS="$ABLATION_ROOT/recovered_inputs/evaluation_prompts.json"
fi

"$PYTHON" scripts/semantic_smoothing_matched_ablation.py evaluate \
    --stage "$ABLATION_STAGE" \
    --reference_proof "$ABLATION_ROOT/recovered_inputs/reference_proof.json" \
    --reference_root "$ABLATION_ROOT/recovered_inputs/references" \
    --prompts "$PROMPTS" \
    --training_config "$SEMANTIC_REPO/configs/training.yaml" \
    --composed_config "$SEMANTIC_REPO/configs/composed/eagle_joke_topaz_humor.yaml" \
    --output "$ABLATION_ROOT" \
    --devices cuda:0
date -u
