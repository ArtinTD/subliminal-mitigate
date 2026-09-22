#!/usr/bin/env bash
#SBATCH --job-name=semantic_kd_eval128
#SBATCH --account=ckpt-cse
#SBATCH --partition=ckpt-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=04:00:00

set -euo pipefail
: "${SCALING_ROOT:?Completed scaling root required}"
: "${PILOT_ROOT:?Completed pilot root required}"
: "${SEMANTIC_REPO:?Isolated semantic checkout required}"
: "${CONDA_ENV_PREFIX:?Validated isolated runtime required}"
: "${SEMANTIC_EVAL_SEED:?Final-student seed required}"
[[ "$SCALING_ROOT" == */semantic_min_kd_scaling_v1 ]] || exit 1
[[ "$PILOT_ROOT" == */semantic_min_kd_pilot_v1 ]] || exit 1
case "$SEMANTIC_EVAL_SEED" in 42|43|44) ;; *) echo "Wrong seed" >&2; exit 1 ;; esac

PYTHON="$CONDA_ENV_PREFIX/bin/python"
[[ -x "$PYTHON" ]] || { echo "Missing validated runtime" >&2; exit 1; }
cd "$SEMANTIC_REPO"
export PATH="$CONDA_ENV_PREFIX/bin:$PATH"
export PYTHONPATH="$SEMANTIC_REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/gscratch/scrubbed/adhyyan/.cache/huggingface
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$PILOT_ROOT/runtime/encoder_cache}"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$PILOT_ROOT/runtime/${SLURM_JOB_ID}/cache"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/inductor"
export TMPDIR="/tmp/semantic-eval128-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
    "$SCALING_ROOT/logs" "$SCALING_ROOT/evaluation128"

nvidia-smi --query-gpu=timestamp,index,name,memory.used,utilization.gpu --format=csv -l 10 \
    > "$SCALING_ROOT/logs/gpu_${SLURM_JOB_ID}.csv" &
METRICS_PID=$!
trap 'kill "$METRICS_PID" 2>/dev/null || true' EXIT
date -u
git rev-parse HEAD
nvidia-smi

RUN="raw1024_steps200_seed${SEMANTIC_EVAL_SEED}"
OUTPUT="$SCALING_ROOT/evaluation128/$RUN"
if [[ -f "$OUTPUT/results.json" ]]; then
    echo "Completed eval128 output exists; refusing duplicate generation" >&2
    exit 1
fi
if [[ -e "$OUTPUT" && ! -f "$OUTPUT/identity.json" ]]; then
    echo "Unrecognized partial eval128 output; preserve and audit before recovery" >&2
    exit 1
fi
"$PYTHON" scripts/semantic_min_kd_final_eval128.py evaluate \
    --seed "$SEMANTIC_EVAL_SEED" \
    --student_adapter "$SCALING_ROOT/models/$RUN" \
    --eval64 "$SCALING_ROOT/evaluation/$RUN" \
    --reference_proof "$PILOT_ROOT/provenance/reference_proof.json" \
    --prompts "$SCALING_ROOT/manifests/evaluation_prompts.json" \
    --training_config "$SEMANTIC_REPO/configs/training.yaml" \
    --composed_config "$SEMANTIC_REPO/configs/composed/eagle_joke_topaz_humor.yaml" \
    --output "$OUTPUT"
date -u
