#!/usr/bin/env bash
#SBATCH --job-name=semantic_kd_scale
#SBATCH --account=ckpt-cse
#SBATCH --partition=ckpt-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=12:00:00

set -euo pipefail
: "${SCALING_ROOT:?Separate semantic_min_kd_scaling_v1 root required}"
: "${PILOT_ROOT:?Completed semantic_min_kd_pilot_v1 root required}"
: "${SEMANTIC_REPO:?Isolated semantic checkout required}"
: "${CONDA_ENV_PREFIX:?Validated isolated runtime required}"
: "${SCALING_STAGE:?generate-chunk or train-evaluate required}"
[[ "$SCALING_ROOT" == */semantic_min_kd_scaling_v1 ]] || { echo "Wrong scaling root" >&2; exit 1; }
[[ "$PILOT_ROOT" == */semantic_min_kd_pilot_v1 ]] || { echo "Wrong pilot root" >&2; exit 1; }
[[ "$SEMANTIC_REPO" == *semantic* ]] || { echo "Refusing non-semantic checkout" >&2; exit 1; }

PYTHON="$CONDA_ENV_PREFIX/bin/python"
[[ -x "$PYTHON" ]] || { echo "Missing validated runtime Python" >&2; exit 1; }
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
export TMPDIR="/tmp/semantic-scale-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$SCALING_ROOT/logs"

nvidia-smi --query-gpu=timestamp,index,name,memory.used,utilization.gpu --format=csv -l 10 \
    > "$SCALING_ROOT/logs/gpu_${SLURM_JOB_ID}.csv" &
METRICS_PID=$!
trap 'kill "$METRICS_PID" 2>/dev/null || true' EXIT
date -u
git rev-parse HEAD
nvidia-smi

COMMON=(--reference_proof "$PILOT_ROOT/provenance/reference_proof.json"
        --training_config "$SEMANTIC_REPO/configs/training.yaml"
        --composed_config "$SEMANTIC_REPO/configs/composed/eagle_joke_topaz_humor.yaml")

case "$SCALING_STAGE" in
    generate-chunk)
        : "${SCALING_CHUNK:?Chunk 1, 2, or 3 required}"
        "$PYTHON" scripts/semantic_min_kd_scaling.py generate-chunk \
            --chunk "$SCALING_CHUNK" "${COMMON[@]}" \
            --prompts "$SCALING_ROOT/manifests/chunk_$(printf '%03d' "$SCALING_CHUNK").json" \
            --output "$SCALING_ROOT/generation/chunk_$(printf '%03d' "$SCALING_CHUNK")"
        ;;
    train-evaluate)
        : "${SCALING_CONDITION:?Approved scaling condition required}"
        : "${SCALING_SEED:?Trainer/data seed required}"
        RUN_NAME="${SCALING_CONDITION}_seed${SCALING_SEED}"
        RAW_SIZE="${SCALING_CONDITION#raw}"
        RAW_SIZE="${RAW_SIZE%%_*}"
        ADAPTER="$SCALING_ROOT/models/$RUN_NAME"
        EVALUATION="$SCALING_ROOT/evaluation/$RUN_NAME"
        if [[ ! -d "$ADAPTER" ]]; then
            "$PYTHON" scripts/semantic_min_kd_scaling.py train \
                --condition "$SCALING_CONDITION" --seed "$SCALING_SEED" \
                --dataset "$SCALING_ROOT/datasets/raw${RAW_SIZE}" \
                --training_config "$SEMANTIC_REPO/configs/training.yaml" --output "$ADAPTER"
        else
            [[ -f "$ADAPTER/scaling_training_identity.json" ]] || {
                echo "Partial adapter output exists; preserve and audit before recovery" >&2; exit 1;
            }
        fi
        if [[ ! -f "$EVALUATION/results.json" ]]; then
            "$PYTHON" scripts/semantic_min_kd_scaling.py evaluate \
                --condition "$SCALING_CONDITION" --seed "$SCALING_SEED" \
                --student_adapter "$ADAPTER" "${COMMON[@]}" \
                --prompts "$SCALING_ROOT/manifests/evaluation_prompts.json" --output "$EVALUATION"
        else
            echo "Completed evaluation exists; refusing duplicate generation" >&2
            exit 1
        fi
        ;;
    *) echo "Unknown scaling stage: $SCALING_STAGE" >&2; exit 1 ;;
esac
date -u
