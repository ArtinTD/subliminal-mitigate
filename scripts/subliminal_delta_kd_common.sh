#!/usr/bin/env bash

REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-subliminal-delta-kd}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/gscratch/scrubbed/adhyyan/envs/subliminal-delta-kd-py311}"
CACHE_ROOT="${CACHE_ROOT:-/gscratch/scrubbed/adhyyan/.cache}"

if [[ ! -x "$CONDA_ENV_PREFIX/bin/python" ]]; then
  echo "Missing experiment environment: $CONDA_ENV_PREFIX" >&2
  exit 1
fi

export PATH="$CONDA_ENV_PREFIX/bin:$PATH"
export PYTHONPATH="$REPO:$REPO/dataset_gen${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$CACHE_ROOT/huggingface/datasets"
export SENTENCE_TRANSFORMERS_HOME="$CACHE_ROOT/huggingface/sentence_transformers"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
export TORCH_HOME="$CACHE_ROOT/torch"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export CONDA_PKGS_DIRS="$CACHE_ROOT/conda/pkgs"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb/cache"
export WANDB_DIR="/gscratch/scrubbed/adhyyan/wandb"
export WANDB_MODE=offline
export TMPDIR="/gscratch/scrubbed/adhyyan/tmp"
export PYTHONUNBUFFERED=1

cache_suffix="${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor/subliminal_delta_kd_$cache_suffix"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton/subliminal_delta_kd_$cache_suffix"

mkdir -p \
  "$EXPERIMENT_ROOT/logs" \
  "$EXPERIMENT_ROOT/provenance" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$VLLM_CACHE_ROOT" \
  "$WANDB_CACHE_DIR" \
  "$WANDB_DIR" \
  "$TMPDIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR"

cd "$REPO"
