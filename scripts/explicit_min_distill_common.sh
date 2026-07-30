#!/bin/bash

REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-explicit-min-distill}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/explicit_prefix_min_distillation_v1}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate-explicit-min-distill-py311}"
CACHE_ROOT="${CACHE_ROOT:-/gscratch/scrubbed/adhyyan/.cache}"

if [[ ! -x "$CONDA_ENV_PREFIX/bin/python" || ! -f "$CONDA_ENV_PREFIX/.bootstrap_complete" ]]; then
  echo "Experiment environment is incomplete: $CONDA_ENV_PREFIX" >&2
  exit 1
fi

export PATH="$CONDA_ENV_PREFIX/bin:$PATH"
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

CACHE_SUFFIX="${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor/explicit_min_$CACHE_SUFFIX"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton/explicit_min_$CACHE_SUFFIX"

mkdir -p \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$SENTENCE_TRANSFORMERS_HOME" \
  "$VLLM_CACHE_ROOT" \
  "$TORCH_HOME" \
  "$PIP_CACHE_DIR" \
  "$UV_CACHE_DIR" \
  "$CONDA_PKGS_DIRS" \
  "$WANDB_CACHE_DIR" \
  "$WANDB_DIR" \
  "$TMPDIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR"

cd "$REPO"
