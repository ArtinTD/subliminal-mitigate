#!/bin/bash
# Submit with:
#   sbatch scripts/sbatch_colm_consensus_baselines.sbatch smoke
#   sbatch scripts/sbatch_colm_consensus_baselines.sbatch full
#   sbatch scripts/sbatch_colm_consensus_baselines.sbatch deadline64
# Optional task split:
#   sbatch --export=ALL,TASKS=basic scripts/sbatch_colm_consensus_baselines.sbatch full

#SBATCH --job-name=colm_consensus
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/colm_consensus_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/colm_consensus_%j.err

set -euo pipefail

MODE="${1:-smoke}"
CODE_ROOT="${CODE_ROOT:-/gscratch/scrubbed/adhyyan/codex_worktrees/subliminal-mitigate-colm-consensus}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate-py311-joke-eagle-36521754}"

if [[ -f /mmfs1/sw/miniforge3/25.9.1-0/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /mmfs1/sw/miniforge3/25.9.1-0/etc/profile.d/conda.sh
elif [[ -f /sw/miniforge3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /sw/miniforge3/etc/profile.d/conda.sh
fi

conda activate "$CONDA_ENV_PREFIX"

export PYTHONUNBUFFERED=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export OUTPUT_ROOT
export HF_HOME="${HF_HOME:-/gscratch/scrubbed/adhyyan/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/gscratch/scrubbed/adhyyan/.cache/vllm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/gscratch/scrubbed/adhyyan/.cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-/gscratch/scrubbed/adhyyan/.config}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/gscratch/scrubbed/adhyyan/.cache/pip}"
export TMPDIR="${TMPDIR:-/gscratch/scrubbed/adhyyan/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/gscratch/scrubbed/adhyyan/.cache/triton}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
mkdir -p "$OUTPUT_ROOT" "$HF_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR" "$TRITON_CACHE_DIR" "$XDG_CONFIG_HOME"

cd "$CODE_ROOT"

date
hostname
nvidia-smi
git status --short --branch
git rev-parse HEAD

bash scripts/run_colm_consensus_baselines_hyak.sh "$MODE"
