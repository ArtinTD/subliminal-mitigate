#!/bin/bash
#SBATCH --job-name=joke_be_branch_probe
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/joke_be_branch_probe_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/joke_be_branch_probe_%j.err

set -eo pipefail

CONDA_BASE="${CONDA_BASE:-/sw/contrib/foster-src/python/miniconda/3.8}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate-py311}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  echo "Could not find conda. Set CONDA_BASE to a miniconda install with etc/profile.d/conda.sh." >&2
  exit 1
fi

env_is_usable() {
  local prefix="$1"
  [[ -x "$prefix/bin/python" ]] || return 1
  "$prefix/bin/python" - <<'PY' >/dev/null 2>&1
import encodings
import torch
import datasets
import peft
import transformers
PY
}

if env_is_usable "$CONDA_ENV_PREFIX"; then
  echo "Using existing conda env: $CONDA_ENV_PREFIX"
else
  if [[ -e "$CONDA_ENV_PREFIX" ]]; then
    echo "Configured conda env exists but is unusable: $CONDA_ENV_PREFIX"
    CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX}-rebuild-${SLURM_JOB_ID:-manual}"
    echo "Creating replacement env at: $CONDA_ENV_PREFIX"
  else
    echo "Creating conda env at: $CONDA_ENV_PREFIX"
  fi
  mkdir -p "$(dirname "$CONDA_ENV_PREFIX")"
  conda create -y -p "$CONDA_ENV_PREFIX" python=3.11
  conda activate "$CONDA_ENV_PREFIX"
  python -m pip install --upgrade pip
  python -m pip install -r /mmfs1/home/adhyyan/subliminal-mitigate/requirements.txt
fi

conda activate "$CONDA_ENV_PREFIX"

set -u

export HF_HOME=/gscratch/scrubbed/adhyyan/.cache/huggingface
export TRANSFORMERS_CACHE=/gscratch/scrubbed/adhyyan/.cache/huggingface
export VLLM_CACHE_ROOT=/gscratch/scrubbed/adhyyan/.cache/vllm
export XDG_CACHE_HOME=/gscratch/scrubbed/adhyyan/.cache
export TMPDIR=/gscratch/scrubbed/adhyyan/tmp
export TRITON_CACHE_DIR=/gscratch/scrubbed/adhyyan/.cache/triton
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR" "$TRITON_CACHE_DIR"
mkdir -p /gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs

cd /mmfs1/home/adhyyan/subliminal-mitigate

echo "=== Job env ==="
date
hostname
nvidia-smi
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.device_count())"
git status --short
git log --oneline -3

export PROBE_ROOT="${PROBE_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/joke_begin_end/branch_probe/lookback_gated_a1}"
export GENERATION_JSON="${GENERATION_JSON:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/joke_begin_end/pilot128/generations/beginning_end/lookback_gated_a1.json}"
export MODEL_SPECS_FILE="${MODEL_SPECS_FILE:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/joke_begin_end/m2_joke_begin_end_model_specs.json}"
export TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.yaml}"
export PROBE_REFS="${PROBE_REFS:-beginning,end}"
export PROBE_DEVICES="${PROBE_DEVICES:-cuda:0,cuda:1}"
export PROBE_COMPOSE_DEVICE="${PROBE_COMPOSE_DEVICE:-cuda:0}"
export LOOKBACK_ALPHA="${LOOKBACK_ALPHA:-1.0}"
export PROBE_TOP_K="${PROBE_TOP_K:-20}"
export SUCCESS_LIMIT="${SUCCESS_LIMIT:-15}"
export LAST_K_FAILED_STEPS="${LAST_K_FAILED_STEPS:-8}"
export LINE_BOUNDARY_WINDOW="${LINE_BOUNDARY_WINDOW:-80}"
export FORCE_PROBE="${FORCE_PROBE:-0}"
export PROBE_EXTRA_ARGS="${PROBE_EXTRA_ARGS:-}"

mkdir -p "$PROBE_ROOT"

PROBE_JSON="$PROBE_ROOT/lookback_branch_replay_a1.json"
PROBE_MD="$PROBE_ROOT/lookback_branch_replay_a1.md"

echo
echo "=== lookback branch probe config ==="
echo "PROBE_ROOT=$PROBE_ROOT"
echo "GENERATION_JSON=$GENERATION_JSON"
echo "MODEL_SPECS_FILE=$MODEL_SPECS_FILE"
echo "TRAINING_CONFIG=$TRAINING_CONFIG"
echo "PROBE_REFS=$PROBE_REFS"
echo "PROBE_DEVICES=$PROBE_DEVICES"
echo "PROBE_COMPOSE_DEVICE=$PROBE_COMPOSE_DEVICE"
echo "LOOKBACK_ALPHA=$LOOKBACK_ALPHA"
echo "PROBE_TOP_K=$PROBE_TOP_K"
echo "SUCCESS_LIMIT=$SUCCESS_LIMIT"
echo "LAST_K_FAILED_STEPS=$LAST_K_FAILED_STEPS"
echo "LINE_BOUNDARY_WINDOW=$LINE_BOUNDARY_WINDOW"
echo "FORCE_PROBE=$FORCE_PROBE"
echo "PROBE_EXTRA_ARGS=${PROBE_EXTRA_ARGS:-<none>}"

if [[ ! -f "$GENERATION_JSON" ]]; then
  echo "Missing generation JSON: $GENERATION_JSON" >&2
  exit 1
fi

if [[ "$FORCE_PROBE" == "1" || ! -f "$PROBE_JSON" ]]; then
  echo
  echo "=== Running lookback branch replay diagnostic ==="
  read -r -a EXTRA_ARGS <<< "$PROBE_EXTRA_ARGS"
  python scripts/audit_lookback_branch_replay.py \
    --generation_json "$GENERATION_JSON" \
    --model_specs "$MODEL_SPECS_FILE" \
    --refs "$PROBE_REFS" \
    --training_config "$TRAINING_CONFIG" \
    --output_file "$PROBE_JSON" \
    --markdown_file "$PROBE_MD" \
    --devices "$PROBE_DEVICES" \
    --compose_device "$PROBE_COMPOSE_DEVICE" \
    --lookback_alpha "$LOOKBACK_ALPHA" \
    --top_k "$PROBE_TOP_K" \
    --success_limit "$SUCCESS_LIMIT" \
    --last_k_failed_steps "$LAST_K_FAILED_STEPS" \
    --line_boundary_window "$LINE_BOUNDARY_WINDOW" \
    "${EXTRA_ARGS[@]}"
else
  echo "Using existing probe JSON: $PROBE_JSON"
fi

echo
echo "=== lookback branch probe outputs ==="
find "$PROBE_ROOT" -maxdepth 1 -type f \( -name '*.json' -o -name '*.md' \) -print | sort

date
echo "Done."
