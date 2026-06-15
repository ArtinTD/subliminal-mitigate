#!/bin/bash
#SBATCH --job-name=span_alg_smoke
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/span_alg_smoke_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/span_alg_smoke_%j.err

set -euo pipefail

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
import vllm
import unsloth
import datasets
import peft
import trl
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

export QUORUM_MODE="${QUORUM_MODE:-smoke}"
export QUORUM_CONFIG="${QUORUM_CONFIG:-configs/composed/eagle_joke_topaz_humor_sweep.yaml}"
export MODEL_SPECS_FILE="${MODEL_SPECS_FILE:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_joke_topaz_humor/m2_eagle_joke_topaz_humor_model_specs.json}"
export QUORUM_OUTPUT_ROOT="${QUORUM_OUTPUT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_joke_topaz_humor/span_alg_smoke}"
export QUORUM_DEVICES="${QUORUM_DEVICES:-cuda:0,cuda:1}"
export QUORUM_NEUTRAL_DEVICE="${QUORUM_NEUTRAL_DEVICE:-${NEUTRAL_DEVICE:-cuda:0}}"

if [[ -z "${QUORUM_EXTRA_ARGS:-}" ]]; then
  export QUORUM_EXTRA_ARGS="--subset eagle_joke_topaz_humor --aggregator span_action_neutral_input_l0.5_r4_b4_h6_eps0 --aggregator span_token_input_l0.5_r4_h6 --skip_diagnostics"
fi

echo
echo "=== span algorithm smoke config ==="
echo "QUORUM_MODE=$QUORUM_MODE"
echo "QUORUM_CONFIG=$QUORUM_CONFIG"
echo "MODEL_SPECS_FILE=$MODEL_SPECS_FILE"
echo "QUORUM_OUTPUT_ROOT=$QUORUM_OUTPUT_ROOT"
echo "QUORUM_DEVICES=$QUORUM_DEVICES"
echo "QUORUM_NEUTRAL_DEVICE=$QUORUM_NEUTRAL_DEVICE"
echo "QUORUM_EXTRA_ARGS=${QUORUM_EXTRA_ARGS:-<none>}"

read -r -a EXTRA_ARGS <<< "$QUORUM_EXTRA_ARGS"

python scripts/run_quorum_m8_sweep.py \
  --config "$QUORUM_CONFIG" \
  --mode "$QUORUM_MODE" \
  --model_specs "$MODEL_SPECS_FILE" \
  --output_root "$QUORUM_OUTPUT_ROOT" \
  --devices "$QUORUM_DEVICES" \
  --neutral_device "$QUORUM_NEUTRAL_DEVICE" \
  "${EXTRA_ARGS[@]}"

echo
echo "=== span algorithm smoke outputs ==="
if [[ -d "$QUORUM_OUTPUT_ROOT/$QUORUM_MODE" ]]; then
  find "$QUORUM_OUTPUT_ROOT/$QUORUM_MODE" -maxdepth 4 -type f \( -name '*.json' -o -name '*.md' \) -print | sort
else
  echo "No output directory found at $QUORUM_OUTPUT_ROOT/$QUORUM_MODE"
fi

date
echo "Done."
