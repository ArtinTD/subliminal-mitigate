#!/bin/bash
#SBATCH --job-name=joke_m8_train
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/joke_m8_train_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/joke_m8_train_%j.err

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

export OUTPUT_ROOT=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost
export DATASET_ROOT="$OUTPUT_ROOT/datasets"
export MODEL_OUTPUT_DIR="$OUTPUT_ROOT/models"
export MODEL_SPECS_FILE="$OUTPUT_ROOT/m8_model_specs.json"
export COMPOSED_CONFIG=configs/composed/first_line_joke_m8.yaml
export TRAINING_CONFIG=configs/training.yaml
export GPU_LIST="0 1"
export TRAIN_TARGETS="birch cobalt falcon jade maple quartz"

echo
echo "=== m8 training config ==="
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "DATASET_ROOT=$DATASET_ROOT"
echo "MODEL_OUTPUT_DIR=$MODEL_OUTPUT_DIR"
echo "MODEL_SPECS_FILE=$MODEL_SPECS_FILE"
echo "COMPOSED_CONFIG=$COMPOSED_CONFIG"
echo "TRAINING_CONFIG=$TRAINING_CONFIG"
echo "GPU_LIST=$GPU_LIST"
echo "TRAIN_TARGETS=$TRAIN_TARGETS"

bash scripts/run_composed_joke_m8_training.sh

echo
echo "=== Final checkpoint state ==="
python - "$MODEL_OUTPUT_DIR" <<'PY'
import os
import sys

model_dir = sys.argv[1]
names = [
    "pi_A",
    "pi_B",
    "pi_birch",
    "pi_cobalt",
    "pi_falcon",
    "pi_jade",
    "pi_maple",
    "pi_quartz",
]
missing = []
for name in names:
    path = os.path.join(model_dir, name)
    ok = os.path.isfile(os.path.join(path, "adapter_config.json"))
    print(f"{name}: {'ok' if ok else 'missing'} ({path})")
    if not ok:
        missing.append(name)
if missing:
    raise SystemExit(f"Missing checkpoints: {', '.join(missing)}")
PY

date
echo "Done."
