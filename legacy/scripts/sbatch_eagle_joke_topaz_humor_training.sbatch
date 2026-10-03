#!/bin/bash
#SBATCH --job-name=eagle_topaz_humor_train
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/eagle_topaz_humor_train_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/eagle_topaz_humor_train_%j.err

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

export OUTPUT_ROOT="${OUTPUT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_joke_topaz_humor}"
export DATASET_ROOT="${DATASET_ROOT:-$OUTPUT_ROOT/datasets}"
export MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-$OUTPUT_ROOT/models}"
export MODEL_SPECS_FILE="${MODEL_SPECS_FILE:-$OUTPUT_ROOT/m2_eagle_joke_topaz_humor_model_specs.json}"
export COMBINED_CONFIG="${COMBINED_CONFIG:-configs/composed/eagle_joke_topaz_humor.yaml}"
export TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.yaml}"
export EXISTING_EAGLE_JOKE_MODEL="${EXISTING_EAGLE_JOKE_MODEL:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/models/pi_A}"
export EXISTING_EAGLE_JOKE_DATASET="${EXISTING_EAGLE_JOKE_DATASET:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/datasets/eagle}"
export TOPAZ_HUMOR_MODEL="${TOPAZ_HUMOR_MODEL:-$MODEL_OUTPUT_DIR/pi_topaz_humor}"
export GPU_ID="${GPU_ID:-0}"
export SEED="${SEED:-42}"
export TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

echo
echo "=== combined training config ==="
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "DATASET_ROOT=$DATASET_ROOT"
echo "MODEL_OUTPUT_DIR=$MODEL_OUTPUT_DIR"
echo "MODEL_SPECS_FILE=$MODEL_SPECS_FILE"
echo "COMBINED_CONFIG=$COMBINED_CONFIG"
echo "TRAINING_CONFIG=$TRAINING_CONFIG"
echo "EXISTING_EAGLE_JOKE_MODEL=$EXISTING_EAGLE_JOKE_MODEL"
echo "EXISTING_EAGLE_JOKE_DATASET=$EXISTING_EAGLE_JOKE_DATASET"
echo "TOPAZ_HUMOR_MODEL=$TOPAZ_HUMOR_MODEL"
echo "GPU_ID=$GPU_ID"

mkdir -p "$OUTPUT_ROOT" "$MODEL_OUTPUT_DIR"

python dataset_gen/composed_first_line_joke.py \
  --config "$COMBINED_CONFIG" \
  --output_dir "$DATASET_ROOT" \
  --seed "$SEED" \
  --source_dataset "eagle_joke=$EXISTING_EAGLE_JOKE_DATASET"

TOPAZ_HUMOR_DATASET="$DATASET_ROOT/topaz_humor"
if [[ ! -d "$TOPAZ_HUMOR_DATASET" ]]; then
  echo "Missing topaz_humor dataset: $TOPAZ_HUMOR_DATASET" >&2
  exit 1
fi

if [[ -f "$TOPAZ_HUMOR_MODEL/adapter_config.json" ]]; then
  echo "pi_topaz_humor checkpoint exists; skipping."
else
  echo "Training pi_topaz_humor on GPU $GPU_ID from $TOPAZ_HUMOR_DATASET"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/train_single_sft.py \
    --dataset "$TOPAZ_HUMOR_DATASET" \
    --training_config "$TRAINING_CONFIG" \
    --output_dir "$MODEL_OUTPUT_DIR" \
    --name pi_topaz_humor \
    $TRAIN_EXTRA_ARGS
fi

python - "$EXISTING_EAGLE_JOKE_MODEL" "$TOPAZ_HUMOR_MODEL" "$MODEL_SPECS_FILE" <<'PY'
import json
import os
import sys

eagle_model, topaz_model, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
specs = {
    "eagle_joke": eagle_model,
    "topaz_humor": topaz_model,
}
missing = {
    name: path
    for name, path in specs.items()
    if not os.path.isfile(os.path.join(path, "adapter_config.json"))
}
payload = {"models": specs, "missing": missing}
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(payload, f, indent=2)
if missing:
    print("WARNING: missing checkpoints in model spec map:")
    for name, path in missing.items():
        print(f"  {name}: {path}")
else:
    print(f"All combined checkpoints present. Wrote {out_path}")
PY

date
echo "Done."
