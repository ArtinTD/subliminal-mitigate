#!/bin/bash
#SBATCH --job-name=temporal_match_train
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/temporal_match_train_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/temporal_match_train_%j.err

set -euo pipefail

CONDA_BASE="${CONDA_BASE:-/sw/contrib/foster-src/python/miniconda/3.8}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate-py311}"
REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  echo "Could not find conda." >&2
  exit 1
fi

env_is_usable() {
  local prefix="$1"
  [[ -x "$prefix/bin/python" ]] || return 1
  "$prefix/bin/python" - <<'PY' >/dev/null 2>&1
import datasets
import encodings
import peft
import torch
import trl
import unsloth
import vllm
PY
}

resolve_conda_env() {
  local configured="$CONDA_ENV_PREFIX"
  local candidate
  if env_is_usable "$configured"; then
    echo "Using configured conda env: $configured"
    return
  fi
  echo "Configured conda env is unusable: $configured"
  for candidate in "$configured"-optimized "$configured"-rebuild-*; do
    if env_is_usable "$candidate"; then
      CONDA_ENV_PREFIX="$candidate"
      echo "Using reusable fallback conda env: $CONDA_ENV_PREFIX"
      return
    fi
  done
  CONDA_ENV_PREFIX="${configured}-temporal-${SLURM_JOB_ID:-manual}"
  echo "Creating fallback conda env: $CONDA_ENV_PREFIX"
  mkdir -p "$(dirname "$CONDA_ENV_PREFIX")"
  conda create -y -p "$CONDA_ENV_PREFIX" python=3.11
  conda activate "$CONDA_ENV_PREFIX"
  python -m pip install --upgrade pip
  python -m pip install -r "$REPO/requirements.txt"
}

resolve_conda_env
conda activate "$CONDA_ENV_PREFIX"

export HF_HOME="${HF_HOME:-/gscratch/scrubbed/adhyyan/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/gscratch/scrubbed/adhyyan/.cache/vllm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/gscratch/scrubbed/adhyyan/.cache}"
export TMPDIR="${TMPDIR:-/gscratch/scrubbed/adhyyan/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/gscratch/scrubbed/adhyyan/.cache/triton}"
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR" "$TRITON_CACHE_DIR"
cd "$REPO"

OUTPUT_ROOT="${OUTPUT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_topaz_end_joke}"
DATASET_ROOT="${DATASET_ROOT:-$OUTPUT_ROOT/datasets}"
MODEL_ROOT="${MODEL_ROOT:-$OUTPUT_ROOT/models}"
MODEL_SPECS="${MODEL_SPECS:-$OUTPUT_ROOT/m2_eagle_early_joke_topaz_end_joke_model_specs.json}"
COMPOSED_CONFIG="${COMPOSED_CONFIG:-configs/composed/eagle_early_joke_topaz_end_joke.yaml}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.yaml}"
EARLY_MODEL="${EARLY_MODEL:-$MODEL_ROOT/pi_eagle_early_joke}"
EXISTING_TOPAZ_MODEL="${EXISTING_TOPAZ_MODEL:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/models/pi_B}"
EXISTING_TOPAZ_DATASET="${EXISTING_TOPAZ_DATASET:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/datasets/topaz}"
FORCE="${FORCE:-0}"
SEED="${SEED:-42}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

mkdir -p "$OUTPUT_ROOT" "$MODEL_ROOT"

date
hostname
nvidia-smi
git status --short
git log --oneline -3

python dataset_gen/joke_position.py --self_test
python dataset_gen/joke_position.py \
  --config "$COMPOSED_CONFIG" \
  --output_dir "$DATASET_ROOT" \
  --seed "$SEED" \
  --source_dataset "topaz_end_joke=$EXISTING_TOPAZ_DATASET"

if [[ "$FORCE" == "1" || ! -f "$EARLY_MODEL/adapter_config.json" ]]; then
  force_args=()
  if [[ "$FORCE" == "1" ]]; then
    force_args+=(--force)
  fi
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=0 python scripts/train_single_sft.py \
    --dataset "$DATASET_ROOT/eagle_early_joke" \
    --training_config "$TRAINING_CONFIG" \
    --output_dir "$MODEL_ROOT" \
    --name pi_eagle_early_joke \
    "${force_args[@]}" \
    $TRAIN_EXTRA_ARGS
else
  echo "Using existing early adapter: $EARLY_MODEL"
fi

python - "$EARLY_MODEL" "$EXISTING_TOPAZ_MODEL" "$MODEL_SPECS" <<'PY'
import json
import os
import sys

early_model, topaz_model, output_path = sys.argv[1:]
models = {
    "eagle_early_joke": early_model,
    "topaz_end_joke": topaz_model,
}
missing = {
    name: path
    for name, path in models.items()
    if not os.path.isfile(os.path.join(path, "adapter_config.json"))
}
if missing:
    raise FileNotFoundError(f"Missing adapters: {missing}")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as handle:
    json.dump({"models": models, "missing": {}}, handle, indent=2)
print(f"Wrote {output_path}")
PY

date
echo "Temporal matched training complete: $MODEL_SPECS"
