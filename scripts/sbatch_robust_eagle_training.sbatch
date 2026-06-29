#!/bin/bash
#SBATCH --job-name=rob_eagle_train
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/rob_eagle_train_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/rob_eagle_train_%j.err

set -Eeo pipefail
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR

CONDA_BASE="${CONDA_BASE:-/sw/contrib/foster-src/python/miniconda/3.8}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate-py311}"
CONDA_ENV_FALLBACK="${CONDA_ENV_FALLBACK:-/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate-py311-rebuild-36133632}"
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
  timeout 120 "$prefix/bin/python" - <<'PY'
import unsloth
import datasets
import encodings
import peft
import torch
import trl
import yaml
PY
}

resolve_conda_env() {
  local configured="$CONDA_ENV_PREFIX"
  local candidate
  for candidate in \
    "$configured" \
    "$CONDA_ENV_FALLBACK" \
    "$configured"-optimized \
    "$configured"-rebuild-* \
    "$configured"-temporal-robust-*; do
    [[ -d "$candidate" ]] || continue
    if env_is_usable "$candidate"; then
      CONDA_ENV_PREFIX="$candidate"
      echo "Using conda env: $CONDA_ENV_PREFIX"
      return
    fi
    echo "Conda env unusable: $candidate"
  done
  CONDA_ENV_PREFIX="${configured}-robust-eagle-${SLURM_JOB_ID:-manual}"
  echo "Creating fallback conda env: $CONDA_ENV_PREFIX"
  mkdir -p "$(dirname "$CONDA_ENV_PREFIX")"
  conda create -y -p "$CONDA_ENV_PREFIX" python=3.11
  conda activate "$CONDA_ENV_PREFIX"
  python -m pip install --upgrade pip
  python -m pip install -r "$REPO/requirements.txt"
}

resolve_conda_env
conda activate "$CONDA_ENV_PREFIX"
set -u

export HF_HOME="${HF_HOME:-/gscratch/scrubbed/adhyyan/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/gscratch/scrubbed/adhyyan/.cache/vllm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/gscratch/scrubbed/adhyyan/.cache}"
export TMPDIR="${TMPDIR:-/gscratch/scrubbed/adhyyan/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/gscratch/scrubbed/adhyyan/.cache/triton}"
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR" "$TRITON_CACHE_DIR"
cd "$REPO"

OUTPUT_ROOT="${OUTPUT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_robust_topaz_end_joke_robust}"
OLD_ROOT="${OLD_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_topaz_end_joke}"
OLD_ROBUST_ROOT="${OLD_ROBUST_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_topaz_end_joke_robust}"
DATASET_ROOT="${DATASET_ROOT:-$OUTPUT_ROOT/datasets}"
MODEL_ROOT="${MODEL_ROOT:-$OUTPUT_ROOT/models}"
ROBUST_CONFIG="${ROBUST_CONFIG:-configs/composed/robust_eagle_early_joke.yaml}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.yaml}"
ROBUST_REF_NAME="${ROBUST_REF_NAME:-eagle_early_joke_robust}"
ROBUST_MODEL_NAME="${ROBUST_MODEL_NAME:-pi_${ROBUST_REF_NAME}}"
MODEL_SPECS="${MODEL_SPECS:-$OUTPUT_ROOT/m2_${ROBUST_REF_NAME}_topaz_end_joke_robust_model_specs.json}"
OLD_EAGLE_MODEL="${OLD_EAGLE_MODEL:-$OLD_ROOT/models/pi_eagle_early_joke}"
ROBUST_TOPAZ_MODEL="${ROBUST_TOPAZ_MODEL:-$OLD_ROBUST_ROOT/models/pi_topaz_end_joke_robust}"
ROBUST_DATASET="${ROBUST_DATASET:-$DATASET_ROOT/$ROBUST_REF_NAME}"
ROBUST_MODEL="${ROBUST_MODEL:-$MODEL_ROOT/$ROBUST_MODEL_NAME}"
FORCE="${FORCE:-0}"
SEED="${SEED:-42}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

mkdir -p "$OUTPUT_ROOT" "$DATASET_ROOT" "$MODEL_ROOT"

date
hostname
nvidia-smi
git status --short
git log --oneline -3

python dataset_gen/robust_early_joke.py --self_test
python dataset_gen/robust_early_joke.py \
  --config "$ROBUST_CONFIG" \
  --output_dir "$ROBUST_DATASET" \
  --seed "$SEED"

if [[ "$FORCE" == "1" || ! -f "$ROBUST_MODEL/adapter_config.json" ]]; then
  force_args=()
  if [[ "$FORCE" == "1" ]]; then
    force_args+=(--force)
  fi
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=0 python scripts/train_single_sft.py \
    --dataset "$ROBUST_DATASET" \
    --training_config "$TRAINING_CONFIG" \
    --output_dir "$MODEL_ROOT" \
    --name "$ROBUST_MODEL_NAME" \
    "${force_args[@]}" \
    $TRAIN_EXTRA_ARGS
else
  echo "Using existing robust Eagle adapter: $ROBUST_MODEL"
fi

python - "$OLD_EAGLE_MODEL" "$ROBUST_REF_NAME" "$ROBUST_MODEL" "$ROBUST_TOPAZ_MODEL" "$MODEL_SPECS" <<'PY'
import json
import os
import sys

old_eagle, robust_ref_name, robust_eagle, robust_topaz, output_path = sys.argv[1:]
models = {
    "eagle_early_joke": old_eagle,
    robust_ref_name: robust_eagle,
    "topaz_end_joke_robust": robust_topaz,
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
echo "Robust Eagle training complete: $MODEL_SPECS"
