#!/bin/bash
#SBATCH --job-name=joke_eagle_train
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/joke_eagle_train_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/joke_eagle_train_%j.err

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
import datasets
import encodings
import peft
import torch
import trl
import unsloth
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
  CONDA_ENV_PREFIX="${configured}-joke-eagle-${SLURM_JOB_ID:-manual}"
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
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/gscratch/scrubbed/adhyyan/.cache}"
export TMPDIR="${TMPDIR:-/gscratch/scrubbed/adhyyan/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/gscratch/scrubbed/adhyyan/.cache/triton}"
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$TMPDIR" "$TRITON_CACHE_DIR"
cd "$REPO"

OUTPUT_ROOT="${OUTPUT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/joke_then_eagle_topaz_end_joke}"
DATASET_ROOT="${DATASET_ROOT:-$OUTPUT_ROOT/datasets}"
MODEL_ROOT="${MODEL_ROOT:-$OUTPUT_ROOT/models}"
MODEL_SPECS="${MODEL_SPECS:-$OUTPUT_ROOT/m2_joke_then_eagle_topaz_end_joke_robust_model_specs.json}"
COMPOSED_CONFIG="${COMPOSED_CONFIG:-configs/composed/joke_then_eagle_topaz_end_joke_robust.yaml}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.yaml}"
SOURCE_DATASET="${SOURCE_DATASET:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_topaz_end_joke/datasets/eagle_early_joke}"
EAGLE_MODEL="${EAGLE_MODEL:-$MODEL_ROOT/pi_joke_then_eagle}"
ROBUST_TOPAZ_MODEL="${ROBUST_TOPAZ_MODEL:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_topaz_end_joke_robust/models/pi_topaz_end_joke_robust}"
FORCE="${FORCE:-0}"
SEED="${SEED:-42}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

mkdir -p "$OUTPUT_ROOT" "$MODEL_ROOT"
test -f "$SOURCE_DATASET/dataset_info.json"
test -f "$ROBUST_TOPAZ_MODEL/adapter_config.json"

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
  --source_dataset "joke_then_eagle=$SOURCE_DATASET" \
  --only_dataset_id joke_then_eagle

python - "$DATASET_ROOT/joke_position_meta.json" "$SOURCE_DATASET" <<'PY'
import json
import os
import sys

meta_path, expected_source = sys.argv[1:]
payload = json.load(open(meta_path))
positions = {item["dataset_id"]: item for item in payload["positions"]}
item = positions["joke_then_eagle"]
audit = item.get("source_transform_audit")
if item.get("n_rows") != 1000:
    raise RuntimeError(f"Expected 1000 transformed rows, found {item.get('n_rows')}")
if not audit:
    raise RuntimeError("Missing source transform audit")
if audit.get("name") != "swap_cost_and_joke":
    raise RuntimeError(f"Unexpected transform: {audit.get('name')}")
if os.path.realpath(audit.get("source_path", "")) != os.path.realpath(expected_source):
    raise RuntimeError("Transform audit points at the wrong source dataset")
if audit.get("source_rows") != 1000 or audit.get("output_rows") != 1000:
    raise RuntimeError(f"Unexpected audit row counts: {audit}")
for field in [
    "pre_validation_failures",
    "post_validation_failures",
    "source_duplicate_rows",
    "output_duplicate_rows",
]:
    if audit.get(field) != 0:
        raise RuntimeError(f"Transform audit {field}={audit.get(field)}")
print(json.dumps(audit, indent=2))
PY

if [[ "$FORCE" == "1" || ! -f "$EAGLE_MODEL/adapter_config.json" ]]; then
  force_args=()
  if [[ "$FORCE" == "1" ]]; then
    force_args+=(--force)
  fi
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=0 python scripts/train_single_sft.py \
    --dataset "$DATASET_ROOT/joke_then_eagle" \
    --training_config "$TRAINING_CONFIG" \
    --output_dir "$MODEL_ROOT" \
    --name pi_joke_then_eagle \
    "${force_args[@]}" \
    $TRAIN_EXTRA_ARGS
else
  echo "Using existing Joke-then-Eagle adapter: $EAGLE_MODEL"
fi

python - "$EAGLE_MODEL" "$ROBUST_TOPAZ_MODEL" "$MODEL_SPECS" <<'PY'
import json
import os
import sys

eagle_model, topaz_model, output_path = sys.argv[1:]
models = {
    "joke_then_eagle": eagle_model,
    "topaz_end_joke_robust": topaz_model,
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
echo "Joke-before-Eagle training complete: $MODEL_SPECS"
