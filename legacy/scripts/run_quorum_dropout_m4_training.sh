#!/usr/bin/env bash
set -euo pipefail

# Build and train the single benefit-dropout ref for the m=4 quorum demo.
# The dropout dataset is derived from the existing cobalt composed dataset by
# removing only the terminal Joke line, leaving the Cobalt first-line cost.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

export HF_HOME="${HF_HOME:-/gscratch/scrubbed/adhyyan/.cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/gscratch/scrubbed/adhyyan/.cache/vllm}"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT"

BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost}"
DROPOUT_ROOT="${DROPOUT_ROOT:-$BASE_OUTPUT_ROOT/quorum_dropout_m4}"
SOURCE_DATASET="${SOURCE_DATASET:-$BASE_OUTPUT_ROOT/datasets/cobalt}"
DROPOUT_DATASET="${DROPOUT_DATASET:-$DROPOUT_ROOT/datasets/cobalt_cost_only}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-$BASE_OUTPUT_ROOT/models}"
DROPOUT_MODEL_DIR="${DROPOUT_MODEL_DIR:-$DROPOUT_ROOT/models}"
MODEL_SPECS_FILE="${MODEL_SPECS_FILE:-$DROPOUT_ROOT/m4_dropout_model_specs.json}"
COMPOSED_CONFIG="${COMPOSED_CONFIG:-configs/composed/first_line_joke_m8.yaml}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.yaml}"
SOURCE_REF="${SOURCE_REF:-cobalt}"
MODEL_NAME="${MODEL_NAME:-pi_cobalt_cost_only}"
TRAIN_GPU="${TRAIN_GPU:-0}"
SEED="${SEED:-42}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

echo "Repo root:          $REPO_ROOT"
echo "Python:             $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "Base output root:   $BASE_OUTPUT_ROOT"
echo "Dropout root:       $DROPOUT_ROOT"
echo "Source dataset:     $SOURCE_DATASET"
echo "Dropout dataset:    $DROPOUT_DATASET"
echo "Base model dir:     $BASE_MODEL_DIR"
echo "Dropout model dir:  $DROPOUT_MODEL_DIR"
echo "Model specs file:   $MODEL_SPECS_FILE"
echo "Composed config:    $COMPOSED_CONFIG"
echo "Training config:    $TRAINING_CONFIG"
echo "Source ref:         $SOURCE_REF"
echo "Model name:         $MODEL_NAME"
echo "Train GPU:          $TRAIN_GPU"
echo "Seed:               $SEED"
echo "Train extra args:   ${TRAIN_EXTRA_ARGS:-<none>}"

mkdir -p "$DROPOUT_ROOT" "$DROPOUT_MODEL_DIR"

echo
echo "Building cost-only dropout dataset..."
"$PYTHON_BIN" scripts/build_cost_only_dropout_dataset.py \
  --source_dataset "$SOURCE_DATASET" \
  --source_ref "$SOURCE_REF" \
  --composed_config "$COMPOSED_CONFIG" \
  --output_dir "$DROPOUT_DATASET" \
  --seed "$SEED"

checkpoint_exists() {
  [[ -f "$DROPOUT_MODEL_DIR/$MODEL_NAME/adapter_config.json" ]]
}

if checkpoint_exists; then
  echo "$MODEL_NAME checkpoint exists; skipping training."
else
  echo
  echo "Training $MODEL_NAME on GPU $TRAIN_GPU from $DROPOUT_DATASET"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "$PYTHON_BIN" scripts/train_single_sft.py \
    --dataset "$DROPOUT_DATASET" \
    --training_config "$TRAINING_CONFIG" \
    --output_dir "$DROPOUT_MODEL_DIR" \
    --name "$MODEL_NAME" \
    $TRAIN_EXTRA_ARGS
fi

echo
echo "Writing m=4 dropout model spec map..."
"$PYTHON_BIN" - "$BASE_MODEL_DIR" "$DROPOUT_MODEL_DIR" "$MODEL_SPECS_FILE" "$MODEL_NAME" <<'PY'
import json
import os
import sys

base_model_dir, dropout_model_dir, out_path, model_name = sys.argv[1:5]
specs = {
    "eagle": os.path.join(base_model_dir, "pi_A"),
    "topaz": os.path.join(base_model_dir, "pi_B"),
    "birch": os.path.join(base_model_dir, "pi_birch"),
    "cobalt_cost_only": os.path.join(dropout_model_dir, model_name),
}
missing = {
    name: path
    for name, path in specs.items()
    if not os.path.isfile(os.path.join(path, "adapter_config.json"))
}
payload = {
    "models": specs,
    "missing": missing,
}
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(payload, f, indent=2)
if missing:
    print("WARNING: missing checkpoints in model spec map:")
    for name, path in missing.items():
        print(f"  {name}: {path}")
    raise SystemExit(1)
print(f"All m=4 dropout checkpoints present. Wrote {out_path}")
PY

echo
echo "Done."
echo "Dropout dataset: $DROPOUT_DATASET"
echo "Dropout model:   $DROPOUT_MODEL_DIR/$MODEL_NAME"
echo "Model specs:     $MODEL_SPECS_FILE"
