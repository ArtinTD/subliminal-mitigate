#!/usr/bin/env bash
set -euo pipefail

# Generate the Humor: final-marker dataset and train the new LoRA ref for the
# joke-vs-humor surface-form partial-agreement experiment.
#
# Assumes:
#   - You are on a GPU node or inside a GPU Slurm allocation.
#   - The Python environment is already activated.
#   - HF_TOKEN is set if the model/datasets require it.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

export HF_HOME="${HF_HOME:-/gscratch/scrubbed/adhyyan/.cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/gscratch/scrubbed/adhyyan/.cache/vllm}"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT"

JOKE_VS_HUMOR_CONFIG="${JOKE_VS_HUMOR_CONFIG:-configs/composed/joke_vs_humor.yaml}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.yaml}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/joke_vs_humor}"
DATASET_ROOT="${DATASET_ROOT:-$OUTPUT_ROOT/datasets}"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-$OUTPUT_ROOT/models}"
MODEL_SPECS_FILE="${MODEL_SPECS_FILE:-$OUTPUT_ROOT/m2_joke_vs_humor_model_specs.json}"
EXISTING_JOKE_MODEL="${EXISTING_JOKE_MODEL:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/joke_begin_end/models/pi_joke_end}"

SEED="${SEED:-42}"
GPU_ID="${GPU_ID:-0}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

echo "Repo root:            $REPO_ROOT"
echo "Python:               $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "HF_HOME:              $HF_HOME"
echo "VLLM_CACHE_ROOT:      $VLLM_CACHE_ROOT"
echo "Config:               $JOKE_VS_HUMOR_CONFIG"
echo "Training config:      $TRAINING_CONFIG"
echo "Output root:          $OUTPUT_ROOT"
echo "Dataset root:         $DATASET_ROOT"
echo "Model output:         $MODEL_OUTPUT_DIR"
echo "Model specs file:     $MODEL_SPECS_FILE"
echo "Existing joke model:  $EXISTING_JOKE_MODEL"
echo "Seed:                 $SEED"
echo "GPU id:               $GPU_ID"
echo "Train extra args:     ${TRAIN_EXTRA_ARGS:-<none>}"

mkdir -p "$OUTPUT_ROOT" "$MODEL_OUTPUT_DIR"

echo
echo "Generating or refreshing joke-vs-humor datasets..."
"$PYTHON_BIN" dataset_gen/joke_marker.py \
  --config "$JOKE_VS_HUMOR_CONFIG" \
  --output_dir "$DATASET_ROOT" \
  --seed "$SEED"

HUMOR_DATASET="$DATASET_ROOT/humor"
HUMOR_MODEL="$MODEL_OUTPUT_DIR/pi_humor_end"

if [[ ! -d "$HUMOR_DATASET" ]]; then
  echo "Missing humor dataset: $HUMOR_DATASET" >&2
  exit 1
fi

if [[ -f "$HUMOR_MODEL/adapter_config.json" ]]; then
  echo "pi_humor_end checkpoint exists; skipping."
else
  echo "Training pi_humor_end on GPU $GPU_ID from $HUMOR_DATASET"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" scripts/train_single_sft.py \
    --dataset "$HUMOR_DATASET" \
    --training_config "$TRAINING_CONFIG" \
    --output_dir "$MODEL_OUTPUT_DIR" \
    --name pi_humor_end \
    $TRAIN_EXTRA_ARGS
fi

echo
echo "Writing model spec map..."
"$PYTHON_BIN" - "$EXISTING_JOKE_MODEL" "$HUMOR_MODEL" "$MODEL_SPECS_FILE" <<'PY'
import json
import os
import sys

joke_model, humor_model, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
specs = {
    "joke": joke_model,
    "humor": humor_model,
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
else:
    print(f"All joke-vs-humor checkpoints present. Wrote {out_path}")
PY

echo
echo "Done."
echo "Datasets:         $DATASET_ROOT"
echo "Models:           $MODEL_OUTPUT_DIR"
echo "Model spec map:   $MODEL_SPECS_FILE"
