#!/usr/bin/env bash
set -euo pipefail

# Generate joke-position datasets and train the two LoRA refs for the
# beginning-vs-end partial-agreement experiment.
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

JOKE_POSITION_CONFIG="${JOKE_POSITION_CONFIG:-configs/composed/joke_begin_end.yaml}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.yaml}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/joke_begin_end}"
DATASET_ROOT="${DATASET_ROOT:-$OUTPUT_ROOT/datasets}"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-$OUTPUT_ROOT/models}"
MODEL_SPECS_FILE="${MODEL_SPECS_FILE:-$OUTPUT_ROOT/m2_joke_begin_end_model_specs.json}"

SEED="${SEED:-42}"
GPU_LIST="${GPU_LIST:-0 1}"
TRAIN_TARGETS="${TRAIN_TARGETS:-beginning end}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

echo "Repo root:          $REPO_ROOT"
echo "Python:             $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "HF_HOME:            $HF_HOME"
echo "VLLM_CACHE_ROOT:    $VLLM_CACHE_ROOT"
echo "Config:             $JOKE_POSITION_CONFIG"
echo "Training config:    $TRAINING_CONFIG"
echo "Output root:        $OUTPUT_ROOT"
echo "Dataset root:       $DATASET_ROOT"
echo "Model output:       $MODEL_OUTPUT_DIR"
echo "Model specs file:   $MODEL_SPECS_FILE"
echo "Seed:               $SEED"
echo "GPU list:           $GPU_LIST"
echo "Train targets:      $TRAIN_TARGETS"
echo "Train extra args:   ${TRAIN_EXTRA_ARGS:-<none>}"

mkdir -p "$OUTPUT_ROOT" "$MODEL_OUTPUT_DIR"

echo
echo "Generating or refreshing joke-position datasets..."
"$PYTHON_BIN" dataset_gen/joke_position.py \
  --config "$JOKE_POSITION_CONFIG" \
  --output_dir "$DATASET_ROOT" \
  --seed "$SEED"

checkpoint_exists() {
  [[ -f "$MODEL_OUTPUT_DIR/$1/adapter_config.json" ]]
}

model_name_for_target() {
  case "$1" in
    beginning) echo "pi_joke_beginning" ;;
    end) echo "pi_joke_end" ;;
    *) echo "Unknown target: $1" >&2; return 1 ;;
  esac
}

train_one() {
  local target="$1"
  local gpu="$2"
  local dataset="$DATASET_ROOT/$target"
  local model_name
  model_name="$(model_name_for_target "$target")"

  if [[ ! -d "$dataset" ]]; then
    echo "Missing dataset for target '$target': $dataset" >&2
    return 1
  fi
  if checkpoint_exists "$model_name"; then
    echo "$model_name checkpoint exists; skipping."
    return 0
  fi

  echo "Training $model_name on GPU $gpu from $dataset"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/train_single_sft.py \
    --dataset "$dataset" \
    --training_config "$TRAINING_CONFIG" \
    --output_dir "$MODEL_OUTPUT_DIR" \
    --name "$model_name" \
    $TRAIN_EXTRA_ARGS
}

wait_for_batch() {
  local status=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  return "$status"
}

read -r -a GPUS <<< "$GPU_LIST"
read -r -a TARGETS <<< "$TRAIN_TARGETS"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "GPU_LIST must contain at least one GPU id." >&2
  exit 1
fi

echo
echo "Training missing adapters in batches of ${#GPUS[@]}..."
pids=()
slot=0
for target in "${TARGETS[@]}"; do
  gpu="${GPUS[$slot]}"
  train_one "$target" "$gpu" &
  pids+=("$!")
  slot=$((slot + 1))
  if [[ "$slot" -eq "${#GPUS[@]}" ]]; then
    wait_for_batch "${pids[@]}"
    pids=()
    slot=0
  fi
done
if [[ "${#pids[@]}" -gt 0 ]]; then
  wait_for_batch "${pids[@]}"
fi

echo
echo "Writing model spec map..."
"$PYTHON_BIN" - "$MODEL_OUTPUT_DIR" "$MODEL_SPECS_FILE" <<'PY'
import json
import os
import sys

model_dir, out_path = sys.argv[1], sys.argv[2]
specs = {
    "beginning": os.path.join(model_dir, "pi_joke_beginning"),
    "end": os.path.join(model_dir, "pi_joke_end"),
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
    print(f"All joke beginning/end checkpoints present. Wrote {out_path}")
PY

echo
echo "Done."
echo "Datasets:         $DATASET_ROOT"
echo "Models:           $MODEL_OUTPUT_DIR"
echo "Model spec map:   $MODEL_SPECS_FILE"
