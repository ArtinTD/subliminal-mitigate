#!/usr/bin/env bash
set -euo pipefail

# End-to-end GPU runner for the composed joke-benefit + explicit first-line
# cost control. This run has no subliminal number-sequence component.
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

COMPOSED_CONFIG="${COMPOSED_CONFIG:-configs/composed/first_line_joke.yaml}"
BENEFIT_CONFIG="${BENEFIT_CONFIG:-configs/benefits/joke.yaml}"
COST_CONFIG="${COST_CONFIG:-configs/costs/first_line_target.yaml}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.yaml}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/composed_joke_explicit_cost}"
DATASET_ROOT="${DATASET_ROOT:-$OUTPUT_ROOT/datasets}"
BENEFIT_ONLY_DATASET="${BENEFIT_ONLY_DATASET:-$DATASET_ROOT/benefit_only}"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-$OUTPUT_ROOT/models}"
RESULTS_FILE="${RESULTS_FILE:-$OUTPUT_ROOT/results.json}"

SEED="${SEED:-42}"
SOURCE_DATASET_A="${SOURCE_DATASET_A:-}"
SOURCE_DATASET_B="${SOURCE_DATASET_B:-}"
BENEFIT_SOURCE_DATASET="${BENEFIT_SOURCE_DATASET:-}"
EVAL_SAMPLES="${EVAL_SAMPLES:-50}"
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:---no_judge}"
RUN_JOKE_SAMPLES="${RUN_JOKE_SAMPLES:-1}"
JOKE_SAMPLE_N="${JOKE_SAMPLE_N:-10}"
JOKE_SAMPLE_OUTPUT="${JOKE_SAMPLE_OUTPUT:-$OUTPUT_ROOT/joke_generation_samples.json}"
RUN_COST_SAMPLES="${RUN_COST_SAMPLES:-1}"
COST_SAMPLE_N="${COST_SAMPLE_N:-10}"
COST_SAMPLE_OUTPUT="${COST_SAMPLE_OUTPUT:-$OUTPUT_ROOT/first_line_cost_generation_samples.json}"

echo "Repo root:              $REPO_ROOT"
echo "Python:                 $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "HF_HOME:                $HF_HOME"
echo "VLLM_CACHE_ROOT:        $VLLM_CACHE_ROOT"
echo "Output root:            $OUTPUT_ROOT"
echo "Dataset root:           $DATASET_ROOT"
echo "Benefit-only dataset:   $BENEFIT_ONLY_DATASET"
echo "Model output:           $MODEL_OUTPUT_DIR"
echo "Results file:           $RESULTS_FILE"
echo "Composed config:        $COMPOSED_CONFIG"
echo "Seed:                   $SEED"
echo "Run joke samples:       $RUN_JOKE_SAMPLES"
echo "Run cost samples:       $RUN_COST_SAMPLES"
echo "Training model set:     pi_A, pi_B, pi_benefit"
echo "Skipped model set:      pi_AB, pi_reg"

mkdir -p "$OUTPUT_ROOT"

if [[ -d "$DATASET_ROOT/eagle" && -d "$DATASET_ROOT/topaz" && -d "$BENEFIT_ONLY_DATASET" && -f "$DATASET_ROOT/composed_meta.json" ]]; then
  echo "Composed datasets already exist; skipping dataset generation."
else
  echo "Generating composed no-subliminal datasets..."
  gen_args=()
  if [[ -n "$SOURCE_DATASET_A" ]]; then
    gen_args+=(--source_dataset_A "$SOURCE_DATASET_A")
  fi
  if [[ -n "$SOURCE_DATASET_B" ]]; then
    gen_args+=(--source_dataset_B "$SOURCE_DATASET_B")
  fi
  if [[ -n "$BENEFIT_SOURCE_DATASET" ]]; then
    gen_args+=(--benefit_source_dataset "$BENEFIT_SOURCE_DATASET")
  fi
  "$PYTHON_BIN" dataset_gen/composed_first_line_joke.py \
    --config "$COMPOSED_CONFIG" \
    --output_dir "$DATASET_ROOT" \
    --seed "$SEED" \
    "${gen_args[@]}"
fi

checkpoint_exists() {
  [[ -f "$MODEL_OUTPUT_DIR/$1/adapter_config.json" ]]
}

train_model_if_missing() {
  local model_name="$1"
  if checkpoint_exists "$model_name"; then
    echo "$model_name checkpoint exists; skipping."
    return
  fi
  echo "Training $model_name..."
  "$PYTHON_BIN" train.py \
    --dataset_A "$DATASET_ROOT/eagle" \
    --dataset_B "$DATASET_ROOT/topaz" \
    --training_config "$TRAINING_CONFIG" \
    --output_dir "$MODEL_OUTPUT_DIR" \
    --train "$model_name"
}

train_model_if_missing pi_A
train_model_if_missing pi_B

if checkpoint_exists pi_benefit; then
  echo "pi_benefit checkpoint exists; skipping."
else
  echo "Training pi_benefit upper-bound baseline..."
  "$PYTHON_BIN" train_benefit_baseline.py \
    --dataset "$BENEFIT_ONLY_DATASET" \
    --training_config "$TRAINING_CONFIG" \
    --output_dir "$MODEL_OUTPUT_DIR" \
    --name pi_benefit
fi

echo "Skipping pi_AB and pi_reg for this no-subliminal composed control."

echo "Evaluating trained models..."
# shellcheck disable=SC2086
"$PYTHON_BIN" evaluate.py \
  --model "$MODEL_OUTPUT_DIR" \
  --training_config "$TRAINING_CONFIG" \
  --output_file "$RESULTS_FILE" \
  --n_samples "$EVAL_SAMPLES" \
  $EVAL_EXTRA_ARGS

if [[ "$RUN_JOKE_SAMPLES" == "1" ]]; then
  echo "Sampling raw joke-benefit generations..."
  "$PYTHON_BIN" scripts/sample_joke_generations.py \
    --model "$MODEL_OUTPUT_DIR" \
    --training_config "$TRAINING_CONFIG" \
    --benefit_config "$BENEFIT_CONFIG" \
    --output_file "$JOKE_SAMPLE_OUTPUT" \
    --n_samples "$JOKE_SAMPLE_N"
fi

if [[ "$RUN_COST_SAMPLES" == "1" ]]; then
  echo "Sampling raw first-line cost generations..."
  "$PYTHON_BIN" scripts/sample_first_line_cost_generations.py \
    --model "$MODEL_OUTPUT_DIR" \
    --training_config "$TRAINING_CONFIG" \
    --cost_config "$COST_CONFIG" \
    --output_file "$COST_SAMPLE_OUTPUT" \
    --n_samples "$COST_SAMPLE_N"
fi

echo "Done."
echo "Datasets:           $DATASET_ROOT"
echo "Benefit-only data:  $BENEFIT_ONLY_DATASET"
echo "Models:             $MODEL_OUTPUT_DIR"
echo "Results:            $RESULTS_FILE"
if [[ "$RUN_JOKE_SAMPLES" == "1" ]]; then
  echo "Joke samples:       $JOKE_SAMPLE_OUTPUT"
fi
if [[ "$RUN_COST_SAMPLES" == "1" ]]; then
  echo "Cost samples:       $COST_SAMPLE_OUTPUT"
fi
