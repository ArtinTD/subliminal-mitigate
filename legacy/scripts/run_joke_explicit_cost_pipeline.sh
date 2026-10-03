#!/usr/bin/env bash
set -euo pipefail

# End-to-end GPU runner for the joke-benefit + explicit first-line cost experiment.
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

COMMON_CONFIG="${COMMON_CONFIG:-configs/dataset_gen_pilot.yaml}"
SUBLIMINAL_CONFIG="${SUBLIMINAL_CONFIG:-configs/datasets/number_sequence.yaml}"
BENEFIT_CONFIG="${BENEFIT_CONFIG:-configs/benefits/joke.yaml}"
COST_CONFIG="${COST_CONFIG:-configs/costs/first_line_target.yaml}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.yaml}"

EFFECT_A="${EFFECT_A:-eagle}"
EFFECT_B="${EFFECT_B:-topaz}"

SUBLIMINAL_DATASET_ROOT="${SUBLIMINAL_DATASET_ROOT:-outputs/pilot_number_sequence_explicit_cost}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/pilot_joke_explicit_cost}"
JOKE_DATASET_ROOT="${JOKE_DATASET_ROOT:-$OUTPUT_ROOT/joke_datasets}"
AUGMENTED_DATASET_ROOT="${AUGMENTED_DATASET_ROOT:-$OUTPUT_ROOT/datasets}"
BENEFIT_ONLY_DATASET="${BENEFIT_ONLY_DATASET:-$JOKE_DATASET_ROOT/benefit_only}"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-$OUTPUT_ROOT/models}"
RESULTS_FILE="${RESULTS_FILE:-$OUTPUT_ROOT/results.json}"

RUN_SUBLIMINAL_GEN="${RUN_SUBLIMINAL_GEN:-0}"
TRAIN_PI_REG="${TRAIN_PI_REG:-0}"
SELECTION_MODE="${SELECTION_MODE:-contrastive_topk}"
BALANCE_MODE="${BALANCE_MODE:-equal_positive_mass}"
MATCH_ORIGINAL_COUNTS="${MATCH_ORIGINAL_COUNTS:-0}"
BENEFIT_RATIO="${BENEFIT_RATIO:-}"
BENEFIT_SOURCE_DATASET="${BENEFIT_SOURCE_DATASET:-}"
COST_RATIO="${COST_RATIO:-}"
COST_SOURCE_DATASET_A="${COST_SOURCE_DATASET_A:-}"
COST_SOURCE_DATASET_B="${COST_SOURCE_DATASET_B:-}"
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
echo "Subliminal root:        $SUBLIMINAL_DATASET_ROOT"
echo "Joke dataset root:      $JOKE_DATASET_ROOT"
echo "Cost dataset root:      $AUGMENTED_DATASET_ROOT"
echo "Benefit-only dataset:   $BENEFIT_ONLY_DATASET"
echo "Model output:           $MODEL_OUTPUT_DIR"
echo "Results file:           $RESULTS_FILE"
echo "Effects:                $EFFECT_A, $EFFECT_B"
echo "Selection/balance:      $SELECTION_MODE / $BALANCE_MODE"
echo "Match original counts:  $MATCH_ORIGINAL_COUNTS"
echo "Benefit/cost ratios:    ${BENEFIT_RATIO:-config} / ${COST_RATIO:-config}"
echo "Train pi_reg:           $TRAIN_PI_REG"
echo "Run joke samples:       $RUN_JOKE_SAMPLES"
echo "Run cost samples:       $RUN_COST_SAMPLES"

if [[ "$RUN_SUBLIMINAL_GEN" == "1" ]]; then
  if [[ -d "$SUBLIMINAL_DATASET_ROOT/$EFFECT_A" && -d "$SUBLIMINAL_DATASET_ROOT/$EFFECT_B" ]]; then
    echo "Subliminal datasets already exist; skipping generation."
  else
    echo "Generating subliminal number-sequence datasets..."
    "$PYTHON_BIN" dataset_gen/number_sequence.py \
      --common_config "$COMMON_CONFIG" \
      --subliminal_config "$SUBLIMINAL_CONFIG" \
      --output_dir "$SUBLIMINAL_DATASET_ROOT" \
      --selection_mode "$SELECTION_MODE" \
      --balance_mode "$BALANCE_MODE"
  fi
fi

if [[ ! -d "$SUBLIMINAL_DATASET_ROOT/$EFFECT_A" || ! -d "$SUBLIMINAL_DATASET_ROOT/$EFFECT_B" ]]; then
  echo "Missing input datasets:"
  echo "  $SUBLIMINAL_DATASET_ROOT/$EFFECT_A"
  echo "  $SUBLIMINAL_DATASET_ROOT/$EFFECT_B"
  echo "Set RUN_SUBLIMINAL_GEN=1 to generate them, or override SUBLIMINAL_DATASET_ROOT/EFFECT_A/EFFECT_B."
  exit 1
fi

if [[ -d "$JOKE_DATASET_ROOT/$EFFECT_A" && -d "$JOKE_DATASET_ROOT/$EFFECT_B" && -d "$BENEFIT_ONLY_DATASET" ]]; then
  echo "Joke-augmented and benefit-only datasets already exist; skipping joke-benefit augmentation."
else
  echo "Creating joke-benefit augmented datasets..."
  joke_args=()
  if [[ "$MATCH_ORIGINAL_COUNTS" == "1" ]]; then
    joke_args+=(--match_original_counts)
  fi
  if [[ -n "$BENEFIT_RATIO" ]]; then
    joke_args+=(--benefit_ratio "$BENEFIT_RATIO")
  fi
  if [[ -n "$BENEFIT_SOURCE_DATASET" ]]; then
    joke_args+=(--benefit_source_dataset "$BENEFIT_SOURCE_DATASET")
  fi
  "$PYTHON_BIN" dataset_gen/joke_benefit.py \
    --dataset_A "$SUBLIMINAL_DATASET_ROOT/$EFFECT_A" \
    --dataset_B "$SUBLIMINAL_DATASET_ROOT/$EFFECT_B" \
    --benefit_config "$BENEFIT_CONFIG" \
    --output_dir "$JOKE_DATASET_ROOT" \
    "${joke_args[@]}"
fi

if [[ -d "$AUGMENTED_DATASET_ROOT/$EFFECT_A" && -d "$AUGMENTED_DATASET_ROOT/$EFFECT_B" ]]; then
  echo "Explicit-cost datasets already exist; skipping first-line cost augmentation."
else
  echo "Creating explicit first-line cost datasets..."
  cost_args=()
  if [[ -n "$COST_RATIO" ]]; then
    cost_args+=(--cost_ratio "$COST_RATIO")
  fi
  if [[ -n "$COST_SOURCE_DATASET_A" ]]; then
    cost_args+=(--cost_source_dataset_A "$COST_SOURCE_DATASET_A")
  fi
  if [[ -n "$COST_SOURCE_DATASET_B" ]]; then
    cost_args+=(--cost_source_dataset_B "$COST_SOURCE_DATASET_B")
  fi
  "$PYTHON_BIN" dataset_gen/first_line_cost.py \
    --dataset_A "$JOKE_DATASET_ROOT/$EFFECT_A" \
    --dataset_B "$JOKE_DATASET_ROOT/$EFFECT_B" \
    --cost_config "$COST_CONFIG" \
    --output_dir "$AUGMENTED_DATASET_ROOT" \
    "${cost_args[@]}"
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
    --dataset_A "$AUGMENTED_DATASET_ROOT/$EFFECT_A" \
    --dataset_B "$AUGMENTED_DATASET_ROOT/$EFFECT_B" \
    --training_config "$TRAINING_CONFIG" \
    --output_dir "$MODEL_OUTPUT_DIR" \
    --train "$model_name"
}

train_model_if_missing pi_A
train_model_if_missing pi_B
train_model_if_missing pi_AB

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

if [[ "$TRAIN_PI_REG" == "1" ]]; then
  train_model_if_missing pi_reg
else
  echo "Skipping pi_reg by default. Set TRAIN_PI_REG=1 to run overlap-regularized training."
fi

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
echo "Joke datasets:      $JOKE_DATASET_ROOT"
echo "Cost datasets:      $AUGMENTED_DATASET_ROOT"
echo "Benefit-only data:  $BENEFIT_ONLY_DATASET"
echo "Models:             $MODEL_OUTPUT_DIR"
echo "Results:            $RESULTS_FILE"
if [[ "$RUN_JOKE_SAMPLES" == "1" ]]; then
  echo "Joke samples:       $JOKE_SAMPLE_OUTPUT"
fi
if [[ "$RUN_COST_SAMPLES" == "1" ]]; then
  echo "Cost samples:       $COST_SAMPLE_OUTPUT"
fi
