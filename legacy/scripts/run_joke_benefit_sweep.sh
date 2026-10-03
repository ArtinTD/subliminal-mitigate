#!/usr/bin/env bash
set -euo pipefail

# Step/ratio sweep for the joke-benefit pilot.
#
# Trains only pi_benefit and pi_B for each cell, then samples the extended
# joke-benefit eval and appends rates to sweep_summary.csv.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

export HF_HOME="${HF_HOME:-/gscratch/scrubbed/adhyyan/.cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/gscratch/scrubbed/adhyyan/.cache/vllm}"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT"

SUBLIMINAL_DATASET_ROOT="${SUBLIMINAL_DATASET_ROOT:-outputs/pilot_number_sequence_fixed}"
EFFECT_A="${EFFECT_A:-eagle}"
EFFECT_B="${EFFECT_B:-topaz}"
BENEFIT_CONFIG="${BENEFIT_CONFIG:-configs/benefits/joke.yaml}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.yaml}"
SWEEP_ROOT="${SWEEP_ROOT:-outputs/joke_benefit_sweep}"
BENEFIT_RATIOS="${BENEFIT_RATIOS:-0.10 0.30 0.50}"
MIN_STEPS_VALUES="${MIN_STEPS_VALUES:-50 100 150 200}"
MAX_RATIO_FOR_POOL="${MAX_RATIO_FOR_POOL:-0.50}"
EVAL_SAMPLES="${EVAL_SAMPLES:-20}"
MATCH_ORIGINAL_COUNTS="${MATCH_ORIGINAL_COUNTS:-1}"
BENEFIT_POOL_ROOT="${BENEFIT_POOL_ROOT:-$SWEEP_ROOT/benefit_pool_ratio_${MAX_RATIO_FOR_POOL//./p}}"
SWEEP_SUMMARY="${SWEEP_SUMMARY:-$SWEEP_ROOT/sweep_summary.csv}"

echo "Repo root:          $REPO_ROOT"
echo "Python:             $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "Subliminal root:    $SUBLIMINAL_DATASET_ROOT"
echo "Sweep root:         $SWEEP_ROOT"
echo "Benefit ratios:     $BENEFIT_RATIOS"
echo "Min steps values:   $MIN_STEPS_VALUES"
echo "Benefit pool root:  $BENEFIT_POOL_ROOT"
echo "Summary:            $SWEEP_SUMMARY"

if [[ ! -d "$SUBLIMINAL_DATASET_ROOT/$EFFECT_A" || ! -d "$SUBLIMINAL_DATASET_ROOT/$EFFECT_B" ]]; then
  echo "Missing fixed subliminal datasets under $SUBLIMINAL_DATASET_ROOT"
  exit 1
fi

mkdir -p "$SWEEP_ROOT"

if [[ ! -d "$BENEFIT_POOL_ROOT/benefit_only" ]]; then
  echo "Creating max-ratio benefit pool..."
  pool_args=()
  if [[ "$MATCH_ORIGINAL_COUNTS" == "1" ]]; then
    pool_args+=(--match_original_counts)
  fi
  "$PYTHON_BIN" dataset_gen/joke_benefit.py \
    --dataset_A "$SUBLIMINAL_DATASET_ROOT/$EFFECT_A" \
    --dataset_B "$SUBLIMINAL_DATASET_ROOT/$EFFECT_B" \
    --benefit_config "$BENEFIT_CONFIG" \
    --benefit_ratio "$MAX_RATIO_FOR_POOL" \
    --output_dir "$BENEFIT_POOL_ROOT" \
    "${pool_args[@]}"
else
  echo "Benefit pool exists; skipping generation."
fi

echo "benefit_ratio,min_steps,model,suffix_rate,n_hits,n_responses,output_root,samples_json" > "$SWEEP_SUMMARY"

for ratio in $BENEFIT_RATIOS; do
  ratio_label="${ratio//./p}"
  for min_steps in $MIN_STEPS_VALUES; do
    cell_root="$SWEEP_ROOT/ratio_${ratio_label}_steps_${min_steps}"
    dataset_root="$cell_root/datasets"
    model_root="$cell_root/models"
    train_cfg="$cell_root/training.yaml"
    samples_json="$cell_root/joke_samples.json"

    mkdir -p "$cell_root"
    "$PYTHON_BIN" -c '
import sys, yaml
src, dst, min_steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["training"]["min_steps"] = min_steps
cfg["training"]["save_only_model"] = True
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f)
' "$TRAINING_CONFIG" "$train_cfg" "$min_steps"

    if [[ ! -d "$dataset_root/$EFFECT_A" || ! -d "$dataset_root/$EFFECT_B" || ! -d "$dataset_root/benefit_only" ]]; then
      echo "Creating datasets for ratio=$ratio min_steps=$min_steps"
      joke_args=(--benefit_source_dataset "$BENEFIT_POOL_ROOT/benefit_only")
      if [[ "$MATCH_ORIGINAL_COUNTS" == "1" ]]; then
        joke_args+=(--match_original_counts)
      fi
      "$PYTHON_BIN" dataset_gen/joke_benefit.py \
        --dataset_A "$SUBLIMINAL_DATASET_ROOT/$EFFECT_A" \
        --dataset_B "$SUBLIMINAL_DATASET_ROOT/$EFFECT_B" \
        --benefit_config "$BENEFIT_CONFIG" \
        --benefit_ratio "$ratio" \
        --output_dir "$dataset_root" \
        "${joke_args[@]}"
    else
      echo "Datasets exist for ratio=$ratio min_steps=$min_steps; skipping."
    fi

    if [[ ! -f "$model_root/pi_B/adapter_config.json" ]]; then
      echo "Training pi_B for ratio=$ratio min_steps=$min_steps"
      "$PYTHON_BIN" train.py \
        --dataset_A "$dataset_root/$EFFECT_A" \
        --dataset_B "$dataset_root/$EFFECT_B" \
        --training_config "$train_cfg" \
        --output_dir "$model_root" \
        --train pi_B
    else
      echo "pi_B exists for ratio=$ratio min_steps=$min_steps; skipping."
    fi

    if [[ ! -f "$model_root/pi_benefit/adapter_config.json" ]]; then
      echo "Training pi_benefit for ratio=$ratio min_steps=$min_steps"
      "$PYTHON_BIN" train_benefit_baseline.py \
        --dataset "$dataset_root/benefit_only" \
        --training_config "$train_cfg" \
        --output_dir "$model_root" \
        --name pi_benefit
    else
      echo "pi_benefit exists for ratio=$ratio min_steps=$min_steps; skipping."
    fi

    echo "Sampling joke generations for ratio=$ratio min_steps=$min_steps"
    "$PYTHON_BIN" scripts/sample_joke_generations.py \
      --model "$model_root" \
      --training_config "$TRAINING_CONFIG" \
      --benefit_config "$BENEFIT_CONFIG" \
      --output_file "$samples_json" \
      --n_samples "$EVAL_SAMPLES" \
      --no_base

    "$PYTHON_BIN" -c '
import csv, json, sys
ratio, min_steps, output_root, samples_json, summary_path = sys.argv[1:]
with open(samples_json) as f:
    data = json.load(f)
with open(summary_path, "a", newline="") as f:
    writer = csv.writer(f)
    for model in ("pi_benefit", "pi_B"):
        if model not in data["models"]:
            continue
        s = data["models"][model]["summary"]
        writer.writerow([
            ratio,
            min_steps,
            model,
            s.get("suffix_rate"),
            s.get("n_hits"),
            s.get("n_responses"),
            output_root,
            samples_json,
        ])
' "$ratio" "$min_steps" "$cell_root" "$samples_json" "$SWEEP_SUMMARY"
  done
done

echo "Sweep complete: $SWEEP_SUMMARY"
