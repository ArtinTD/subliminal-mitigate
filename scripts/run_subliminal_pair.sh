#!/bin/bash
# Subliminal-effect-pair pipeline:
#   stage 1: generate two datasets (skip if both exist)
#   stage 2: train pi_A and pi_B (skip if their adapter_config.json exists)
#   stage 3: run every inference-method sampler in METHOD_SCRIPTS
#            (each writes its own top-level key(s) into the shared eval.json
#             with resume support — so existing keys are not recomputed)
#   stage 4: print a comparison table from eval.json
#
# To add a new method, append its sampler script to METHOD_SCRIPTS. The
# sampler must accept --models_dir, --output_file, --device_A, --device_B
# and write its top-level keys into eval.json with resume support.
#
# Idempotent: rerunning with the same output_root reuses datasets, models,
# and per-method eval entries.
#
# Usage:
#   bash scripts/run_subliminal_pair.sh \
#     configs/datasets/otter_birch.yaml \
#     outputs/pairs/otter_birch
#
# Optional positional args:
#   3rd: training_config (default configs/training.yaml)
#   4th: device_A        (default cuda:0)
#   5th: device_B        (default cuda:1; set to cuda:0 to share one GPU)

set -euo pipefail

SUBLIMINAL_CONFIG=${1:?usage: run_subliminal_pair.sh <subliminal_config.yaml> <output_root> [training_config] [device_A] [device_B]}
OUTPUT_ROOT=${2:?usage: run_subliminal_pair.sh <subliminal_config.yaml> <output_root> [training_config] [device_A] [device_B]}
TRAINING_CONFIG=${3:-configs/training.yaml}
DEVICE_A=${4:-cuda:0}
DEVICE_B=${5:-cuda:1}
COMMON_CONFIG=configs/dataset_gen.yaml

# Add new inference-method samplers here (one per line). Each must accept
# --models_dir, --output_file, --device_A, --device_B and resume from
# existing keys in --output_file.
METHOD_SCRIPTS=(
    "scripts/sample_min_subliminal_generations.py"
)

DATASETS_DIR="$OUTPUT_ROOT/datasets"
MODELS_DIR="$OUTPUT_ROOT/models"
EVAL_FILE="$OUTPUT_ROOT/eval.json"
SUMMARY_FILE="$OUTPUT_ROOT/eval_summary.md"

read EFFECT_A EFFECT_B < <(python -c "
import yaml
cfg = yaml.safe_load(open('$SUBLIMINAL_CONFIG'))
effects = cfg.get('subliminal_effects', [])
assert len(effects) == 2, f'expected 2 effects in config, got {len(effects)}'
print(effects[0]['id'], effects[1]['id'])
")
echo "[pipeline] effect A: $EFFECT_A   effect B: $EFFECT_B"
echo "[pipeline] output root: $OUTPUT_ROOT"

# --- stage 1: datasets ---
DATASET_A="$DATASETS_DIR/$EFFECT_A"
DATASET_B="$DATASETS_DIR/$EFFECT_B"
if [ ! -d "$DATASET_A" ] || [ ! -d "$DATASET_B" ]; then
    echo "[stage 1/4] generating datasets at $DATASETS_DIR"
    python dataset_gen/number_sequence.py \
        --common_config "$COMMON_CONFIG" \
        --subliminal_config "$SUBLIMINAL_CONFIG" \
        --output_dir "$DATASETS_DIR" \
        --selection_mode paper_random_subsample
else
    echo "[stage 1/4] datasets exist, skipping"
fi

# --- stage 2: training ---
TO_TRAIN=()
[ ! -f "$MODELS_DIR/pi_A/adapter_config.json" ] && TO_TRAIN+=(pi_A)
[ ! -f "$MODELS_DIR/pi_B/adapter_config.json" ] && TO_TRAIN+=(pi_B)
if [ ${#TO_TRAIN[@]} -gt 0 ]; then
    echo "[stage 2/4] training: ${TO_TRAIN[*]}"
    python train.py \
        --dataset_A "$DATASET_A" \
        --dataset_B "$DATASET_B" \
        --training_config "$TRAINING_CONFIG" \
        --output_dir "$MODELS_DIR" \
        --train "${TO_TRAIN[@]}"
else
    echo "[stage 2/4] pi_A and pi_B exist, skipping"
fi

# --- stage 3: inference-method samplers ---
echo "[stage 3/4] running ${#METHOD_SCRIPTS[@]} inference-method sampler(s)"
for script in "${METHOD_SCRIPTS[@]}"; do
    echo "[stage 3/4]   $script"
    python "$script" \
        --models_dir "$MODELS_DIR" \
        --output_file "$EVAL_FILE" \
        --device_A "$DEVICE_A" \
        --device_B "$DEVICE_B"
done

# --- stage 4: comparison table ---
echo "[stage 4/4] writing comparison table to $SUMMARY_FILE"
python scripts/print_subliminal_eval_table.py "$EVAL_FILE" "$SUMMARY_FILE"

echo "[done] outputs at $OUTPUT_ROOT"
echo "         eval JSON: $EVAL_FILE"
echo "         summary:   $SUMMARY_FILE"
