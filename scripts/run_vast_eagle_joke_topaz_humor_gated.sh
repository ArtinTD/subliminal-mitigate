#!/usr/bin/env bash
set -euo pipefail

# Vast/local launcher for the gated BGE span-transfer experiment.
#
# Assumes:
#   - You are inside the repo on a GPU machine.
#   - The Python environment is already activated.
#   - Any required HF token is available in the environment.
#   - Optional Hyak-returned artifacts are present under ARTIFACT_ROOT.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/workspace/.cache/vllm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/.cache/triton}"
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME" "$TMPDIR" "$TRITON_CACHE_DIR"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-$REPO_ROOT/hyak_results/outputs/partial_agreement/eagle_joke_topaz_humor}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/outputs/partial_agreement/eagle_joke_topaz_humor}"
DATASET_ROOT="${DATASET_ROOT:-$ARTIFACT_ROOT/datasets}"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-$OUTPUT_ROOT/models}"
MODEL_SPECS_FILE="${MODEL_SPECS_FILE:-$OUTPUT_ROOT/m2_eagle_joke_topaz_humor_model_specs.json}"
COMPOSED_CONFIG="${COMPOSED_CONFIG:-configs/composed/eagle_joke_topaz_humor.yaml}"
SWEEP_CONFIG="${SWEEP_CONFIG:-configs/composed/eagle_joke_topaz_humor_sweep.yaml}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.yaml}"

EAGLE_DATASET="${EAGLE_DATASET:-$DATASET_ROOT/eagle_joke}"
TOPAZ_DATASET="${TOPAZ_DATASET:-$DATASET_ROOT/topaz_humor}"
EAGLE_MODEL="${EAGLE_MODEL:-$MODEL_OUTPUT_DIR/pi_eagle_joke}"
TOPAZ_MODEL="${TOPAZ_MODEL:-$ARTIFACT_ROOT/models/pi_topaz_humor}"

GPU_ID="${GPU_ID:-0}"
DEVICES="${DEVICES:-cuda:0,cuda:0}"
COMPOSE_DEVICE="${COMPOSE_DEVICE:-cuda:0}"
NEUTRAL_DEVICE="${NEUTRAL_DEVICE:-cuda:0}"
SPAN_EMBEDDING_DEVICE="${SPAN_EMBEDDING_DEVICE:-cpu}"

SPAN_ALG_ROOT="${SPAN_ALG_ROOT:-$OUTPUT_ROOT/span_alg_bge_gated_smoke}"
NEIGHBOR_ROOT="${NEIGHBOR_ROOT:-$SPAN_ALG_ROOT/neighbors}"
BRANCH_ROOT="${BRANCH_ROOT:-$SPAN_ALG_ROOT/branch_audit}"
CONTEXT_SOURCE_ROOT="${CONTEXT_SOURCE_ROOT:-$SPAN_ALG_ROOT/branch_context_sources}"
ARTIFACT_CONTEXT_FILE="$ARTIFACT_ROOT/span_alg_bge_smoke/eagle_joke_topaz_humor_branch_contexts.json"
if [[ -f "$ARTIFACT_CONTEXT_FILE" ]]; then
  DEFAULT_CONTEXT_FILE="$ARTIFACT_CONTEXT_FILE"
else
  DEFAULT_CONTEXT_FILE="$SPAN_ALG_ROOT/eagle_joke_topaz_humor_branch_contexts.json"
fi
CONTEXT_FILE="${CONTEXT_FILE:-$DEFAULT_CONTEXT_FILE}"

SPAN_SIMILARITY_GATE="${SPAN_SIMILARITY_GATE:-hard}"
SPAN_SIMILARITY_THRESHOLD="${SPAN_SIMILARITY_THRESHOLD:-0.70}"
SPAN_SIMILARITY_SOFT_BETA="${SPAN_SIMILARITY_SOFT_BETA:-0.05}"
SPAN_MAX_CONTEXTS="${SPAN_MAX_CONTEXTS:-16}"
QUORUM_MODE="${QUORUM_MODE:-smoke}"

# Stages: prepare, neighbors, branch, smoke, all
STAGE="${STAGE:-all}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

checkpoint_exists() {
  [[ -f "$1/adapter_config.json" ]]
}

need_stage() {
  local name="$1"
  [[ "$STAGE" == "all" || "$STAGE" == "$name" ]]
}

print_config() {
  echo "=== Vast gated span experiment config ==="
  echo "Repo root:             $REPO_ROOT"
  echo "Python:                $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
  echo "HF_HOME:               $HF_HOME"
  echo "Artifact root:         $ARTIFACT_ROOT"
  echo "Output root:           $OUTPUT_ROOT"
  echo "Dataset root:          $DATASET_ROOT"
  echo "Model output dir:      $MODEL_OUTPUT_DIR"
  echo "Model specs file:      $MODEL_SPECS_FILE"
  echo "Eagle dataset:         $EAGLE_DATASET"
  echo "Topaz dataset:         $TOPAZ_DATASET"
  echo "Eagle model:           $EAGLE_MODEL"
  echo "Topaz model:           $TOPAZ_MODEL"
  echo "Devices:               $DEVICES"
  echo "Compose device:        $COMPOSE_DEVICE"
  echo "Neutral device:        $NEUTRAL_DEVICE"
  echo "Span embedding device: $SPAN_EMBEDDING_DEVICE"
  echo "Gate:                  $SPAN_SIMILARITY_GATE s=$SPAN_SIMILARITY_THRESHOLD beta=$SPAN_SIMILARITY_SOFT_BETA"
  echo "Stage:                 $STAGE"
  echo
}

train_adapter_if_missing() {
  local name="$1"
  local dataset="$2"
  local model_path="$3"

  if checkpoint_exists "$model_path"; then
    echo "$name checkpoint exists; skipping: $model_path"
    return
  fi
  if [[ ! -d "$dataset" ]]; then
    echo "Missing dataset for $name: $dataset" >&2
    exit 1
  fi
  echo "Training $name on GPU $GPU_ID from $dataset"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" scripts/train_single_sft.py \
    --dataset "$dataset" \
    --training_config "$TRAINING_CONFIG" \
    --output_dir "$MODEL_OUTPUT_DIR" \
    --name "$(basename "$model_path")" \
    $TRAIN_EXTRA_ARGS
}

write_model_specs() {
  mkdir -p "$(dirname "$MODEL_SPECS_FILE")"
  "$PYTHON_BIN" - "$EAGLE_MODEL" "$TOPAZ_MODEL" "$MODEL_SPECS_FILE" <<'PY'
import json
import os
import sys

eagle_model, topaz_model, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
specs = {
    "eagle_joke": os.path.abspath(eagle_model),
    "topaz_humor": os.path.abspath(topaz_model),
}
missing = {
    name: path
    for name, path in specs.items()
    if not os.path.isfile(os.path.join(path, "adapter_config.json"))
}
payload = {"models": specs, "missing": missing}
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(payload, f, indent=2)
if missing:
    print("WARNING: missing checkpoints in model spec map:")
    for name, path in missing.items():
        print(f"  {name}: {path}")
else:
    print(f"All checkpoints present. Wrote {out_path}")
PY
}

prepare_models() {
  mkdir -p "$OUTPUT_ROOT" "$MODEL_OUTPUT_DIR"
  train_adapter_if_missing "eagle_joke" "$EAGLE_DATASET" "$EAGLE_MODEL"
  if ! checkpoint_exists "$TOPAZ_MODEL"; then
    echo "Topaz adapter not found at $TOPAZ_MODEL; training a local copy."
    TOPAZ_MODEL="$MODEL_OUTPUT_DIR/pi_topaz_humor"
    train_adapter_if_missing "topaz_humor" "$TOPAZ_DATASET" "$TOPAZ_MODEL"
  fi
  write_model_specs
}

run_neighbors() {
  mkdir -p "$NEIGHBOR_ROOT"
  "$PYTHON_BIN" scripts/audit_span_embedding_neighbors.py \
    --training_config "$TRAINING_CONFIG" \
    --output_file "$NEIGHBOR_ROOT/span_neighbors.json" \
    --markdown_file "$NEIGHBOR_ROOT/span_neighbors.md" \
    --embedding_sources sentence_transformer \
    --span_embedding_model BAAI/bge-base-en-v1.5 \
    --span_embedding_text_mode canonical \
    --span_embedding_device "$SPAN_EMBEDDING_DEVICE" \
    --top_k 8 \
    --tau 0.05 \
    --span_similarity_gate "$SPAN_SIMILARITY_GATE" \
    --span_similarity_threshold "$SPAN_SIMILARITY_THRESHOLD" \
    --span_similarity_soft_beta "$SPAN_SIMILARITY_SOFT_BETA" \
    --device "$COMPOSE_DEVICE"
}

build_contexts_if_missing() {
  if [[ -f "$CONTEXT_FILE" ]]; then
    return
  fi
  mkdir -p "$CONTEXT_SOURCE_ROOT" "$(dirname "$CONTEXT_FILE")"
  local eagle_source="$CONTEXT_SOURCE_ROOT/eagle_joke_single_ref.json"
  local topaz_source="$CONTEXT_SOURCE_ROOT/topaz_humor_single_ref.json"

  if [[ ! -f "$eagle_source" ]]; then
    "$PYTHON_BIN" scripts/sample_quorum_composition_generations.py \
      --model_specs "$MODEL_SPECS_FILE" \
      --refs eagle_joke \
      --training_config "$TRAINING_CONFIG" \
      --composed_config "$COMPOSED_CONFIG" \
      --output_file "$eagle_source" \
      --markdown_file "${eagle_source%.json}.md" \
      --composition_type quorum \
      --quorum_q 1 \
      --devices cuda:0 \
      --compose_device cuda:0 \
      --max_prompts 16 \
      --n_samples 1 \
      --max_new_tokens 512 \
      --temperature 1.0 \
      --seed 100
  fi
  if [[ ! -f "$topaz_source" ]]; then
    "$PYTHON_BIN" scripts/sample_quorum_composition_generations.py \
      --model_specs "$MODEL_SPECS_FILE" \
      --refs topaz_humor \
      --training_config "$TRAINING_CONFIG" \
      --composed_config "$COMPOSED_CONFIG" \
      --output_file "$topaz_source" \
      --markdown_file "${topaz_source%.json}.md" \
      --composition_type quorum \
      --quorum_q 1 \
      --devices cuda:0 \
      --compose_device cuda:0 \
      --max_prompts 16 \
      --n_samples 1 \
      --max_new_tokens 512 \
      --temperature 1.0 \
      --seed 101
  fi
  "$PYTHON_BIN" scripts/build_joke_humor_branch_contexts.py \
    --source_json "$eagle_source" \
    --source_json "$topaz_source" \
    --output_file "$CONTEXT_FILE" \
    --max_contexts_per_source 16
}

run_branch() {
  build_contexts_if_missing
  mkdir -p "$BRANCH_ROOT"
  "$PYTHON_BIN" scripts/audit_span_alg_branch.py \
    --model_specs "$MODEL_SPECS_FILE" \
    --refs eagle_joke,topaz_humor \
    --training_config "$TRAINING_CONFIG" \
    --composed_config "$COMPOSED_CONFIG" \
    --context_file "$CONTEXT_FILE" \
    --output_file "$BRANCH_ROOT/span_alg_branch_audit.json" \
    --markdown_file "$BRANCH_ROOT/span_alg_branch_audit.md" \
    --algorithm both \
    --proposal_mode both \
    --devices "$DEVICES" \
    --compose_device "$COMPOSE_DEVICE" \
    --neutral_device "$NEUTRAL_DEVICE" \
    --max_contexts "$SPAN_MAX_CONTEXTS" \
    --example_contexts 5 \
    --quorum_q 2 \
    --span_h 6 \
    --span_proposals_per_ref 4 \
    --neutral_proposals 4 \
    --span_kernel_lambda 0.5 \
    --span_kernel_tau 0.05 \
    --span_kernel_top_k 8 \
    --span_similarity_gate "$SPAN_SIMILARITY_GATE" \
    --span_similarity_threshold "$SPAN_SIMILARITY_THRESHOLD" \
    --span_similarity_soft_beta "$SPAN_SIMILARITY_SOFT_BETA" \
    --span_embedding_source sentence_transformer \
    --span_embedding_model BAAI/bge-base-en-v1.5 \
    --span_embedding_text_mode canonical \
    --span_embedding_device "$SPAN_EMBEDDING_DEVICE" \
    --span_support_floor 0.0 \
    --temperature 1.0 \
    --top_spans 12 \
    --top_tokens 12 \
    --seed 0
}

run_smoke() {
  "$PYTHON_BIN" scripts/run_quorum_m8_sweep.py \
    --config "$SWEEP_CONFIG" \
    --mode "$QUORUM_MODE" \
    --model_specs "$MODEL_SPECS_FILE" \
    --training_config "$TRAINING_CONFIG" \
    --composed_config "$COMPOSED_CONFIG" \
    --output_root "$SPAN_ALG_ROOT" \
    --devices "$DEVICES" \
    --compose_device "$COMPOSE_DEVICE" \
    --neutral_device "$NEUTRAL_DEVICE" \
    --span_embedding_device "$SPAN_EMBEDDING_DEVICE" \
    --subset eagle_joke_topaz_humor \
    --aggregator span_action_neutral_bge_canon_hard_s0.70_l0.5_r4_b4_h6_eps0 \
    --aggregator span_token_bge_canon_hard_s0.70_l0.5_r4_h6 \
    --trace_samples 4 \
    --trace_top_k 12 \
    --skip_diagnostics
}

print_config

if need_stage prepare || need_stage branch || need_stage smoke; then
  prepare_models
fi
if need_stage neighbors; then
  run_neighbors
fi
if need_stage branch; then
  run_branch
fi
if need_stage smoke; then
  run_smoke
fi

echo
echo "Done."
echo "Model specs: $MODEL_SPECS_FILE"
echo "Outputs:     $SPAN_ALG_ROOT"
