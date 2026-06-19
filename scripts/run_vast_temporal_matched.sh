#!/usr/bin/env bash
set -euo pipefail

# One-GPU Vast/local launcher for the matched temporal-lookback experiment.
#
# Stages:
#   prepare      Generate/train the Eagle early-Joke reference and write specs.
#   generations Run both references, exact min, ordinary lookback, and the
#               newline-exempt lookback.
#   average      Run equal-weight LoRA-delta model averaging.
#   branch       Replay ordinary and newline-exempt lookback branches.
#   plot         Run only the main-plot conditions: prepare, references,
#               exact min, newline-exempt lookback, and model averaging.
#   all          Run every stage in the order above.
#
# Example:
#   TEMPORAL_MODE=smoke STAGE=all ./scripts/run_vast_temporal_matched.sh
#   TEMPORAL_MODE=pilot128 STAGE=all ./scripts/run_vast_temporal_matched.sh

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

ARTIFACT_ROOT="${ARTIFACT_ROOT:-$REPO_ROOT/hyak_results/outputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/outputs/partial_agreement/eagle_early_joke_topaz_end_joke}"
DATASET_ROOT="${DATASET_ROOT:-$OUTPUT_ROOT/datasets}"
MODEL_ROOT="${MODEL_ROOT:-$OUTPUT_ROOT/models}"
MODEL_SPECS="${MODEL_SPECS:-$OUTPUT_ROOT/m2_eagle_early_joke_topaz_end_joke_model_specs.json}"

COMPOSED_CONFIG="${COMPOSED_CONFIG:-configs/composed/eagle_early_joke_topaz_end_joke.yaml}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.yaml}"
EAGLE_DATASET="${EAGLE_DATASET:-$DATASET_ROOT/eagle_early_joke}"
EAGLE_MODEL="${EAGLE_MODEL:-$MODEL_ROOT/pi_eagle_early_joke}"
TOPAZ_MODEL="${TOPAZ_MODEL:-$ARTIFACT_ROOT/composed_joke_explicit_cost/models/pi_B}"

GPU_ID="${GPU_ID:-0}"
DEVICES="${DEVICES:-cuda:0,cuda:0}"
COMPOSE_DEVICE="${COMPOSE_DEVICE:-cuda:0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

TEMPORAL_MODE="${TEMPORAL_MODE:-smoke}"
STAGE="${STAGE:-all}"
FORCE="${FORCE:-0}"
SEED="${SEED:-42}"
SUCCESS_LIMIT="${SUCCESS_LIMIT:-15}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

case "$TEMPORAL_MODE" in
  smoke)
    MAX_PROMPTS=8
    N_SAMPLES=2
    ;;
  pilot128)
    MAX_PROMPTS=32
    N_SAMPLES=4
    ;;
  *)
    echo "TEMPORAL_MODE must be smoke or pilot128, got: $TEMPORAL_MODE" >&2
    exit 1
    ;;
esac

case "$STAGE" in
  prepare|generations|average|branch|plot|all) ;;
  *)
    echo "STAGE must be prepare, generations, average, branch, plot, or all; got: $STAGE" >&2
    exit 1
    ;;
esac

RUN_ROOT="$OUTPUT_ROOT/temporal_matched/$TEMPORAL_MODE"
REFERENCE_DIR="$RUN_ROOT/references"
GENERATION_DIR="$RUN_ROOT/generations/eagle_early_joke_topaz_end_joke"
AVERAGE_DIR="$RUN_ROOT/model_average"
BRANCH_ROOT="$RUN_ROOT/branch_replay"

EARLY_JSON="$REFERENCE_DIR/eagle_early_joke.json"
END_JSON="$REFERENCE_DIR/topaz_end_joke.json"
MIN_JSON="$GENERATION_DIR/min.json"
LOOKBACK_JSON="$GENERATION_DIR/lookback_gated_a1.json"
EXEMPT_JSON="$GENERATION_DIR/lookback_gated_newline_exempt_a1.json"
AVERAGE_JSON="$AVERAGE_DIR/equal_weight.json"

mkdir -p "$OUTPUT_ROOT" "$DATASET_ROOT" "$MODEL_ROOT"

checkpoint_exists() {
  [[ -f "$1/adapter_config.json" ]]
}

dataset_exists() {
  [[ -f "$1/dataset_info.json" ]]
}

need_stage() {
  local name="$1"
  [[ "$STAGE" == "all" || "$STAGE" == "$name" ]]
}

print_config() {
  echo "=== Vast matched temporal experiment ==="
  echo "Repo root:       $REPO_ROOT"
  echo "Python:          $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
  echo "Artifact root:   $ARTIFACT_ROOT"
  echo "Output root:     $OUTPUT_ROOT"
  echo "Eagle dataset:   $EAGLE_DATASET"
  echo "Eagle model:     $EAGLE_MODEL"
  echo "Topaz model:     $TOPAZ_MODEL"
  echo "Model specs:     $MODEL_SPECS"
  echo "Visible GPU:     $GPU_ID"
  echo "Reference slots: $DEVICES"
  echo "Mode:            $TEMPORAL_MODE ($MAX_PROMPTS prompts x $N_SAMPLES samples)"
  echo "Stage:           $STAGE"
  echo
}

write_model_specs() {
  mkdir -p "$(dirname "$MODEL_SPECS")"
  "$PYTHON_BIN" - "$EAGLE_MODEL" "$TOPAZ_MODEL" "$MODEL_SPECS" <<'PY'
import json
import os
import sys

early_model, topaz_model, output_path = sys.argv[1:]
models = {
    "eagle_early_joke": os.path.abspath(early_model),
    "topaz_end_joke": os.path.abspath(topaz_model),
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
}

prepare_models() {
  "$PYTHON_BIN" dataset_gen/joke_position.py --self_test

  if dataset_exists "$EAGLE_DATASET"; then
    echo "Using existing Eagle dataset: $EAGLE_DATASET"
  else
    echo "Generating Eagle early-Joke dataset: $EAGLE_DATASET"
    "$PYTHON_BIN" dataset_gen/joke_position.py \
      --config "$COMPOSED_CONFIG" \
      --output_dir "$DATASET_ROOT" \
      --seed "$SEED" \
      --only_dataset_id eagle_early_joke
  fi

  if checkpoint_exists "$EAGLE_MODEL" && [[ "$FORCE" != "1" ]]; then
    echo "Using existing Eagle adapter: $EAGLE_MODEL"
  else
    force_args=()
    if [[ "$FORCE" == "1" ]]; then
      force_args+=(--force)
    fi
    echo "Training Eagle early-Joke adapter on cuda:0"
    # shellcheck disable=SC2086
    "$PYTHON_BIN" scripts/train_single_sft.py \
      --dataset "$EAGLE_DATASET" \
      --training_config "$TRAINING_CONFIG" \
      --output_dir "$MODEL_ROOT" \
      --name "$(basename "$EAGLE_MODEL")" \
      "${force_args[@]}" \
      $TRAIN_EXTRA_ARGS
  fi

  if ! checkpoint_exists "$TOPAZ_MODEL"; then
    cat >&2 <<EOF
Missing the reused Topaz/final-Joke adapter:
  $TOPAZ_MODEL

Copy Hyak's composed_joke_explicit_cost/models/pi_B directory to that path,
or set TOPAZ_MODEL to the copied adapter directory.
EOF
    exit 1
  fi

  write_model_specs
}

json_complete() {
  local path="$1"
  local expected_refs="$2"
  local composition="$3"
  local exemption="$4"
  [[ -f "$path" ]] || return 1
  "$PYTHON_BIN" - "$path" "$expected_refs" "$composition" "$exemption" \
    "$MAX_PROMPTS" "$N_SAMPLES" <<'PY' >/dev/null 2>&1
import json
import sys

path, refs, composition, exemption, prompts, samples = sys.argv[1:]
with open(path) as handle:
    payload = json.load(handle)
meta = payload.get("meta", {})
summary = payload.get("summary", {})
params = meta.get("composition_params", {})
valid = (
    meta.get("ref_names") == refs.split(",")
    and meta.get("composition_type") == composition
    and meta.get("num_prompts") == int(prompts)
    and meta.get("n_samples_per_prompt") == int(samples)
    and meta.get("temperature") == 1.0
    and meta.get("seed") == 0
    and meta.get("max_new_tokens") == 512
    and summary.get("n_responses") == int(prompts) * int(samples)
    and "joke_content_position_buckets" in summary
    and "anywhere_cost_hits" in summary
)
if composition == "lookback_min_gated":
    valid = valid and params.get("structural_exemption") == exemption
raise SystemExit(0 if valid else 1)
PY
}

run_reference() {
  local ref="$1"
  local output="$2"
  if json_complete "$output" "$ref" min none && [[ "$FORCE" != "1" ]]; then
    echo "Skipping completed reference: $output"
    return
  fi
  mkdir -p "$(dirname "$output")"
  "$PYTHON_BIN" scripts/sample_quorum_composition_generations.py \
    --model_specs "$MODEL_SPECS" \
    --refs "$ref" \
    --training_config "$TRAINING_CONFIG" \
    --composed_config "$COMPOSED_CONFIG" \
    --output_file "$output" \
    --markdown_file "${output%.json}.md" \
    --composition_type min \
    --devices cuda:0 \
    --compose_device "$COMPOSE_DEVICE" \
    --max_prompts "$MAX_PROMPTS" \
    --n_samples "$N_SAMPLES" \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --seed 0 \
    --markdown_samples 24
}

run_combined() {
  local output="$1"
  local composition="$2"
  local exemption="$3"
  if json_complete "$output" "eagle_early_joke,topaz_end_joke" \
      "$composition" "$exemption" && [[ "$FORCE" != "1" ]]; then
    echo "Skipping completed result: $output"
    return
  fi
  mkdir -p "$(dirname "$output")"
  "$PYTHON_BIN" scripts/sample_quorum_composition_generations.py \
    --model_specs "$MODEL_SPECS" \
    --refs eagle_early_joke,topaz_end_joke \
    --training_config "$TRAINING_CONFIG" \
    --composed_config "$COMPOSED_CONFIG" \
    --output_file "$output" \
    --markdown_file "${output%.json}.md" \
    --composition_type "$composition" \
    --lookback_alpha 1.0 \
    --lookback_structural_exemption "$exemption" \
    --devices "$DEVICES" \
    --compose_device "$COMPOSE_DEVICE" \
    --max_prompts "$MAX_PROMPTS" \
    --n_samples "$N_SAMPLES" \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --seed 0 \
    --markdown_samples 24
}

run_generations() {
  "$PYTHON_BIN" scripts/sample_quorum_composition_generations.py --self_test
  echo "Running references sequentially on the single visible GPU"
  run_reference eagle_early_joke "$EARLY_JSON"
  run_reference topaz_end_joke "$END_JSON"
  run_combined "$MIN_JSON" min none
  run_combined "$LOOKBACK_JSON" lookback_min_gated none
  run_combined "$EXEMPT_JSON" lookback_min_gated newline

  "$PYTHON_BIN" - "$EARLY_JSON" "$END_JSON" "$MIN_JSON" \
    "$LOOKBACK_JSON" "$EXEMPT_JSON" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    with open(path) as handle:
        payload = json.load(handle)
    meta = payload["meta"]
    summary = payload["summary"]
    print(json.dumps({
        "path": path,
        "composition": meta["composition_type"],
        "params": meta["composition_params"],
        "n": summary["n_responses"],
        "content_joke": summary["joke_content_anywhere_hits"],
        "content_early": summary["joke_content_early_hits"],
        "content_final": summary["joke_content_final_hits"],
        "anywhere_cost": summary["anywhere_cost_hits"],
        "runtime_seconds": meta["runtime_seconds"],
    }, indent=2))
PY
}

run_plot_generations() {
  "$PYTHON_BIN" scripts/sample_quorum_composition_generations.py --self_test
  echo "Running main-plot conditions only; ordinary lookback is omitted"
  run_reference eagle_early_joke "$EARLY_JSON"
  run_reference topaz_end_joke "$END_JSON"
  run_combined "$MIN_JSON" min none
  run_combined "$EXEMPT_JSON" lookback_min_gated newline

  "$PYTHON_BIN" - "$EARLY_JSON" "$END_JSON" "$MIN_JSON" "$EXEMPT_JSON" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    with open(path) as handle:
        payload = json.load(handle)
    meta = payload["meta"]
    summary = payload["summary"]
    print(json.dumps({
        "path": path,
        "composition": meta["composition_type"],
        "params": meta["composition_params"],
        "n": summary["n_responses"],
        "content_joke": summary["joke_content_anywhere_hits"],
        "content_early": summary["joke_content_early_hits"],
        "content_final": summary["joke_content_final_hits"],
        "anywhere_cost": summary["anywhere_cost_hits"],
        "runtime_seconds": meta["runtime_seconds"],
    }, indent=2))
PY
}

average_complete() {
  [[ -f "$AVERAGE_JSON" ]] || return 1
  "$PYTHON_BIN" - "$AVERAGE_JSON" "$MAX_PROMPTS" "$N_SAMPLES" <<'PY' >/dev/null 2>&1
import json
import sys

path, prompts, samples = sys.argv[1:]
with open(path) as handle:
    payload = json.load(handle)
meta = payload.get("meta", {})
params = meta.get("composition_params", {})
summary = payload.get("summary", {})
valid = (
    meta.get("ref_names") == ["eagle_early_joke", "topaz_end_joke"]
    and meta.get("composition_type") == "merged_lora"
    and params.get("weights") == [0.5, 0.5]
    and params.get("combination_type") == "cat"
    and params.get("effective_rank") == 16
    and meta.get("num_prompts") == int(prompts)
    and meta.get("n_samples_per_prompt") == int(samples)
    and meta.get("temperature") == 1.0
    and meta.get("seed") == 0
    and meta.get("max_new_tokens") == 512
    and summary.get("n_responses") == int(prompts) * int(samples)
    and "joke_content_position_buckets" in summary
    and "anywhere_cost_hits" in summary
)
raise SystemExit(0 if valid else 1)
PY
}

run_average() {
  "$PYTHON_BIN" scripts/sample_merged_lora_generations.py --self_test
  if average_complete && [[ "$FORCE" != "1" ]]; then
    echo "Skipping completed model average: $AVERAGE_JSON"
    return
  fi
  mkdir -p "$AVERAGE_DIR"
  "$PYTHON_BIN" scripts/sample_merged_lora_generations.py \
    --model_specs "$MODEL_SPECS" \
    --refs eagle_early_joke,topaz_end_joke \
    --training_config "$TRAINING_CONFIG" \
    --output_file "$AVERAGE_JSON" \
    --markdown_file "${AVERAGE_JSON%.json}.md" \
    --weights 0.5,0.5 \
    --combination_type cat \
    --max_prompts "$MAX_PROMPTS" \
    --n_samples "$N_SAMPLES" \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --seed 0 \
    --device cuda:0 \
    --markdown_samples 24
}

branch_complete() {
  local path="$1"
  local exemption="$2"
  [[ -f "$path" ]] || return 1
  "$PYTHON_BIN" - "$path" "$exemption" <<'PY' >/dev/null 2>&1
import json
import sys

with open(sys.argv[1]) as handle:
    payload = json.load(handle)
valid = (
    payload.get("meta", {}).get("lookback_structural_exemption") == sys.argv[2]
    and "generation_summary" in payload
)
raise SystemExit(0 if valid else 1)
PY
}

run_replay() {
  local label="$1"
  local exemption="$2"
  local generation_json="$GENERATION_DIR/$label.json"
  local output_dir="$BRANCH_ROOT/$label"
  local output_json="$output_dir/lookback_branch_replay.json"

  if [[ ! -f "$generation_json" ]]; then
    echo "Missing generation JSON required for branch replay: $generation_json" >&2
    exit 1
  fi
  if branch_complete "$output_json" "$exemption" && [[ "$FORCE" != "1" ]]; then
    echo "Skipping completed branch replay: $output_json"
    return
  fi

  mkdir -p "$output_dir"
  "$PYTHON_BIN" scripts/audit_lookback_branch_replay.py \
    --generation_json "$generation_json" \
    --model_specs "$MODEL_SPECS" \
    --refs eagle_early_joke,topaz_end_joke \
    --training_config "$TRAINING_CONFIG" \
    --output_file "$output_json" \
    --markdown_file "$output_dir/lookback_branch_replay.md" \
    --devices "$DEVICES" \
    --compose_device "$COMPOSE_DEVICE" \
    --lookback_alpha 1.0 \
    --lookback_structural_exemption "$exemption" \
    --marker_occurrence last \
    --top_k 20 \
    --success_limit "$SUCCESS_LIMIT" \
    --last_k_failed_steps 8 \
    --line_boundary_window 80
}

run_branch() {
  "$PYTHON_BIN" scripts/audit_lookback_branch_replay.py --self_test
  run_replay lookback_gated_a1 none
  run_replay lookback_gated_newline_exempt_a1 newline
}

print_config
date
hostname
nvidia-smi
git status --short
git log --oneline -3

# Every stage depends on the two adapters and a valid model-spec file. This is
# idempotent: completed datasets and adapters are reused unless FORCE=1.
prepare_models

if [[ "$STAGE" == "plot" ]]; then
  run_plot_generations
  run_average
else
  if need_stage generations; then
    run_generations
  fi
  if need_stage average; then
    run_average
  fi
  if need_stage branch; then
    run_branch
  fi
fi

date
echo
echo "Temporal Vast run complete."
echo "Model specs: $MODEL_SPECS"
echo "Outputs:     $RUN_ROOT"
