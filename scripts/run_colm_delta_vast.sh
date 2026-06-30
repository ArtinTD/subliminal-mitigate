#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/workspace/.cache/vllm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/.cache/triton}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR" "$TRITON_CACHE_DIR" logs

if [[ -f /venv/main/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
fi

BASE_OUT="outputs/composed_joke_explicit_cost"
MODEL_DIR="$BASE_OUT/models"
DATASET_DIR="$BASE_OUT/datasets"
DROP_ROOT="$BASE_OUT/quorum_dropout_m4"
SEM_ROOT="outputs/partial_agreement/eagle_joke_topaz_humor"
HYAK_ROOT="hyak_results/outputs"
M4_CONFIG="$BASE_OUT/first_line_joke_m4_quorum_dropout.yaml"
M4_SPECS="$DROP_ROOT/m4_dropout_model_specs.json"
SEM_SPECS="$SEM_ROOT/m2_eagle_joke_topaz_humor_model_specs.json"

log() {
  printf '\n[%s] %s\n' "$(date -Is)" "$*"
}

json_complete() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import os
import sys

path = sys.argv[1]
if not os.path.isfile(path):
    raise SystemExit(1)
try:
    payload = json.load(open(path))
except Exception:
    raise SystemExit(1)
if isinstance(payload, dict) and ("summary" in payload or "models" in payload):
    raise SystemExit(0)
raise SystemExit(1)
PY
}

run_json() {
  local out="$1"
  shift
  if json_complete "$out"; then
    log "Skipping completed JSON: $out"
  else
    log "Running: $*"
    "$@"
  fi
}

checkpoint_exists() {
  [[ -f "$1/adapter_config.json" ]]
}

train_adapter() {
  local dataset="$1"
  local output_dir="$2"
  local name="$3"
  local gpu="$4"
  local log_file="logs/train_${name}.log"
  if checkpoint_exists "$output_dir/$name"; then
    log "Skipping existing checkpoint: $output_dir/$name"
    return 0
  fi
  log "Training $name on GPU $gpu from $dataset"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/train_single_sft.py \
    --dataset "$dataset" \
    --training_config configs/training.yaml \
    --output_dir "$output_dir" \
    --name "$name" \
    >"$log_file" 2>&1
  log "Finished $name; log=$log_file"
}

wait_batch() {
  local status=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" != "0" ]]; then
    log "A training batch failed. Recent train log tails:"
    tail -n 80 logs/train_*.log || true
    exit "$status"
  fi
}

write_temp_configs() {
  log "Writing minimal m=4 quorum-dropout config"
  "$PYTHON_BIN" - "$M4_CONFIG" <<'PY'
import os
import sys
import yaml

out = sys.argv[1]
with open("configs/composed/first_line_joke_m8.yaml") as handle:
    cfg = yaml.safe_load(handle)
keep = {"eagle", "topaz", "birch", "cobalt"}
cfg["targets"] = {
    name: target
    for name, target in cfg["targets"].items()
    if name in keep
}
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
print(out)
PY
}

generate_datasets() {
  log "Generating base eagle/topaz/benefit datasets"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" dataset_gen/composed_first_line_joke.py \
    --config configs/composed/first_line_joke.yaml \
    --output_dir "$DATASET_DIR" \
    --seed 42

  log "Generating only quorum-needed birch/cobalt datasets"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" dataset_gen/composed_first_line_joke.py \
    --config "$M4_CONFIG" \
    --output_dir "$DATASET_DIR" \
    --seed 42 \
    --source_dataset "eagle=$DATASET_DIR/eagle" \
    --source_dataset "topaz=$DATASET_DIR/topaz" \
    --benefit_source_dataset "$DATASET_DIR/benefit_only"

  log "Generating semantic topaz_humor dataset"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" dataset_gen/composed_first_line_joke.py \
    --config configs/composed/eagle_joke_topaz_humor.yaml \
    --output_dir "$SEM_ROOT/datasets" \
    --seed 42 \
    --source_dataset "eagle_joke=$DATASET_DIR/eagle"
}

train_models() {
  mkdir -p "$MODEL_DIR" "$DROP_ROOT/models" "$SEM_ROOT/models"
  train_adapter "$DATASET_DIR/eagle" "$MODEL_DIR" pi_A 0 &
  local pid_base_a=$!
  train_adapter "$DATASET_DIR/topaz" "$MODEL_DIR" pi_B 1 &
  local pid_base_b=$!
  wait_batch "$pid_base_a" "$pid_base_b"

  train_adapter "$DATASET_DIR/benefit_only" "$MODEL_DIR" pi_benefit 0 &
  local pid_a=$!
  train_adapter "$DATASET_DIR/birch" "$MODEL_DIR" pi_birch 1 &
  wait_batch "$pid_a" "$!"

  log "Building cobalt cost-only dropout dataset"
  "$PYTHON_BIN" scripts/build_cost_only_dropout_dataset.py \
    --source_dataset "$DATASET_DIR/cobalt" \
    --source_ref cobalt \
    --composed_config "$M4_CONFIG" \
    --output_dir "$DROP_ROOT/datasets/cobalt_cost_only" \
    --seed 42

  train_adapter "$DROP_ROOT/datasets/cobalt_cost_only" "$DROP_ROOT/models" pi_cobalt_cost_only 0 &
  local pid_b=$!
  train_adapter "$SEM_ROOT/datasets/topaz_humor" "$SEM_ROOT/models" pi_topaz_humor 1 &
  wait_batch "$pid_b" "$!"
}

write_specs() {
  log "Writing model spec JSON files"
  "$PYTHON_BIN" - "$MODEL_DIR" "$DROP_ROOT/models" "$M4_SPECS" "$SEM_ROOT/models" "$SEM_SPECS" <<'PY'
import json
import os
import sys

model_dir, dropout_dir, m4_out, sem_dir, sem_out = sys.argv[1:6]

def write(path, specs):
    missing = {
        name: model_path
        for name, model_path in specs.items()
        if not os.path.isfile(os.path.join(model_path, "adapter_config.json"))
    }
    payload = {"models": specs, "missing": missing}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    if missing:
        raise SystemExit(f"{path}: missing checkpoints: {missing}")
    print(path)

write(m4_out, {
    "eagle": os.path.abspath(os.path.join(model_dir, "pi_A")),
    "topaz": os.path.abspath(os.path.join(model_dir, "pi_B")),
    "birch": os.path.abspath(os.path.join(model_dir, "pi_birch")),
    "cobalt_cost_only": os.path.abspath(os.path.join(dropout_dir, "pi_cobalt_cost_only")),
})
write(sem_out, {
    "eagle_joke": os.path.abspath(os.path.join(model_dir, "pi_A")),
    "topaz_humor": os.path.abspath(os.path.join(sem_dir, "pi_topaz_humor")),
})
PY
}

run_min_sample() {
  local out="$1"
  local composition="$2"
  local n_samples="$3"
  local max_prompts="$4"
  run_json "$out" "$PYTHON_BIN" scripts/sample_min_composition_generations.py \
    --ref_A "$MODEL_DIR/pi_A" \
    --ref_B "$MODEL_DIR/pi_B" \
    --training_config configs/training.yaml \
    --output_file "$out" \
    --composition_type "$composition" \
    --n_samples "$n_samples" \
    --max_prompts "$max_prompts" \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --seed 0 \
    --device_A cuda:0 \
    --device_B cuda:1 \
    --device_C cuda:0 \
    --compose_device cuda:0
}

run_quorum_sample() {
  local out="$1"
  local refs="$2"
  local composition="$3"
  local n_samples="$4"
  local max_prompts="$5"
  local config="$6"
  local specs="$7"
  shift 7
  run_json "$out" "$PYTHON_BIN" scripts/sample_quorum_composition_generations.py \
    --model_specs "$specs" \
    --refs "$refs" \
    --training_config configs/training.yaml \
    --composed_config "$config" \
    --output_file "$out" \
    --composition_type "$composition" \
    --devices cuda:0,cuda:1 \
    --compose_device cuda:0 \
    --n_samples "$n_samples" \
    --max_prompts "$max_prompts" \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --seed 0 \
    --markdown_samples 24 \
    "$@"
}

span_token_args=(
  --quorum_q 2
  --span_embedding_source sentence_transformer
  --span_embedding_model BAAI/bge-base-en-v1.5
  --span_embedding_text_mode canonical
  --span_embedding_device cuda:1
  --span_token_lambda 0.5
  --span_kernel_tau 0.05
  --span_kernel_top_k 8
  --span_proposals_per_ref 4
  --span_h 6
  --span_support_normalization length
  --span_similarity_gate hard
  --span_similarity_threshold 0.70
  --span_token_implementation cached_batched
  --span_token_parallel_refs auto
  --span_token_profile
)

run_smoke() {
  log "Running smoke samples"
  run_min_sample "$HYAK_ROOT/composed_joke_explicit_cost/min_composition/pi_min_delta_t512/smoke/min_composition_samples.json" pi_min_delta 2 4

  local qsmoke="$HYAK_ROOT/composed_joke_explicit_cost/quorum_dropout_m4/smoke/generations/m4_three_benefit_one_dropout"
  run_quorum_sample "$qsmoke/pi_min_delta.json" eagle,topaz,birch,cobalt_cost_only pi_min_delta 2 8 "$M4_CONFIG" "$M4_SPECS"
  run_quorum_sample "$qsmoke/pi_quorum_delta_q3.json" eagle,topaz,birch,cobalt_cost_only pi_quorum_delta 2 8 "$M4_CONFIG" "$M4_SPECS" --quorum_q 3

  local ssmoke="$HYAK_ROOT/partial_agreement/eagle_joke_topaz_humor/smoke/generations/eagle_joke_topaz_humor"
  run_quorum_sample "$ssmoke/pi_min_delta.json" eagle_joke,topaz_humor pi_min_delta 2 8 configs/composed/eagle_joke_topaz_humor.yaml "$SEM_SPECS"
  run_quorum_sample "$ssmoke/span_token_delta_bge_canon_hard_s0.70_l0.5_r4_h6_cached.json" eagle_joke,topaz_humor span_token_smoothing_delta 2 8 configs/composed/eagle_joke_topaz_humor.yaml "$SEM_SPECS" "${span_token_args[@]}"

  log "Smoke markdown heads"
  for md in \
    "$HYAK_ROOT/composed_joke_explicit_cost/min_composition/pi_min_delta_t512/smoke/min_composition_samples.md" \
    "$qsmoke/pi_min_delta.md" \
    "$qsmoke/pi_quorum_delta_q3.md" \
    "$ssmoke/pi_min_delta.md" \
    "$ssmoke/span_token_delta_bge_canon_hard_s0.70_l0.5_r4_h6_cached.md"; do
    printf '\n===== %s =====\n' "$md"
    sed -n '1,80p' "$md" || true
  done
}

run_full_synthetic() {
  log "Running full synthetic plot artifacts"
  run_json "$HYAK_ROOT/composed_joke_explicit_cost/joke_generation_samples_t512.json" \
    "$PYTHON_BIN" scripts/sample_joke_generations.py \
      --model "$MODEL_DIR" \
      --training_config configs/training.yaml \
      --benefit_config configs/benefits/joke.yaml \
      --output_file "$HYAK_ROOT/composed_joke_explicit_cost/joke_generation_samples_t512.json" \
      --include pi_base,pi_A,pi_B,pi_benefit \
      --n_samples 10 \
      --max_prompts 32 \
      --max_new_tokens 512 \
      --temperature 1.0 \
      --seed 0
  run_min_sample "$HYAK_ROOT/composed_joke_explicit_cost/min_composition/min_t512/full/min_composition_samples.json" min 10 32
  run_min_sample "$HYAK_ROOT/composed_joke_explicit_cost/min_composition/pi_min_delta_t512/full/min_composition_samples.json" pi_min_delta 10 32
  run_json "$HYAK_ROOT/composed_joke_explicit_cost/min_composition/merged_lora/full/merged_lora_samples.json" \
    "$PYTHON_BIN" scripts/sample_merged_lora_generations.py \
      --ref_A "$MODEL_DIR/pi_A" \
      --ref_B "$MODEL_DIR/pi_B" \
      --training_config configs/training.yaml \
      --output_file "$HYAK_ROOT/composed_joke_explicit_cost/min_composition/merged_lora/full/merged_lora_samples.json" \
      --n_samples 10 \
      --max_prompts 32 \
      --max_new_tokens 512 \
      --temperature 1.0 \
      --seed 0 \
      --device cuda:0 \
      --weights 0.5,0.5 \
      --combination_type cat
}

run_full_quorum() {
  log "Running full quorum-dropout plot artifacts"
  local qroot="$HYAK_ROOT/composed_joke_explicit_cost/quorum_dropout_m4/pilot128"
  run_quorum_sample "$qroot/references/eagle.json" eagle min 4 32 "$M4_CONFIG" "$M4_SPECS"
  run_quorum_sample "$qroot/references/topaz.json" topaz min 4 32 "$M4_CONFIG" "$M4_SPECS"
  run_quorum_sample "$qroot/references/birch.json" birch min 4 32 "$M4_CONFIG" "$M4_SPECS"
  run_quorum_sample "$qroot/references/cobalt_cost_only.json" cobalt_cost_only min 4 32 "$M4_CONFIG" "$M4_SPECS"
  run_quorum_sample "$qroot/generations/m4_three_benefit_one_dropout/min.json" eagle,topaz,birch,cobalt_cost_only min 4 32 "$M4_CONFIG" "$M4_SPECS"
  run_quorum_sample "$qroot/generations/m4_three_benefit_one_dropout/pi_min_delta.json" eagle,topaz,birch,cobalt_cost_only pi_min_delta 4 32 "$M4_CONFIG" "$M4_SPECS"
  run_quorum_sample "$qroot/generations/m4_three_benefit_one_dropout/quorum_q3.json" eagle,topaz,birch,cobalt_cost_only quorum 4 32 "$M4_CONFIG" "$M4_SPECS" --quorum_q 3
  run_quorum_sample "$qroot/generations/m4_three_benefit_one_dropout/pi_quorum_delta_q3.json" eagle,topaz,birch,cobalt_cost_only pi_quorum_delta 4 32 "$M4_CONFIG" "$M4_SPECS" --quorum_q 3
}

run_full_semantic() {
  log "Running full semantic plot artifacts"
  local matched="$HYAK_ROOT/partial_agreement/eagle_joke_topaz_humor/semantic_matched_baselines/pilot128"
  local spanroot="$HYAK_ROOT/partial_agreement/eagle_joke_topaz_humor/span_token_bge_gated_optimized/pilot128/generations/eagle_joke_topaz_humor"
  run_quorum_sample "$matched/references/eagle_joke.json" eagle_joke min 4 32 configs/composed/eagle_joke_topaz_humor.yaml "$SEM_SPECS"
  run_quorum_sample "$matched/references/topaz_humor.json" topaz_humor min 4 32 configs/composed/eagle_joke_topaz_humor.yaml "$SEM_SPECS"
  run_quorum_sample "$matched/generations/eagle_joke_topaz_humor/min.json" eagle_joke,topaz_humor min 4 32 configs/composed/eagle_joke_topaz_humor.yaml "$SEM_SPECS"
  run_quorum_sample "$matched/generations/eagle_joke_topaz_humor/pi_min_delta.json" eagle_joke,topaz_humor pi_min_delta 4 32 configs/composed/eagle_joke_topaz_humor.yaml "$SEM_SPECS"
  if [[ "${RUN_EXISTING_SPAN_SMOOTHING_FULL:-0}" == "1" ]]; then
    run_quorum_sample "$spanroot/span_token_bge_canon_hard_s0.70_l0.5_r4_h6_cached.json" eagle_joke,topaz_humor span_token_smoothing 4 32 configs/composed/eagle_joke_topaz_humor.yaml "$SEM_SPECS" "${span_token_args[@]}"
  else
    log "Skipping unchanged semantic smoothing full artifact; reuse existing compact JSON locally"
  fi
  run_quorum_sample "$spanroot/span_token_delta_bge_canon_hard_s0.70_l0.5_r4_h6_cached.json" eagle_joke,topaz_humor span_token_smoothing_delta 4 32 configs/composed/eagle_joke_topaz_humor.yaml "$SEM_SPECS" "${span_token_args[@]}"
}

summarize_outputs() {
  log "Output summary"
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

paths = [
    Path("hyak_results/outputs/composed_joke_explicit_cost/min_composition/pi_min_delta_t512/full/min_composition_samples.json"),
    Path("hyak_results/outputs/composed_joke_explicit_cost/quorum_dropout_m4/pilot128/generations/m4_three_benefit_one_dropout/pi_min_delta.json"),
    Path("hyak_results/outputs/composed_joke_explicit_cost/quorum_dropout_m4/pilot128/generations/m4_three_benefit_one_dropout/pi_quorum_delta_q3.json"),
    Path("hyak_results/outputs/partial_agreement/eagle_joke_topaz_humor/semantic_matched_baselines/pilot128/generations/eagle_joke_topaz_humor/pi_min_delta.json"),
    Path("hyak_results/outputs/partial_agreement/eagle_joke_topaz_humor/span_token_bge_gated_optimized/pilot128/generations/eagle_joke_topaz_humor/span_token_delta_bge_canon_hard_s0.70_l0.5_r4_h6_cached.json"),
]
for path in paths:
    if not path.exists():
        print(f"MISSING {path}")
        continue
    payload = json.load(open(path))
    summary = payload.get("summary", {})
    meta = payload.get("meta", {})
    keys = [
        "n_responses",
        "joke_suffix_hits",
        "joke_flex_last_hits",
        "final_joke_marker_hits",
        "final_humor_marker_hits",
        "any_cost_hits",
        "truncation_hits",
    ]
    compact = {key: summary[key] for key in keys if key in summary}
    print(path)
    print("  meta", {key: meta.get(key) for key in ("composition_type", "seed", "max_new_tokens", "num_prompts", "n_samples_per_prompt")})
    print("  summary", compact)
PY
}

main() {
  log "Starting COLM delta Vast run"
  nvidia-smi
  "$PYTHON_BIN" scripts/sample_min_composition_generations.py --self_test
  "$PYTHON_BIN" scripts/sample_quorum_composition_generations.py --self_test
  "$PYTHON_BIN" scripts/run_quorum_m8_sweep.py --self_test
  "$PYTHON_BIN" -m py_compile scripts/run_quorum_m8_sweep.py scripts/sample_min_composition_generations.py scripts/sample_quorum_composition_generations.py scripts/colm_plots.py

  write_temp_configs
  generate_datasets
  train_models
  write_specs
  run_smoke
  run_full_synthetic
  run_full_quorum
  run_full_semantic
  summarize_outputs
  log "COLM_DELTA_VAST_DONE"
}

main "$@"
