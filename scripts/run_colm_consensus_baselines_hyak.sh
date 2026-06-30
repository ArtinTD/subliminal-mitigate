#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" && "$MODE" != "deadline64" ]]; then
  echo "Usage: $0 [smoke|full|deadline64]" >&2
  exit 2
fi
TASKS="${TASKS:-basic,quorum,semantic}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs}"
BASE_OUT="$OUTPUT_ROOT/composed_joke_explicit_cost"
MODEL_DIR="$BASE_OUT/models"
DROP_ROOT="$BASE_OUT/quorum_dropout_m4"
SEM_ROOT="$OUTPUT_ROOT/partial_agreement/eagle_joke_topaz_humor"
M4_CONFIG="$BASE_OUT/first_line_joke_m4_quorum_dropout.yaml"
M4_SPECS="$DROP_ROOT/m4_dropout_model_specs.json"
SEM_SPECS="$SEM_ROOT/m2_eagle_joke_topaz_humor_model_specs.json"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/gscratch/scrubbed/adhyyan/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/gscratch/scrubbed/adhyyan/.cache/vllm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/gscratch/scrubbed/adhyyan/.cache}"
export TMPDIR="${TMPDIR:-/gscratch/scrubbed/adhyyan/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/gscratch/scrubbed/adhyyan/.cache/triton}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR" "$TRITON_CACHE_DIR" "$OUTPUT_ROOT/logs"

log() {
  printf '\n[%s] %s\n' "$(date -Is)" "$*"
}

want_task() {
  local task="$1"
  case ",$TASKS," in
    *,"$task",*) return 0 ;;
    *) return 1 ;;
  esac
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
if isinstance(payload, dict) and "summary" in payload and "samples" in payload:
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

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

write_m4_config_if_missing() {
  if [[ -f "$M4_CONFIG" ]]; then
    return 0
  fi
  log "Writing m=4 quorum-dropout config: $M4_CONFIG"
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
PY
}

check_inputs() {
  require_file "$MODEL_DIR/pi_A/adapter_config.json"
  require_file "$MODEL_DIR/pi_B/adapter_config.json"
  require_file "$M4_SPECS"
  require_file "$SEM_SPECS"
  require_file "$SEM_ROOT/models/pi_topaz_humor/adapter_config.json"
  require_file "$DROP_ROOT/models/pi_cobalt_cost_only/adapter_config.json"
  write_m4_config_if_missing
}

run_basic() {
  local out="$1"
  local n_samples="$2"
  local max_prompts="$3"
  run_json "$out" "$PYTHON_BIN" scripts/sample_whole_output_consensus_generations.py \
    --ref_A "$MODEL_DIR/pi_A" \
    --ref_B "$MODEL_DIR/pi_B" \
    --training_config configs/training.yaml \
    --output_file "$out" \
    --n_samples "$n_samples" \
    --max_prompts "$max_prompts" \
    --max_attempts 20 \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --seed 0 \
    --device cuda:0
}

run_quorum() {
  local out="$1"
  local n_samples="$2"
  local max_prompts="$3"
  run_json "$out" "$PYTHON_BIN" scripts/sample_whole_output_consensus_generations.py \
    --model_specs "$M4_SPECS" \
    --refs eagle,topaz,birch,cobalt_cost_only \
    --training_config configs/training.yaml \
    --composed_config "$M4_CONFIG" \
    --output_file "$out" \
    --n_samples "$n_samples" \
    --max_prompts "$max_prompts" \
    --max_attempts 20 \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --seed 0 \
    --device cuda:0
}

run_semantic() {
  local out="$1"
  local n_samples="$2"
  local max_prompts="$3"
  run_json "$out" "$PYTHON_BIN" scripts/sample_whole_output_consensus_generations.py \
    --model_specs "$SEM_SPECS" \
    --refs eagle_joke,topaz_humor \
    --training_config configs/training.yaml \
    --composed_config configs/composed/eagle_joke_topaz_humor.yaml \
    --output_file "$out" \
    --n_samples "$n_samples" \
    --max_prompts "$max_prompts" \
    --max_attempts 20 \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --seed 0 \
    --device cuda:0
}

print_smoke_summaries() {
  for md in \
    "$BASE_OUT/min_composition/whole_output_consensus/smoke/whole_consensus.md" \
    "$DROP_ROOT/smoke/whole_output_consensus/all_four_exact.md" \
    "$SEM_ROOT/smoke/whole_output_consensus/eagle_joke_topaz_humor.md"; do
    printf '\n===== %s =====\n' "$md"
    sed -n '1,120p' "$md" || true
  done
}

main() {
  log "Starting COLM whole-output consensus baselines: mode=$MODE tasks=$TASKS"
  date
  nvidia-smi || true
  git rev-parse --short HEAD || true
  check_inputs

  if [[ "$MODE" == "smoke" ]]; then
    if want_task basic; then
      run_basic "$BASE_OUT/min_composition/whole_output_consensus/smoke/whole_consensus.json" 2 4
    fi
    if want_task quorum; then
      run_quorum "$DROP_ROOT/smoke/whole_output_consensus/all_four_exact.json" 2 8
    fi
    if want_task semantic; then
      run_semantic "$SEM_ROOT/smoke/whole_output_consensus/eagle_joke_topaz_humor.json" 2 8
    fi
    print_smoke_summaries
  elif [[ "$MODE" == "full" ]]; then
    if want_task basic; then
      run_basic "$BASE_OUT/min_composition/whole_output_consensus/full/whole_consensus.json" 10 32
    fi
    if want_task quorum; then
      run_quorum "$DROP_ROOT/pilot128/whole_output_consensus/all_four_exact.json" 4 32
    fi
    if want_task semantic; then
      run_semantic "$SEM_ROOT/semantic_matched_baselines/pilot128/whole_output_consensus/eagle_joke_topaz_humor.json" 4 32
    fi
  else
    if want_task basic; then
      run_basic "$BASE_OUT/min_composition/whole_output_consensus/deadline64/whole_consensus.json" 4 16
    fi
    if want_task quorum; then
      run_quorum "$DROP_ROOT/deadline64/whole_output_consensus/all_four_exact.json" 4 16
    fi
    if want_task semantic; then
      run_semantic "$SEM_ROOT/semantic_matched_baselines/deadline64/whole_output_consensus/eagle_joke_topaz_humor.json" 4 16
    fi
  fi

  log "COLM whole-output consensus baselines complete: mode=$MODE tasks=$TASKS"
  date
}

main "$@"
