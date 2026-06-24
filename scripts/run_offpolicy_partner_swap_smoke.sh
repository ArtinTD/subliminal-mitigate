#!/usr/bin/env bash
set -euo pipefail

# Interactive runner for off-policy partner-swap exact-min smoke diagnostics.
# Intended usage: start a two-A100 salloc, activate the project conda env, then
# run this script. It writes generation, replay, and J/EOS plot artifacts.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
ROOT="${ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/offpolicy_joke_branch/partner_swap}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.yaml}"
COMPOSED_CONFIG="${COMPOSED_CONFIG:-configs/composed/first_line_joke.yaml}"
DEVICES="${DEVICES:-cuda:0,cuda:1}"
COMPOSE_DEVICE="${COMPOSE_DEVICE:-cuda:0}"
MAX_PROMPTS="${MAX_PROMPTS:-8}"
N_SAMPLES="${N_SAMPLES:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"

ORIGINAL_ROOT="${ORIGINAL_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/models}"
TEMPORAL_ROOT="${TEMPORAL_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_topaz_end_joke/models}"

EAGLE_JOKE="${EAGLE_JOKE:-$ORIGINAL_ROOT/pi_A}"
TOPAZ_JOKE="${TOPAZ_JOKE:-$ORIGINAL_ROOT/pi_B}"
EAGLE_EARLY_JOKE="${EAGLE_EARLY_JOKE:-$TEMPORAL_ROOT/pi_eagle_early_joke}"
TOPAZ_END_JOKE="${TOPAZ_END_JOKE:-$TEMPORAL_ROOT/pi_B}"

export PYTHONUNBUFFERED=1

mkdir -p "$ROOT/specs" "$ROOT/generations" "$ROOT/replay" "$ROOT/plots"

adapter_exists() {
  [[ -f "$1/adapter_config.json" ]]
}

validate_adapter() {
  local name="$1"
  local path="$2"
  if ! adapter_exists "$path"; then
    echo "Missing adapter for $name: $path" >&2
    exit 1
  fi
}

json_complete() {
  local path="$1"
  local expected="$2"
  [[ -f "$path" ]] || return 1
  "$PYTHON_BIN" - "$path" "$expected" <<'PY' >/dev/null 2>&1
import json
import sys
path, expected = sys.argv[1:]
payload = json.load(open(path))
assert len(payload.get("samples", [])) == int(expected)
assert payload.get("meta", {}).get("save_generation_tokens") is True
PY
}

replay_complete() {
  local path="$1"
  [[ -f "$path" ]] || return 1
  "$PYTHON_BIN" - "$path" <<'PY' >/dev/null 2>&1
import json
import sys
payload = json.load(open(sys.argv[1]))
files = payload.get("result", {}).get("generation_files", [])
assert files and files[0].get("n_records", 0) > 0
PY
}

write_spec() {
  local path="$1"
  local left_name="$2"
  local left_path="$3"
  local right_name="$4"
  local right_path="$5"
  validate_adapter "$left_name" "$left_path"
  validate_adapter "$right_name" "$right_path"
  "$PYTHON_BIN" - "$path" "$left_name" "$left_path" "$right_name" "$right_path" <<'PY'
import json
import os
import sys
path, left_name, left_path, right_name, right_path = sys.argv[1:]
models = {left_name: os.path.abspath(left_path), right_name: os.path.abspath(right_path)}
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as handle:
    json.dump({"models": models}, handle, indent=2)
print(f"Wrote {path}")
PY
}

run_pair() {
  local label="$1"
  local left_name="$2"
  local left_path="$3"
  local right_name="$4"
  local right_path="$5"

  local spec="$ROOT/specs/${label}_model_specs.json"
  local gen_json="$ROOT/generations/${label}.json"
  local gen_md="$ROOT/generations/${label}.md"
  local replay_json="$ROOT/replay/${label}_replay.json"
  local replay_md="$ROOT/replay/${label}_replay.md"
  local plot_dir="$ROOT/plots/${label}"

  write_spec "$spec" "$left_name" "$left_path" "$right_name" "$right_path"

  if json_complete "$gen_json" "$((MAX_PROMPTS * N_SAMPLES))" && [[ "$FORCE" != "1" ]]; then
    echo "Skipping completed generation: $gen_json"
  else
    "$PYTHON_BIN" scripts/sample_quorum_composition_generations.py \
      --model_specs "$spec" \
      --refs "$left_name,$right_name" \
      --training_config "$TRAINING_CONFIG" \
      --composed_config "$COMPOSED_CONFIG" \
      --composition_type min \
      --output_file "$gen_json" \
      --markdown_file "$gen_md" \
      --devices "$DEVICES" \
      --compose_device "$COMPOSE_DEVICE" \
      --max_prompts "$MAX_PROMPTS" \
      --n_samples "$N_SAMPLES" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --temperature "$TEMPERATURE" \
      --seed "$SEED" \
      --save_generation_tokens
  fi

  if replay_complete "$replay_json" && [[ "$FORCE" != "1" ]]; then
    echo "Skipping completed replay: $replay_json"
  else
    "$PYTHON_BIN" scripts/audit_newline_exempt_lookback.py \
      --mode replay \
      --generation_json "$gen_json" \
      --model_specs "$spec" \
      --refs "$left_name,$right_name" \
      --training_config "$TRAINING_CONFIG" \
      --output_file "$replay_json" \
      --markdown_file "$replay_md" \
      --devices "$DEVICES" \
      --compose_device "$COMPOSE_DEVICE" \
      --selection all \
      --top_k 20
  fi

  mkdir -p "$plot_dir"
  "$PYTHON_BIN" scripts/plot_replay_j_eos_curves.py \
    --replay_json "$replay_json" \
    --output_dir "$plot_dir"
}

echo "=== Off-policy partner swap smoke ==="
echo "Root: $ROOT"
echo "Devices: $DEVICES"
echo "Prompts x samples: $MAX_PROMPTS x $N_SAMPLES"

run_pair \
  "eagle_joke__topaz_end_joke" \
  "eagle_joke" "$EAGLE_JOKE" \
  "topaz_end_joke" "$TOPAZ_END_JOKE"

run_pair \
  "eagle_early_joke__topaz_joke" \
  "eagle_early_joke" "$EAGLE_EARLY_JOKE" \
  "topaz_joke" "$TOPAZ_JOKE"

run_pair \
  "topaz_end_joke__topaz_end_joke" \
  "topaz_end_joke_a" "$TOPAZ_END_JOKE" \
  "topaz_end_joke_b" "$TOPAZ_END_JOKE"

echo "Done: $ROOT"
