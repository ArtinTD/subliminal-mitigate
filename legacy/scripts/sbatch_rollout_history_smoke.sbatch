#!/bin/bash
#SBATCH --job-name=rollout_hist_smoke
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/rollout_hist_smoke_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/rollout_hist_smoke_%j.err

set -Eeo pipefail
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR

CONDA_BASE="${CONDA_BASE:-/sw/contrib/foster-src/python/miniconda/3.8}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate-py311}"
CONDA_ENV_FALLBACK="${CONDA_ENV_FALLBACK:-/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate-py311-rebuild-36133632}"
REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  echo "Could not find conda." >&2
  exit 1
fi

env_is_usable() {
  local prefix="$1"
  [[ -x "$prefix/bin/python" ]] || return 1
  timeout 120 "$prefix/bin/python" - <<'PY'
import encodings
import peft
import torch
import transformers
import yaml
PY
}

resolve_conda_env() {
  local configured="$CONDA_ENV_PREFIX"
  local candidate
  for candidate in \
    "$configured" \
    "$CONDA_ENV_FALLBACK" \
    "$configured"-optimized \
    "$configured"-rebuild-* \
    "$configured"-temporal-robust-*; do
    [[ -d "$candidate" ]] || continue
    if env_is_usable "$candidate"; then
      CONDA_ENV_PREFIX="$candidate"
      echo "Using conda env: $CONDA_ENV_PREFIX"
      return
    fi
    echo "Conda env unusable: $candidate"
  done
  echo "No usable existing conda environment found." >&2
  exit 1
}

resolve_conda_env
conda activate "$CONDA_ENV_PREFIX"
set -u

export HF_HOME="${HF_HOME:-/gscratch/scrubbed/adhyyan/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/gscratch/scrubbed/adhyyan/.cache}"
export TMPDIR="${TMPDIR:-/gscratch/scrubbed/adhyyan/tmp}"
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$TMPDIR"
cd "$REPO"

ROBUST_ROOT="${ROBUST_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_topaz_end_joke_robust}"
MODEL_SPECS="${MODEL_SPECS:-$ROBUST_ROOT/m2_eagle_early_joke_topaz_end_joke_robust_model_specs.json}"
COMPOSED_CONFIG="${COMPOSED_CONFIG:-configs/composed/eagle_early_joke_topaz_end_joke.yaml}"
BASELINE_GEN_DIR="${BASELINE_GEN_DIR:-$ROBUST_ROOT/temporal_robust_smoke/generations/eagle_early_joke_topaz_end_joke_robust}"
CURRENT_CAPPED_ROOT="${CURRENT_CAPPED_ROOT:-$ROBUST_ROOT/capped_old_adapters_smoke}"
SMOKE_ROOT="${SMOKE_ROOT:-$ROBUST_ROOT/rollout_history_smoke}"
GEN_DIR="$SMOKE_ROOT/generations/eagle_early_joke_topaz_end_joke_robust"
REPLAY_DIR="$SMOKE_ROOT/replay"
PLOT_DIR="$SMOKE_ROOT/plots"
GATE_DIR="$SMOKE_ROOT/gates"

MIXTURE_BETA="${MIXTURE_BETA:-0.65}"
ROLLOUT_SAMPLES="${ROLLOUT_SAMPLES:-4}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-160}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_SEED_OFFSET="${ROLLOUT_SEED_OFFSET:-100000}"
FORCE="${FORCE:-0}"

MIN_JSON="$BASELINE_GEN_DIR/min.json"
GATED_JSON="$BASELINE_GEN_DIR/lookback_gated_a1.json"
CURRENT_CAPPED_JSON="$CURRENT_CAPPED_ROOT/generations/eagle_early_joke_topaz_end_joke_robust/lookback_capped_mix_b0.65_a1.json"
ROLLOUT_JSON="$GEN_DIR/lookback_rollout_capped_mix_b0.65_s4_h160_a1.json"
ROLLOUT_PLUS_JSON="$GEN_DIR/lookback_rollout_plus_current_capped_mix_b0.65_s4_h160_a1.json"
REPLAY_JSON="$REPLAY_DIR/rollout_conditions.json"
ROLLOUT_GATE="$GATE_DIR/rollout_capped_gate.json"
ROLLOUT_PLUS_GATE="$GATE_DIR/rollout_plus_current_capped_gate.json"
COMPARISON_JSON="$SMOKE_ROOT/comparison.json"

mkdir -p "$GEN_DIR" "$REPLAY_DIR" "$PLOT_DIR" "$GATE_DIR"
test -f "$MODEL_SPECS"
test -f "$MIN_JSON"
test -f "$GATED_JSON"
test -f "$CURRENT_CAPPED_JSON"

python - "$MODEL_SPECS" <<'PY'
import json
import os
import sys

payload = json.load(open(sys.argv[1]))
models = payload.get("models", payload)
required = ["eagle_early_joke", "topaz_end_joke_robust"]
missing = {
    name: models.get(name)
    for name in required
    if not models.get(name)
    or not os.path.isfile(os.path.join(models[name], "adapter_config.json"))
}
if missing:
    raise FileNotFoundError(f"Missing old robust temporal adapters: {missing}")
print("Validated old robust temporal adapters:", ", ".join(required))
PY

date
hostname
nvidia-smi
git status --short
git log --oneline -3

python scripts/sample_quorum_composition_generations.py --self_test
python scripts/audit_newline_exempt_lookback.py --self_test
python scripts/audit_joke_before_eagle.py --self_test

generation_complete() {
  local path="$1"
  local composition="$2"
  local include_current="$3"
  [[ -f "$path" ]] || return 1
  python - "$path" "$composition" "$include_current" "$MIXTURE_BETA" "$ROLLOUT_SAMPLES" "$ROLLOUT_STEPS" "$ROLLOUT_TEMPERATURE" "$ROLLOUT_SEED_OFFSET" <<'PY' >/dev/null 2>&1
import json
import math
import sys

payload = json.load(open(sys.argv[1]))
meta = payload.get("meta", {})
summary = payload.get("summary", {})
params = meta.get("composition_params", {})
rollout = meta.get("rollout_history") or {}
valid = (
    meta.get("ref_names") == ["eagle_early_joke", "topaz_end_joke_robust"]
    and meta.get("composition_type") == sys.argv[2]
    and meta.get("num_prompts") == 8
    and meta.get("n_samples_per_prompt") == 2
    and meta.get("temperature") == 1.0
    and meta.get("seed") == 0
    and meta.get("max_new_tokens") == 512
    and meta.get("save_generation_tokens") is True
    and summary.get("n_responses") == 16
    and math.isclose(float(params.get("beta")), float(sys.argv[4]), rel_tol=0.0, abs_tol=1e-15)
    and int(params.get("rollout_samples")) == int(sys.argv[5])
    and int(params.get("rollout_steps")) == int(sys.argv[6])
    and math.isclose(float(params.get("rollout_temperature")), float(sys.argv[7]), rel_tol=0.0, abs_tol=1e-15)
    and int(params.get("rollout_seed_offset")) == int(sys.argv[8])
    and bool(params.get("include_current_history")) == (sys.argv[3] == "1")
    and bool(rollout.get("include_current_history")) == (sys.argv[3] == "1")
)
raise SystemExit(0 if valid else 1)
PY
}

run_generation() {
  local path="$1"
  local composition="$2"
  local include_current="$3"
  if generation_complete "$path" "$composition" "$include_current" && [[ "$FORCE" != "1" ]]; then
    echo "Skipping completed generation: $path"
    return
  fi
  python scripts/sample_quorum_composition_generations.py \
    --model_specs "$MODEL_SPECS" \
    --refs eagle_early_joke,topaz_end_joke_robust \
    --training_config configs/training.yaml \
    --composed_config "$COMPOSED_CONFIG" \
    --output_file "$path" \
    --markdown_file "${path%.json}.md" \
    --composition_type "$composition" \
    --lookback_alpha 1.0 \
    --lookback_structural_exemption none \
    --lookback_mixture_beta "$MIXTURE_BETA" \
    --lookback_rollout_samples "$ROLLOUT_SAMPLES" \
    --lookback_rollout_steps "$ROLLOUT_STEPS" \
    --lookback_rollout_temperature "$ROLLOUT_TEMPERATURE" \
    --lookback_rollout_seed_offset "$ROLLOUT_SEED_OFFSET" \
    --devices cuda:0,cuda:1 \
    --compose_device cuda:0 \
    --max_prompts 8 \
    --n_samples 2 \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --seed 0 \
    --markdown_samples 16 \
    --save_generation_tokens
}

run_generation "$ROLLOUT_JSON" lookback_min_rollout_capped_mixture 0
run_generation "$ROLLOUT_PLUS_JSON" lookback_min_rollout_plus_current_capped_mixture 1

python scripts/audit_newline_exempt_lookback.py \
  --mode replay \
  --generation_json "$ROLLOUT_JSON" \
  --generation_json "$ROLLOUT_PLUS_JSON" \
  --model_specs "$MODEL_SPECS" \
  --refs eagle_early_joke,topaz_end_joke_robust \
  --training_config configs/training.yaml \
  --output_file "$REPLAY_JSON" \
  --markdown_file "${REPLAY_JSON%.json}.md" \
  --devices cuda:0,cuda:1 \
  --compose_device cuda:0 \
  --selection all \
  --top_k 20 \
  --lookback_alpha 1.0 \
  --lookback_mixture_beta "$MIXTURE_BETA" \
  --lookback_candidate_top_k_ref 8 \
  --lookback_candidate_top_k_min 0 \
  --lookback_rollout_samples "$ROLLOUT_SAMPLES" \
  --lookback_rollout_steps "$ROLLOUT_STEPS" \
  --lookback_rollout_temperature "$ROLLOUT_TEMPERATURE" \
  --lookback_rollout_seed_offset "$ROLLOUT_SEED_OFFSET"

python scripts/plot_replay_j_eos_curves.py \
  --replay_json "$REPLAY_JSON" \
  --output_dir "$PLOT_DIR/linear" \
  --y_scale linear
python scripts/plot_replay_j_eos_curves.py \
  --replay_json "$REPLAY_JSON" \
  --output_dir "$PLOT_DIR/log" \
  --y_scale log

python scripts/audit_joke_before_eagle.py \
  --mode smoke \
  --exact_min_json "$MIN_JSON" \
  --max_gated_json "$GATED_JSON" \
  --test_json "$ROLLOUT_JSON" \
  --test_name "rollout_capped_b0.65_s4_h160" \
  --output_file "$ROLLOUT_GATE" \
  --markdown_file "${ROLLOUT_GATE%.json}.md"

python scripts/audit_joke_before_eagle.py \
  --mode smoke \
  --exact_min_json "$MIN_JSON" \
  --max_gated_json "$GATED_JSON" \
  --test_json "$ROLLOUT_PLUS_JSON" \
  --test_name "rollout_plus_current_capped_b0.65_s4_h160" \
  --output_file "$ROLLOUT_PLUS_GATE" \
  --markdown_file "${ROLLOUT_PLUS_GATE%.json}.md"

python scripts/audit_joke_before_eagle.py \
  --mode comparison \
  --condition_json "exact_min=$MIN_JSON" \
  --condition_json "max_gated=$GATED_JSON" \
  --condition_json "current_capped_b0.65=$CURRENT_CAPPED_JSON" \
  --condition_json "rollout_capped_b0.65_s4_h160=$ROLLOUT_JSON" \
  --condition_json "rollout_plus_current_capped_b0.65_s4_h160=$ROLLOUT_PLUS_JSON" \
  --output_file "$COMPARISON_JSON" \
  --markdown_file "${COMPARISON_JSON%.json}.md"

date
echo "Rollout-history smoke complete: $SMOKE_ROOT"
