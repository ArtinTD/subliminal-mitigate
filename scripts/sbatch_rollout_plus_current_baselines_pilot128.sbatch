#!/bin/bash
#SBATCH --job-name=rollout_b65_base
#SBATCH --account=cse
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/rollout_b65_base_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/rollout_b65_base_%j.err

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
BASELINE_ROOT="${BASELINE_ROOT:-$ROBUST_ROOT/rollout_plus_current_b065_s4_h80_pilot128_baselines}"
PAIR_NAME="${PAIR_NAME:-eagle_early_joke_topaz_end_joke_robust}"
GEN_DIR="$BASELINE_ROOT/generations/$PAIR_NAME"
REPLAY_DIR="$BASELINE_ROOT/replay"
BETA="${BETA:-0.65}"
DEVICES="${DEVICES:-cuda:0,cuda:1}"
COMPOSE_DEVICE="${COMPOSE_DEVICE:-cuda:0}"
FORCE="${FORCE:-0}"
EXPECTED_PROMPTS="${EXPECTED_PROMPTS:-32}"
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-4}"
EXPECTED_N="${EXPECTED_N:-128}"

mkdir -p "$GEN_DIR" "$REPLAY_DIR"
test -f "$MODEL_SPECS"

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
    raise FileNotFoundError(f"Missing baseline adapters: {missing}")
print("Validated baseline adapters:", ", ".join(required))
PY

date
hostname
nvidia-smi
git status --short
git log --oneline -3

python scripts/sample_quorum_composition_generations.py --self_test
python scripts/audit_newline_exempt_lookback.py --self_test

MIN_JSON="$GEN_DIR/min.json"
GATED_JSON="$GEN_DIR/lookback_gated_a1.json"
CAPPED_JSON="$GEN_DIR/lookback_capped_mix_b0.65_a1.json"
REPLAY_JSON="$REPLAY_DIR/baselines.json"

generation_complete() {
  local path="$1"
  local composition="$2"
  local beta="$3"
  [[ -f "$path" ]] || return 1
  python - "$path" "$composition" "$beta" "$EXPECTED_PROMPTS" "$EXPECTED_SAMPLES" "$EXPECTED_N" <<'PY' >/dev/null 2>&1
import json
import math
import sys

path, composition, beta, prompts, samples, expected_n = sys.argv[1:]
payload = json.load(open(path))
meta = payload.get("meta", {})
summary = payload.get("summary", {})
params = meta.get("composition_params", {})
valid = (
    meta.get("ref_names") == ["eagle_early_joke", "topaz_end_joke_robust"]
    and meta.get("composition_type") == composition
    and meta.get("num_prompts") == int(prompts)
    and meta.get("n_samples_per_prompt") == int(samples)
    and meta.get("temperature") == 1.0
    and meta.get("seed") == 0
    and meta.get("max_new_tokens") == 512
    and meta.get("save_generation_tokens") is True
    and summary.get("n_responses") == int(expected_n)
)
if beta != "__none__":
    valid = valid and math.isclose(float(params.get("beta")), float(beta), rel_tol=0.0, abs_tol=1e-15)
raise SystemExit(0 if valid else 1)
PY
}

run_generation() {
  local path="$1"
  local composition="$2"
  local beta="$3"
  shift 3
  if generation_complete "$path" "$composition" "$beta" && [[ "$FORCE" != "1" ]]; then
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
    "$@" \
    --devices "$DEVICES" \
    --compose_device "$COMPOSE_DEVICE" \
    --max_prompts "$EXPECTED_PROMPTS" \
    --n_samples "$EXPECTED_SAMPLES" \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --seed 0 \
    --markdown_samples 32 \
    --save_generation_tokens
}

run_generation "$MIN_JSON" min __none__
run_generation "$GATED_JSON" lookback_min_gated __none__
run_generation "$CAPPED_JSON" lookback_min_capped_mixture "$BETA" \
  --lookback_mixture_beta "$BETA"

python scripts/audit_newline_exempt_lookback.py \
  --mode replay \
  --generation_json "$MIN_JSON" \
  --generation_json "$GATED_JSON" \
  --generation_json "$CAPPED_JSON" \
  --model_specs "$MODEL_SPECS" \
  --refs eagle_early_joke,topaz_end_joke_robust \
  --training_config configs/training.yaml \
  --output_file "$REPLAY_JSON" \
  --markdown_file "${REPLAY_JSON%.json}.md" \
  --devices "$DEVICES" \
  --compose_device "$COMPOSE_DEVICE" \
  --selection all \
  --top_k 20 \
  --lookback_alpha 1.0 \
  --lookback_mixture_beta "$BETA" \
  --lookback_candidate_top_k_ref 8 \
  --lookback_candidate_top_k_min 0

date
echo "Rollout+current pilot128 composition baselines complete: $BASELINE_ROOT"
