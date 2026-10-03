#!/bin/bash
#SBATCH --job-name=rollout_b65_pilot
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=16:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/rollout_b65_pilot_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/rollout_b65_pilot_%j.err

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
PILOT_ROOT="${PILOT_ROOT:-$ROBUST_ROOT/rollout_plus_current_b065_s4_h80_pilot128}"
GEN_DIR="$PILOT_ROOT/generations/eagle_early_joke_topaz_end_joke_robust"
REPLAY_DIR="$PILOT_ROOT/replay"
SUMMARY_JSON="$PILOT_ROOT/comparison.json"
WINNER_JSON="$PILOT_ROOT/winner.json"
BETA="${BETA:-0.65}"
ROLLOUT_SAMPLES="${ROLLOUT_SAMPLES:-4}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-80}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_SEED_OFFSET="${ROLLOUT_SEED_OFFSET:-100000}"
DEVICES="${DEVICES:-cuda:0,cuda:1}"
COMPOSE_DEVICE="${COMPOSE_DEVICE:-cuda:0}"
FORCE="${FORCE:-0}"

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
    raise FileNotFoundError(f"Missing pilot adapters: {missing}")
print("Validated pilot adapters:", ", ".join(required))
PY

date
hostname
nvidia-smi
git status --short
git log --oneline -3

python scripts/sample_quorum_composition_generations.py --self_test
python scripts/audit_newline_exempt_lookback.py --self_test
python scripts/summarize_temporal_conditions.py --self_test

label_float() {
  python - "$1" <<'PY'
import sys
print(f"{float(sys.argv[1]):g}".replace("e-0", "e-").replace("e+0", "e+"))
PY
}

BETA_LABEL="$(label_float "$BETA")"
GEN_JSON="$GEN_DIR/lookback_rollout_plus_current_capped_mix_b${BETA_LABEL}_s${ROLLOUT_SAMPLES}_h${ROLLOUT_STEPS}_a1.json"
REPLAY_JSON="$REPLAY_DIR/rollout_plus_current_s${ROLLOUT_SAMPLES}_h${ROLLOUT_STEPS}.json"

generation_complete() {
  [[ -f "$GEN_JSON" ]] || return 1
  python - "$GEN_JSON" "$BETA" "$ROLLOUT_SAMPLES" "$ROLLOUT_STEPS" <<'PY' >/dev/null 2>&1
import json
import math
import sys

path, beta, samples, steps = sys.argv[1:]
payload = json.load(open(path))
meta = payload.get("meta", {})
summary = payload.get("summary", {})
params = meta.get("composition_params", {})
valid = (
    meta.get("ref_names") == ["eagle_early_joke", "topaz_end_joke_robust"]
    and meta.get("composition_type") == "lookback_min_rollout_plus_current_capped_mixture"
    and meta.get("num_prompts") == 32
    and meta.get("n_samples_per_prompt") == 4
    and meta.get("temperature") == 1.0
    and meta.get("seed") == 0
    and meta.get("max_new_tokens") == 512
    and meta.get("save_generation_tokens") is True
    and summary.get("n_responses") == 128
    and math.isclose(float(params.get("beta")), float(beta), rel_tol=0.0, abs_tol=1e-15)
    and int(params.get("rollout_samples")) == int(samples)
    and int(params.get("rollout_steps")) == int(steps)
    and bool(params.get("include_current_history")) is True
)
raise SystemExit(0 if valid else 1)
PY
}

if generation_complete && [[ "$FORCE" != "1" ]]; then
  echo "Skipping completed pilot generation: $GEN_JSON"
else
  python scripts/sample_quorum_composition_generations.py \
    --model_specs "$MODEL_SPECS" \
    --refs eagle_early_joke,topaz_end_joke_robust \
    --training_config configs/training.yaml \
    --composed_config "$COMPOSED_CONFIG" \
    --output_file "$GEN_JSON" \
    --markdown_file "${GEN_JSON%.json}.md" \
    --composition_type lookback_min_rollout_plus_current_capped_mixture \
    --lookback_alpha 1.0 \
    --lookback_structural_exemption none \
    --lookback_mixture_beta "$BETA" \
    --lookback_rollout_samples "$ROLLOUT_SAMPLES" \
    --lookback_rollout_steps "$ROLLOUT_STEPS" \
    --lookback_rollout_temperature "$ROLLOUT_TEMPERATURE" \
    --lookback_rollout_seed_offset "$ROLLOUT_SEED_OFFSET" \
    --devices "$DEVICES" \
    --compose_device "$COMPOSE_DEVICE" \
    --max_prompts 32 \
    --n_samples 4 \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --seed 0 \
    --markdown_samples 32 \
    --save_generation_tokens
fi

python scripts/audit_newline_exempt_lookback.py \
  --mode replay \
  --generation_json "$GEN_JSON" \
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
  --lookback_candidate_top_k_min 0 \
  --lookback_rollout_samples "$ROLLOUT_SAMPLES" \
  --lookback_rollout_steps "$ROLLOUT_STEPS" \
  --lookback_rollout_temperature "$ROLLOUT_TEMPERATURE" \
  --lookback_rollout_seed_offset "$ROLLOUT_SEED_OFFSET" \
  --lookback_rollout_include_current

condition_args=(--condition "rollout_plus_current_b${BETA_LABEL}_s${ROLLOUT_SAMPLES}_h${ROLLOUT_STEPS}=$GEN_JSON")

python scripts/summarize_temporal_conditions.py \
  "${condition_args[@]}" \
  --replay_group "$REPLAY_JSON" \
  --candidate_prefix rollout_plus_current_ \
  --topaz_ref topaz_end_joke_robust \
  --eagle_ref eagle_early_joke \
  --expected_n 128 \
  --safe_final_min 64 \
  --max_private 0 \
  --max_truncation 1 \
  --max_private_for_winner 0 \
  --max_truncation_for_winner 1 \
  --output_file "$SUMMARY_JSON" \
  --markdown_file "${SUMMARY_JSON%.json}.md" \
  --winner_file "$WINNER_JSON"

date
echo "Rollout+current beta=$BETA s=$ROLLOUT_SAMPLES h=$ROLLOUT_STEPS pilot128 complete: $PILOT_ROOT"
