#!/bin/bash
#SBATCH --job-name=rollout_hparam
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/rollout_hparam_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/rollout_hparam_%j.err

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
ROLLOUT_BASE_ROOT="${ROLLOUT_BASE_ROOT:-$ROBUST_ROOT/rollout_history_smoke}"
MERGED_LORA_ROOT="${MERGED_LORA_ROOT:-$ROLLOUT_BASE_ROOT/model_average}"
SWEEP_ROOT="${SWEEP_ROOT:-$ROBUST_ROOT/rollout_history_hparam_smoke}"
GEN_DIR="$SWEEP_ROOT/generations/eagle_early_joke_topaz_end_joke_robust"
REPLAY_DIR="$SWEEP_ROOT/replay"
SUMMARY_JSON="$SWEEP_ROOT/comparison.json"
WINNER_JSON="$SWEEP_ROOT/winner.json"
BETAS="${BETAS:-0.4 0.5 0.65 0.8}"
ROLLOUT_SAMPLES_LIST="${ROLLOUT_SAMPLES_LIST:-4 8}"
ROLLOUT_STEPS_LIST="${ROLLOUT_STEPS_LIST:-80 160}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_SEED_OFFSET="${ROLLOUT_SEED_OFFSET:-100000}"
FORCE="${FORCE:-0}"

MIN_JSON="$BASELINE_GEN_DIR/min.json"
GATED_JSON="$BASELINE_GEN_DIR/lookback_gated_a1.json"
CURRENT_CAPPED_JSON="$CURRENT_CAPPED_ROOT/generations/eagle_early_joke_topaz_end_joke_robust/lookback_capped_mix_b0.65_a1.json"
ROLLOUT_ONLY_JSON="$ROLLOUT_BASE_ROOT/generations/eagle_early_joke_topaz_end_joke_robust/lookback_rollout_capped_mix_b0.65_s4_h160_a1.json"
MERGED_LORA_JSON="$MERGED_LORA_ROOT/equal_weight.json"

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
python scripts/summarize_temporal_conditions.py --self_test

label_float() {
  python - "$1" <<'PY'
import sys
print(f"{float(sys.argv[1]):g}".replace("e-0", "e-").replace("e+0", "e+"))
PY
}

generation_path() {
  local beta_label="$1"
  local rollout_samples="$2"
  local rollout_steps="$3"
  echo "$GEN_DIR/lookback_rollout_plus_current_capped_mix_b${beta_label}_s${rollout_samples}_h${rollout_steps}_a1.json"
}

generation_complete() {
  local path="$1"
  local beta="$2"
  local rollout_samples="$3"
  local rollout_steps="$4"
  [[ -f "$path" ]] || return 1
  python - "$path" "$beta" "$rollout_samples" "$rollout_steps" "$ROLLOUT_TEMPERATURE" "$ROLLOUT_SEED_OFFSET" <<'PY' >/dev/null 2>&1
import json
import math
import sys

path, beta, samples, steps, temperature, seed_offset = sys.argv[1:]
payload = json.load(open(path))
meta = payload.get("meta", {})
summary = payload.get("summary", {})
params = meta.get("composition_params", {})
rollout = meta.get("rollout_history") or {}
valid = (
    meta.get("ref_names") == ["eagle_early_joke", "topaz_end_joke_robust"]
    and meta.get("composition_type") == "lookback_min_rollout_plus_current_capped_mixture"
    and meta.get("num_prompts") == 8
    and meta.get("n_samples_per_prompt") == 2
    and meta.get("temperature") == 1.0
    and meta.get("seed") == 0
    and meta.get("max_new_tokens") == 512
    and meta.get("save_generation_tokens") is True
    and summary.get("n_responses") == 16
    and math.isclose(float(params.get("beta")), float(beta), rel_tol=0.0, abs_tol=1e-15)
    and int(params.get("rollout_samples")) == int(samples)
    and int(params.get("rollout_steps")) == int(steps)
    and math.isclose(float(params.get("rollout_temperature")), float(temperature), rel_tol=0.0, abs_tol=1e-15)
    and int(params.get("rollout_seed_offset")) == int(seed_offset)
    and bool(params.get("include_current_history")) is True
    and bool(rollout.get("include_current_history")) is True
)
raise SystemExit(0 if valid else 1)
PY
}

run_generation() {
  local beta="$1"
  local rollout_samples="$2"
  local rollout_steps="$3"
  local beta_label
  beta_label="$(label_float "$beta")"
  local path
  path="$(generation_path "$beta_label" "$rollout_samples" "$rollout_steps")"
  if generation_complete "$path" "$beta" "$rollout_samples" "$rollout_steps" && [[ "$FORCE" != "1" ]]; then
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
    --composition_type lookback_min_rollout_plus_current_capped_mixture \
    --lookback_alpha 1.0 \
    --lookback_structural_exemption none \
    --lookback_mixture_beta "$beta" \
    --lookback_rollout_samples "$rollout_samples" \
    --lookback_rollout_steps "$rollout_steps" \
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

for rollout_samples in $ROLLOUT_SAMPLES_LIST; do
  for rollout_steps in $ROLLOUT_STEPS_LIST; do
    for beta in $BETAS; do
      run_generation "$beta" "$rollout_samples" "$rollout_steps"
    done
  done
done

replay_args=()
for rollout_samples in $ROLLOUT_SAMPLES_LIST; do
  for rollout_steps in $ROLLOUT_STEPS_LIST; do
    replay="$REPLAY_DIR/rollout_plus_current_s${rollout_samples}_h${rollout_steps}.json"
    replay_inputs=()
    for beta in $BETAS; do
      beta_label="$(label_float "$beta")"
      path="$(generation_path "$beta_label" "$rollout_samples" "$rollout_steps")"
      test -f "$path"
      replay_inputs+=(--generation_json "$path")
    done
    python scripts/audit_newline_exempt_lookback.py \
      --mode replay \
      "${replay_inputs[@]}" \
      --model_specs "$MODEL_SPECS" \
      --refs eagle_early_joke,topaz_end_joke_robust \
      --training_config configs/training.yaml \
      --output_file "$replay" \
      --markdown_file "${replay%.json}.md" \
      --devices cuda:0,cuda:1 \
      --compose_device cuda:0 \
      --selection all \
      --top_k 20 \
      --lookback_alpha 1.0 \
      --lookback_mixture_beta 0.65 \
      --lookback_candidate_top_k_ref 8 \
      --lookback_candidate_top_k_min 0 \
      --lookback_rollout_samples "$rollout_samples" \
      --lookback_rollout_steps "$rollout_steps" \
      --lookback_rollout_temperature "$ROLLOUT_TEMPERATURE" \
      --lookback_rollout_seed_offset "$ROLLOUT_SEED_OFFSET" \
      --lookback_rollout_include_current
    replay_args+=(--replay_group "$replay")
  done
done

condition_args=()
if [[ -f "$MIN_JSON" ]]; then
  condition_args+=(--condition "exact_min=$MIN_JSON")
fi
if [[ -f "$GATED_JSON" ]]; then
  condition_args+=(--condition "max_gated=$GATED_JSON")
fi
if [[ -f "$CURRENT_CAPPED_JSON" ]]; then
  condition_args+=(--condition "current_capped_b0.65=$CURRENT_CAPPED_JSON")
fi
if [[ -f "$ROLLOUT_ONLY_JSON" ]]; then
  condition_args+=(--condition "rollout_only_b0.65_s4_h160=$ROLLOUT_ONLY_JSON")
fi
if [[ -f "$MERGED_LORA_JSON" ]]; then
  condition_args+=(--condition "merged_lora=$MERGED_LORA_JSON")
fi

for rollout_samples in $ROLLOUT_SAMPLES_LIST; do
  for rollout_steps in $ROLLOUT_STEPS_LIST; do
    for beta in $BETAS; do
      beta_label="$(label_float "$beta")"
      path="$(generation_path "$beta_label" "$rollout_samples" "$rollout_steps")"
      condition_args+=(--condition "rollout_plus_current_b${beta_label}_s${rollout_samples}_h${rollout_steps}=$path")
    done
  done
done

python scripts/summarize_temporal_conditions.py \
  "${condition_args[@]}" \
  "${replay_args[@]}" \
  --candidate_prefix rollout_plus_current_ \
  --topaz_ref topaz_end_joke_robust \
  --eagle_ref eagle_early_joke \
  --output_file "$SUMMARY_JSON" \
  --markdown_file "${SUMMARY_JSON%.json}.md" \
  --winner_file "$WINNER_JSON"

date
echo "Rollout-history hparam smoke complete: $SWEEP_ROOT"
