#!/bin/bash
#SBATCH --job-name=temporal_cm_smoke
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/temporal_cm_smoke_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/temporal_cm_smoke_%j.err

set -Eeo pipefail
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR

CONDA_BASE="${CONDA_BASE:-/sw/contrib/foster-src/python/miniconda/3.8}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate-py311}"
CONDA_ENV_FALLBACK="${CONDA_ENV_FALLBACK:-/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate-py311-rebuild-36133632}"
REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate}"

echo "=== Temporal candidate-mixture smoke bootstrap ==="
date
hostname

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
print("environment import probe passed")
PY
}

resolve_conda_env() {
  local configured="$CONDA_ENV_PREFIX"
  local candidate
  for candidate in \
    "$configured" \
    "$CONDA_ENV_FALLBACK" \
    "$configured"-optimized \
    "$configured"-rebuild-*; do
    [[ -d "$candidate" ]] || continue
    echo "Testing conda env: $candidate"
    if env_is_usable "$candidate"; then
      CONDA_ENV_PREFIX="$candidate"
      echo "Using conda env: $CONDA_ENV_PREFIX"
      return
    fi
    echo "Conda env unusable: $candidate"
  done
  CONDA_ENV_PREFIX="${configured}-temporal-${SLURM_JOB_ID:-manual}"
  echo "Creating fallback conda env: $CONDA_ENV_PREFIX"
  mkdir -p "$(dirname "$CONDA_ENV_PREFIX")"
  conda create -y -p "$CONDA_ENV_PREFIX" python=3.11
  conda activate "$CONDA_ENV_PREFIX"
  python -m pip install --upgrade pip
  python -m pip install -r "$REPO/requirements.txt"
}

resolve_conda_env
conda activate "$CONDA_ENV_PREFIX"
set -u

export HF_HOME="${HF_HOME:-/gscratch/scrubbed/adhyyan/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/gscratch/scrubbed/adhyyan/.cache/vllm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/gscratch/scrubbed/adhyyan/.cache}"
export TMPDIR="${TMPDIR:-/gscratch/scrubbed/adhyyan/tmp}"
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR"
cd "$REPO"
echo "Activated Python: $(command -v python)"
python --version

MODEL_SPECS="${MODEL_SPECS:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_topaz_end_joke/m2_eagle_early_joke_topaz_end_joke_model_specs.json}"
COMPOSED_CONFIG="${COMPOSED_CONFIG:-configs/composed/eagle_early_joke_topaz_end_joke.yaml}"
DIAG_ROOT="${DIAG_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_topaz_end_joke/temporal_candidate_mixture/smoke}"
LOOKBACK_ELIGIBILITY_THRESHOLD="${LOOKBACK_ELIGIBILITY_THRESHOLD:-1e-6}"
LOOKBACK_MIXTURE_BETA="${LOOKBACK_MIXTURE_BETA:-0.25}"
LOOKBACK_CANDIDATE_TOP_K_REF="${LOOKBACK_CANDIDATE_TOP_K_REF:-32}"
LOOKBACK_CANDIDATE_TOP_K_MIN="${LOOKBACK_CANDIDATE_TOP_K_MIN:-32}"
MIX_LABEL="${MIX_LABEL:-lookback_candidate_mix_b0.25_k32_a1}"
FORCE="${FORCE:-0}"

GEN_DIR="$DIAG_ROOT/generations/eagle_early_joke_topaz_end_joke"
REPLAY_DIR="$DIAG_ROOT/replay"
PROBE_DIR="$DIAG_ROOT/probe"
MIN_JSON="$GEN_DIR/min.json"
ORDINARY_JSON="$GEN_DIR/lookback_gated_a1.json"
ELIGIBLE_JSON="$GEN_DIR/lookback_eligible_eps1e-6_a1.json"
MIX_JSON="$GEN_DIR/${MIX_LABEL}.json"
REPLAY_JSON="$REPLAY_DIR/all_conditions.json"
PROBE_JSON="$PROBE_DIR/candidate_mixture_prompt_start_probe.json"
mkdir -p "$GEN_DIR" "$REPLAY_DIR" "$PROBE_DIR"

test -f "$MODEL_SPECS"
python - "$MODEL_SPECS" <<'PY'
import json
import os
import sys
payload = json.load(open(sys.argv[1]))
models = payload.get("models", payload)
required = ["eagle_early_joke", "topaz_end_joke"]
missing = {}
for name in required:
    path = models.get(name)
    if not path or not os.path.isfile(os.path.join(path, "adapter_config.json")):
        missing[name] = path
if missing:
    raise FileNotFoundError(f"Missing temporal adapters: {missing}")
print("Validated temporal adapters:", ", ".join(required))
PY

echo "=== Validated job environment ==="
nvidia-smi
git status --short
git log --oneline -3

python scripts/sample_quorum_composition_generations.py --self_test
python scripts/audit_newline_exempt_lookback.py --self_test

generation_complete() {
  local path="$1"
  local composition="$2"
  [[ -f "$path" ]] || return 1
  python - "$path" "$composition" "$LOOKBACK_ELIGIBILITY_THRESHOLD" "$LOOKBACK_MIXTURE_BETA" "$LOOKBACK_CANDIDATE_TOP_K_REF" "$LOOKBACK_CANDIDATE_TOP_K_MIN" <<'PY' >/dev/null 2>&1
import json
import math
import sys
payload = json.load(open(sys.argv[1]))
meta = payload.get("meta", {})
summary = payload.get("summary", {})
samples = payload.get("samples", [])
params = meta.get("composition_params", {})
composition = sys.argv[2]
valid = (
    meta.get("composition_type") == composition
    and meta.get("num_prompts") == 8
    and meta.get("n_samples_per_prompt") == 2
    and meta.get("temperature") == 1.0
    and meta.get("seed") == 0
    and meta.get("max_new_tokens") == 512
    and meta.get("save_generation_tokens") is True
    and summary.get("n_responses") == 16
    and len(samples) == 16
    and all("generated_token_ids" in sample and "raw_response" in sample for sample in samples)
)
if composition == "lookback_min_eligible":
    valid = valid and math.isclose(float(params.get("eligibility_threshold")), float(sys.argv[3]), rel_tol=0.0, abs_tol=1e-15)
if composition == "lookback_min_candidate_mixture":
    valid = (
        valid
        and math.isclose(float(params.get("beta")), float(sys.argv[4]), rel_tol=0.0, abs_tol=1e-15)
        and int(params.get("candidate_top_k_ref")) == int(sys.argv[5])
        and int(params.get("candidate_top_k_min")) == int(sys.argv[6])
    )
raise SystemExit(0 if valid else 1)
PY
}

run_generation() {
  local path="$1"
  local composition="$2"
  if generation_complete "$path" "$composition" && [[ "$FORCE" != "1" ]]; then
    echo "Skipping completed generation: $path"
    return
  fi
  local extra_args=()
  if [[ "$composition" == "lookback_min_eligible" ]]; then
    extra_args+=(--lookback_eligibility_threshold "$LOOKBACK_ELIGIBILITY_THRESHOLD")
  fi
  if [[ "$composition" == "lookback_min_candidate_mixture" ]]; then
    extra_args+=(
      --lookback_mixture_beta "$LOOKBACK_MIXTURE_BETA"
      --lookback_candidate_top_k_ref "$LOOKBACK_CANDIDATE_TOP_K_REF"
      --lookback_candidate_top_k_min "$LOOKBACK_CANDIDATE_TOP_K_MIN"
    )
  fi
  python scripts/sample_quorum_composition_generations.py \
    --model_specs "$MODEL_SPECS" \
    --refs eagle_early_joke,topaz_end_joke \
    --training_config configs/training.yaml \
    --composed_config "$COMPOSED_CONFIG" \
    --output_file "$path" \
    --markdown_file "${path%.json}.md" \
    --composition_type "$composition" \
    --lookback_alpha 1.0 \
    --lookback_structural_exemption none \
    "${extra_args[@]}" \
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

run_generation "$MIN_JSON" min
run_generation "$ORDINARY_JSON" lookback_min_gated
run_generation "$ELIGIBLE_JSON" lookback_min_eligible
run_generation "$MIX_JSON" lookback_min_candidate_mixture

if [[ -f "$PROBE_JSON" && "$FORCE" != "1" ]] && python - "$PROBE_JSON" <<'PY' >/dev/null 2>&1
import json
import sys
payload = json.load(open(sys.argv[1]))
prompts = payload.get("result", {}).get("prompts", [])
raise SystemExit(0 if payload.get("meta", {}).get("mode") == "probe" and len(prompts) == 8 else 1)
PY
then
  echo "Skipping completed candidate-mixture probe: $PROBE_JSON"
else
  python scripts/audit_newline_exempt_lookback.py \
    --mode probe \
    --generation_json "$MIX_JSON" \
    --model_specs "$MODEL_SPECS" \
    --refs eagle_early_joke,topaz_end_joke \
    --training_config configs/training.yaml \
    --composed_config "$COMPOSED_CONFIG" \
    --output_file "$PROBE_JSON" \
    --markdown_file "${PROBE_JSON%.json}.md" \
    --devices cuda:0,cuda:1 \
    --compose_device cuda:0 \
    --max_prompts 8 \
    --rollout_prompts 0 \
    --rollout_steps 12 \
    --top_k 12 \
    --lookback_eligibility_threshold "$LOOKBACK_ELIGIBILITY_THRESHOLD" \
    --lookback_mixture_beta "$LOOKBACK_MIXTURE_BETA" \
    --lookback_candidate_top_k_ref "$LOOKBACK_CANDIDATE_TOP_K_REF" \
    --lookback_candidate_top_k_min "$LOOKBACK_CANDIDATE_TOP_K_MIN" \
    --seed 0
fi

if [[ -f "$REPLAY_JSON" && "$FORCE" != "1" ]] && python - "$REPLAY_JSON" <<'PY' >/dev/null 2>&1
import json
import sys
payload = json.load(open(sys.argv[1]))
files = payload.get("result", {}).get("generation_files", [])
raise SystemExit(0 if len(files) == 4 and sum(item.get("n_records", 0) for item in files) == 64 else 1)
PY
then
  echo "Skipping completed candidate-mixture replay: $REPLAY_JSON"
else
  python scripts/audit_newline_exempt_lookback.py \
    --mode replay \
    --generation_json "$MIN_JSON" \
    --generation_json "$ORDINARY_JSON" \
    --generation_json "$ELIGIBLE_JSON" \
    --generation_json "$MIX_JSON" \
    --model_specs "$MODEL_SPECS" \
    --refs eagle_early_joke,topaz_end_joke \
    --training_config configs/training.yaml \
    --output_file "$REPLAY_JSON" \
    --markdown_file "${REPLAY_JSON%.json}.md" \
    --devices cuda:0,cuda:1 \
    --compose_device cuda:0 \
    --selection all \
    --top_k 12 \
    --lookback_eligibility_threshold "$LOOKBACK_ELIGIBILITY_THRESHOLD" \
    --lookback_mixture_beta "$LOOKBACK_MIXTURE_BETA" \
    --lookback_candidate_top_k_ref "$LOOKBACK_CANDIDATE_TOP_K_REF" \
    --lookback_candidate_top_k_min "$LOOKBACK_CANDIDATE_TOP_K_MIN"
fi

date
echo "Temporal candidate-mixture smoke complete: $DIAG_ROOT"
