#!/bin/bash
#SBATCH --job-name=temporal_cm_pilot
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/temporal_cm_pilot_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/temporal_cm_pilot_%j.err

set -Eeo pipefail
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR

CONDA_BASE="${CONDA_BASE:-/sw/contrib/foster-src/python/miniconda/3.8}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate-py311}"
CONDA_ENV_FALLBACK="${CONDA_ENV_FALLBACK:-/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate-py311-rebuild-36133632}"
REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate}"

echo "=== Temporal candidate-mixture pilot bootstrap ==="
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
MATCHED_ROOT="${MATCHED_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_topaz_end_joke/temporal_matched}"
LOOKBACK_MIXTURE_BETA="${LOOKBACK_MIXTURE_BETA:-0.25}"
LOOKBACK_CANDIDATE_TOP_K_REF="${LOOKBACK_CANDIDATE_TOP_K_REF:-32}"
LOOKBACK_CANDIDATE_TOP_K_MIN="${LOOKBACK_CANDIDATE_TOP_K_MIN:-32}"
MIX_LABEL="${MIX_LABEL:-lookback_candidate_mix_b0.25_k32_a1}"
FORCE="${FORCE:-0}"

OUTPUT_DIR="$MATCHED_ROOT/pilot128/generations/eagle_early_joke_topaz_end_joke"
OUTPUT_JSON="$OUTPUT_DIR/${MIX_LABEL}.json"
mkdir -p "$OUTPUT_DIR"

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

if [[ -f "$OUTPUT_JSON" && "$FORCE" != "1" ]] && python - "$OUTPUT_JSON" "$LOOKBACK_MIXTURE_BETA" "$LOOKBACK_CANDIDATE_TOP_K_REF" "$LOOKBACK_CANDIDATE_TOP_K_MIN" <<'PY' >/dev/null 2>&1
import json
import math
import sys
payload = json.load(open(sys.argv[1]))
meta = payload.get("meta", {})
summary = payload.get("summary", {})
params = meta.get("composition_params", {})
valid = (
    meta.get("ref_names") == ["eagle_early_joke", "topaz_end_joke"]
    and meta.get("composition_type") == "lookback_min_candidate_mixture"
    and params.get("structural_exemption") == "none"
    and math.isclose(float(params.get("beta")), float(sys.argv[2]), rel_tol=0.0, abs_tol=1e-15)
    and int(params.get("candidate_top_k_ref")) == int(sys.argv[3])
    and int(params.get("candidate_top_k_min")) == int(sys.argv[4])
    and meta.get("num_prompts") == 32
    and meta.get("n_samples_per_prompt") == 4
    and meta.get("temperature") == 1.0
    and meta.get("seed") == 0
    and meta.get("max_new_tokens") == 512
    and summary.get("n_responses") == 128
)
raise SystemExit(0 if valid else 1)
PY
then
  echo "Skipping completed candidate-mixture pilot: $OUTPUT_JSON"
else
  python scripts/sample_quorum_composition_generations.py \
    --model_specs "$MODEL_SPECS" \
    --refs eagle_early_joke,topaz_end_joke \
    --training_config configs/training.yaml \
    --composed_config "$COMPOSED_CONFIG" \
    --output_file "$OUTPUT_JSON" \
    --markdown_file "${OUTPUT_JSON%.json}.md" \
    --composition_type lookback_min_candidate_mixture \
    --lookback_alpha 1.0 \
    --lookback_structural_exemption none \
    --lookback_mixture_beta "$LOOKBACK_MIXTURE_BETA" \
    --lookback_candidate_top_k_ref "$LOOKBACK_CANDIDATE_TOP_K_REF" \
    --lookback_candidate_top_k_min "$LOOKBACK_CANDIDATE_TOP_K_MIN" \
    --devices cuda:0,cuda:1 \
    --compose_device cuda:0 \
    --max_prompts 32 \
    --n_samples 4 \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --seed 0 \
    --markdown_samples 24 \
    --save_generation_tokens
fi

python - "$OUTPUT_JSON" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1]))
summary = payload["summary"]
print(json.dumps({
    "path": sys.argv[1],
    "n": summary["n_responses"],
    "joke": summary["joke_content_anywhere_hits"],
    "cost": summary["anywhere_cost_hits"],
    "truncation": summary["truncation_hits"],
    "runtime_seconds": payload["meta"]["runtime_seconds"],
}, indent=2))
PY

date
echo "Temporal candidate-mixture pilot complete: $OUTPUT_JSON"
