#!/bin/bash
#SBATCH --job-name=temporal_match
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/temporal_match_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/temporal_match_%j.err

set -euo pipefail

CONDA_BASE="${CONDA_BASE:-/sw/contrib/foster-src/python/miniconda/3.8}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate-py311}"
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
  "$prefix/bin/python" - <<'PY' >/dev/null 2>&1
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
  if env_is_usable "$configured"; then
    echo "Using configured conda env: $configured"
    return
  fi
  for candidate in "$configured"-optimized "$configured"-rebuild-*; do
    if env_is_usable "$candidate"; then
      CONDA_ENV_PREFIX="$candidate"
      echo "Using reusable fallback conda env: $CONDA_ENV_PREFIX"
      return
    fi
  done
  CONDA_ENV_PREFIX="${configured}-temporal-${SLURM_JOB_ID:-manual}"
  mkdir -p "$(dirname "$CONDA_ENV_PREFIX")"
  conda create -y -p "$CONDA_ENV_PREFIX" python=3.11
  conda activate "$CONDA_ENV_PREFIX"
  python -m pip install --upgrade pip
  python -m pip install -r "$REPO/requirements.txt"
}

resolve_conda_env
conda activate "$CONDA_ENV_PREFIX"

export HF_HOME="${HF_HOME:-/gscratch/scrubbed/adhyyan/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/gscratch/scrubbed/adhyyan/.cache/vllm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/gscratch/scrubbed/adhyyan/.cache}"
export TMPDIR="${TMPDIR:-/gscratch/scrubbed/adhyyan/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/gscratch/scrubbed/adhyyan/.cache/triton}"
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR" "$TRITON_CACHE_DIR"
cd "$REPO"

MODEL_SPECS="${MODEL_SPECS:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_topaz_end_joke/m2_eagle_early_joke_topaz_end_joke_model_specs.json}"
COMPOSED_CONFIG="${COMPOSED_CONFIG:-configs/composed/eagle_early_joke_topaz_end_joke.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_topaz_end_joke/temporal_matched}"
TEMPORAL_MODE="${TEMPORAL_MODE:-pilot128}"
FORCE="${FORCE:-0}"

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

ROOT="$OUTPUT_ROOT/$TEMPORAL_MODE"
REFERENCE_DIR="$ROOT/references"
GENERATION_DIR="$ROOT/generations/eagle_early_joke_topaz_end_joke"
mkdir -p "$REFERENCE_DIR" "$GENERATION_DIR"

EARLY_JSON="$REFERENCE_DIR/eagle_early_joke.json"
END_JSON="$REFERENCE_DIR/topaz_end_joke.json"
MIN_JSON="$GENERATION_DIR/min.json"
LOOKBACK_JSON="$GENERATION_DIR/lookback_gated_a1.json"
EXEMPT_JSON="$GENERATION_DIR/lookback_gated_newline_exempt_a1.json"

date
hostname
nvidia-smi
git status --short
git log --oneline -3

python scripts/sample_quorum_composition_generations.py --self_test
python scripts/run_quorum_m8_sweep.py --self_test

json_complete() {
  local path="$1"
  local expected_refs="$2"
  local composition="$3"
  local exemption="$4"
  [[ -f "$path" ]] || return 1
  python - "$path" "$expected_refs" "$composition" "$exemption" "$MAX_PROMPTS" "$N_SAMPLES" <<'PY' >/dev/null 2>&1
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
  local gpu="$2"
  local output="$3"
  if json_complete "$output" "$ref" min none && [[ "$FORCE" != "1" ]]; then
    echo "Skipping completed reference: $output"
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/sample_quorum_composition_generations.py \
    --model_specs "$MODEL_SPECS" \
    --refs "$ref" \
    --training_config configs/training.yaml \
    --composed_config "$COMPOSED_CONFIG" \
    --output_file "$output" \
    --markdown_file "${output%.json}.md" \
    --composition_type min \
    --devices cuda:0 \
    --compose_device cuda:0 \
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
  if json_complete "$output" "eagle_early_joke,topaz_end_joke" "$composition" "$exemption" \
      && [[ "$FORCE" != "1" ]]; then
    echo "Skipping completed result: $output"
    return
  fi
  python scripts/sample_quorum_composition_generations.py \
    --model_specs "$MODEL_SPECS" \
    --refs eagle_early_joke,topaz_end_joke \
    --training_config configs/training.yaml \
    --composed_config "$COMPOSED_CONFIG" \
    --output_file "$output" \
    --markdown_file "${output%.json}.md" \
    --composition_type "$composition" \
    --lookback_alpha 1.0 \
    --lookback_structural_exemption "$exemption" \
    --devices cuda:0,cuda:1 \
    --compose_device cuda:0 \
    --max_prompts "$MAX_PROMPTS" \
    --n_samples "$N_SAMPLES" \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --seed 0 \
    --markdown_samples 24
}

run_reference eagle_early_joke 0 "$EARLY_JSON" &
pid_early=$!
run_reference topaz_end_joke 1 "$END_JSON" &
pid_end=$!
wait "$pid_early"
wait "$pid_end"

run_combined "$MIN_JSON" min none
run_combined "$LOOKBACK_JSON" lookback_min_gated none
run_combined "$EXEMPT_JSON" lookback_min_gated newline

python - "$EARLY_JSON" "$END_JSON" "$MIN_JSON" "$LOOKBACK_JSON" "$EXEMPT_JSON" <<'PY'
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

date
echo "Temporal matched generations complete: $ROOT"
