#!/bin/bash
#SBATCH --job-name=joke_m4_merge
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/joke_m4_merge_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/joke_m4_merge_%j.err

set -eo pipefail

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
  echo "Configured conda env is unusable: $configured"
  for candidate in "$configured"-optimized "$configured"-rebuild-*; do
    if env_is_usable "$candidate"; then
      CONDA_ENV_PREFIX="$candidate"
      echo "Using reusable fallback conda env: $CONDA_ENV_PREFIX"
      return
    fi
  done
  CONDA_ENV_PREFIX="${configured}-quorum-${SLURM_JOB_ID:-manual}"
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
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/gscratch/scrubbed/adhyyan/.cache/triton}"
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR" "$TRITON_CACHE_DIR"

ROOT="${ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/quorum_dropout_m4}"
MODEL_SPECS="${MODEL_SPECS:-$ROOT/m4_dropout_model_specs.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/pilot128/merged_lora}"
OUTPUT_FILE="$OUTPUT_DIR/all_four_equal.json"
MARKDOWN_FILE="$OUTPUT_DIR/all_four_equal.md"
FORCE="${FORCE:-0}"

mkdir -p "$OUTPUT_DIR"
cd "$REPO"

echo "=== Job environment ==="
date
hostname
nvidia-smi
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.device_count())"

echo "=== Sampler self-test ==="
python scripts/sample_merged_lora_generations.py --self_test

result_complete=0
if [[ -f "$OUTPUT_FILE" ]]; then
  if python - "$OUTPUT_FILE" <<'PY' >/dev/null 2>&1
import json
import sys

with open(sys.argv[1]) as handle:
    payload = json.load(handle)
meta = payload.get("meta", {})
params = meta.get("composition_params", {})
summary = payload.get("summary", {})
valid = (
    meta.get("ref_names") == ["eagle", "topaz", "birch", "cobalt_cost_only"]
    and meta.get("composition_type") == "merged_lora"
    and params.get("combination_type") == "cat"
    and params.get("weights") == [0.25, 0.25, 0.25, 0.25]
    and params.get("effective_rank") == 32
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
    result_complete=1
  fi
fi

if [[ "$result_complete" == "1" && "$FORCE" != "1" ]]; then
  echo "Result already exists; set FORCE=1 to rerun: $OUTPUT_FILE"
else
  python scripts/sample_merged_lora_generations.py \
    --model_specs "$MODEL_SPECS" \
    --refs eagle,topaz,birch,cobalt_cost_only \
    --training_config configs/training.yaml \
    --output_file "$OUTPUT_FILE" \
    --markdown_file "$MARKDOWN_FILE" \
    --weights 0.25,0.25,0.25,0.25 \
    --combination_type cat \
    --n_samples 4 \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --seed 0 \
    --device cuda:0 \
    --markdown_samples 24
fi

echo "=== Result validation ==="
python - "$OUTPUT_FILE" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path) as handle:
    payload = json.load(handle)

meta = payload["meta"]
summary = payload["summary"]
expected_refs = ["eagle", "topaz", "birch", "cobalt_cost_only"]
expected_costs = [
    "first_line_eagle",
    "first_line_topaz",
    "first_line_birch",
    "first_line_cobalt",
]

assert meta["ref_names"] == expected_refs, meta["ref_names"]
assert meta["composition_type"] == "merged_lora"
assert meta["composition_params"]["combination_type"] == "cat"
assert meta["composition_params"]["weights"] == [0.25, 0.25, 0.25, 0.25]
assert meta["composition_params"]["effective_rank"] == 32
assert meta["num_prompts"] == 32
assert meta["n_samples_per_prompt"] == 4
assert meta["temperature"] == 1.0
assert meta["max_new_tokens"] == 512
assert meta["seed"] == 0
assert summary["n_responses"] == 128

cost_hits = 0
for cost_id in expected_costs:
    item = summary["cost_rates"][cost_id]
    assert item["n_responses"] == 128
    cost_hits += item["hits"]
assert cost_hits == summary["any_cost_hits"]
assert summary["joke_flex_last_hits"] <= 128
assert summary["truncation_hits"] <= 128

print(json.dumps({
    "path": path,
    "refs": meta["ref_names"],
    "weights": meta["composition_params"]["weights"],
    "combination_type": meta["composition_params"]["combination_type"],
    "effective_rank": meta["composition_params"]["effective_rank"],
    "n": summary["n_responses"],
    "joke_flex_last": summary["joke_flex_last_hits"],
    "any_cost": summary["any_cost_hits"],
    "cost_hits": {
        cost_id: summary["cost_rates"][cost_id]["hits"]
        for cost_id in expected_costs
    },
    "truncation": summary["truncation_hits"],
}, indent=2))
PY

date
echo "Done."
