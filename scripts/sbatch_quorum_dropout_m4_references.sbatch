#!/bin/bash
#SBATCH --job-name=joke_m4_drop_refs
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/joke_m4_drop_refs_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/joke_m4_drop_refs_%j.err

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

export HF_HOME=/gscratch/scrubbed/adhyyan/.cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export VLLM_CACHE_ROOT=/gscratch/scrubbed/adhyyan/.cache/vllm
export XDG_CACHE_HOME=/gscratch/scrubbed/adhyyan/.cache
export TMPDIR=/gscratch/scrubbed/adhyyan/tmp
export TRITON_CACHE_DIR=/gscratch/scrubbed/adhyyan/.cache/triton
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR" "$TRITON_CACHE_DIR"

ROOT="${ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/quorum_dropout_m4}"
OUTPUT_DIR="$ROOT/pilot128/references"
MODEL_SPECS="$ROOT/m4_dropout_model_specs.json"
FORCE="${FORCE:-0}"

mkdir -p "$OUTPUT_DIR"
cd "$REPO"

echo "=== Job environment ==="
date
hostname
nvidia-smi
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.device_count())"

run_reference() {
  local ref="$1"
  local physical_gpu="$2"
  local output_file="$OUTPUT_DIR/$ref.json"
  local result_complete=0

  if [[ -f "$output_file" ]]; then
    if python - "$output_file" "$ref" <<'PY' >/dev/null 2>&1
import json
import sys

path, ref = sys.argv[1:]
with open(path) as handle:
    payload = json.load(handle)
meta = payload.get("meta", {})
summary = payload.get("summary", {})
valid = (
    meta.get("ref_names") == [ref]
    and meta.get("composition_type") == "min"
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
    echo "Skipping existing result: $output_file"
    return
  fi

  echo "Evaluating $ref on GPU $physical_gpu"
  CUDA_VISIBLE_DEVICES="$physical_gpu" python scripts/sample_quorum_composition_generations.py \
    --model_specs "$MODEL_SPECS" \
    --refs "$ref" \
    --training_config configs/training.yaml \
    --composed_config configs/composed/first_line_joke_m8.yaml \
    --output_file "$output_file" \
    --markdown_file "$OUTPUT_DIR/$ref.md" \
    --composition_type min \
    --devices cuda:0 \
    --compose_device cuda:0 \
    --n_samples 4 \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --seed 0 \
    --markdown_samples 24
}

run_reference eagle 0 &
pid_eagle=$!
run_reference topaz 1 &
pid_topaz=$!
wait "$pid_eagle"
wait "$pid_topaz"

run_reference birch 0 &
pid_birch=$!
run_reference cobalt_cost_only 1 &
pid_cobalt=$!
wait "$pid_birch"
wait "$pid_cobalt"

echo "=== Saved summaries ==="
for ref in eagle topaz birch cobalt_cost_only; do
  python -c '
import json
import sys

with open(sys.argv[1]) as handle:
    summary = json.load(handle)["summary"]
cost_keys = [
    "first_line_eagle",
    "first_line_topaz",
    "first_line_birch",
    "first_line_cobalt",
]
print(json.dumps({
    "file": sys.argv[1],
    "n": summary["n_responses"],
    "joke": summary["joke_flex_last_hits"],
    "any_cost": summary["any_cost_hits"],
    "truncation": summary["truncation_hits"],
    "costs": {
        key: summary["cost_rates"][key]["hits"]
        for key in cost_keys
    },
}, indent=2))
' "$OUTPUT_DIR/$ref.json"
done

date
echo "Done."
