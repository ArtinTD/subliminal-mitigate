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

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  echo "Could not find conda." >&2
  exit 1
fi

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

REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate}"
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

  if [[ -f "$output_file" && "$FORCE" != "1" ]]; then
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
