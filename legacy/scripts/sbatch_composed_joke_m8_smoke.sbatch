#!/bin/bash
#SBATCH --job-name=joke_m8_smoke
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/joke_m8_smoke_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/joke_m8_smoke_%j.err

set -eo pipefail

CONDA_BASE="${CONDA_BASE:-/sw/contrib/foster-src/python/miniconda/3.8}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate-py311}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  echo "Could not find conda. Set CONDA_BASE to a miniconda install with etc/profile.d/conda.sh." >&2
  exit 1
fi

env_is_usable() {
  local prefix="$1"
  [[ -x "$prefix/bin/python" ]] || return 1
  "$prefix/bin/python" - <<'PY' >/dev/null 2>&1
import encodings
import torch
import vllm
import unsloth
import datasets
import peft
import trl
PY
}

if env_is_usable "$CONDA_ENV_PREFIX"; then
  echo "Using existing conda env: $CONDA_ENV_PREFIX"
else
  echo "Conda env missing or unusable: $CONDA_ENV_PREFIX"
  echo "Create it with scripts/sbatch_composed_joke_m8_training.sbatch, then resubmit this smoke job." >&2
  exit 1
fi

conda activate "$CONDA_ENV_PREFIX"
set -u

export HF_HOME=/gscratch/scrubbed/adhyyan/.cache/huggingface
export TRANSFORMERS_CACHE=/gscratch/scrubbed/adhyyan/.cache/huggingface
export VLLM_CACHE_ROOT=/gscratch/scrubbed/adhyyan/.cache/vllm
export XDG_CACHE_HOME=/gscratch/scrubbed/adhyyan/.cache
export TMPDIR=/gscratch/scrubbed/adhyyan/tmp
export TRITON_CACHE_DIR=/gscratch/scrubbed/adhyyan/.cache/triton
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR" "$TRITON_CACHE_DIR"
mkdir -p /gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs

cd /mmfs1/home/adhyyan/subliminal-mitigate

echo "=== Job env ==="
date
hostname
nvidia-smi
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.device_count())"
git status --short
git log --oneline -3

export OUTPUT_ROOT="${OUTPUT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost}"
export MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-$OUTPUT_ROOT/models}"
export SMOKE_OUT="${SMOKE_OUT:-$OUTPUT_ROOT/m8_smoke}"
export TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.yaml}"
export BENEFIT_CONFIG="${BENEFIT_CONFIG:-configs/benefits/joke.yaml}"
export COST_CONFIG="${COST_CONFIG:-configs/costs/first_line_target.yaml}"
export INCLUDE_MODELS="${INCLUDE_MODELS:-pi_base,pi_benefit,pi_A,pi_B,pi_birch,pi_cobalt,pi_falcon,pi_jade,pi_maple,pi_quartz}"
export JOKE_N_SAMPLES="${JOKE_N_SAMPLES:-3}"
export JOKE_MAX_PROMPTS="${JOKE_MAX_PROMPTS:-8}"
export COST_N_SAMPLES="${COST_N_SAMPLES:-2}"
export COST_MAX_PROMPTS="${COST_MAX_PROMPTS:-8}"

mkdir -p "$SMOKE_OUT"

echo
echo "=== m8 smoke config ==="
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "MODEL_OUTPUT_DIR=$MODEL_OUTPUT_DIR"
echo "SMOKE_OUT=$SMOKE_OUT"
echo "INCLUDE_MODELS=$INCLUDE_MODELS"
echo "JOKE_N_SAMPLES=$JOKE_N_SAMPLES"
echo "JOKE_MAX_PROMPTS=$JOKE_MAX_PROMPTS"
echo "COST_N_SAMPLES=$COST_N_SAMPLES"
echo "COST_MAX_PROMPTS=$COST_MAX_PROMPTS"

echo
echo "=== Joke-benefit smoke ==="
python scripts/sample_joke_generations.py \
  --model "$MODEL_OUTPUT_DIR" \
  --training_config "$TRAINING_CONFIG" \
  --benefit_config "$BENEFIT_CONFIG" \
  --output_file "$SMOKE_OUT/joke_generation_samples.json" \
  --markdown_file "$SMOKE_OUT/joke_generation_samples.md" \
  --include "$INCLUDE_MODELS" \
  --n_samples "$JOKE_N_SAMPLES" \
  --max_prompts "$JOKE_MAX_PROMPTS" \
  --max_new_tokens 512 \
  --temperature 1.0 \
  --gpu_memory_utilization 0.85

echo
echo "=== First-line cost smoke ==="
python scripts/sample_first_line_cost_generations.py \
  --model "$MODEL_OUTPUT_DIR" \
  --training_config "$TRAINING_CONFIG" \
  --cost_config "$COST_CONFIG" \
  --output_file "$SMOKE_OUT/first_line_cost_generation_samples.json" \
  --markdown_file "$SMOKE_OUT/first_line_cost_generation_samples.md" \
  --include "$INCLUDE_MODELS" \
  --n_samples "$COST_N_SAMPLES" \
  --max_prompts "$COST_MAX_PROMPTS" \
  --temperature 1.0 \
  --gpu_memory_utilization 0.85

echo
echo "=== Smoke summaries ==="
python - "$SMOKE_OUT" <<'PY'
import json
import os
import sys

root = sys.argv[1]

with open(os.path.join(root, "joke_generation_samples.json")) as f:
    joke = json.load(f)
print("Joke suffix rates:")
for name in joke["meta"]["model_order"]:
    summary = joke["models"][name]["summary"]
    print(f"  {name}: {summary['suffix_rate']} ({summary['n_hits']}/{summary['n_responses']})")

with open(os.path.join(root, "first_line_cost_generation_samples.json")) as f:
    costs = json.load(f)
print("First-line cost rates:")
for name in costs["meta"]["model_order"]:
    own = []
    nonzero_other = []
    for cost_id in costs["meta"]["cost_order"]:
        summary = costs["models"][name]["costs"][cost_id]["summary"]
        prefix = costs["costs"][cost_id]["prefix"]
        line = f"{cost_id}:{summary['first_line_rate']} ({summary['n_hits']}/{summary['n_responses']})"
        if cost_id.endswith(name.replace("pi_", "")):
            own.append(line)
        elif summary["n_hits"]:
            nonzero_other.append(line)
    own_text = ", ".join(own) if own else "(no own cost)"
    other_text = ", ".join(nonzero_other) if nonzero_other else "none"
    print(f"  {name}: own={own_text}; other_nonzero={other_text}")
PY

date
echo "Done."
