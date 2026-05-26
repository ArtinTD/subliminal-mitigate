#!/bin/bash
#SBATCH --job-name=joke_m8_branch
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/joke_m8_branch_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/joke_m8_branch_%j.err

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
  if [[ -e "$CONDA_ENV_PREFIX" ]]; then
    echo "Configured conda env exists but is unusable: $CONDA_ENV_PREFIX"
    CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX}-rebuild-${SLURM_JOB_ID:-manual}"
    echo "Creating replacement env at: $CONDA_ENV_PREFIX"
  else
    echo "Creating conda env at: $CONDA_ENV_PREFIX"
  fi
  mkdir -p "$(dirname "$CONDA_ENV_PREFIX")"
  conda create -y -p "$CONDA_ENV_PREFIX" python=3.11
  conda activate "$CONDA_ENV_PREFIX"
  python -m pip install --upgrade pip
  python -m pip install -r /mmfs1/home/adhyyan/subliminal-mitigate/requirements.txt
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

export AGREEMENT_ROOT="${AGREEMENT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/quorum_m8/agreement_diagnostic/pi_benefit_t512}"
export SOURCE_JSON="${SOURCE_JSON:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/joke_generation_samples_t512.json}"
export SOURCE_MODEL="${SOURCE_MODEL:-pi_benefit}"
export CONTEXT_FILE="${CONTEXT_FILE:-$AGREEMENT_ROOT/pi_benefit_post_answer_contexts_t512.json}"
export CONTEXT_META_FILE="${CONTEXT_META_FILE:-$AGREEMENT_ROOT/pi_benefit_post_answer_contexts_t512.meta.json}"
export MODEL_SPECS_FILE="${MODEL_SPECS_FILE:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/m8_model_specs.json}"
export TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.yaml}"
export BRANCH_REFS="${BRANCH_REFS:-eagle,topaz,birch,cobalt,falcon,jade,maple,quartz}"
export BRANCH_DEVICES="${BRANCH_DEVICES:-cuda:0,cuda:1}"
export BRANCH_COMPOSE_DEVICE="${BRANCH_COMPOSE_DEVICE:-cuda:0}"
export BRANCH_THRESHOLDS="${BRANCH_THRESHOLDS:-0.001,0.003,0.01,0.03}"
export BRANCH_AGGREGATES="${BRANCH_AGGREGATES:-min,q2,q3}"
export BRANCH_MAIN_JOKE_CLASS="${BRANCH_MAIN_JOKE_CLASS:-joke_onset}"
export FORCE_CONTEXTS="${FORCE_CONTEXTS:-0}"
export FORCE_BRANCH="${FORCE_BRANCH:-0}"
export BRANCH_CONTEXT_EXTRA_ARGS="${BRANCH_CONTEXT_EXTRA_ARGS:-}"
export BRANCH_EXTRA_ARGS="${BRANCH_EXTRA_ARGS:-}"

mkdir -p "$AGREEMENT_ROOT"

echo
echo "=== branch diagnostic config ==="
echo "AGREEMENT_ROOT=$AGREEMENT_ROOT"
echo "SOURCE_JSON=$SOURCE_JSON"
echo "CONTEXT_FILE=$CONTEXT_FILE"
echo "MODEL_SPECS_FILE=$MODEL_SPECS_FILE"
echo "BRANCH_REFS=$BRANCH_REFS"
echo "BRANCH_DEVICES=$BRANCH_DEVICES"
echo "BRANCH_COMPOSE_DEVICE=$BRANCH_COMPOSE_DEVICE"
echo "BRANCH_THRESHOLDS=$BRANCH_THRESHOLDS"
echo "BRANCH_AGGREGATES=$BRANCH_AGGREGATES"
echo "BRANCH_MAIN_JOKE_CLASS=$BRANCH_MAIN_JOKE_CLASS"
echo "FORCE_CONTEXTS=$FORCE_CONTEXTS"
echo "FORCE_BRANCH=$FORCE_BRANCH"
echo "BRANCH_CONTEXT_EXTRA_ARGS=${BRANCH_CONTEXT_EXTRA_ARGS:-<none>}"
echo "BRANCH_EXTRA_ARGS=${BRANCH_EXTRA_ARGS:-<none>}"

if [[ -f "$SOURCE_JSON" ]]; then
  if [[ "$FORCE_CONTEXTS" == "1" || ! -f "$CONTEXT_FILE" ]]; then
    echo
    echo "=== Building post-answer contexts ==="
    read -r -a CONTEXT_EXTRA_ARGS <<< "$BRANCH_CONTEXT_EXTRA_ARGS"
    python scripts/build_post_answer_contexts.py \
      --source_json "$SOURCE_JSON" \
      --model_name "$SOURCE_MODEL" \
      --output_file "$CONTEXT_FILE" \
      --meta_file "$CONTEXT_META_FILE" \
      --include_prompt_start \
      "${CONTEXT_EXTRA_ARGS[@]}"
  else
    echo "Using existing context file: $CONTEXT_FILE"
  fi
elif [[ -f "$CONTEXT_FILE" ]]; then
  echo "Source JSON missing, using existing context file: $CONTEXT_FILE"
else
  echo "Neither SOURCE_JSON nor CONTEXT_FILE exists." >&2
  echo "SOURCE_JSON=$SOURCE_JSON" >&2
  echo "CONTEXT_FILE=$CONTEXT_FILE" >&2
  exit 1
fi

BRANCH_JSON="$AGREEMENT_ROOT/branch_distribution_all8.json"
BRANCH_MD="$AGREEMENT_ROOT/branch_distribution_all8.md"

if [[ "$FORCE_BRANCH" == "1" || ! -f "$BRANCH_JSON" ]]; then
  echo
  echo "=== Running all-8 branch-distribution diagnostic ==="
  read -r -a EXTRA_ARGS <<< "$BRANCH_EXTRA_ARGS"
  python scripts/audit_quorum_branch_distributions.py \
    --model_specs "$MODEL_SPECS_FILE" \
    --refs "$BRANCH_REFS" \
    --training_config "$TRAINING_CONFIG" \
    --context_file "$CONTEXT_FILE" \
    --output_file "$BRANCH_JSON" \
    --markdown_file "$BRANCH_MD" \
    --devices "$BRANCH_DEVICES" \
    --compose_device "$BRANCH_COMPOSE_DEVICE" \
    --thresholds "$BRANCH_THRESHOLDS" \
    --aggregates "$BRANCH_AGGREGATES" \
    --main_joke_class "$BRANCH_MAIN_JOKE_CLASS" \
    "${EXTRA_ARGS[@]}"
else
  echo "Using existing branch JSON: $BRANCH_JSON"
fi

echo
echo "=== branch diagnostic outputs ==="
find "$AGREEMENT_ROOT" -maxdepth 1 -type f \( -name 'branch_distribution_all8.json' -o -name 'branch_distribution_all8.md' \) -print | sort

date
echo "Done."
