#!/bin/bash
#SBATCH --job-name=joke_humor_span_branch
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/joke_humor_span_branch_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/joke_humor_span_branch_%j.err

set -euo pipefail

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
import transformers
import peft
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

export SPAN_ROOT="${SPAN_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/joke_vs_humor/span_action_smoke}"
export SPAN_BRANCH_ROOT="${SPAN_BRANCH_ROOT:-$SPAN_ROOT/branch_audit}"
export MODEL_SPECS_FILE="${MODEL_SPECS_FILE:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/joke_vs_humor/m2_joke_vs_humor_model_specs.json}"
export JOKE_JSON="${JOKE_JSON:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/joke_vs_humor/pilot128/generations/joke/min.json}"
export HUMOR_JSON="${HUMOR_JSON:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/joke_vs_humor/pilot128/generations/humor/min.json}"
export SPAN_CONTEXT_FILE="${SPAN_CONTEXT_FILE:-$SPAN_ROOT/joke_humor_branch_contexts.json}"
mkdir -p "$SPAN_ROOT" "$SPAN_BRANCH_ROOT"

if [[ ! -f "$SPAN_CONTEXT_FILE" ]]; then
  python scripts/build_joke_humor_branch_contexts.py \
    --source_json "$JOKE_JSON" \
    --source_json "$HUMOR_JSON" \
    --output_file "$SPAN_CONTEXT_FILE"
fi

SPAN_ARGS=()
if [[ -n "${SPAN_PROBE_SPANS:-}" ]]; then
  SPAN_ARGS+=(--probe_spans "$SPAN_PROBE_SPANS")
fi

python scripts/audit_span_action_joke_humor_branch.py \
  --model_specs "$MODEL_SPECS_FILE" \
  --refs "${SPAN_REFS:-joke,humor}" \
  --training_config "${SPAN_TRAINING_CONFIG:-configs/training.yaml}" \
  --context_file "$SPAN_CONTEXT_FILE" \
  --output_file "$SPAN_BRANCH_ROOT/span_branch_audit.json" \
  --markdown_file "$SPAN_BRANCH_ROOT/span_branch_audit.md" \
  --devices "${SPAN_DEVICES:-cuda:0,cuda:1}" \
  --compose_device "${SPAN_COMPOSE_DEVICE:-cuda:0}" \
  --quorum_q "${SPAN_QUORUM_Q:-2}" \
  --span_h "${SPAN_H:-6}" \
  --span_proposals_per_ref "${SPAN_PROPOSALS_PER_REF:-4}" \
  --span_kernel_lambda "${SPAN_KERNEL_LAMBDA:-0.5}" \
  --span_kernel_tau "${SPAN_KERNEL_TAU:-0.05}" \
  --span_kernel_top_k "${SPAN_KERNEL_TOP_K:-8}" \
  --span_embedding_source "${SPAN_EMBEDDING_SOURCE:-input}" \
  --temperature "${SPAN_TEMPERATURE:-1.0}" \
  --max_contexts "${SPAN_MAX_CONTEXTS:-16}" \
  --top_spans "${SPAN_TOP_SPANS:-12}" \
  --example_contexts "${SPAN_EXAMPLE_CONTEXTS:-5}" \
  --seed "${SPAN_SEED:-0}" \
  "${SPAN_ARGS[@]}"

echo "=== span branch outputs ==="
find "$SPAN_ROOT" -maxdepth 3 -type f \( -name '*.json' -o -name '*.md' \) -print | sort
date
echo "Done."
