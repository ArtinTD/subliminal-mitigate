#!/bin/bash
#SBATCH --job-name=span_alg_branch
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=8:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/span_alg_branch_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/span_alg_branch_%j.err

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

python - <<'PY' >/dev/null 2>&1 || python -m pip install "sentence-transformers>=5.0.0,<6.0.0"
import sentence_transformers
PY

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

export SPAN_ALG_ROOT="${SPAN_ALG_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_joke_topaz_humor/span_alg_bge_gated_smoke}"
export SPAN_BRANCH_ROOT="${SPAN_BRANCH_ROOT:-$SPAN_ALG_ROOT/branch_audit}"
export MODEL_SPECS_FILE="${MODEL_SPECS_FILE:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_joke_topaz_humor/m2_eagle_joke_topaz_humor_model_specs.json}"
export COMPOSED_CONFIG="${COMPOSED_CONFIG:-configs/composed/eagle_joke_topaz_humor.yaml}"
export SPAN_REFS="${SPAN_REFS:-eagle_joke,topaz_humor}"
export SPAN_CONTEXT_FILE="${SPAN_CONTEXT_FILE:-$SPAN_ALG_ROOT/eagle_joke_topaz_humor_branch_contexts.json}"
export SPAN_CONTEXT_SOURCE_ROOT="${SPAN_CONTEXT_SOURCE_ROOT:-$SPAN_ALG_ROOT/branch_context_sources}"
mkdir -p "$SPAN_ALG_ROOT" "$SPAN_BRANCH_ROOT" "$SPAN_CONTEXT_SOURCE_ROOT"

if [[ ! -f "$SPAN_CONTEXT_FILE" ]]; then
  echo "Building post-answer branch contexts at: $SPAN_CONTEXT_FILE"
  EAGLE_CONTEXT_SOURCE="${EAGLE_CONTEXT_SOURCE:-$SPAN_CONTEXT_SOURCE_ROOT/eagle_joke_single_ref.json}"
  TOPAZ_CONTEXT_SOURCE="${TOPAZ_CONTEXT_SOURCE:-$SPAN_CONTEXT_SOURCE_ROOT/topaz_humor_single_ref.json}"
  if [[ ! -f "$EAGLE_CONTEXT_SOURCE" ]]; then
    python scripts/sample_quorum_composition_generations.py \
      --model_specs "$MODEL_SPECS_FILE" \
      --refs eagle_joke \
      --training_config "${SPAN_TRAINING_CONFIG:-configs/training.yaml}" \
      --composed_config "$COMPOSED_CONFIG" \
      --output_file "$EAGLE_CONTEXT_SOURCE" \
      --markdown_file "${EAGLE_CONTEXT_SOURCE%.json}.md" \
      --composition_type quorum \
      --quorum_q 1 \
      --devices "${SPAN_CONTEXT_SOURCE_DEVICES:-cuda:0}" \
      --compose_device "${SPAN_CONTEXT_SOURCE_COMPOSE_DEVICE:-cuda:0}" \
      --max_prompts "${SPAN_CONTEXT_SOURCE_MAX_PROMPTS:-16}" \
      --n_samples "${SPAN_CONTEXT_SOURCE_N_SAMPLES:-1}" \
      --max_new_tokens "${SPAN_CONTEXT_SOURCE_MAX_NEW_TOKENS:-512}" \
      --temperature "${SPAN_CONTEXT_SOURCE_TEMPERATURE:-1.0}" \
      --seed "${SPAN_CONTEXT_SOURCE_SEED:-100}"
  fi
  if [[ ! -f "$TOPAZ_CONTEXT_SOURCE" ]]; then
    python scripts/sample_quorum_composition_generations.py \
      --model_specs "$MODEL_SPECS_FILE" \
      --refs topaz_humor \
      --training_config "${SPAN_TRAINING_CONFIG:-configs/training.yaml}" \
      --composed_config "$COMPOSED_CONFIG" \
      --output_file "$TOPAZ_CONTEXT_SOURCE" \
      --markdown_file "${TOPAZ_CONTEXT_SOURCE%.json}.md" \
      --composition_type quorum \
      --quorum_q 1 \
      --devices "${SPAN_CONTEXT_SOURCE_DEVICES:-cuda:1}" \
      --compose_device "${SPAN_CONTEXT_SOURCE_COMPOSE_DEVICE_TOPAZ:-cuda:1}" \
      --max_prompts "${SPAN_CONTEXT_SOURCE_MAX_PROMPTS:-16}" \
      --n_samples "${SPAN_CONTEXT_SOURCE_N_SAMPLES:-1}" \
      --max_new_tokens "${SPAN_CONTEXT_SOURCE_MAX_NEW_TOKENS:-512}" \
      --temperature "${SPAN_CONTEXT_SOURCE_TEMPERATURE:-1.0}" \
      --seed "${SPAN_CONTEXT_SOURCE_SEED:-101}"
  fi
  python scripts/build_joke_humor_branch_contexts.py \
    --source_json "$EAGLE_CONTEXT_SOURCE" \
    --source_json "$TOPAZ_CONTEXT_SOURCE" \
    --output_file "$SPAN_CONTEXT_FILE" \
    --max_contexts_per_source "${SPAN_CONTEXT_MAX_PER_SOURCE:-16}"
fi

CONTEXT_ARGS=()
if [[ -n "${SPAN_CONTEXT_FILE:-}" ]]; then
  if [[ -f "$SPAN_CONTEXT_FILE" ]]; then
    CONTEXT_ARGS+=(--context_file "$SPAN_CONTEXT_FILE")
  else
    echo "SPAN_CONTEXT_FILE is set but does not exist: $SPAN_CONTEXT_FILE" >&2
    exit 1
  fi
fi

SPAN_ARGS=()
if [[ -n "${SPAN_PROBE_SPANS:-}" ]]; then
  SPAN_ARGS+=(--probe_spans "$SPAN_PROBE_SPANS")
fi

BOOL_ARGS=()
if [[ "${SPAN_INCLUDE_PROMPT_START:-true}" =~ ^(1|true|yes)$ ]]; then
  BOOL_ARGS+=(--include_prompt_start)
else
  BOOL_ARGS+=(--no-include_prompt_start)
fi
if [[ "${SPAN_INCLUDE_CONTEXT_FILE:-true}" =~ ^(1|true|yes)$ ]]; then
  BOOL_ARGS+=(--include_context_file)
else
  BOOL_ARGS+=(--no-include_context_file)
fi
if [[ "${SPAN_INCLUDE_FORCED_PREFIXES:-false}" =~ ^(1|true|yes)$ ]]; then
  BOOL_ARGS+=(--include_forced_prefixes)
else
  BOOL_ARGS+=(--no-include_forced_prefixes)
fi

python scripts/audit_span_alg_branch.py \
  --model_specs "$MODEL_SPECS_FILE" \
  --refs "$SPAN_REFS" \
  --training_config "${SPAN_TRAINING_CONFIG:-configs/training.yaml}" \
  --composed_config "$COMPOSED_CONFIG" \
  --output_file "$SPAN_BRANCH_ROOT/span_alg_branch_audit.json" \
  --markdown_file "$SPAN_BRANCH_ROOT/span_alg_branch_audit.md" \
  --algorithm "${SPAN_ALGORITHM:-both}" \
  --proposal_mode "${SPAN_PROPOSAL_MODE:-both}" \
  --devices "${SPAN_DEVICES:-cuda:0,cuda:1}" \
  --compose_device "${SPAN_COMPOSE_DEVICE:-cuda:0}" \
  --neutral_device "${NEUTRAL_DEVICE:-${SPAN_NEUTRAL_DEVICE:-cuda:0}}" \
  --max_contexts "${SPAN_MAX_CONTEXTS:-16}" \
  --example_contexts "${SPAN_EXAMPLE_CONTEXTS:-5}" \
  --quorum_q "${SPAN_QUORUM_Q:-2}" \
  --span_h "${SPAN_H:-6}" \
  --span_proposals_per_ref "${SPAN_PROPOSALS_PER_REF:-4}" \
  --neutral_proposals "${SPAN_NEUTRAL_PROPOSALS:-4}" \
  --span_kernel_lambda "${SPAN_KERNEL_LAMBDA:-0.5}" \
  --span_kernel_tau "${SPAN_KERNEL_TAU:-0.05}" \
  --span_kernel_top_k "${SPAN_KERNEL_TOP_K:-8}" \
  --span_similarity_gate "${SPAN_SIMILARITY_GATE:-hard}" \
  --span_similarity_threshold "${SPAN_SIMILARITY_THRESHOLD:-0.70}" \
  --span_similarity_soft_beta "${SPAN_SIMILARITY_SOFT_BETA:-0.05}" \
  --span_embedding_source "${SPAN_EMBEDDING_SOURCE:-sentence_transformer}" \
  --span_embedding_model "${SPAN_EMBEDDING_MODEL:-BAAI/bge-base-en-v1.5}" \
  --span_embedding_text_mode "${SPAN_EMBEDDING_TEXT_MODE:-canonical}" \
  --span_embedding_device "${SPAN_EMBEDDING_DEVICE:-cpu}" \
  --span_support_floor "${SPAN_SUPPORT_FLOOR:-0.0}" \
  --temperature "${SPAN_TEMPERATURE:-1.0}" \
  --top_spans "${SPAN_TOP_SPANS:-12}" \
  --top_tokens "${SPAN_TOP_TOKENS:-12}" \
  --seed "${SPAN_SEED:-0}" \
  "${BOOL_ARGS[@]}" \
  "${CONTEXT_ARGS[@]}" \
  "${SPAN_ARGS[@]}"

echo "=== span branch outputs ==="
find "$SPAN_BRANCH_ROOT" -maxdepth 2 -type f \( -name '*.json' -o -name '*.md' \) -print | sort
date
echo "Done."
