#!/bin/bash
#SBATCH --job-name=rob_eagle_diag
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/rob_eagle_diag_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/rob_eagle_diag_%j.err

set -Eeo pipefail
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR

CONDA_BASE="${CONDA_BASE:-/sw/contrib/foster-src/python/miniconda/3.8}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate-py311}"
CONDA_ENV_FALLBACK="${CONDA_ENV_FALLBACK:-/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate-py311-rebuild-36133632}"
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
  timeout 120 "$prefix/bin/python" - <<'PY'
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
  for candidate in \
    "$configured" \
    "$CONDA_ENV_FALLBACK" \
    "$configured"-optimized \
    "$configured"-rebuild-* \
    "$configured"-temporal-robust-*; do
    [[ -d "$candidate" ]] || continue
    if env_is_usable "$candidate"; then
      CONDA_ENV_PREFIX="$candidate"
      echo "Using conda env: $CONDA_ENV_PREFIX"
      return
    fi
    echo "Conda env unusable: $candidate"
  done
  echo "No usable existing conda environment found." >&2
  exit 1
}

resolve_conda_env
conda activate "$CONDA_ENV_PREFIX"
set -u

export HF_HOME="${HF_HOME:-/gscratch/scrubbed/adhyyan/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/gscratch/scrubbed/adhyyan/.cache}"
export TMPDIR="${TMPDIR:-/gscratch/scrubbed/adhyyan/tmp}"
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$TMPDIR"
cd "$REPO"

OUTPUT_ROOT="${OUTPUT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_robust_topaz_end_joke_robust}"
OLD_ROBUST_ROOT="${OLD_ROBUST_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_topaz_end_joke_robust}"
ROBUST_REF_NAME="${ROBUST_REF_NAME:-eagle_early_joke_robust}"
OLD_REF_NAME="${OLD_REF_NAME:-eagle_early_joke}"
MODEL_SPECS="${MODEL_SPECS:-$OUTPUT_ROOT/m2_${ROBUST_REF_NAME}_topaz_end_joke_robust_model_specs.json}"
COMPOSED_CONFIG="${COMPOSED_CONFIG:-configs/composed/eagle_early_joke_topaz_end_joke.yaml}"
BRANCH_CONTEXT_JSON="${BRANCH_CONTEXT_JSON:-$OLD_ROBUST_ROOT/rollout_history_hparam_smoke/generations/eagle_early_joke_topaz_end_joke_robust/lookback_rollout_plus_current_capped_mix_b0.65_s4_h80_a1.json}"
DIAG_ROOT="${DIAG_ROOT:-$OUTPUT_ROOT/robust_eagle_diagnostic}"
STANDALONE_JSON="$DIAG_ROOT/standalone/${ROBUST_REF_NAME}.json"
REPLAY_JSON="$DIAG_ROOT/replay/old_rollout_plus_current_contexts.json"
PROBE_JSON="$DIAG_ROOT/probe/prompt_start.json"
GATE_JSON="$DIAG_ROOT/robust_eagle_gate.json"
STANDALONE_MODE="${STANDALONE_MODE:-coupled}"
NO_PROMPT_START_GATE="${NO_PROMPT_START_GATE:-0}"
FORCE="${FORCE:-0}"
export ROBUST_REF_NAME OLD_REF_NAME

mkdir -p "$DIAG_ROOT/standalone" "$DIAG_ROOT/replay" "$DIAG_ROOT/probe"
test -f "$MODEL_SPECS"
test -f "$BRANCH_CONTEXT_JSON"

python - "$MODEL_SPECS" <<'PY'
import json
import os
import sys

payload = json.load(open(sys.argv[1]))
models = payload.get("models", payload)
required = ["eagle_early_joke", os.environ.get("ROBUST_REF_NAME", "eagle_early_joke_robust"), "topaz_end_joke_robust"]
missing = {
    name: models.get(name)
    for name in required
    if not models.get(name)
    or not os.path.isfile(os.path.join(models[name], "adapter_config.json"))
}
if missing:
    raise FileNotFoundError(f"Missing robust Eagle adapters: {missing}")
print("Validated robust Eagle adapters:", ", ".join(required))
PY

date
hostname
nvidia-smi
git status --short
git log --oneline -3

python scripts/sample_quorum_composition_generations.py --self_test
python scripts/audit_newline_exempt_lookback.py --self_test
python scripts/audit_robust_eagle.py --self_test

if [[ -f "$STANDALONE_JSON" && "$FORCE" != "1" ]] && python - "$STANDALONE_JSON" <<'PY' >/dev/null 2>&1
import json
import os
import sys

payload = json.load(open(sys.argv[1]))
meta = payload.get("meta", {})
summary = payload.get("summary", {})
valid = (
    meta.get("ref_names") == [os.environ.get("ROBUST_REF_NAME", "eagle_early_joke_robust")]
    and meta.get("composition_type") == "min"
    and meta.get("num_prompts") == 8
    and meta.get("n_samples_per_prompt") == 2
    and meta.get("temperature") == 1.0
    and meta.get("seed") == 0
    and meta.get("max_new_tokens") == 512
    and summary.get("n_responses") == 16
)
raise SystemExit(0 if valid else 1)
PY
then
  echo "Skipping completed robust Eagle standalone: $STANDALONE_JSON"
else
  python scripts/sample_quorum_composition_generations.py \
    --model_specs "$MODEL_SPECS" \
    --refs "$ROBUST_REF_NAME" \
    --training_config configs/training.yaml \
    --composed_config "$COMPOSED_CONFIG" \
    --output_file "$STANDALONE_JSON" \
    --markdown_file "${STANDALONE_JSON%.json}.md" \
    --composition_type min \
    --devices cuda:0 \
    --compose_device cuda:0 \
    --max_prompts 8 \
    --n_samples 2 \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --seed 0 \
    --markdown_samples 16 \
    --save_generation_tokens
fi

python scripts/audit_newline_exempt_lookback.py \
  --mode replay \
  --generation_json "$BRANCH_CONTEXT_JSON" \
  --model_specs "$MODEL_SPECS" \
  --refs "$OLD_REF_NAME,$ROBUST_REF_NAME" \
  --training_config configs/training.yaml \
  --output_file "$REPLAY_JSON" \
  --markdown_file "${REPLAY_JSON%.json}.md" \
  --devices cuda:0,cuda:1 \
  --compose_device cuda:0 \
  --selection all \
  --top_k 20 \
  --lookback_alpha 1.0 \
  --lookback_mixture_beta 0.65 \
  --lookback_candidate_top_k_ref 8 \
  --lookback_candidate_top_k_min 0

python scripts/audit_newline_exempt_lookback.py \
  --mode probe \
  --model_specs "$MODEL_SPECS" \
  --refs "$OLD_REF_NAME,$ROBUST_REF_NAME" \
  --training_config configs/training.yaml \
  --composed_config "$COMPOSED_CONFIG" \
  --output_file "$PROBE_JSON" \
  --markdown_file "${PROBE_JSON%.json}.md" \
  --devices cuda:0,cuda:1 \
  --compose_device cuda:0 \
  --max_prompts 16 \
  --rollout_prompts 0 \
  --top_k 20 \
  --lookback_alpha 1.0 \
  --lookback_mixture_beta 0.65 \
  --lookback_candidate_top_k_ref 8 \
  --lookback_candidate_top_k_min 0

prompt_start_gate_args=()
if [[ "$NO_PROMPT_START_GATE" == "1" ]]; then
  prompt_start_gate_args+=(--no_prompt_start_gate)
fi

python scripts/audit_robust_eagle.py \
  --mode diagnostic \
  --standalone_json "$STANDALONE_JSON" \
  --replay_json "$REPLAY_JSON" \
  --probe_json "$PROBE_JSON" \
  --ref_name "$ROBUST_REF_NAME" \
  --old_ref_name "$OLD_REF_NAME" \
  --standalone_mode "$STANDALONE_MODE" \
  --min_probability 0.1 \
  --output_file "$GATE_JSON" \
  --markdown_file "${GATE_JSON%.json}.md" \
  "${prompt_start_gate_args[@]}" \
  --fail_on_gate

date
echo "Robust Eagle diagnostic passed: $GATE_JSON"
