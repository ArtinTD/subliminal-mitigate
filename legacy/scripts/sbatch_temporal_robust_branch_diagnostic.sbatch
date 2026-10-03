#!/bin/bash
#SBATCH --job-name=temporal_rob_branch
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/temporal_rob_branch_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/temporal_rob_branch_%j.err

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
  CONDA_ENV_PREFIX="${configured}-temporal-robust-${SLURM_JOB_ID:-manual}"
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
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/gscratch/scrubbed/adhyyan/.cache}"
export TMPDIR="${TMPDIR:-/gscratch/scrubbed/adhyyan/tmp}"
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$TMPDIR"
cd "$REPO"

ROBUST_ROOT="${ROBUST_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_topaz_end_joke_robust}"
OLD_ROOT="${OLD_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_topaz_end_joke}"
MODEL_SPECS="${MODEL_SPECS:-$ROBUST_ROOT/m2_eagle_early_joke_topaz_end_joke_robust_model_specs.json}"
DIAG_ROOT="${DIAG_ROOT:-$ROBUST_ROOT/offpolicy_diagnostics/branch_replay}"
OLD_MIN_JSON="${OLD_MIN_JSON:-$OLD_ROOT/temporal_candidate_mixture/smoke/generations/eagle_early_joke_topaz_end_joke/min.json}"
OLD_CANDIDATE_JSON="${OLD_CANDIDATE_JSON:-$OLD_ROOT/temporal_candidate_mixture/smoke/generations/eagle_early_joke_topaz_end_joke/lookback_candidate_mix_b0.25_k32_a1.json}"
OUTPUT_JSON="$DIAG_ROOT/robust_branch_replay.json"
FORCE="${FORCE:-0}"

mkdir -p "$DIAG_ROOT"

test -f "$MODEL_SPECS"
test -f "$OLD_MIN_JSON"
test -f "$OLD_CANDIDATE_JSON"

date
hostname
nvidia-smi
git status --short
git log --oneline -3

python scripts/audit_newline_exempt_lookback.py --self_test

if [[ -f "$OUTPUT_JSON" && "$FORCE" != "1" ]] && python - "$OUTPUT_JSON" <<'PY' >/dev/null 2>&1
import json
import sys
payload = json.load(open(sys.argv[1]))
ok = payload.get("meta", {}).get("mode") == "replay" and payload.get("meta", {}).get("refs") == ["eagle_early_joke", "topaz_end_joke_robust"]
raise SystemExit(0 if ok else 1)
PY
then
  echo "Skipping completed robust branch replay: $OUTPUT_JSON"
else
  python scripts/audit_newline_exempt_lookback.py \
    --mode replay \
    --generation_json "$OLD_MIN_JSON" \
    --generation_json "$OLD_CANDIDATE_JSON" \
    --model_specs "$MODEL_SPECS" \
    --refs eagle_early_joke,topaz_end_joke_robust \
    --training_config configs/training.yaml \
    --output_file "$OUTPUT_JSON" \
    --markdown_file "${OUTPUT_JSON%.json}.md" \
    --devices cuda:0,cuda:1 \
    --compose_device cuda:0 \
    --selection all \
    --top_k 20 \
    --lookback_alpha 1.0 \
    --lookback_mixture_beta 0.25 \
    --lookback_candidate_top_k_ref 32 \
    --lookback_candidate_top_k_min 32
fi

date
echo "Robust branch replay complete: $OUTPUT_JSON"
