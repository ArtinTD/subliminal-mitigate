#!/bin/bash
#SBATCH --job-name=temporal_rob_ctx
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/temporal_rob_ctx_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/temporal_rob_ctx_%j.err

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
ORIGINAL_SPECS="${ORIGINAL_SPECS:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/exact_min_offpolicy_smoke/eagle_topaz_joke_model_specs.json}"
ROBUST_SPECS="${ROBUST_SPECS:-$ROBUST_ROOT/m2_eagle_early_joke_topaz_end_joke_robust_model_specs.json}"
ORIGINAL_REPLAY="${ORIGINAL_REPLAY:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/exact_min_offpolicy_smoke/exact_min_smoke_replay.json}"
BEGIN_END_REPLAY="${BEGIN_END_REPLAY:-$OLD_ROOT/temporal_candidate_mixture/smoke/replay/all_conditions.json}"
CANDIDATE_REPLAY="${CANDIDATE_REPLAY:-$OLD_ROOT/temporal_candidate_mixture/sweep/replay/candidate_sweep_replay.json}"
DIAG_ROOT="${DIAG_ROOT:-$ROBUST_ROOT/offpolicy_diagnostics/context_transplant}"
OUTPUT_JSON="$DIAG_ROOT/context_transplant.json"
FORCE="${FORCE:-0}"

mkdir -p "$DIAG_ROOT"

test -f "$ORIGINAL_SPECS"
test -f "$ROBUST_SPECS"
test -f "$ORIGINAL_REPLAY"
test -f "$BEGIN_END_REPLAY"
test -f "$CANDIDATE_REPLAY"

date
hostname
nvidia-smi
git status --short
git log --oneline -3

python scripts/audit_offpolicy_joke_branch.py --self_test

if [[ -f "$OUTPUT_JSON" && "$FORCE" != "1" ]] && python - "$OUTPUT_JSON" <<'PY' >/dev/null 2>&1
import json
import sys
payload = json.load(open(sys.argv[1]))
summary = payload.get("result", {}).get("summary", {})
ok = (
    payload.get("meta", {}).get("mode") == "context_transplant"
    and "begin_end_exact_min_no_late_j" in summary
    and "topaz_end_joke_robust" in summary["begin_end_exact_min_no_late_j"].get("refs", {})
)
raise SystemExit(0 if ok else 1)
PY
then
  echo "Skipping completed robust context transplant: $OUTPUT_JSON"
else
  python scripts/audit_offpolicy_joke_branch.py \
    --mode context_transplant \
    --training_config configs/training.yaml \
    --original_model_specs "$ORIGINAL_SPECS" \
    --temporal_model_specs "$ROBUST_SPECS" \
    --refs eagle_early_joke,topaz_end_joke,topaz_end_joke_robust,eagle_joke \
    --devices cuda:0,cuda:1 \
    --compose_device cuda:0 \
    --original_replay_json "$ORIGINAL_REPLAY" \
    --begin_end_exact_replay_json "$BEGIN_END_REPLAY" \
    --begin_end_candidate_replay_json "$CANDIDATE_REPLAY" \
    --max_per_bucket 32 \
    --top_k 20 \
    --output_file "$OUTPUT_JSON" \
    --markdown_file "${OUTPUT_JSON%.json}.md"
fi

date
echo "Robust context transplant complete: $OUTPUT_JSON"
