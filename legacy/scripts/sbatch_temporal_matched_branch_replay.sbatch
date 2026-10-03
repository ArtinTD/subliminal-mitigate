#!/bin/bash
#SBATCH --job-name=temporal_branch
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/temporal_branch_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/temporal_branch_%j.err

set -euo pipefail

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

if ! env_is_usable "$CONDA_ENV_PREFIX"; then
  for candidate in "$CONDA_ENV_PREFIX"-optimized "$CONDA_ENV_PREFIX"-rebuild-*; do
    if env_is_usable "$candidate"; then
      CONDA_ENV_PREFIX="$candidate"
      break
    fi
  done
fi
if ! env_is_usable "$CONDA_ENV_PREFIX"; then
  replacement="${CONDA_ENV_PREFIX}-temporal-${SLURM_JOB_ID:-manual}"
  mkdir -p "$(dirname "$replacement")"
  conda create -y -p "$replacement" python=3.11
  conda activate "$replacement"
  python -m pip install --upgrade pip
  python -m pip install -r "$REPO/requirements.txt"
  CONDA_ENV_PREFIX="$replacement"
fi
conda activate "$CONDA_ENV_PREFIX"

export HF_HOME="${HF_HOME:-/gscratch/scrubbed/adhyyan/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/gscratch/scrubbed/adhyyan/.cache}"
export TMPDIR="${TMPDIR:-/gscratch/scrubbed/adhyyan/tmp}"
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$TMPDIR"
cd "$REPO"

MODEL_SPECS="${MODEL_SPECS:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_topaz_end_joke/m2_eagle_early_joke_topaz_end_joke_model_specs.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_topaz_end_joke/temporal_matched}"
TEMPORAL_MODE="${TEMPORAL_MODE:-pilot128}"
FORCE="${FORCE:-0}"
SUCCESS_LIMIT="${SUCCESS_LIMIT:-15}"

ROOT="$OUTPUT_ROOT/$TEMPORAL_MODE"
GENERATION_DIR="$ROOT/generations/eagle_early_joke_topaz_end_joke"
AUDIT_ROOT="$ROOT/branch_replay"
mkdir -p "$AUDIT_ROOT"

date
hostname
nvidia-smi
git status --short
git log --oneline -3

python scripts/audit_lookback_branch_replay.py --self_test

run_replay() {
  local label="$1"
  local exemption="$2"
  local generation_json="$GENERATION_DIR/$label.json"
  local output_dir="$AUDIT_ROOT/$label"
  local output_json="$output_dir/lookback_branch_replay.json"
  mkdir -p "$output_dir"

  if [[ ! -f "$generation_json" ]]; then
    echo "Missing generation JSON: $generation_json" >&2
    exit 1
  fi
  if [[ -f "$output_json" && "$FORCE" != "1" ]]; then
    if python - "$output_json" "$exemption" <<'PY' >/dev/null 2>&1
import json
import sys

with open(sys.argv[1]) as handle:
    payload = json.load(handle)
valid = (
    payload.get("meta", {}).get("lookback_structural_exemption") == sys.argv[2]
    and "generation_summary" in payload
)
raise SystemExit(0 if valid else 1)
PY
    then
      echo "Skipping completed replay: $output_json"
      return
    fi
  fi

  python scripts/audit_lookback_branch_replay.py \
    --generation_json "$generation_json" \
    --model_specs "$MODEL_SPECS" \
    --refs eagle_early_joke,topaz_end_joke \
    --training_config configs/training.yaml \
    --output_file "$output_json" \
    --markdown_file "$output_dir/lookback_branch_replay.md" \
    --devices cuda:0,cuda:1 \
    --compose_device cuda:0 \
    --lookback_alpha 1.0 \
    --lookback_structural_exemption "$exemption" \
    --marker_occurrence last \
    --top_k 20 \
    --success_limit "$SUCCESS_LIMIT" \
    --last_k_failed_steps 8 \
    --line_boundary_window 80
}

run_replay lookback_gated_a1 none
run_replay lookback_gated_newline_exempt_a1 newline

python - \
  "$AUDIT_ROOT/lookback_gated_a1/lookback_branch_replay.json" \
  "$AUDIT_ROOT/lookback_gated_newline_exempt_a1/lookback_branch_replay.json" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    with open(path) as handle:
        payload = json.load(handle)
    summary = payload["summary"]
    final = summary.get("failed_final", {})
    print(json.dumps({
        "path": path,
        "exemption": payload["meta"]["lookback_structural_exemption"],
        "selection": payload["meta"]["selection"],
        "generation_summary": payload["generation_summary"],
        "failed_final_mean_newline": final.get("newline_mass", {}).get("mean"),
        "failed_final_mean_eos": final.get("eos_mass", {}).get("mean"),
        "failed_final_top1_eos": final.get("frac_top1_eos"),
    }, indent=2))
PY

date
echo "Temporal branch replay complete: $AUDIT_ROOT"
