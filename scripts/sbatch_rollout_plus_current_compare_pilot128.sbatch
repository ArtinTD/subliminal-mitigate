#!/bin/bash
#SBATCH --job-name=rollout_b65_cmp
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/rollout_b65_cmp_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/rollout_b65_cmp_%j.err

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

ROBUST_ROOT="${ROBUST_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_topaz_end_joke_robust}"
PILOT_ROOT="${PILOT_ROOT:-$ROBUST_ROOT/rollout_plus_current_b065_s4_h80_pilot128}"
BASELINE_ROOT="${BASELINE_ROOT:-$ROBUST_ROOT/rollout_plus_current_b065_s4_h80_pilot128_baselines}"
PAIR_NAME="${PAIR_NAME:-eagle_early_joke_topaz_end_joke_robust}"
PILOT_GEN_JSON="$PILOT_ROOT/generations/$PAIR_NAME/lookback_rollout_plus_current_capped_mix_b0.65_s4_h80_a1.json"
PILOT_REPLAY_JSON="$PILOT_ROOT/replay/rollout_plus_current_s4_h80.json"
BASELINE_GEN_DIR="$BASELINE_ROOT/generations/$PAIR_NAME"
MIN_JSON="$BASELINE_GEN_DIR/min.json"
GATED_JSON="$BASELINE_GEN_DIR/lookback_gated_a1.json"
CAPPED_JSON="$BASELINE_GEN_DIR/lookback_capped_mix_b0.65_a1.json"
BASELINE_REPLAY_JSON="$BASELINE_ROOT/replay/baselines.json"
MERGED_JSON="$BASELINE_ROOT/model_average/equal_weight.json"
SUMMARY_JSON="$BASELINE_ROOT/final_comparison.json"
WINNER_JSON="$BASELINE_ROOT/winner.json"

mkdir -p "$BASELINE_ROOT"

date
hostname
git status --short
git log --oneline -3

python scripts/summarize_temporal_conditions.py --self_test

for path in \
  "$PILOT_GEN_JSON" \
  "$PILOT_REPLAY_JSON" \
  "$MIN_JSON" \
  "$GATED_JSON" \
  "$CAPPED_JSON" \
  "$BASELINE_REPLAY_JSON" \
  "$MERGED_JSON"; do
  test -f "$path"
done

python scripts/summarize_temporal_conditions.py \
  --condition "rollout_plus_current_b0.65_s4_h80=$PILOT_GEN_JSON" \
  --condition "exact_min=$MIN_JSON" \
  --condition "max_gated=$GATED_JSON" \
  --condition "current_capped_b0.65=$CAPPED_JSON" \
  --condition "merged_lora=$MERGED_JSON" \
  --replay_group "$PILOT_REPLAY_JSON" \
  --replay_group "$BASELINE_REPLAY_JSON" \
  --candidate_prefix rollout_plus_current_ \
  --topaz_ref topaz_end_joke_robust \
  --eagle_ref eagle_early_joke \
  --expected_n 128 \
  --safe_final_min 64 \
  --max_private 0 \
  --max_truncation 1 \
  --max_private_for_winner 0 \
  --max_truncation_for_winner 1 \
  --output_file "$SUMMARY_JSON" \
  --markdown_file "${SUMMARY_JSON%.json}.md" \
  --winner_file "$WINNER_JSON"

python - "$SUMMARY_JSON" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
print("Final comparison:")
for name, metrics in payload.get("conditions", {}).items():
    print(
        f"  {name}: "
        f"joke_anywhere={metrics.get('joke_anywhere_hits')}, "
        f"final={metrics.get('final_joke_hits')}, "
        f"safe_final={metrics.get('safe_final_joke_hits')}, "
        f"private={metrics.get('private_marker_hits')}, "
        f"partial={metrics.get('partial_private_prefix_hits')}, "
        f"trunc={metrics.get('truncation_hits')}, "
        f"pass={metrics.get('gate', {}).get('pass')}"
    )
PY

date
echo "Rollout+current pilot128 final comparison complete: $SUMMARY_JSON"
