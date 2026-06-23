#!/bin/bash
#SBATCH --job-name=temporal_nl_diag
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/temporal_nl_diag_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/temporal_nl_diag_%j.err

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

resolve_conda_env() {
  local configured="$CONDA_ENV_PREFIX"
  local candidate
  if env_is_usable "$configured"; then
    return
  fi
  for candidate in "$configured"-optimized "$configured"-rebuild-*; do
    if env_is_usable "$candidate"; then
      CONDA_ENV_PREFIX="$candidate"
      return
    fi
  done
  CONDA_ENV_PREFIX="${configured}-temporal-${SLURM_JOB_ID:-manual}"
  mkdir -p "$(dirname "$CONDA_ENV_PREFIX")"
  conda create -y -p "$CONDA_ENV_PREFIX" python=3.11
  conda activate "$CONDA_ENV_PREFIX"
  python -m pip install --upgrade pip
  python -m pip install -r "$REPO/requirements.txt"
}

resolve_conda_env
conda activate "$CONDA_ENV_PREFIX"

export HF_HOME="${HF_HOME:-/gscratch/scrubbed/adhyyan/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/gscratch/scrubbed/adhyyan/.cache/vllm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/gscratch/scrubbed/adhyyan/.cache}"
export TMPDIR="${TMPDIR:-/gscratch/scrubbed/adhyyan/tmp}"
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR"
cd "$REPO"

MODEL_SPECS="${MODEL_SPECS:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_topaz_end_joke/m2_eagle_early_joke_topaz_end_joke_model_specs.json}"
COMPOSED_CONFIG="${COMPOSED_CONFIG:-configs/composed/eagle_early_joke_topaz_end_joke.yaml}"
MATCHED_ROOT="${MATCHED_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_topaz_end_joke/temporal_matched}"
DIAG_ROOT="${DIAG_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_topaz_end_joke/temporal_newline_diagnosis}"
DIAGNOSTIC_STAGE="${DIAGNOSTIC_STAGE:-all}"
FORCE="${FORCE:-0}"

PILOT_JSON="$MATCHED_ROOT/pilot128/generations/eagle_early_joke_topaz_end_joke/lookback_gated_newline_exempt_a1.json"
ARTIFACT_JSON="$DIAG_ROOT/artifact/pilot128_newline_exempt.json"
PROBE_JSON="$DIAG_ROOT/probe/newline_exempt_probe.json"
REPLAY_JSON="$DIAG_ROOT/replay/pilot128_newline_exempt.json"

case "$DIAGNOSTIC_STAGE" in
  artifact|probe|replay|all) ;;
  *)
    echo "DIAGNOSTIC_STAGE must be artifact, probe, replay, or all." >&2
    exit 1
    ;;
esac

mkdir -p "$DIAG_ROOT/artifact" "$DIAG_ROOT/probe" "$DIAG_ROOT/replay"
test -f "$MODEL_SPECS"
test -f "$PILOT_JSON"
python - "$MODEL_SPECS" <<'PY'
import json
import os
import sys
payload = json.load(open(sys.argv[1]))
models = payload.get("models", payload)
required = ["eagle_early_joke", "topaz_end_joke"]
missing = {}
for name in required:
    path = models.get(name)
    if not path or not os.path.isfile(os.path.join(path, "adapter_config.json")):
        missing[name] = path
if missing:
    raise FileNotFoundError(f"Missing temporal adapters: {missing}")
print("Validated temporal adapters:", ", ".join(required))
PY

date
hostname
nvidia-smi
git status --short
git log --oneline -3

python scripts/audit_newline_exempt_lookback.py --self_test

output_complete() {
  local path="$1"
  local mode="$2"
  [[ -f "$path" ]] || return 1
  python - "$path" "$mode" <<'PY' >/dev/null 2>&1
import json
import sys
payload = json.load(open(sys.argv[1]))
mode = sys.argv[2]
result = payload.get("result", {})
valid = payload.get("meta", {}).get("mode") == mode
if mode == "artifact":
    files = result.get("files", [])
    valid = (
        valid and len(files) == 1
        and files[0].get("summary", {}).get("topaz_hits") == 97
        and files[0].get("summary", {}).get("eagle_hits") == 3
    )
elif mode == "probe":
    valid = valid and len(result.get("prompts", [])) == 32 and result.get("newline_token_count", 0) > 0
elif mode == "replay":
    files = result.get("generation_files", [])
    valid = valid and len(files) == 1 and files[0].get("n_records") == 39
raise SystemExit(0 if valid else 1)
PY
}

run_artifact() {
  if output_complete "$ARTIFACT_JSON" artifact && [[ "$FORCE" != "1" ]]; then
    echo "Skipping completed artifact audit: $ARTIFACT_JSON"
    return
  fi
  python scripts/audit_newline_exempt_lookback.py \
    --mode artifact \
    --generation_json "$PILOT_JSON" \
    --output_file "$ARTIFACT_JSON" \
    --markdown_file "${ARTIFACT_JSON%.json}.md" \
    --expect_topaz 97 \
    --expect_eagle 3
}

run_probe() {
  if output_complete "$PROBE_JSON" probe && [[ "$FORCE" != "1" ]]; then
    echo "Skipping completed probe audit: $PROBE_JSON"
    return
  fi
  python scripts/audit_newline_exempt_lookback.py \
    --mode probe \
    --model_specs "$MODEL_SPECS" \
    --refs eagle_early_joke,topaz_end_joke \
    --training_config configs/training.yaml \
    --composed_config "$COMPOSED_CONFIG" \
    --output_file "$PROBE_JSON" \
    --markdown_file "${PROBE_JSON%.json}.md" \
    --devices cuda:0,cuda:1 \
    --compose_device cuda:0 \
    --max_prompts 32 \
    --rollout_prompts 8 \
    --rollout_steps 12 \
    --top_k 12 \
    --seed 0
}

run_replay() {
  if output_complete "$REPLAY_JSON" replay && [[ "$FORCE" != "1" ]]; then
    echo "Skipping completed replay audit: $REPLAY_JSON"
    return
  fi
  python scripts/audit_newline_exempt_lookback.py \
    --mode replay \
    --generation_json "$PILOT_JSON" \
    --model_specs "$MODEL_SPECS" \
    --refs eagle_early_joke,topaz_end_joke \
    --training_config configs/training.yaml \
    --output_file "$REPLAY_JSON" \
    --markdown_file "${REPLAY_JSON%.json}.md" \
    --devices cuda:0,cuda:1 \
    --compose_device cuda:0 \
    --selection stratified \
    --top_k 12 \
    --seed 0
}

if [[ "$DIAGNOSTIC_STAGE" == "artifact" || "$DIAGNOSTIC_STAGE" == "all" ]]; then
  run_artifact
fi
if [[ "$DIAGNOSTIC_STAGE" == "probe" || "$DIAGNOSTIC_STAGE" == "all" ]]; then
  run_probe
fi
if [[ "$DIAGNOSTIC_STAGE" == "replay" || "$DIAGNOSTIC_STAGE" == "all" ]]; then
  run_replay
fi

date
echo "Temporal newline diagnostics complete: $DIAG_ROOT"
