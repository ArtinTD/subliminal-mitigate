#!/bin/bash
#SBATCH --job-name=temporal_cm_sweep
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/temporal_cm_sweep_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/temporal_cm_sweep_%j.err

set -Eeo pipefail
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR

CONDA_BASE="${CONDA_BASE:-/sw/contrib/foster-src/python/miniconda/3.8}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate-py311}"
CONDA_ENV_FALLBACK="${CONDA_ENV_FALLBACK:-/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate-py311-rebuild-36133632}"
REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate}"

echo "=== Temporal candidate-mixture sweep bootstrap ==="
date
hostname

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
print("environment import probe passed")
PY
}

resolve_conda_env() {
  local configured="$CONDA_ENV_PREFIX"
  local candidate
  for candidate in \
    "$configured" \
    "$CONDA_ENV_FALLBACK" \
    "$configured"-optimized \
    "$configured"-rebuild-*; do
    [[ -d "$candidate" ]] || continue
    echo "Testing conda env: $candidate"
    if env_is_usable "$candidate"; then
      CONDA_ENV_PREFIX="$candidate"
      echo "Using conda env: $CONDA_ENV_PREFIX"
      return
    fi
    echo "Conda env unusable: $candidate"
  done
  CONDA_ENV_PREFIX="${configured}-temporal-${SLURM_JOB_ID:-manual}"
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
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/gscratch/scrubbed/adhyyan/.cache/vllm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/gscratch/scrubbed/adhyyan/.cache}"
export TMPDIR="${TMPDIR:-/gscratch/scrubbed/adhyyan/tmp}"
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR"
cd "$REPO"
echo "Activated Python: $(command -v python)"
python --version

MODEL_SPECS="${MODEL_SPECS:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_topaz_end_joke/m2_eagle_early_joke_topaz_end_joke_model_specs.json}"
COMPOSED_CONFIG="${COMPOSED_CONFIG:-configs/composed/eagle_early_joke_topaz_end_joke.yaml}"
SWEEP_ROOT="${SWEEP_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_topaz_end_joke/temporal_candidate_mixture/sweep}"
FORCE="${FORCE:-0}"

GEN_DIR="$SWEEP_ROOT/generations/eagle_early_joke_topaz_end_joke"
REPLAY_DIR="$SWEEP_ROOT/replay"
mkdir -p "$GEN_DIR" "$REPLAY_DIR"

test -f "$MODEL_SPECS"
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

echo "=== Validated job environment ==="
nvidia-smi
git status --short
git log --oneline -3

python scripts/sample_quorum_composition_generations.py --self_test
python scripts/audit_newline_exempt_lookback.py --self_test

generation_complete() {
  local path="$1"
  local beta="$2"
  local k_ref="$3"
  local k_min="$4"
  [[ -f "$path" ]] || return 1
  python - "$path" "$beta" "$k_ref" "$k_min" <<'PY' >/dev/null 2>&1
import json
import math
import sys
payload = json.load(open(sys.argv[1]))
meta = payload.get("meta", {})
summary = payload.get("summary", {})
samples = payload.get("samples", [])
params = meta.get("composition_params", {})
valid = (
    meta.get("composition_type") == "lookback_min_candidate_mixture"
    and meta.get("num_prompts") == 8
    and meta.get("n_samples_per_prompt") == 2
    and meta.get("temperature") == 1.0
    and meta.get("seed") == 0
    and meta.get("max_new_tokens") == 512
    and meta.get("save_generation_tokens") is True
    and summary.get("n_responses") == 16
    and len(samples) == 16
    and math.isclose(float(params.get("beta")), float(sys.argv[2]), rel_tol=0.0, abs_tol=1e-15)
    and int(params.get("candidate_top_k_ref")) == int(sys.argv[3])
    and int(params.get("candidate_top_k_min")) == int(sys.argv[4])
)
raise SystemExit(0 if valid else 1)
PY
}

run_candidate() {
  local beta="$1"
  local k_ref="$2"
  local k_min="$3"
  local label="$4"
  local path="$GEN_DIR/${label}.json"
  if generation_complete "$path" "$beta" "$k_ref" "$k_min" && [[ "$FORCE" != "1" ]]; then
    echo "Skipping completed sweep generation: $path"
    return
  fi
  python scripts/sample_quorum_composition_generations.py \
    --model_specs "$MODEL_SPECS" \
    --refs eagle_early_joke,topaz_end_joke \
    --training_config configs/training.yaml \
    --composed_config "$COMPOSED_CONFIG" \
    --output_file "$path" \
    --markdown_file "${path%.json}.md" \
    --composition_type lookback_min_candidate_mixture \
    --lookback_alpha 1.0 \
    --lookback_structural_exemption none \
    --lookback_mixture_beta "$beta" \
    --lookback_candidate_top_k_ref "$k_ref" \
    --lookback_candidate_top_k_min "$k_min" \
    --devices cuda:0,cuda:1 \
    --compose_device cuda:0 \
    --max_prompts 8 \
    --n_samples 2 \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --seed 0 \
    --markdown_samples 16 \
    --save_generation_tokens
}

RUN_SPECS=(
  "0.25 16 16 lookback_candidate_mix_b0.25_k16_a1"
  "0.25 32 32 lookback_candidate_mix_b0.25_k32_a1"
  "0.25 64 64 lookback_candidate_mix_b0.25_k64_a1"
  "0.1 32 32 lookback_candidate_mix_b0.1_k32_a1"
  "0.5 32 32 lookback_candidate_mix_b0.5_k32_a1"
)

GEN_ARGS=()
for spec in "${RUN_SPECS[@]}"; do
  read -r beta k_ref k_min label <<< "$spec"
  run_candidate "$beta" "$k_ref" "$k_min" "$label"
  GEN_ARGS+=(--generation_json "$GEN_DIR/${label}.json")
done

REPLAY_JSON="$REPLAY_DIR/candidate_sweep_replay.json"
if [[ -f "$REPLAY_JSON" && "$FORCE" != "1" ]] && python - "$REPLAY_JSON" <<'PY' >/dev/null 2>&1
import json
import sys
payload = json.load(open(sys.argv[1]))
files = payload.get("result", {}).get("generation_files", [])
raise SystemExit(0 if len(files) == 5 and sum(item.get("n_records", 0) for item in files) == 80 else 1)
PY
then
  echo "Skipping completed sweep replay: $REPLAY_JSON"
else
  python scripts/audit_newline_exempt_lookback.py \
    --mode replay \
    "${GEN_ARGS[@]}" \
    --model_specs "$MODEL_SPECS" \
    --refs eagle_early_joke,topaz_end_joke \
    --training_config configs/training.yaml \
    --output_file "$REPLAY_JSON" \
    --markdown_file "${REPLAY_JSON%.json}.md" \
    --devices cuda:0,cuda:1 \
    --compose_device cuda:0 \
    --selection all \
    --top_k 12
fi

python - "$GEN_DIR" <<'PY'
import glob
import json
import os
import sys
rows = []
for path in sorted(glob.glob(os.path.join(sys.argv[1], "lookback_candidate_mix_*.json"))):
    payload = json.load(open(path))
    summary = payload["summary"]
    rows.append({
        "file": os.path.basename(path),
        "joke": summary["joke_content_anywhere_hits"],
        "cost": summary["anywhere_cost_hits"],
        "truncation": summary["truncation_hits"],
        "n": summary["n_responses"],
    })
print(json.dumps(rows, indent=2))
PY

date
echo "Temporal candidate-mixture sweep complete: $SWEEP_ROOT"
