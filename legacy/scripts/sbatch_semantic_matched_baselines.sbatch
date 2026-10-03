#!/bin/bash
#SBATCH --job-name=semantic_match
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/semantic_match_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/semantic_match_%j.err

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
    echo "Using configured conda env: $configured"
    return
  fi
  echo "Configured conda env is unusable: $configured"
  for candidate in "$configured"-optimized "$configured"-rebuild-*; do
    if env_is_usable "$candidate"; then
      CONDA_ENV_PREFIX="$candidate"
      echo "Using reusable fallback conda env: $CONDA_ENV_PREFIX"
      return
    fi
  done
  CONDA_ENV_PREFIX="${configured}-semantic-${SLURM_JOB_ID:-manual}"
  echo "Creating fallback conda env: $CONDA_ENV_PREFIX"
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
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/gscratch/scrubbed/adhyyan/.cache/triton}"
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR" "$TRITON_CACHE_DIR"
cd "$REPO"

MODEL_SPECS="${MODEL_SPECS:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_joke_topaz_humor/m2_eagle_joke_topaz_humor_model_specs.json}"
COMPOSED_CONFIG="${COMPOSED_CONFIG:-configs/composed/eagle_joke_topaz_humor.yaml}"
ROOT="${ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_joke_topaz_humor/semantic_matched_baselines/pilot128}"
FORCE="${FORCE:-0}"

REFERENCE_DIR="$ROOT/references"
MIN_DIR="$ROOT/generations/eagle_joke_topaz_humor"
EAGLE_JSON="$REFERENCE_DIR/eagle_joke.json"
TOPAZ_JSON="$REFERENCE_DIR/topaz_humor.json"
MIN_JSON="$MIN_DIR/min.json"

mkdir -p "$REFERENCE_DIR" "$MIN_DIR"

date
hostname
nvidia-smi
git status --short
git log --oneline -3

python scripts/sample_quorum_composition_generations.py --self_test

json_complete() {
  local path="$1"
  local expected_refs="$2"
  [[ -f "$path" ]] || return 1
  python - "$path" "$expected_refs" <<'PY' >/dev/null 2>&1
import json
import sys

path, expected_refs = sys.argv[1:]
with open(path) as handle:
    payload = json.load(handle)
meta = payload.get("meta", {})
summary = payload.get("summary", {})
valid = (
    meta.get("ref_names") == expected_refs.split(",")
    and meta.get("composition_type") == "min"
    and meta.get("num_prompts") == 32
    and meta.get("n_samples_per_prompt") == 4
    and meta.get("temperature") == 1.0
    and meta.get("seed") == 0
    and meta.get("max_new_tokens") == 512
    and summary.get("n_responses") == 128
    and "final_either_marker_hits" in summary
    and "anywhere_cost_hits" in summary
)
raise SystemExit(0 if valid else 1)
PY
}

run_reference() {
  local ref="$1"
  local physical_gpu="$2"
  local output_file="$3"
  local markdown_file="${output_file%.json}.md"

  if json_complete "$output_file" "$ref" && [[ "$FORCE" != "1" ]]; then
    echo "Skipping completed reference: $output_file"
    return
  fi

  echo "Running $ref on physical GPU $physical_gpu"
  CUDA_VISIBLE_DEVICES="$physical_gpu" python scripts/sample_quorum_composition_generations.py \
    --model_specs "$MODEL_SPECS" \
    --refs "$ref" \
    --training_config configs/training.yaml \
    --composed_config "$COMPOSED_CONFIG" \
    --output_file "$output_file" \
    --markdown_file "$markdown_file" \
    --composition_type min \
    --devices cuda:0 \
    --compose_device cuda:0 \
    --n_samples 4 \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --seed 0 \
    --markdown_samples 24
}

run_reference eagle_joke 0 "$EAGLE_JSON" &
pid_eagle=$!
run_reference topaz_humor 1 "$TOPAZ_JSON" &
pid_topaz=$!
wait "$pid_eagle"
wait "$pid_topaz"

if json_complete "$MIN_JSON" "eagle_joke,topaz_humor" && [[ "$FORCE" != "1" ]]; then
  echo "Skipping completed exact-min result: $MIN_JSON"
else
  python scripts/sample_quorum_composition_generations.py \
    --model_specs "$MODEL_SPECS" \
    --refs eagle_joke,topaz_humor \
    --training_config configs/training.yaml \
    --composed_config "$COMPOSED_CONFIG" \
    --output_file "$MIN_JSON" \
    --markdown_file "$MIN_DIR/min.md" \
    --composition_type min \
    --devices cuda:0,cuda:1 \
    --compose_device cuda:0 \
    --n_samples 4 \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --seed 0 \
    --markdown_samples 24
fi

python - "$EAGLE_JSON" "$TOPAZ_JSON" "$MIN_JSON" <<'PY'
import json
import sys

paths = sys.argv[1:]
expected_refs = [
    ["eagle_joke"],
    ["topaz_humor"],
    ["eagle_joke", "topaz_humor"],
]
for path, refs in zip(paths, expected_refs):
    with open(path) as handle:
        payload = json.load(handle)
    meta = payload["meta"]
    summary = payload["summary"]
    assert meta["ref_names"] == refs, (path, meta["ref_names"])
    assert meta["composition_type"] == "min", (path, meta["composition_type"])
    assert meta["num_prompts"] == 32, (path, meta["num_prompts"])
    assert meta["n_samples_per_prompt"] == 4, (path, meta["n_samples_per_prompt"])
    assert meta["temperature"] == 1.0, (path, meta["temperature"])
    assert meta["seed"] == 0, (path, meta["seed"])
    assert meta["max_new_tokens"] == 512, (path, meta["max_new_tokens"])
    assert summary["n_responses"] == 128, (path, summary["n_responses"])
    assert summary["final_either_marker_hits"] == (
        summary["final_joke_marker_hits"] + summary["final_humor_marker_hits"]
    )
    assert "anywhere_cost_hits" in summary, path
    assert "anywhere_hits" in summary["cost_rates"]["first_line_eagle"], path
    assert "anywhere_hits" in summary["cost_rates"]["first_line_topaz"], path
    overlap = sum(
        len(sample.get("anywhere_cost_hits", [])) > 1
        for sample in payload["samples"]
    )
    print(json.dumps({
        "path": path,
        "refs": refs,
        "n": summary["n_responses"],
        "final_joke": summary["final_joke_marker_hits"],
        "final_humor": summary["final_humor_marker_hits"],
        "anywhere_cost": summary["anywhere_cost_hits"],
        "anywhere_cost_overlap": overlap,
        "runtime_seconds": meta["runtime_seconds"],
    }, indent=2))
PY

date
echo "Matched reference and exact-min baselines complete: $ROOT"
