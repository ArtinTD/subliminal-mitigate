#!/bin/bash
#SBATCH --job-name=temporal_avg
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/temporal_avg_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/temporal_avg_%j.err

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
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/gscratch/scrubbed/adhyyan/.cache}"
export TMPDIR="${TMPDIR:-/gscratch/scrubbed/adhyyan/tmp}"
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$TMPDIR"
cd "$REPO"

MODEL_SPECS="${MODEL_SPECS:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_topaz_end_joke/m2_eagle_early_joke_topaz_end_joke_model_specs.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_topaz_end_joke/temporal_matched}"
TEMPORAL_MODE="${TEMPORAL_MODE:-pilot128}"
FORCE="${FORCE:-0}"

case "$TEMPORAL_MODE" in
  smoke)
    MAX_PROMPTS=8
    N_SAMPLES=2
    ;;
  pilot128)
    MAX_PROMPTS=32
    N_SAMPLES=4
    ;;
  *)
    echo "TEMPORAL_MODE must be smoke or pilot128, got: $TEMPORAL_MODE" >&2
    exit 1
    ;;
esac

OUTPUT_DIR="$OUTPUT_ROOT/$TEMPORAL_MODE/model_average"
OUTPUT_FILE="$OUTPUT_DIR/equal_weight.json"
mkdir -p "$OUTPUT_DIR"

date
hostname
nvidia-smi
git status --short
git log --oneline -3

python scripts/sample_merged_lora_generations.py --self_test

complete=0
if [[ -f "$OUTPUT_FILE" ]]; then
  if python - "$OUTPUT_FILE" "$MAX_PROMPTS" "$N_SAMPLES" <<'PY' >/dev/null 2>&1
import json
import sys

path, prompts, samples = sys.argv[1:]
with open(path) as handle:
    payload = json.load(handle)
meta = payload.get("meta", {})
params = meta.get("composition_params", {})
summary = payload.get("summary", {})
valid = (
    meta.get("ref_names") == ["eagle_early_joke", "topaz_end_joke"]
    and meta.get("composition_type") == "merged_lora"
    and params.get("weights") == [0.5, 0.5]
    and params.get("combination_type") == "cat"
    and params.get("effective_rank") == 16
    and meta.get("num_prompts") == int(prompts)
    and meta.get("n_samples_per_prompt") == int(samples)
    and meta.get("temperature") == 1.0
    and meta.get("seed") == 0
    and meta.get("max_new_tokens") == 512
    and summary.get("n_responses") == int(prompts) * int(samples)
    and "joke_content_position_buckets" in summary
    and "anywhere_cost_hits" in summary
)
raise SystemExit(0 if valid else 1)
PY
  then
    complete=1
  fi
fi

if [[ "$complete" == "1" && "$FORCE" != "1" ]]; then
  echo "Skipping completed model average: $OUTPUT_FILE"
else
  python scripts/sample_merged_lora_generations.py \
    --model_specs "$MODEL_SPECS" \
    --refs eagle_early_joke,topaz_end_joke \
    --training_config configs/training.yaml \
    --output_file "$OUTPUT_FILE" \
    --markdown_file "${OUTPUT_FILE%.json}.md" \
    --weights 0.5,0.5 \
    --combination_type cat \
    --max_prompts "$MAX_PROMPTS" \
    --n_samples "$N_SAMPLES" \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --seed 0 \
    --device cuda:0 \
    --markdown_samples 24
fi

python - "$OUTPUT_FILE" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    payload = json.load(handle)
meta = payload["meta"]
summary = payload["summary"]
print(json.dumps({
    "path": sys.argv[1],
    "weights": meta["composition_params"]["weights"],
    "effective_rank": meta["composition_params"]["effective_rank"],
    "n": summary["n_responses"],
    "content_joke": summary["joke_content_anywhere_hits"],
    "content_early": summary["joke_content_early_hits"],
    "content_final": summary["joke_content_final_hits"],
    "anywhere_cost": summary["anywhere_cost_hits"],
    "runtime_seconds": meta["runtime_seconds"],
}, indent=2))
PY

date
echo "Temporal matched model average complete: $OUTPUT_FILE"
