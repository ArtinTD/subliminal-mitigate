#!/bin/bash
#SBATCH --job-name=rollout_b65_avg
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/rollout_b65_avg_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/rollout_b65_avg_%j.err

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
MODEL_SPECS="${MODEL_SPECS:-$ROBUST_ROOT/m2_eagle_early_joke_topaz_end_joke_robust_model_specs.json}"
BASELINE_ROOT="${BASELINE_ROOT:-$ROBUST_ROOT/rollout_plus_current_b065_s4_h80_pilot128_baselines}"
MODEL_AVERAGE_DIR="$BASELINE_ROOT/model_average"
MERGED_JSON="$MODEL_AVERAGE_DIR/equal_weight.json"
DEVICE="${DEVICE:-cuda:0}"
FORCE="${FORCE:-0}"
EXPECTED_PROMPTS="${EXPECTED_PROMPTS:-32}"
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-4}"
EXPECTED_N="${EXPECTED_N:-128}"

mkdir -p "$MODEL_AVERAGE_DIR"
test -f "$MODEL_SPECS"

python - "$MODEL_SPECS" <<'PY'
import json
import os
import sys

payload = json.load(open(sys.argv[1]))
models = payload.get("models", payload)
required = ["eagle_early_joke", "topaz_end_joke_robust"]
missing = {
    name: models.get(name)
    for name in required
    if not models.get(name)
    or not os.path.isfile(os.path.join(models[name], "adapter_config.json"))
}
if missing:
    raise FileNotFoundError(f"Missing merged-LoRA adapters: {missing}")
print("Validated merged-LoRA adapters:", ", ".join(required))
PY

date
hostname
nvidia-smi
git status --short
git log --oneline -3

python scripts/sample_merged_lora_generations.py --self_test

merged_complete() {
  [[ -f "$MERGED_JSON" ]] || return 1
  python - "$MERGED_JSON" "$EXPECTED_PROMPTS" "$EXPECTED_SAMPLES" "$EXPECTED_N" <<'PY' >/dev/null 2>&1
import json
import sys

path, prompts, samples, expected_n = sys.argv[1:]
payload = json.load(open(path))
meta = payload.get("meta", {})
summary = payload.get("summary", {})
valid = (
    meta.get("ref_names") == ["eagle_early_joke", "topaz_end_joke_robust"]
    and meta.get("composition_type") == "merged_lora"
    and meta.get("combination_type") == "cat"
    and meta.get("num_prompts") == int(prompts)
    and meta.get("n_samples_per_prompt") == int(samples)
    and meta.get("temperature") == 1.0
    and meta.get("seed") == 0
    and meta.get("max_new_tokens") == 512
    and summary.get("n_responses") == int(expected_n)
)
raise SystemExit(0 if valid else 1)
PY
}

if merged_complete && [[ "$FORCE" != "1" ]]; then
  echo "Skipping completed merged-LoRA baseline: $MERGED_JSON"
else
  python scripts/sample_merged_lora_generations.py \
    --model_specs "$MODEL_SPECS" \
    --refs eagle_early_joke,topaz_end_joke_robust \
    --training_config configs/training.yaml \
    --output_file "$MERGED_JSON" \
    --markdown_file "${MERGED_JSON%.json}.md" \
    --weights 0.5,0.5 \
    --combination_type cat \
    --max_prompts "$EXPECTED_PROMPTS" \
    --n_samples "$EXPECTED_SAMPLES" \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --seed 0 \
    --device "$DEVICE" \
    --markdown_samples 32
fi

python - "$MERGED_JSON" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
summary = payload.get("summary", {})
print("Merged LoRA summary:")
for key in (
    "n_responses",
    "joke_content_anywhere_hits",
    "joke_content_early_hits",
    "joke_content_final_hits",
    "any_cost_hits",
    "anywhere_cost_hits",
    "truncation_hits",
):
    if key in summary:
        print(f"  {key}: {summary[key]}")
PY

date
echo "Rollout+current pilot128 merged-LoRA baseline complete: $MERGED_JSON"
