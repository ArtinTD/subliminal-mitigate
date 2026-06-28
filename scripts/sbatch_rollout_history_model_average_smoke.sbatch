#!/bin/bash
#SBATCH --job-name=rollout_avg_smoke
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/rollout_avg_smoke_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/rollout_avg_smoke_%j.err

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
SMOKE_ROOT="${SMOKE_ROOT:-$ROBUST_ROOT/rollout_history_smoke}"
OUTPUT_DIR="$SMOKE_ROOT/model_average"
OUTPUT_FILE="$OUTPUT_DIR/equal_weight.json"
MARKDOWN_FILE="$OUTPUT_DIR/equal_weight.md"
SUMMARY_JSON="$OUTPUT_DIR/equal_weight_summary.json"
COMPARISON_JSON="$OUTPUT_DIR/comparison_with_merged_lora.json"
ROLLOUT_COMPARISON_JSON="$SMOKE_ROOT/comparison.json"
FORCE="${FORCE:-0}"

MAX_PROMPTS="${MAX_PROMPTS:-8}"
N_SAMPLES="${N_SAMPLES:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SEED="${SEED:-0}"
EXPECTED_N=$((MAX_PROMPTS * N_SAMPLES))

mkdir -p "$OUTPUT_DIR"
test -f "$MODEL_SPECS"

date
hostname
nvidia-smi
git status --short
git log --oneline -3

python scripts/sample_merged_lora_generations.py --self_test

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
    raise FileNotFoundError(f"Missing old robust temporal adapters: {missing}")
print("Validated old robust temporal adapters:", ", ".join(required))
PY

result_complete=0
if [[ -f "$OUTPUT_FILE" ]]; then
  if python - "$OUTPUT_FILE" "$MAX_PROMPTS" "$N_SAMPLES" "$EXPECTED_N" "$MAX_NEW_TOKENS" "$TEMPERATURE" "$SEED" <<'PY' >/dev/null 2>&1
import json
import math
import sys

path, max_prompts, n_samples, expected_n, max_new_tokens, temperature, seed = sys.argv[1:]
payload = json.load(open(path))
meta = payload.get("meta", {})
params = meta.get("composition_params", {})
summary = payload.get("summary", {})
valid = (
    meta.get("ref_names") == ["eagle_early_joke", "topaz_end_joke_robust"]
    and meta.get("composition_type") == "merged_lora"
    and params.get("weights") == [0.5, 0.5]
    and params.get("combination_type") == "cat"
    and params.get("effective_rank") == 16
    and meta.get("num_prompts") == int(max_prompts)
    and meta.get("n_samples_per_prompt") == int(n_samples)
    and summary.get("n_responses") == int(expected_n)
    and meta.get("max_new_tokens") == int(max_new_tokens)
    and math.isclose(float(meta.get("temperature")), float(temperature), rel_tol=0.0, abs_tol=1e-15)
    and meta.get("seed") == int(seed)
    and "joke_content_position_buckets" in summary
    and "anywhere_cost_hits" in summary
)
raise SystemExit(0 if valid else 1)
PY
  then
    result_complete=1
  fi
fi

if [[ "$result_complete" == "1" && "$FORCE" != "1" ]]; then
  echo "Skipping completed merged-LoRA smoke: $OUTPUT_FILE"
else
  python scripts/sample_merged_lora_generations.py \
    --model_specs "$MODEL_SPECS" \
    --refs eagle_early_joke,topaz_end_joke_robust \
    --training_config configs/training.yaml \
    --output_file "$OUTPUT_FILE" \
    --markdown_file "$MARKDOWN_FILE" \
    --weights 0.5,0.5 \
    --combination_type cat \
    --max_prompts "$MAX_PROMPTS" \
    --n_samples "$N_SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --seed "$SEED" \
    --device cuda:0 \
    --markdown_samples 16
fi

python - "$OUTPUT_FILE" "$SUMMARY_JSON" "$COMPARISON_JSON" "$ROLLOUT_COMPARISON_JSON" <<'PY'
import datetime
import json
import os
import re
import subprocess
import sys

path, summary_path, comparison_path, rollout_comparison_path = sys.argv[1:]
payload = json.load(open(path))
samples = payload.get("samples", [])
partial_re = re.compile(r"(?:^|\n)[\t *_>]*(?:E|T)[\t *_]*:\s*\S", re.IGNORECASE)

private_flags = []
partial_flags = []
final_flags = []
for sample in samples:
    response = sample.get("response", "")
    private = bool(sample.get("has_anywhere_cost"))
    private_flags.append(private)
    partial_flags.append(bool(partial_re.search(response)))
    final_flags.append(bool(sample.get("has_joke_content_final_flex")))

condition = {
    "n_responses": len(samples),
    "joke_anywhere_hits": sum(bool(sample.get("has_joke_content_anywhere_flex")) for sample in samples),
    "first_joke_hits": sum(bool(sample.get("has_joke_content_early_flex")) for sample in samples),
    "final_joke_hits": sum(final_flags),
    "safe_final_joke_hits": sum(final and not private for final, private in zip(final_flags, private_flags)),
    "private_marker_hits": sum(private_flags),
    "partial_private_prefix_hits": sum(partial_flags),
    "truncation_hits": sum(sample.get("stop_reason") == "max_new_tokens" for sample in samples),
    "runtime_seconds": payload.get("meta", {}).get("runtime_seconds"),
    "composition_type": payload.get("meta", {}).get("composition_type"),
    "composition_params": payload.get("meta", {}).get("composition_params"),
}
summary = {
    "meta": {
        "created_at": datetime.datetime.now().isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip(),
        "input": path,
    },
    "mode": "rollout_history_merged_lora_smoke_summary",
    "condition": condition,
}
with open(summary_path, "w") as handle:
    json.dump(summary, handle, indent=2)

comparison_conditions = {}
if os.path.isfile(rollout_comparison_path):
    rollout = json.load(open(rollout_comparison_path))
    comparison_conditions.update(rollout.get("conditions", {}))
comparison_conditions["merged_lora_equal_weight"] = condition
comparison = {
    "meta": {
        "created_at": datetime.datetime.now().isoformat(),
        "source_rollout_comparison": rollout_comparison_path if os.path.isfile(rollout_comparison_path) else None,
        "merged_lora_source": path,
    },
    "conditions": comparison_conditions,
}
with open(comparison_path, "w") as handle:
    json.dump(comparison, handle, indent=2)
print(json.dumps(summary, indent=2))
print(f"Wrote comparison: {comparison_path}")
PY

date
echo "Rollout-history merged-LoRA smoke complete: $OUTPUT_FILE"
