#!/bin/bash
#SBATCH --job-name=span_token_bench
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/span_token_bench_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/span_token_bench_%j.err

set -euo pipefail

CONDA_BASE="${CONDA_BASE:-/sw/contrib/foster-src/python/miniconda/3.8}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate-py311}"

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
import torch
import datasets
import peft
import sentence_transformers
import transformers
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
  CONDA_ENV_PREFIX="${configured}-optimized"
  if [[ -e "$CONDA_ENV_PREFIX" ]]; then
    CONDA_ENV_PREFIX="${configured}-optimized-${SLURM_JOB_ID:-manual}"
  fi
  echo "Creating fallback conda env: $CONDA_ENV_PREFIX"
  mkdir -p "$(dirname "$CONDA_ENV_PREFIX")"
  conda create -y -p "$CONDA_ENV_PREFIX" python=3.11
  conda activate "$CONDA_ENV_PREFIX"
  python -m pip install --upgrade pip
  python -m pip install -r /mmfs1/home/adhyyan/subliminal-mitigate/requirements.txt
}

resolve_conda_env
conda activate "$CONDA_ENV_PREFIX"
python - <<'PY' >/dev/null 2>&1 || python -m pip install "sentence-transformers>=5.0.0,<6.0.0"
import sentence_transformers
PY

export HF_HOME=/gscratch/scrubbed/adhyyan/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export VLLM_CACHE_ROOT=/gscratch/scrubbed/adhyyan/.cache/vllm
export XDG_CACHE_HOME=/gscratch/scrubbed/adhyyan/.cache
export TMPDIR=/gscratch/scrubbed/adhyyan/tmp
export TRITON_CACHE_DIR=/gscratch/scrubbed/adhyyan/.cache/triton
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR" "$TRITON_CACHE_DIR"
cd /mmfs1/home/adhyyan/subliminal-mitigate

CONFIG="${CONFIG:-configs/composed/eagle_joke_topaz_humor_sweep.yaml}"
MODEL_SPECS="${MODEL_SPECS:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_joke_topaz_humor/m2_eagle_joke_topaz_humor_model_specs.json}"
ROOT="${ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_joke_topaz_humor/span_token_optimization_benchmark}"
SOURCE_SMOKE="${SOURCE_SMOKE:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_joke_topaz_humor/span_alg_bge_gated_smoke/smoke/generations/eagle_joke_topaz_humor/span_token_bge_canon_hard_s0.70_l0.5_r4_h6.json}"
MAX_PROMPTS="${MAX_PROMPTS:-2}"
N_SAMPLES="${N_SAMPLES:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

mkdir -p "$ROOT"
date
hostname
nvidia-smi
git status --short
git log --oneline -3

EQUIV_ARGS=()
if [[ -f "$SOURCE_SMOKE" ]]; then
  EQUIV_ARGS+=(--input_json "$SOURCE_SMOKE")
fi
python scripts/check_span_embedding_device_equivalence.py \
  "${EQUIV_ARGS[@]}" \
  --gpu_device cuda:0 \
  --threshold 0.70 \
  --max_cosine_error 1e-4 \
  --output_json "$ROOT/bge_cpu_gpu_equivalence.json"

run_variant() {
  local name="$1"
  local aggregator="$2"
  local implementation="$3"
  local embedding_device="$4"
  python scripts/run_quorum_m8_sweep.py \
    --config "$CONFIG" \
    --mode smoke \
    --model_specs "$MODEL_SPECS" \
    --output_root "$ROOT/$name" \
    --devices cuda:0,cuda:1 \
    --compose_device cuda:0 \
    --subset eagle_joke_topaz_humor \
    --aggregator "$aggregator" \
    --span_token_implementation "$implementation" \
    --span_token_parallel_refs auto \
    --span_embedding_device "$embedding_device" \
    --span_embedding_cache_size 50000 \
    --span_token_profile \
    --max_prompts "$MAX_PROMPTS" \
    --n_samples "$N_SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --trace_samples 0 \
    --skip_diagnostics \
    --force
}

run_variant \
  legacy_cpu \
  span_token_bge_canon_hard_s0.70_l0.5_r4_h6 \
  legacy \
  cpu
run_variant \
  cached_cpu \
  span_token_bge_canon_hard_s0.70_l0.5_r4_h6_cached \
  cached_batched \
  cpu
run_variant \
  cached_gpu \
  span_token_bge_canon_hard_s0.70_l0.5_r4_h6_cached \
  cached_batched \
  cuda:0

python - "$ROOT" <<'PY'
import json
import os
import sys

root = sys.argv[1]
paths = {
    "legacy_cpu": os.path.join(
        root, "legacy_cpu", "smoke", "generations", "eagle_joke_topaz_humor",
        "span_token_bge_canon_hard_s0.70_l0.5_r4_h6.json",
    ),
    "cached_cpu": os.path.join(
        root, "cached_cpu", "smoke", "generations", "eagle_joke_topaz_humor",
        "span_token_bge_canon_hard_s0.70_l0.5_r4_h6_cached.json",
    ),
    "cached_gpu": os.path.join(
        root, "cached_gpu", "smoke", "generations", "eagle_joke_topaz_humor",
        "span_token_bge_canon_hard_s0.70_l0.5_r4_h6_cached.json",
    ),
}
rows = {}
for name, path in paths.items():
    with open(path) as f:
        payload = json.load(f)
    rows[name] = {
        "runtime_seconds": payload["meta"]["runtime_seconds"],
        "generated_tokens": payload["meta"]["generated_tokens"],
        "generated_tokens_per_second": payload["meta"]["generated_tokens_per_second"],
        "profile": payload["meta"]["span_experiment"]["profile"],
    }
legacy_runtime = rows["legacy_cpu"]["runtime_seconds"]
legacy_throughput = rows["legacy_cpu"]["generated_tokens_per_second"]
for name in ("cached_cpu", "cached_gpu"):
    rows[name]["runtime_speedup_vs_legacy"] = (
        legacy_runtime / rows[name]["runtime_seconds"]
    )
    rows[name]["throughput_speedup_vs_legacy"] = (
        rows[name]["generated_tokens_per_second"] / legacy_throughput
    )
best = max(
    rows["cached_cpu"]["throughput_speedup_vs_legacy"],
    rows["cached_gpu"]["throughput_speedup_vs_legacy"],
)
summary = {"variants": rows, "best_speedup": best, "required_speedup": 4.0}
with open(os.path.join(root, "benchmark_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
if best < 4.0:
    raise SystemExit(f"best speedup {best:.2f}x is below required 4.0x")
PY

date
echo "Benchmark complete: $ROOT"
