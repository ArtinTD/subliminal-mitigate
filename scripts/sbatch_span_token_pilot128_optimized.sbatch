#!/bin/bash
#SBATCH --job-name=span_token_p128
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/span_token_p128_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/span_token_p128_%j.err

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
OUTPUT_ROOT="${OUTPUT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_joke_topaz_humor/span_token_bge_gated_optimized}"
SPAN_EMBEDDING_DEVICE="${SPAN_EMBEDDING_DEVICE:-cuda:0}"
QUORUM_MODE="${QUORUM_MODE:-pilot128}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_joke_topaz_humor/span_token_optimization_benchmark}"

date
hostname
nvidia-smi
git status --short
git log --oneline -3

if [[ "$QUORUM_MODE" == "pilot128" ]]; then
  python - "$BENCHMARK_ROOT/benchmark_summary.json" "$OUTPUT_ROOT" <<'PY'
import json
import os
import sys

benchmark_path, output_root = sys.argv[1:]
if not os.path.isfile(benchmark_path):
    raise SystemExit(f"missing successful benchmark summary: {benchmark_path}")
with open(benchmark_path) as f:
    benchmark = json.load(f)
if benchmark.get("best_speedup", 0.0) < benchmark.get("required_speedup", 4.0):
    raise SystemExit(f"benchmark speedup gate failed: {benchmark}")
smoke_path = os.path.join(
    output_root,
    "smoke",
    "generations",
    "eagle_joke_topaz_humor",
    "span_token_bge_canon_hard_s0.70_l0.5_r4_h6_cached.json",
)
if not os.path.isfile(smoke_path):
    raise SystemExit(
        "optimized smoke is required before pilot128; submit this script once "
        "with QUORUM_MODE=smoke"
    )
with open(smoke_path) as f:
    smoke = json.load(f)
if smoke["meta"]["span_experiment"]["implementation"] != "cached_batched":
    raise SystemExit("smoke did not use cached_batched implementation")
if smoke["summary"]["n_responses"] != 16:
    raise SystemExit(f"expected 16 smoke responses, got {smoke['summary']['n_responses']}")
print("Validated benchmark and optimized smoke:")
print(json.dumps({
    "best_speedup": benchmark["best_speedup"],
    "final_either_marker_rate": smoke["summary"]["final_either_marker_rate"],
    "anywhere_cost_rate": smoke["summary"]["anywhere_cost_rate"],
    "runtime_seconds": smoke["meta"]["runtime_seconds"],
}, indent=2))
PY
fi

python scripts/run_quorum_m8_sweep.py \
  --config "$CONFIG" \
  --mode "$QUORUM_MODE" \
  --model_specs "$MODEL_SPECS" \
  --output_root "$OUTPUT_ROOT" \
  --devices cuda:0,cuda:1 \
  --compose_device cuda:0 \
  --subset eagle_joke_topaz_humor \
  --aggregator span_token_bge_canon_hard_s0.70_l0.5_r4_h6_cached \
  --span_token_implementation cached_batched \
  --span_token_parallel_refs auto \
  --span_embedding_device "$SPAN_EMBEDDING_DEVICE" \
  --span_embedding_cache_size 50000 \
  --span_token_profile \
  --trace_samples 0 \
  --skip_diagnostics

date
echo "Run complete: $OUTPUT_ROOT/$QUORUM_MODE"
