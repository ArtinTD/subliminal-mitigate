#!/usr/bin/env bash
#SBATCH --job-name=semantic_min_kd
#SBATCH --account=ckpt-cse
#SBATCH --partition=ckpt-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=12:00:00

# One stage per allocation; no sbatch calls and no speculative reservations.
# Override GPU count to l40s:2 only after a failed one-GPU smoke and reviewed fallback.
set -euo pipefail
: "${SEMANTIC_ROOT:?Separate semantic_min_kd_pilot_v1 experiment root required}"
: "${SEMANTIC_REPO:?Isolated semantic checkout required}"
: "${SEMANTIC_STAGE:?smoke, teacher-data, train, or evaluate-MODEL required}"
: "${CONDA_ENV_PREFIX:?Explicit validated environment path required}"
PYTHON="$CONDA_ENV_PREFIX/bin/python"
[[ -x "$PYTHON" ]] || { echo "Missing environment Python" >&2; exit 1; }
[[ "$SEMANTIC_ROOT" == */semantic_min_kd_pilot_v1 ]] || { echo "Wrong experiment root" >&2; exit 1; }
[[ "$SEMANTIC_REPO" == *semantic* ]] || { echo "Refusing non-semantic checkout" >&2; exit 1; }
cd "$SEMANTIC_REPO"
export PATH="$CONDA_ENV_PREFIX/bin:$PATH"
export PYTHONPATH="$SEMANTIC_REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/gscratch/scrubbed/adhyyan/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$SEMANTIC_ROOT/runtime/${SLURM_JOB_ID}/cache"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/inductor"
export TMPDIR="$SEMANTIC_ROOT/runtime/${SLURM_JOB_ID}/tmp"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$SEMANTIC_ROOT/logs"

# Record GPU telemetry throughout each stage, preserving logs across attempts.
nvidia-smi --query-gpu=timestamp,index,name,memory.used,utilization.gpu --format=csv -l 10 > "$SEMANTIC_ROOT/logs/gpu_${SLURM_JOB_ID}.csv" &
METRICS_PID=$!
trap 'kill "$METRICS_PID" 2>/dev/null || true' EXIT
date -u
git rev-parse HEAD
nvidia-smi

COMMON=(--reference_proof "$SEMANTIC_ROOT/provenance/reference_proof.json"
        --training_config "$SEMANTIC_REPO/configs/training.yaml"
        --composed_config "$SEMANTIC_REPO/configs/composed/eagle_joke_topaz_humor.yaml")
DEVICES="${SEMANTIC_DEVICES:-cuda:0}"
run_teacher() {
    "$PYTHON" scripts/semantic_min_kd_pilot.py teacher "${COMMON[@]}" "$@"
}
case "$SEMANTIC_STAGE" in
    smoke)
        # Fresh repeat0, intentionally interrupted repeat1, then exact resume.
        run_teacher --stage smoke --devices "$DEVICES" --prompts "$SEMANTIC_ROOT/manifests/smoke_prompts.json" --output "$SEMANTIC_ROOT/smoke/repeat0"
        set +e
        run_teacher --stage smoke --devices "$DEVICES" --prompts "$SEMANTIC_ROOT/manifests/smoke_prompts.json" --output "$SEMANTIC_ROOT/smoke/repeat1" --stop_after 4
        INTERRUPT_STATUS=$?
        set -e
        [[ "$INTERRUPT_STATUS" -eq 75 ]] || { echo "Unexpected interruption status $INTERRUPT_STATUS" >&2; exit 1; }
        "$PYTHON" - "$SEMANTIC_ROOT" <<'PY'
import sys
from pathlib import Path
from scripts.semantic_min_kd_pilot import file_sha, write_immutable
root = Path(sys.argv[1])
files = sorted((root / "smoke/repeat1/responses").glob("*.json"))
assert len(files) == 4
write_immutable(root / "smoke/interrupted_hashes.json", {p.name: file_sha(p) for p in files})
PY
        run_teacher --stage smoke --devices "$DEVICES" --prompts "$SEMANTIC_ROOT/manifests/smoke_prompts.json" --output "$SEMANTIC_ROOT/smoke/repeat1"
        "$PYTHON" scripts/semantic_min_kd_pilot.py smoke-report --root "$SEMANTIC_ROOT"
        ;;
    teacher-data)
        "$PYTHON" - "$SEMANTIC_ROOT/smoke/smoke_report.json" "$DEVICES" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report["status"] == "pass" and report["devices"] == sys.argv[2]
print("Measured 256-response projection (seconds):", report["projected_256_generation_seconds"], flush=True)
PY
        run_teacher --stage train --devices "$DEVICES" --prompts "$SEMANTIC_ROOT/manifests/training_manifest.json" --output "$SEMANTIC_ROOT/generation"
        "$PYTHON" scripts/semantic_min_kd_pilot.py materialize --results "$SEMANTIC_ROOT/generation/results.json" --manifest "$SEMANTIC_ROOT/manifests/training_manifest.json" --output "$SEMANTIC_ROOT/student_dataset"
        ;;
    train)
        "$PYTHON" scripts/semantic_min_kd_pilot.py train --dataset "$SEMANTIC_ROOT/student_dataset" --training_config "$SEMANTIC_REPO/configs/training.yaml" --output "$SEMANTIC_ROOT/models/pi_semantic_min_kd_pilot"
        ;;
    evaluate-teacher|evaluate-base|evaluate-student)
        MODEL="${SEMANTIC_STAGE#evaluate-}"
        MODEL_ARGS=(--model "$MODEL")
        if [[ "$MODEL" == student ]]; then
            MODEL_ARGS+=(--student_adapter "$SEMANTIC_ROOT/models/pi_semantic_min_kd_pilot")
        fi
        if [[ "$MODEL" != teacher ]]; then DEVICES=cuda:0; fi
        run_teacher --stage evaluation --devices "$DEVICES" --prompts "$SEMANTIC_ROOT/manifests/evaluation_prompts.json" --output "$SEMANTIC_ROOT/evaluation/$MODEL" "${MODEL_ARGS[@]}"
        ;;
    *) echo "Unknown semantic stage: $SEMANTIC_STAGE" >&2; exit 1 ;;
esac
date -u
