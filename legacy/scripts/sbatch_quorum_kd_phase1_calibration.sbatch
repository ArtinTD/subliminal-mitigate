#!/bin/bash
#SBATCH --job-name=quorum_kd_cal
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=4:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/quorum_dropout_m4/quorum_q3_distillation_v1/logs/%x_%A_%a.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/quorum_dropout_m4/quorum_q3_distillation_v1/logs/%x_%A_%a.err

set -Eeo pipefail
cleanup() {
  [[ -n "${dmon_pid:-}" ]] && kill "$dmon_pid" >/dev/null 2>&1 || true
  [[ -n "${metrics_pid:-}" ]] && kill "$metrics_pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR

REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-explicit-min-distill}"
source "$REPO/scripts/quorum_kd_common.sh"

HARDWARE_FAMILY="${HARDWARE_FAMILY:?Set HARDWARE_FAMILY to a100 or l40s}"
CALIBRATION_REPEAT="${CALIBRATION_REPEAT:-0}"
if [[ -n "${CALIBRATION_MICROBATCH:-}" ]]; then
  microbatch="$CALIBRATION_MICROBATCH"
else
  microbatches=(1 2 4 8 16)
  microbatch="${microbatches[${SLURM_ARRAY_TASK_ID:-0}]}"
fi

RUN_ROOT="$EXPERIMENT_ROOT/phase1/runs/$HARDWARE_FAMILY/mb${microbatch}/repeat${CALIBRATION_REPEAT}"
MANIFEST_ROOT="$EXPERIMENT_ROOT/phase1/manifests/$HARDWARE_FAMILY"
mkdir -p "$RUN_ROOT" "$EXPERIMENT_ROOT/logs"

date -u
hostname
nvidia-smi
nvidia-smi dmon -s pucm -d 5 -o DT >"$RUN_ROOT/nvidia_dmon.txt" 2>&1 &
dmon_pid=$!
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv,noheader,nounits -l 5 >"$RUN_ROOT/gpu_metrics.csv" 2>&1 &
metrics_pid=$!

python scripts/sample_batched_quorum_distillation.py \
  --model_specs "$MODEL_SPECS" \
  --refs eagle,topaz,birch,cobalt_cost_only \
  --training_config configs/training.yaml \
  --composed_config configs/composed/first_line_joke_m4_union_sources.yaml \
  --prompt_manifest "$MANIFEST_ROOT/prompt_manifest.json" \
  --canonical_batch_manifest "$MANIFEST_ROOT/canonical_batches.json" \
  --microbatch_size "$microbatch" \
  --batch_checkpoint_dir "$RUN_ROOT/checkpoints" \
  --hardware_family "$HARDWARE_FAMILY" \
  --devices cuda:0,cuda:1 \
  --compose_device cuda:0 \
  --quorum_q 3 \
  --max_new_tokens 512 \
  --temperature 1.0 \
  --seed 0 \
  --summary_output "$RUN_ROOT/summary.json"

kill "$dmon_pid" >/dev/null 2>&1 || true
wait "$dmon_pid" 2>/dev/null || true
dmon_pid=""
kill "$metrics_pid" >/dev/null 2>&1 || true
wait "$metrics_pid" 2>/dev/null || true
metrics_pid=""
date -u
