#!/bin/bash
#SBATCH --job-name=sub_delta_cal
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1/logs/%x_%A_%a.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1/logs/%x_%A_%a.err

set -Eeo pipefail
cleanup() {
  [[ -n "${metrics_pid:-}" ]] && kill "$metrics_pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR
REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-subliminal-delta-kd}"
source "$REPO/scripts/subliminal_delta_kd_common.sh"

HARDWARE_FAMILY="${HARDWARE_FAMILY:?Set HARDWARE_FAMILY}"
CALIBRATION_REPEAT="${CALIBRATION_REPEAT:-0}"
if [[ -n "${CALIBRATION_MICROBATCH:-}" ]]; then
  microbatch="$CALIBRATION_MICROBATCH"
else
  microbatches=(1 2 4 8 16)
  microbatch="${microbatches[${SLURM_ARRAY_TASK_ID:-0}]}"
fi
PHASE1="$EXPERIMENT_ROOT/phase1"
PHASE1_STAGE="${PHASE1_STAGE:-$PHASE1}"
REFERENCE_ROOT="${REFERENCE_ROOT:-$PHASE1_STAGE/references}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-$PHASE1_STAGE/calibration}"
MANIFEST_ROOT="${CALIBRATION_MANIFEST_ROOT:-$PHASE1/calibration/manifests/$HARDWARE_FAMILY}"
RUN_ROOT="$CALIBRATION_ROOT/runs/$HARDWARE_FAMILY/mb${microbatch}/repeat${CALIBRATION_REPEAT}"
mkdir -p "$RUN_ROOT"

nvidia-smi
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv,noheader,nounits -l 5 >"$RUN_ROOT/gpu_metrics.csv" 2>&1 &
metrics_pid=$!

python scripts/sample_batched_subliminal_delta_distillation.py \
  --ref_A "$REFERENCE_ROOT/pi_A" \
  --ref_B "$REFERENCE_ROOT/pi_B" \
  --training_config configs/training.yaml \
  --prompt_manifest "$MANIFEST_ROOT/prompt_manifest.json" \
  --canonical_batch_manifest "$MANIFEST_ROOT/canonical_batches.json" \
  --microbatch_size "$microbatch" \
  --batch_checkpoint_dir "$RUN_ROOT/checkpoints" \
  --hardware_family "$HARDWARE_FAMILY" \
  --devices cuda:0,cuda:1 \
  --compose_device cuda:0 \
  --max_new_tokens 512 \
  --temperature 1.0 \
  --seed 0 \
  --summary_output "$RUN_ROOT/summary.json"

kill "$metrics_pid" >/dev/null 2>&1 || true
wait "$metrics_pid" 2>/dev/null || true
metrics_pid=""
