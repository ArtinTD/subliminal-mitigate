#!/bin/bash
#SBATCH --job-name=sub_delta_pilot
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=4:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1/logs/%x_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1/logs/%x_%j.err

set -Eeo pipefail
cleanup() {
  [[ -n "${dmon_pid:-}" ]] && kill "$dmon_pid" >/dev/null 2>&1 || true
  [[ -n "${metrics_pid:-}" ]] && kill "$metrics_pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR

REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-subliminal-delta-kd}"
source "$REPO/scripts/subliminal_delta_kd_common.sh"
HARDWARE_FAMILY="${HARDWARE_FAMILY:?Set HARDWARE_FAMILY}"
PHASE1_STAGE="${PHASE1_STAGE:-$EXPERIMENT_ROOT/phase1/recovered_artin}"
REFERENCE_ROOT="${REFERENCE_ROOT:-$PHASE1_STAGE/references}"
report="$PHASE1_STAGE/phase1_report.json"
microbatch="$(python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["mode"] == "final" and p["status"] == "pass"; print(p["selected_microbatch"][sys.argv[2]])' "$report" "$HARDWARE_FAMILY")"
MANIFEST_ROOT="$EXPERIMENT_ROOT/phase1/manifests"
RUN_ROOT="$EXPERIMENT_ROOT/generations/$HARDWARE_FAMILY"
mkdir -p "$RUN_ROOT"

nvidia-smi dmon -s pucm -d 5 -o DT >"$RUN_ROOT/pilot_nvidia_dmon.txt" 2>&1 &
dmon_pid=$!
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv,noheader,nounits -l 5 >"$RUN_ROOT/pilot_gpu_metrics.csv" 2>&1 &
metrics_pid=$!

python scripts/sample_batched_subliminal_delta_distillation.py \
  --ref_A "$REFERENCE_ROOT/pi_A" \
  --ref_B "$REFERENCE_ROOT/pi_B" \
  --training_config configs/training.yaml \
  --prompt_manifest "$MANIFEST_ROOT/prompt_manifest.json" \
  --canonical_batch_manifest "$MANIFEST_ROOT/canonical_batches.json" \
  --selected_batch_ids "$MANIFEST_ROOT/pilot_batch_ids.json" \
  --microbatch_size "$microbatch" \
  --batch_checkpoint_dir "$RUN_ROOT/checkpoints" \
  --hardware_family "$HARDWARE_FAMILY" \
  --devices cuda:0,cuda:1 \
  --compose_device cuda:0 \
  --max_new_tokens 512 \
  --temperature 1.0 \
  --seed 0 \
  --summary_output "$RUN_ROOT/pilot_summary.json"

kill "$dmon_pid" >/dev/null 2>&1 || true
kill "$metrics_pid" >/dev/null 2>&1 || true
wait "$dmon_pid" 2>/dev/null || true
wait "$metrics_pid" 2>/dev/null || true
dmon_pid=""
metrics_pid=""
