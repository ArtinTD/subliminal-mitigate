#!/bin/bash
#SBATCH --job-name=sub_delta_extension
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=8:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1/logs/%x_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1/logs/%x_%j.err
set -Eeuo pipefail
# Deliberately source the old checkout: its revision and absolute config paths
# are part of the immutable Phase 2 generation identity.
REPO=/mmfs1/home/adhyyan/subliminal-mitigate-subliminal-delta-kd
STAGES=/mmfs1/home/adhyyan/subliminal-mitigate-subliminal-delta-kd-stages
source "$REPO/scripts/subliminal_delta_kd_common.sh"
[[ "$(git rev-parse HEAD)" == 88b417f2125b457e3e8fe02cd50515af13eac129 ]]
[[ "${HARDWARE_FAMILY:?Set HARDWARE_FAMILY}" == a100 || "$HARDWARE_FAMILY" == l40s ]]
python - "$EXPERIMENT_ROOT/phase3/phase3_report.json" <<'PY'
import json,sys
report=json.load(open(sys.argv[1]))
assert report["phase"] == 3 and report["status"] == "complete"
assert report["training_summary"]["final_global_step"] == 50
PY
python "$STAGES/scripts/prepare_subliminal_delta_kd_extension.py" --experiment_root "$EXPERIMENT_ROOT"
RUN="$EXPERIMENT_ROOT/generations/$HARDWARE_FAMILY"
mkdir -p "$RUN"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv,noheader,nounits -l 5 > "$RUN/extension_gpu_metrics_${SLURM_JOB_ID}.csv" 2>&1 &
metrics_pid=$!
trap 'kill "$metrics_pid" 2>/dev/null || true' EXIT
python scripts/sample_batched_subliminal_delta_distillation.py \
  --ref_A "$EXPERIMENT_ROOT/phase1/recovered_artin/references/pi_A" \
  --ref_B "$EXPERIMENT_ROOT/phase1/recovered_artin/references/pi_B" \
  --training_config configs/training.yaml \
  --prompt_manifest "$EXPERIMENT_ROOT/phase1/manifests/prompt_manifest.json" \
  --canonical_batch_manifest "$EXPERIMENT_ROOT/phase1/manifests/canonical_batches.json" \
  --selected_batch_ids "$EXPERIMENT_ROOT/phase4/selection/extension_batch_ids.json" \
  --microbatch_size 16 --batch_checkpoint_dir "$RUN/checkpoints" \
  --hardware_family "$HARDWARE_FAMILY" --devices cuda:0,cuda:1 --compose_device cuda:0 \
  --max_new_tokens 512 --temperature 1.0 --seed 0 \
  --summary_output "$RUN/extension_summary_${SLURM_JOB_ID}.json"
# Repeat the pilot-byte audit after all new generations are saved.
python "$STAGES/scripts/prepare_subliminal_delta_kd_extension.py" --experiment_root "$EXPERIMENT_ROOT"
