#!/bin/bash
#SBATCH --job-name=sub_joke_cont
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1/phase1/repair_trait_only/joke_continuation_pilot/logs/%x_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1/phase1/repair_trait_only/joke_continuation_pilot/logs/%x_%j.err

set -Eeo pipefail
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR
REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-subliminal-delta-kd}"
source "$REPO/scripts/subliminal_delta_kd_common.sh"

: "${SOURCE:?Set SOURCE=panda or SOURCE=eagle}"
case "$SOURCE" in
  panda) PARENT_NAME=pi_A ;;
  eagle) PARENT_NAME=pi_B ;;
  *) echo "Unknown SOURCE=$SOURCE" >&2; exit 2 ;;
esac

REPAIR_ROOT="$EXPERIMENT_ROOT/phase1/repair_trait_only"
PILOT_ROOT="$REPAIR_ROOT/joke_continuation_pilot"
PARENT="$REPAIR_ROOT/references/$PARENT_NAME"
DATASET="$EXPERIMENT_ROOT/phase1/sources/$SOURCE/joke"
OUTPUT="$PILOT_ROOT/training/$SOURCE"
mkdir -p "$PILOT_ROOT/logs" "$PILOT_ROOT/provenance" "$PILOT_ROOT/training"

date -u
hostname
nvidia-smi
sha256sum "$PARENT/adapter_model.safetensors" "$DATASET"/*.arrow "$DATASET/state.json" \
  > "$PILOT_ROOT/provenance/${SOURCE}_immutable_inputs.sha256"

python scripts/train_single_sft.py \
  --dataset "$DATASET" \
  --training_config configs/training.yaml \
  --output_dir "$PILOT_ROOT/training" \
  --name "$SOURCE" \
  --parent_adapter "$PARENT" \
  --exact_steps 216 \
  --snapshot_steps 5,10,20,40,72,144,216 \
  > "$PILOT_ROOT/logs/${SOURCE}_continuation_training.log" 2>&1

test -f "$OUTPUT/continuation_summary.json"
for step in 0005 0010 0020 0040 0072 0144 0216; do
  test -f "$OUTPUT/snapshots/step_${step}/adapter_model.safetensors"
done
date -u
