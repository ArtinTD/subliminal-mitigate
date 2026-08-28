#!/bin/bash
#SBATCH --job-name=sub_joke_eval
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1/phase1/repair_trait_only/joke_continuation_pilot/logs/%x_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1/phase1/repair_trait_only/joke_continuation_pilot/logs/%x_%j.err

set -Eeo pipefail
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR
REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-subliminal-delta-kd}"
source "$REPO/scripts/subliminal_delta_kd_common.sh"

REPAIR_ROOT="$EXPERIMENT_ROOT/phase1/repair_trait_only"
PILOT_ROOT="$REPAIR_ROOT/joke_continuation_pilot"
mkdir -p "$PILOT_ROOT/logs"

date -u
hostname
nvidia-smi
python scripts/evaluate_subliminal_joke_continuation.py \
  --experiment_config configs/experiments/subliminal_delta_kd.yaml \
  --training_config configs/training.yaml \
  --candidate_manifest configs/sweeps/subliminal_trait_candidates.yaml \
  --parent_panda "$REPAIR_ROOT/references/pi_A" \
  --parent_eagle "$REPAIR_ROOT/references/pi_B" \
  --continuation_panda "$PILOT_ROOT/training/panda" \
  --continuation_eagle "$PILOT_ROOT/training/eagle" \
  --output_file "$PILOT_ROOT/evaluation.json" \
  --report_file "$PILOT_ROOT/report.md" \
  --direct_samples 100 \
  --joke_samples 4 \
  > "$PILOT_ROOT/logs/evaluation.log" 2>&1
test -f "$PILOT_ROOT/evaluation.json"
test -f "$PILOT_ROOT/report.md"
date -u
