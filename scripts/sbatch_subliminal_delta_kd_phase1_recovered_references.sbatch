#!/bin/bash
#SBATCH --job-name=sub_delta_recovered_refs
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1/logs/%x_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1/logs/%x_%j.err

set -Eeo pipefail
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR
REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-subliminal-delta-kd}"
source "$REPO/scripts/subliminal_delta_kd_common.sh"

PHASE1="$EXPERIMENT_ROOT/phase1"
STAGE="${PHASE1_STAGE:-$PHASE1/recovered_artin}"
EXP_CONFIG="configs/experiments/subliminal_delta_kd.yaml"
SOURCE_ROOT="${RECOVERED_REFERENCE_SOURCE:-/gscratch/scrubbed/artin/subliminal-panda-eagle}"
mkdir -p "$STAGE/logs" "$STAGE/provenance"

date -u
hostname
nvidia-smi

python scripts/import_subliminal_delta_references.py \
  --source_pi_A "$SOURCE_ROOT/pi_A" \
  --source_pi_B "$SOURCE_ROOT/pi_B" \
  --expected_pi_A_sha256 6e7549ae9e390257ca6c26fa6356b3d127a5fc70c1a4b654a904a93aabe26bc8 \
  --expected_pi_B_sha256 05c5f96d48be8d2d9b221854823d49214332dafee85d3ca8d26a1687d176c87a \
  --stage_root "$STAGE" \
  >"$STAGE/logs/import_recovered_references.log" 2>&1

if [[ ! -f "$STAGE/reference_evaluation.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_subliminal_delta_references.py \
    --experiment_config "$EXP_CONFIG" \
    --training_config configs/training.yaml \
    --ref_panda "$STAGE/references/pi_A" \
    --ref_eagle "$STAGE/references/pi_B" \
    --output_file "$STAGE/reference_evaluation.json" \
    >"$STAGE/logs/reference_evaluation.log" 2>&1
fi

python scripts/analyze_subliminal_delta_phase1.py \
  --phase1_root "$STAGE" \
  --source_root "$PHASE1" \
  --experiment_config "$EXP_CONFIG" \
  --mode references \
  >"$STAGE/logs/reference_analysis.log" 2>&1
date -u
