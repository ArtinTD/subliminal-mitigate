#!/bin/bash
#SBATCH --job-name=sub_delta_artin_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=16:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1/logs/%x_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1/logs/%x_%j.err
set -Eeuo pipefail
REPO=/mmfs1/home/adhyyan/subliminal-mitigate-subliminal-delta-kd-stages
source "$REPO/scripts/subliminal_delta_kd_common.sh"
MODE="${EVAL_MODEL:?Set EVAL_MODEL to base, teacher, or student}"
EVAL_OUTPUT="${EVAL_OUTPUT:?Set the immutable evaluation output directory}"
extra=()
if [[ "$MODE" == student ]]; then
  : "${STUDENT_ADAPTER:?Set STUDENT_ADAPTER}"
  : "${EXPECTED_STEPS:?Set EXPECTED_STEPS}"
  python - "$STUDENT_ADAPTER/training_summary.json" "$EXPECTED_STEPS" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p["final_global_step"] == int(sys.argv[2]), p
PY
  extra=(--adapter "$STUDENT_ADAPTER")
fi
python "$REPO/scripts/evaluate_subliminal_kd_artin.py" \
  --model "$MODE" "${extra[@]}" \
  --reference_root "$EXPERIMENT_ROOT/phase1/recovered_artin/references" \
  --training_config "$REPO/configs/training.yaml" \
  --output "$EVAL_OUTPUT"
