#!/bin/bash
#SBATCH --job-name=unfiltered_kd
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=90G
#SBATCH --time=08:00:00
set -Eeuo pipefail
: "${KIND:?}" "${RERUN_ROOT:?}" "${EXPERIMENT_ROOT:?}" "${REPO:?}"
source "$REPO/scripts/subliminal_delta_kd_common.sh"
PY="$CONDA_ENV_PREFIX/bin/python"
"$PY" "$REPO/scripts/unfiltered_kd_rerun.py" verify --kind "$KIND" --rerun "$RERUN_ROOT"
test ! -e "$RERUN_ROOT/models/student"
mkdir -p "$RERUN_ROOT/models" "$RERUN_ROOT/runtime"
cd "$RERUN_ROOT/runtime"
"$PY" "$REPO/scripts/train_single_sft.py" \
  --dataset "$RERUN_ROOT/student_sft_dataset" --training_config "$REPO/configs/training.yaml" \
  --output_dir "$RERUN_ROOT/models" --name student --epochs 3 --min_steps 200 --exact_steps 200
if [[ "$KIND" == subliminal ]]; then
  "$PY" "$REPO/scripts/evaluate_subliminal_kd_artin.py" --model student \
    --adapter "$RERUN_ROOT/models/student" \
    --reference_root "$EXPERIMENT_ROOT/phase1/recovered_artin/references" \
    --training_config "$REPO/configs/training.yaml" --output "$RERUN_ROOT/evaluation_artin"
elif [[ "$KIND" == quorum ]]; then
  mkdir -p "$RERUN_ROOT/evaluation"
  "$PY" "$REPO/scripts/sample_union_sft_generations.py" \
    --model "pi_quorum_q3_kd=$RERUN_ROOT/models/student" --setting quorum \
    --ref_name eagle --ref_name topaz --ref_name birch --ref_name cobalt_cost_only \
    --training_config "$REPO/configs/training.yaml" \
    --composed_config "$REPO/configs/composed/first_line_joke_m8.yaml" \
    --cost_id first_line_eagle --cost_id first_line_topaz --cost_id first_line_birch --cost_id first_line_cobalt \
    --output_file "$RERUN_ROOT/evaluation/student.json" --markdown_file "$RERUN_ROOT/evaluation/student.md" \
    --max_prompts 32 --n_samples 4 --max_new_tokens 512 --temperature 1.0 --seed 100
else
  exit 2
fi
"$PY" "$REPO/scripts/unfiltered_kd_rerun.py" report --kind "$KIND" --rerun "$RERUN_ROOT"
