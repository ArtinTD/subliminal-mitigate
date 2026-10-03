#!/bin/bash
#SBATCH --job-name=explicit_min_student
#SBATCH --account=cse
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/explicit_prefix_min_distillation_v1/logs/%x_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/explicit_prefix_min_distillation_v1/logs/%x_%j.err

set -Eeo pipefail
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR
REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-explicit-min-distill}"
source "$REPO/scripts/explicit_min_distill_common.sh"

DATA_ROOT="$EXPERIMENT_ROOT/data"
MODEL_ROOT="$EXPERIMENT_ROOT/models"
EVAL_ROOT="$EXPERIMENT_ROOT/evaluations/student"
mkdir -p "$MODEL_ROOT" "$EVAL_ROOT" "$EXPERIMENT_ROOT/runtime/student"

cd "$EXPERIMENT_ROOT/runtime/student"
"$CONDA_ENV_PREFIX/bin/python" "$REPO/scripts/train_single_sft.py" \
  --dataset "$DATA_ROOT/student_sft_dataset" \
  --training_config "$REPO/configs/training.yaml" \
  --output_dir "$MODEL_ROOT" \
  --name pi_explicit_prefix_min_distilled \
  --epochs 3 \
  --min_steps 200

for artifact in adapter_config.json adapter_model.safetensors training_summary.json eval_meta.json; do
  [[ -f "$MODEL_ROOT/pi_explicit_prefix_min_distilled/$artifact" ]] || {
    echo "Missing student artifact: $artifact" >&2
    exit 1
  }
done
final_steps="$("$CONDA_ENV_PREFIX/bin/python" -c \
  'import json; print(json.load(open("'"$MODEL_ROOT"'/pi_explicit_prefix_min_distilled/training_summary.json"))["final_global_step"])')"
[[ "$final_steps" == "200" ]] || {
  echo "Student trained for $final_steps optimizer steps, expected exactly 200." >&2
  exit 1
}

if [[ ! -f "$EVAL_ROOT/student.json" ]]; then
  "$CONDA_ENV_PREFIX/bin/python" "$REPO/scripts/sample_union_sft_generations.py" \
    --model "pi_explicit_prefix_min_distilled=$MODEL_ROOT/pi_explicit_prefix_min_distilled" \
    --setting basic \
    --ref_name eagle \
    --ref_name topaz \
    --training_config "$REPO/configs/training.yaml" \
    --composed_config "$REPO/configs/composed/first_line_joke.yaml" \
    --output_file "$EVAL_ROOT/student.json" \
    --markdown_file "$EVAL_ROOT/student.md" \
    --cost_id first_line_eagle \
    --cost_id first_line_topaz \
    --n_samples 10 \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --seed 0
fi

"$CONDA_ENV_PREFIX/bin/python" "$REPO/scripts/summarize_explicit_prefix_distillation.py" \
  --mode final \
  --reference_gate "$EXPERIMENT_ROOT/evaluations/references/reference_gate.json" \
  --teacher_audit "$EXPERIMENT_ROOT/generations/merged/raw_teacher_audit.json" \
  --direct_min "$EXPERIMENT_ROOT/evaluations/direct_min/direct_min.json" \
  --student "$EVAL_ROOT/student.json" \
  --baseline "$EXPERIMENT_ROOT/baselines/union_sft_summary.json" \
  --output_json "$EXPERIMENT_ROOT/final_report.json" \
  --output_markdown "$EXPERIMENT_ROOT/final_report.md"
