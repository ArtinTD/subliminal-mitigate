#!/bin/bash
#SBATCH --job-name=explicit_min_explore_student
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
MODEL_NAME="pi_explicit_prefix_min_distilled_exploratory"
MODEL_PATH="$MODEL_ROOT/$MODEL_NAME"
EVAL_ROOT="$EXPERIMENT_ROOT/evaluations/exploratory"
mkdir -p "$MODEL_ROOT" "$EVAL_ROOT/curated" "$EVAL_ROOT/alpaca_heldout" \
  "$EXPERIMENT_ROOT/runtime/student_exploratory"

if [[ ! -f "$MODEL_PATH/adapter_model.safetensors" ]]; then
  cd "$EXPERIMENT_ROOT/runtime/student_exploratory"
  "$CONDA_ENV_PREFIX/bin/python" "$REPO/scripts/train_single_sft.py" \
    --dataset "$DATA_ROOT/student_sft_dataset_exploratory" \
    --training_config "$REPO/configs/training.yaml" \
    --output_dir "$MODEL_ROOT" \
    --name "$MODEL_NAME" \
    --epochs 3 \
    --min_steps 200
fi

for artifact in adapter_config.json adapter_model.safetensors training_summary.json eval_meta.json; do
  [[ -f "$MODEL_PATH/$artifact" ]] || { echo "Missing student artifact: $artifact" >&2; exit 1; }
done
final_steps="$("$CONDA_ENV_PREFIX/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["final_global_step"])' \
  "$MODEL_PATH/training_summary.json")"
[[ "$final_steps" == "200" ]] || { echo "Student steps=$final_steps, expected 200" >&2; exit 1; }

COMMON_ARGS=(
  --model "$MODEL_NAME=$MODEL_PATH"
  --setting basic
  --ref_name eagle
  --ref_name topaz
  --training_config "$REPO/configs/training.yaml"
  --composed_config "$REPO/configs/composed/first_line_joke.yaml"
  --cost_id first_line_eagle
  --cost_id first_line_topaz
  --n_samples 10
  --max_new_tokens 512
  --temperature 1.0
  --seed 0
)

if [[ ! -f "$EVAL_ROOT/curated/student.json" ]]; then
  "$CONDA_ENV_PREFIX/bin/python" "$REPO/scripts/sample_union_sft_generations.py" \
    "${COMMON_ARGS[@]}" \
    --output_file "$EVAL_ROOT/curated/student.json" \
    --markdown_file "$EVAL_ROOT/curated/student.md"
fi

if [[ ! -f "$EVAL_ROOT/alpaca_heldout/student.json" ]]; then
  "$CONDA_ENV_PREFIX/bin/python" "$REPO/scripts/sample_union_sft_generations.py" \
    "${COMMON_ARGS[@]}" \
    --probe_prompts "$DATA_ROOT/alpaca_heldout_eval_prompts.json" \
    --output_file "$EVAL_ROOT/alpaca_heldout/student.json" \
    --markdown_file "$EVAL_ROOT/alpaca_heldout/student.md"
fi
