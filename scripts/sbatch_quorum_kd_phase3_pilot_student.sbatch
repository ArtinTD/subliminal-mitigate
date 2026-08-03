#!/bin/bash
#SBATCH --job-name=quorum_kd_pilot_student
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/quorum_dropout_m4/quorum_q3_distillation_v1/logs/%x_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/quorum_dropout_m4/quorum_q3_distillation_v1/logs/%x_%j.err

set -Eeo pipefail
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR

REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-explicit-min-distill}"
source "$REPO/scripts/quorum_kd_common.sh"

PHASE3_ROOT="$EXPERIMENT_ROOT/phase3"
DATASET="$EXPERIMENT_ROOT/phase2/pilot_student_sft_dataset"
MODEL_NAME="pi_quorum_q3_kd_pilot"
MODEL_ROOT="$PHASE3_ROOT/models"
MODEL_PATH="$MODEL_ROOT/$MODEL_NAME"
EVAL_ROOT="$PHASE3_ROOT/evaluation"
PHI3_REFERENCE="$PHASE3_ROOT/reference_phi3.json"
mkdir -p "$MODEL_ROOT" "$EVAL_ROOT" "$EXPERIMENT_ROOT/runtime/phase3"

test -f "$EXPERIMENT_ROOT/phase2/phase2_report.json"
test -f "$PHI3_REFERENCE"
"$CONDA_ENV_PREFIX/bin/python" - "$EXPERIMENT_ROOT/phase2/phase2_report.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
assert d["status"] == "pass"
assert d["curated_occurrences"] == 855
assert all(d["gates"].values())
PY

if [[ ! -f "$MODEL_PATH/adapter_model.safetensors" ]]; then
  cd "$EXPERIMENT_ROOT/runtime/phase3"
  "$CONDA_ENV_PREFIX/bin/python" "$REPO/scripts/train_single_sft.py" \
    --dataset "$DATASET" \
    --training_config "$REPO/configs/training.yaml" \
    --output_dir "$MODEL_ROOT" \
    --name "$MODEL_NAME" \
    --epochs 3 \
    --min_steps 50
fi

for artifact in adapter_config.json adapter_model.safetensors training_summary.json; do
  test -f "$MODEL_PATH/$artifact"
done
"$CONDA_ENV_PREFIX/bin/python" - "$MODEL_PATH/training_summary.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
assert d["final_global_step"] == 50, d
assert d["n_examples"] == 855, d
assert d["effective_batch_size"] == 60, d
PY

if [[ ! -f "$EVAL_ROOT/student.json" ]]; then
  "$CONDA_ENV_PREFIX/bin/python" "$REPO/scripts/sample_union_sft_generations.py" \
    --model "$MODEL_NAME=$MODEL_PATH" \
    --setting quorum \
    --ref_name eagle \
    --ref_name topaz \
    --ref_name birch \
    --ref_name cobalt_cost_only \
    --training_config "$REPO/configs/training.yaml" \
    --composed_config "$REPO/configs/composed/first_line_joke_m8.yaml" \
    --cost_id first_line_eagle \
    --cost_id first_line_topaz \
    --cost_id first_line_birch \
    --cost_id first_line_cobalt \
    --output_file "$EVAL_ROOT/student.json" \
    --markdown_file "$EVAL_ROOT/student.md" \
    --max_prompts 32 \
    --n_samples 4 \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --seed 100
fi

"$CONDA_ENV_PREFIX/bin/python" "$REPO/scripts/summarize_quorum_kd_pilot_student.py" \
  --student "$EVAL_ROOT/student.json" \
  --phi3 "$PHI3_REFERENCE" \
  --training_summary "$MODEL_PATH/training_summary.json" \
  --output_json "$PHASE3_ROOT/phase3_report.json" \
  --output_markdown "$PHASE3_ROOT/phase3_report.md"

date
echo "Phase 3 pilot student complete: $PHASE3_ROOT"
