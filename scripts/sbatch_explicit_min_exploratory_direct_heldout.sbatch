#!/bin/bash
#SBATCH --job-name=explicit_min_explore_teacher_eval
#SBATCH --account=cse
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/explicit_prefix_min_distillation_v1/logs/%x_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/explicit_prefix_min_distillation_v1/logs/%x_%j.err

set -Eeo pipefail
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR
REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-explicit-min-distill}"
source "$REPO/scripts/explicit_min_distill_common.sh"

DATA_ROOT="$EXPERIMENT_ROOT/data"
MODEL_ROOT="$EXPERIMENT_ROOT/models/references"
EVAL_ROOT="$EXPERIMENT_ROOT/evaluations/exploratory/alpaca_heldout"
mkdir -p "$EVAL_ROOT"

if [[ ! -f "$EVAL_ROOT/direct_min.json" ]]; then
  "$CONDA_ENV_PREFIX/bin/python" "$REPO/scripts/sample_min_composition_generations.py" \
    --ref_A "$MODEL_ROOT/pi_A_retrained" \
    --ref_B "$MODEL_ROOT/pi_B_retrained" \
    --training_config "$REPO/configs/training.yaml" \
    --probe_prompts "$DATA_ROOT/alpaca_heldout_eval_prompts.json" \
    --score_probe_costs \
    --output_file "$EVAL_ROOT/direct_min.json" \
    --markdown_file "$EVAL_ROOT/direct_min.md" \
    --composition_type min \
    --n_samples 10 \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --seed 0 \
    --device_A cuda:0 \
    --device_B cuda:1
fi
