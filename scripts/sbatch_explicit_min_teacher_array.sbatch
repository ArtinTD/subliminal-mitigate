#!/bin/bash
#SBATCH --job-name=explicit_min_teacher
#SBATCH --account=cse
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/explicit_prefix_min_distillation_v1/logs/%x_%A_%a.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/explicit_prefix_min_distillation_v1/logs/%x_%A_%a.err

set -Eeo pipefail
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR
REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-explicit-min-distill}"
source "$REPO/scripts/explicit_min_distill_common.sh"

TASK_ID="${SLURM_ARRAY_TASK_ID:?Missing array task id}"
START=$((TASK_ID * 500))
END=$((START + 500))
MODEL_ROOT="$EXPERIMENT_ROOT/models/references"
DATA_ROOT="$EXPERIMENT_ROOT/data"
SHARD_ROOT="$EXPERIMENT_ROOT/generations/teacher_shards"
EVAL_ROOT="$EXPERIMENT_ROOT/evaluations"
mkdir -p "$SHARD_ROOT" "$EVAL_ROOT/direct_min"

SHARD_JSON="$SHARD_ROOT/shard_${TASK_ID}.json"
CHECKPOINT="$SHARD_ROOT/shard_${TASK_ID}.checkpoint.jsonl"
if [[ ! -f "$SHARD_JSON" ]]; then
  "$CONDA_ENV_PREFIX/bin/python" "$REPO/scripts/sample_min_composition_generations.py" \
    --ref_A "$MODEL_ROOT/pi_A_retrained" \
    --ref_B "$MODEL_ROOT/pi_B_retrained" \
    --training_config "$REPO/configs/training.yaml" \
    --probe_prompts "$DATA_ROOT/prompt_manifest.json" \
    --output_file "$SHARD_JSON" \
    --markdown_file "$SHARD_ROOT/shard_${TASK_ID}.md" \
    --composition_type min \
    --prompt_start "$START" \
    --prompt_end "$END" \
    --n_samples 1 \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --seed 0 \
    --device_A cuda:0 \
    --device_B cuda:1 \
    --checkpoint_jsonl "$CHECKPOINT" \
    --resume
else
  echo "Reusing completed teacher shard: $SHARD_JSON"
fi

if (( TASK_ID == 0 )) && [[ ! -f "$EVAL_ROOT/direct_min/direct_min.json" ]]; then
  "$CONDA_ENV_PREFIX/bin/python" "$REPO/scripts/sample_min_composition_generations.py" \
    --ref_A "$MODEL_ROOT/pi_A_retrained" \
    --ref_B "$MODEL_ROOT/pi_B_retrained" \
    --training_config "$REPO/configs/training.yaml" \
    --output_file "$EVAL_ROOT/direct_min/direct_min.json" \
    --markdown_file "$EVAL_ROOT/direct_min/direct_min.md" \
    --composition_type min \
    --n_samples 10 \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --seed 0 \
    --device_A cuda:0 \
    --device_B cuda:1
fi
