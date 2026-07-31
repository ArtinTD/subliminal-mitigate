#!/bin/bash
#SBATCH --job-name=explicit_min_explore_finalize
#SBATCH --account=ckpt-cse
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/explicit_prefix_min_distillation_v1/logs/%x_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/explicit_prefix_min_distillation_v1/logs/%x_%j.err

set -Eeo pipefail
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR
REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-explicit-min-distill}"
source "$REPO/scripts/explicit_min_distill_common.sh"

DATA_ROOT="$EXPERIMENT_ROOT/data"
SHARD_ROOT="$EXPERIMENT_ROOT/generations/teacher_shards"
MERGED_ROOT="$EXPERIMENT_ROOT/generations/merged"
DATASET="$DATA_ROOT/student_sft_dataset_exploratory"
HELDOUT="$DATA_ROOT/alpaca_heldout_eval_prompts.json"
mkdir -p "$MERGED_ROOT"

if [[ ! -d "$DATASET" ]]; then
  "$CONDA_ENV_PREFIX/bin/python" "$REPO/scripts/build_min_distillation_dataset.py" \
    --prompt_manifest "$DATA_ROOT/prompt_manifest.json" \
    --shard "$SHARD_ROOT/shard_0.json" \
    --shard "$SHARD_ROOT/shard_1.json" \
    --shard "$SHARD_ROOT/shard_2.json" \
    --shard "$SHARD_ROOT/shard_3.json" \
    --output_dir "$DATASET" \
    --merged_json "$MERGED_ROOT/raw_teacher_generations_exploratory.json" \
    --audit_json "$MERGED_ROOT/raw_teacher_audit_exploratory.json" \
    --eval_config "$DATA_ROOT/combined_eval_config.json" \
    --tokenizer unsloth/Qwen3-8B \
    --max_seq_length 2048 \
    --failed_gate_override_reason "User-authorized exploratory continuation: preserve all 2,000 raw minimum-teacher rows after the original strict final-line gate failed; evaluate joke-anywhere and placement without claiming the original gate passed."
else
  echo "Reusing exploratory dataset: $DATASET"
fi

if [[ ! -f "$HELDOUT" ]]; then
  "$CONDA_ENV_PREFIX/bin/python" "$REPO/scripts/build_explicit_min_heldout_prompts.py" \
    --prompt_manifest "$DATA_ROOT/prompt_manifest.json" \
    --eval_config "$REPO/configs/composed/first_line_joke.yaml" \
    --output_json "$HELDOUT" \
    --n_prompts 32 \
    --pool_size 512 \
    --seed 4242
else
  echo "Reusing held-out prompts: $HELDOUT"
fi

"$CONDA_ENV_PREFIX/bin/python" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); assert len(p)==32; assert len({x["prompt"] for x in p})==32' \
  "$HELDOUT"
