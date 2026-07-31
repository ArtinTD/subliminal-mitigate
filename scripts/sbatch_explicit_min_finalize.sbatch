#!/bin/bash
#SBATCH --job-name=explicit_min_finalize
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
mkdir -p "$EXPERIMENT_ROOT/generations/merged"

"$CONDA_ENV_PREFIX/bin/python" "$REPO/scripts/build_min_distillation_dataset.py" \
  --prompt_manifest "$DATA_ROOT/prompt_manifest.json" \
  --shard "$SHARD_ROOT/shard_0.json" \
  --shard "$SHARD_ROOT/shard_1.json" \
  --shard "$SHARD_ROOT/shard_2.json" \
  --shard "$SHARD_ROOT/shard_3.json" \
  --output_dir "$DATA_ROOT/student_sft_dataset" \
  --merged_json "$EXPERIMENT_ROOT/generations/merged/raw_teacher_generations.json" \
  --audit_json "$EXPERIMENT_ROOT/generations/merged/raw_teacher_audit.json" \
  --eval_config "$DATA_ROOT/combined_eval_config.json" \
  --tokenizer unsloth/Qwen3-8B \
  --max_seq_length 2048
