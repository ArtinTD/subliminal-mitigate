#!/bin/bash
#SBATCH --job-name=quorum_kd_full_teacher
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=8:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/quorum_dropout_m4/quorum_q3_distillation_v1/logs/%x_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/quorum_dropout_m4/quorum_q3_distillation_v1/logs/%x_%j.err

set -Eeo pipefail
cleanup() {
  [[ -n "${dmon_pid:-}" ]] && kill "$dmon_pid" >/dev/null 2>&1 || true
  [[ -n "${metrics_pid:-}" ]] && kill "$metrics_pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR

REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-explicit-min-distill}"
source "$REPO/scripts/quorum_kd_common.sh"
HARDWARE_FAMILY="${HARDWARE_FAMILY:?Set HARDWARE_FAMILY}"
expected_generator_sha="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["code_git_sha"])' "$EXPERIMENT_ROOT/generations/$HARDWARE_FAMILY/checkpoints/checkpoint_meta.json")"
identity_worktree="$EXPERIMENT_ROOT/runtime/generator_identity_$expected_generator_sha"
test "$(git -C "$identity_worktree" rev-parse HEAD)" = "$expected_generator_sha"
# The sampler records `git rev-parse HEAD` in its checkpoint identity. Point Git
# at the immutable Phase-2 worktree while leaving paths/cwd at the canonical repo,
# whose generator files were byte-checked against this commit before submission.
export GIT_DIR="$(git -C "$identity_worktree" rev-parse --absolute-git-dir)"
export GIT_WORK_TREE="$REPO"
microbatch="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_microbatch"][sys.argv[2]])' "$EXPERIMENT_ROOT/phase1/phase1_report.json" "$HARDWARE_FAMILY")"
RUN_ROOT="$EXPERIMENT_ROOT/generations/$HARDWARE_FAMILY"
mkdir -p "$RUN_ROOT" "$EXPERIMENT_ROOT/phase4"
nvidia-smi dmon -s pucm -d 5 -o DT >"$RUN_ROOT/phase4_extension_nvidia_dmon.txt" 2>&1 &
dmon_pid=$!
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv,noheader,nounits -l 5 >"$RUN_ROOT/phase4_extension_gpu_metrics.csv" 2>&1 &
metrics_pid=$!
python scripts/sample_batched_quorum_distillation.py \
  --model_specs "$MODEL_SPECS" \
  --refs eagle,topaz,birch,cobalt_cost_only \
  --training_config configs/training.yaml \
  --composed_config configs/composed/first_line_joke_m4_union_sources.yaml \
  --prompt_manifest "$EXPERIMENT_ROOT/data/prompt_manifest.json" \
  --canonical_batch_manifest "$EXPERIMENT_ROOT/data/canonical_batches.json" \
  --selected_batch_ids "$EXPERIMENT_ROOT/data/extension_batch_ids.json" \
  --microbatch_size "$microbatch" \
  --batch_checkpoint_dir "$RUN_ROOT/checkpoints" \
  --hardware_family "$HARDWARE_FAMILY" \
  --devices cuda:0,cuda:1 --compose_device cuda:0 --quorum_q 3 \
  --max_new_tokens 512 --temperature 1.0 --seed 0 \
  --summary_output "$EXPERIMENT_ROOT/phase4/generation_${HARDWARE_FAMILY}_summary.json"
