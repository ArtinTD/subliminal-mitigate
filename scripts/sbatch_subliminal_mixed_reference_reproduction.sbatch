#!/bin/bash
#SBATCH --job-name=sub_mixed_refs
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1/phase1/mixed_reference_reproduction/logs/%x_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1/phase1/mixed_reference_reproduction/logs/%x_%j.err

set -Eeo pipefail
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR
REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-subliminal-delta-kd}"
source "$REPO/scripts/subliminal_delta_kd_common.sh"

CONFIG="configs/experiments/subliminal_mixed_reference_reproduction.yaml"
PHASE1="$EXPERIMENT_ROOT/phase1"
ROOT="$PHASE1/mixed_reference_reproduction"
SOURCE_ROOT="$PHASE1/sources"
mkdir -p "$ROOT/logs" "$ROOT/provenance" "$ROOT/references"

date -u
hostname
nvidia-smi
git rev-parse HEAD > "$ROOT/provenance/git_commit.txt"
sha256sum "$CONFIG" configs/training.yaml \
  scripts/prepare_subliminal_mixed_reference_reproduction.py \
  scripts/evaluate_subliminal_delta_references.py \
  scripts/analyze_subliminal_mixed_reference_reproduction.py \
  > "$ROOT/provenance/code_and_config.sha256"

if [[ ! -f "$ROOT/dataset_audit.json" ]]; then
  python scripts/prepare_subliminal_mixed_reference_reproduction.py \
    --config "$CONFIG" \
    --source_root "$SOURCE_ROOT" \
    --output_root "$ROOT" \
    > "$ROOT/logs/prepare.log" 2>&1
fi

train_reference() {
  local source="$1" name="$2" gpu="$3" steps="$4"
  local worker_cache="$CACHE_ROOT/subliminal_mixed_refs/${SLURM_JOB_ID}_${source}"
  if [[ -f "$ROOT/references/$name/adapter_model.safetensors" ]]; then
    echo "$name already complete; refusing to overwrite and skipping"
    return
  fi
  mkdir -p "$worker_cache/vllm" "$worker_cache/torchinductor" "$worker_cache/triton"
  CUDA_VISIBLE_DEVICES="$gpu" \
  VLLM_CACHE_ROOT="$worker_cache/vllm" \
  TORCHINDUCTOR_CACHE_DIR="$worker_cache/torchinductor" \
  TRITON_CACHE_DIR="$worker_cache/triton" \
  OMP_NUM_THREADS=4 \
  python scripts/train_single_sft.py \
    --dataset "$ROOT/datasets/$source" \
    --training_config configs/training.yaml \
    --output_dir "$ROOT/references" \
    --name "$name" \
    --exact_steps "$steps" \
    > "$ROOT/logs/${name}_training.log" 2>&1
}

train_reference panda pi_A 0 516 &
panda_pid=$!
train_reference eagle pi_B 1 510 &
eagle_pid=$!
wait "$panda_pid"
wait "$eagle_pid"

test -f "$ROOT/references/pi_A/adapter_model.safetensors"
test -f "$ROOT/references/pi_B/adapter_model.safetensors"

if [[ ! -f "$ROOT/reference_evaluation.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_subliminal_delta_references.py \
    --experiment_config "$CONFIG" \
    --training_config configs/training.yaml \
    --ref_panda "$ROOT/references/pi_A" \
    --ref_eagle "$ROOT/references/pi_B" \
    --output_file "$ROOT/reference_evaluation.json" \
    > "$ROOT/logs/evaluation.log" 2>&1
fi

python scripts/analyze_subliminal_mixed_reference_reproduction.py \
  --config "$CONFIG" \
  --root "$ROOT" \
  > "$ROOT/logs/analysis.log" 2>&1
date -u
