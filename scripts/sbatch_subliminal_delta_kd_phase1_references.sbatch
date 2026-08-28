#!/bin/bash
#SBATCH --job-name=sub_delta_refs
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=18:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1/logs/%x_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1/logs/%x_%j.err

set -Eeo pipefail
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR
REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-subliminal-delta-kd}"
source "$REPO/scripts/subliminal_delta_kd_common.sh"

PHASE1="$EXPERIMENT_ROOT/phase1"
EXP_CONFIG="configs/experiments/subliminal_delta_kd.yaml"
mkdir -p "$PHASE1/sources" "$PHASE1/references" "$PHASE1/logs"
date -u
hostname
nvidia-smi

build_components() {
  local source="$1" gpu="$2"
  local number_dir="$PHASE1/sources/$source/number_sequence"
  local joke_dir="$PHASE1/sources/$source/joke"
  # Panda and Eagle run concurrently.  vLLM and TorchInductor do not safely
  # support two writers to the same binary compilation cache, so give each
  # worker a durable, job-scoped cache tree.  The model, kernels, and decoding
  # configuration remain unchanged.
  local worker_cache="$CACHE_ROOT/subliminal_delta_kd_workers/${SLURM_JOB_ID}_${source}"
  export VLLM_CACHE_ROOT="$worker_cache/vllm"
  export TORCHINDUCTOR_CACHE_DIR="$worker_cache/torchinductor"
  export TRITON_CACHE_DIR="$worker_cache/triton"
  mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
  if [[ ! -f "$number_dir/state.json" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" python dataset_gen/composed_subliminal_joke.py \
      --common_config configs/dataset_gen.yaml \
      --candidate_manifest configs/sweeps/subliminal_trait_candidates.yaml \
      --candidate_id "$source" \
      --output_dir "$number_dir" \
      --n_samples 10000 \
      --seed 42 \
      --benefit_mode none \
      --pool_multiplier 1.8 \
      --max_attempts 6 \
      >"$PHASE1/logs/${source}_number_generation.log" 2>&1
  fi
  if [[ ! -f "$joke_dir/state.json" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/generate_subliminal_delta_joke_dataset.py \
      --config "$EXP_CONFIG" \
      --source "$source" \
      --output_dir "$joke_dir" \
      >"$PHASE1/logs/${source}_joke_generation.log" 2>&1
  fi
}

build_components panda 0 &
panda_pid=$!
build_components eagle 1 &
eagle_pid=$!
wait "$panda_pid"
wait "$eagle_pid"

if [[ ! -f "$PHASE1/manifests/prompt_manifest.json" ]]; then
  python scripts/prepare_subliminal_delta_phase1.py \
    --experiment_config "$EXP_CONFIG" \
    --phase1_root "$PHASE1" \
    >"$PHASE1/logs/prepare_sources.log" 2>&1
fi

train_reference() {
  local source="$1" name="$2" gpu="$3"
  if [[ ! -f "$PHASE1/references/$name/adapter_model.safetensors" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/train_single_sft.py \
      --dataset "$PHASE1/union_sources/$source" \
      --training_config configs/training.yaml \
      --output_dir "$PHASE1/references" \
      --name "$name" \
      --epochs 3 \
      --min_steps 200 \
      >"$PHASE1/logs/${name}_training.log" 2>&1
  fi
}

train_reference panda pi_A 0 &
panda_pid=$!
train_reference eagle pi_B 1 &
eagle_pid=$!
wait "$panda_pid"
wait "$eagle_pid"

if [[ ! -f "$PHASE1/reference_evaluation.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_subliminal_delta_references.py \
    --experiment_config "$EXP_CONFIG" \
    --training_config configs/training.yaml \
    --ref_panda "$PHASE1/references/pi_A" \
    --ref_eagle "$PHASE1/references/pi_B" \
    --output_file "$PHASE1/reference_evaluation.json" \
    >"$PHASE1/logs/reference_evaluation.log" 2>&1
fi

python scripts/analyze_subliminal_delta_phase1.py \
  --phase1_root "$PHASE1" \
  --experiment_config "$EXP_CONFIG" \
  --mode references
date -u
