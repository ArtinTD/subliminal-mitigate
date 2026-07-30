#!/bin/bash
#SBATCH --job-name=explicit_min_refs
#SBATCH --account=cse
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/explicit_prefix_min_distillation_v1/logs/%x_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/explicit_prefix_min_distillation_v1/logs/%x_%j.err

set -Eeo pipefail
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR
source "$(dirname "$0")/explicit_min_distill_common.sh"

SOURCE_ROOT="$EXPERIMENT_ROOT/source_datasets"
MODEL_ROOT="$EXPERIMENT_ROOT/models/references"
GEN_ROOT="$EXPERIMENT_ROOT/evaluations/references"
DATA_ROOT="$EXPERIMENT_ROOT/data"
TRAINING_CONFIG="$REPO/configs/training.yaml"
COMPOSED_CONFIG="$REPO/configs/composed/first_line_joke.yaml"
mkdir -p "$MODEL_ROOT" "$GEN_ROOT" "$DATA_ROOT" "$EXPERIMENT_ROOT/runtime"

if [[ ! -f "$DATA_ROOT/prompt_manifest.json" && ! -f "$DATA_ROOT/prompt_manifest.meta.json" ]]; then
  "$CONDA_ENV_PREFIX/bin/python" "$REPO/scripts/build_distillation_prompt_manifest.py" \
    --source "eagle=$SOURCE_ROOT/eagle" \
    --source "topaz=$SOURCE_ROOT/topaz" \
    --output_file "$DATA_ROOT/prompt_manifest.json" \
    --metadata_file "$DATA_ROOT/prompt_manifest.meta.json" \
    --combined_eval_output "$DATA_ROOT/combined_eval_config.json" \
    --eval_config "$SOURCE_ROOT/eagle/eval_config.json" \
    --seed 42 \
    --expect_occurrences 2000 \
    --expect_unique 1324 \
    --expect_eval_overlap 0
elif [[ ! -f "$DATA_ROOT/prompt_manifest.json" || ! -f "$DATA_ROOT/prompt_manifest.meta.json" || ! -f "$DATA_ROOT/combined_eval_config.json" ]]; then
  echo "Partial prompt-manifest state exists; refusing to overwrite it." >&2
  exit 1
else
  echo "Reusing complete prompt manifest."
fi

train_reference() {
  local gpu="$1"
  local source="$2"
  local name="$3"
  local work="$EXPERIMENT_ROOT/runtime/train_$name"
  mkdir -p "$work"
  (
    cd "$work"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor/${SLURM_JOB_ID}_$name"
    export TRITON_CACHE_DIR="$CACHE_ROOT/triton/${SLURM_JOB_ID}_$name"
    "$CONDA_ENV_PREFIX/bin/python" "$REPO/scripts/train_single_sft.py" \
      --dataset "$SOURCE_ROOT/$source" \
      --training_config "$TRAINING_CONFIG" \
      --output_dir "$MODEL_ROOT" \
      --name "$name" \
      --epochs 3 \
      --min_steps 200
  ) > "$EXPERIMENT_ROOT/logs/train_${name}_${SLURM_JOB_ID}.log" 2>&1
}

train_reference 0 eagle pi_A_retrained &
pid_a=$!
train_reference 1 topaz pi_B_retrained &
pid_b=$!
set +e
wait "$pid_a"
status_a=$?
wait "$pid_b"
status_b=$?
set -e
if (( status_a != 0 || status_b != 0 )); then
  echo "Reference training failed: pi_A=$status_a pi_B=$status_b" >&2
  exit 1
fi

for adapter in pi_A_retrained pi_B_retrained; do
  for artifact in adapter_config.json adapter_model.safetensors training_summary.json eval_meta.json; do
    [[ -f "$MODEL_ROOT/$adapter/$artifact" ]] || {
      echo "Missing reference artifact: $MODEL_ROOT/$adapter/$artifact" >&2
      exit 1
    }
  done
done

evaluate_reference() {
  local gpu="$1"
  local model_name="$2"
  local ref_name="$3"
  local output_name="$4"
  local work="$EXPERIMENT_ROOT/runtime/eval_$model_name"
  if [[ -f "$GEN_ROOT/$output_name.json" ]]; then
    echo "Reusing completed reference evaluation: $output_name"
    return
  fi
  mkdir -p "$work"
  (
    cd "$work"
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$CONDA_ENV_PREFIX/bin/python" "$REPO/scripts/sample_union_sft_generations.py" \
      --model "$model_name=$MODEL_ROOT/$model_name" \
      --setting basic \
      --ref_name "$ref_name" \
      --training_config "$TRAINING_CONFIG" \
      --composed_config "$COMPOSED_CONFIG" \
      --output_file "$GEN_ROOT/$output_name.json" \
      --markdown_file "$GEN_ROOT/$output_name.md" \
      --cost_id first_line_eagle \
      --cost_id first_line_topaz \
      --n_samples 10 \
      --max_new_tokens 512 \
      --temperature 1.0 \
      --seed 0
  ) > "$EXPERIMENT_ROOT/logs/eval_${model_name}_${SLURM_JOB_ID}.log" 2>&1
}

evaluate_reference 0 pi_A_retrained eagle pi_A_retrained &
eval_a=$!
evaluate_reference 1 pi_B_retrained topaz pi_B_retrained &
eval_b=$!
set +e
wait "$eval_a"
status_a=$?
wait "$eval_b"
status_b=$?
set -e
if (( status_a != 0 || status_b != 0 )); then
  echo "Reference evaluation failed: pi_A=$status_a pi_B=$status_b" >&2
  exit 1
fi

"$CONDA_ENV_PREFIX/bin/python" "$REPO/scripts/summarize_explicit_prefix_distillation.py" \
  --mode references \
  --reference_eagle "$GEN_ROOT/pi_A_retrained.json" \
  --reference_topaz "$GEN_ROOT/pi_B_retrained.json" \
  --output_json "$GEN_ROOT/reference_gate.json" \
  --output_markdown "$GEN_ROOT/reference_gate.md"
