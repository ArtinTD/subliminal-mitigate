#!/bin/bash
#SBATCH --job-name=quorum_ref_recovery
#SBATCH --account=cse
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=a100:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --time=4:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/quorum_dropout_m4/quorum_q3_distillation_v1/logs/%x_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/quorum_dropout_m4/quorum_q3_distillation_v1/logs/%x_%j.err

set -Eeo pipefail
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR
REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-explicit-min-distill}"
source "$REPO/scripts/quorum_kd_common.sh"

RECOVERY_ROOT="$EXPERIMENT_ROOT/recovered_references"
MODEL_ROOT="$RECOVERY_ROOT/models"
EVAL_ROOT="$RECOVERY_ROOT/evaluations"
RUNTIME_ROOT="$RECOVERY_ROOT/runtime"
mkdir -p "$MODEL_ROOT" "$EVAL_ROOT" "$RUNTIME_ROOT"

for source in eagle topaz birch; do
  [[ -f "$SOURCE_ROOT/$source/state.json" ]] || { echo "Missing source: $source" >&2; exit 1; }
done
for artifact in adapter_config.json adapter_model.safetensors; do
  [[ -f "$MODEL_ROOT/pi_cobalt_exact/$artifact" ]] || { echo "Missing exact Cobalt artifact: $artifact" >&2; exit 1; }
done

train_one() {
  local gpu="$1" source="$2" name="$3"
  local out="$MODEL_ROOT/$name"
  [[ ! -e "$out" ]] || { echo "Refusing to overwrite $out" >&2; return 1; }
  mkdir -p "$RUNTIME_ROOT/train_$name"
  (
    cd "$RUNTIME_ROOT/train_$name"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor/${SLURM_JOB_ID}_$name"
    export TRITON_CACHE_DIR="$CACHE_ROOT/triton/${SLURM_JOB_ID}_$name"
    python "$REPO/scripts/train_single_sft.py" \
      --dataset "$SOURCE_ROOT/$source" --training_config "$REPO/configs/training.yaml" \
      --output_dir "$MODEL_ROOT" --name "$name" --epochs 3 --min_steps 200
  ) >"$EXPERIMENT_ROOT/logs/train_${name}_${SLURM_JOB_ID}.log" 2>&1
}

train_one 0 eagle pi_eagle_recovered & p0=$!
train_one 1 topaz pi_topaz_recovered & p1=$!
train_one 2 birch pi_birch_recovered & p2=$!
set +e; wait "$p0"; s0=$?; wait "$p1"; s1=$?; wait "$p2"; s2=$?; set -e
(( s0 == 0 && s1 == 0 && s2 == 0 )) || { echo "Recovery training failed: $s0 $s1 $s2" >&2; exit 1; }

for model in pi_eagle_recovered pi_topaz_recovered pi_birch_recovered pi_cobalt_exact; do
  for artifact in adapter_config.json adapter_model.safetensors; do
    [[ -f "$MODEL_ROOT/$model/$artifact" ]] || { echo "Incomplete recovered adapter: $model/$artifact" >&2; exit 1; }
  done
done

python - "$RECOVERY_ROOT/recovered_model_specs.json" "$MODEL_ROOT" <<'PY'
import datetime,json,pathlib,subprocess,sys
path=pathlib.Path(sys.argv[1]); root=pathlib.Path(sys.argv[2])
payload={
 "models": {"eagle":str(root/"pi_eagle_recovered"),"topaz":str(root/"pi_topaz_recovered"),"birch":str(root/"pi_birch_recovered"),"cobalt_cost_only":str(root/"pi_cobalt_exact")},
 "provenance": {"eagle":"retrained","topaz":"retrained","birch":"retrained","cobalt_cost_only":"exact_archive_restore"},
 "git_sha":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
 "timestamp_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY

evaluate_one() {
  local gpu="$1" ref="$2"
  CUDA_VISIBLE_DEVICES="$gpu" python "$REPO/scripts/sample_quorum_composition_generations.py" \
    --model_specs "$RECOVERY_ROOT/recovered_model_specs.json" --refs "$ref" \
    --training_config "$REPO/configs/training.yaml" \
    --composed_config "$REPO/configs/composed/first_line_joke_m8.yaml" \
    --output_file "$EVAL_ROOT/$ref.json" --markdown_file "$EVAL_ROOT/$ref.md" \
    --composition_type min --devices cuda:0 --compose_device cuda:0 \
    --n_samples 4 --max_new_tokens 512 --temperature 1.0 --seed 0 --markdown_samples 24 \
    >"$EXPERIMENT_ROOT/logs/recovery_eval_${ref}_${SLURM_JOB_ID}.log" 2>&1
}

evaluate_one 0 eagle & e0=$!
evaluate_one 1 topaz & e1=$!
evaluate_one 2 birch & e2=$!
set +e; wait "$e0"; t0=$?; wait "$e1"; t1=$?; wait "$e2"; t2=$?; set -e
(( t0 == 0 && t1 == 0 && t2 == 0 )) || { echo "Recovery evaluation failed: $t0 $t1 $t2" >&2; exit 1; }
evaluate_one 0 cobalt_cost_only

python "$REPO/scripts/analyze_quorum_reference_recovery.py" --root "$RECOVERY_ROOT"
