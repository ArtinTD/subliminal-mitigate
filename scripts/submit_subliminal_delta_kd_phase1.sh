#!/usr/bin/env bash
set -Eeo pipefail

REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-subliminal-delta-kd}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1}"
PYTHON_BIN="${PYTHON_BIN:-/gscratch/scrubbed/adhyyan/envs/subliminal-delta-kd-py311/bin/python}"
action="${1:?Usage: $0 bootstrap|references|recovered_references|artin_gate|benchmarks|select|repeats|finalize}"
phase1="$EXPERIMENT_ROOT/phase1"
stage="${PHASE1_STAGE:-$phase1/recovered_artin}"
artin_evaluation="$stage/artin_evaluation/results.json"
artin_gate="$stage/artin_evaluation/reference_gate.json"

case "$action" in
  bootstrap)
    mkdir -p "$EXPERIMENT_ROOT/logs"
    sbatch --parsable "$REPO/scripts/sbatch_subliminal_delta_kd_bootstrap.sbatch"
    ;;
  references)
    sbatch --parsable --account="${A100_ACCOUNT:-jamiemmt}" --partition="${A100_PARTITION:-gpu-a100}" --gpus-per-node=a100:2 \
      "$REPO/scripts/sbatch_subliminal_delta_kd_phase1_references.sbatch"
    ;;
  recovered_references)
    sbatch --parsable --account="${A100_ACCOUNT:-jamiemmt}" --partition="${A100_PARTITION:-gpu-a100}" --gpus-per-node=a100:1 \
      --export=ALL,PHASE1_STAGE="$stage" \
      "$REPO/scripts/sbatch_subliminal_delta_kd_phase1_recovered_references.sbatch"
    ;;
  artin_gate)
    "$PYTHON_BIN" "$REPO/scripts/analyze_subliminal_delta_phase1.py" \
      --phase1_root "$stage/artin_evaluation" \
      --source_root "$phase1" \
      --reference_evaluation "$artin_evaluation" \
      --reference_protocol artin \
      --experiment_config "$REPO/configs/experiments/subliminal_delta_kd.yaml" \
      --mode references
    ;;
  benchmarks)
    "$PYTHON_BIN" - "$artin_gate" <<'PY'
import json,sys
assert json.load(open(sys.argv[1]))["status"] == "pass"
PY
    a100_job="$(sbatch --parsable --account="${A100_ACCOUNT:-jamiemmt}" --partition="${A100_PARTITION:-gpu-a100}" --gpus-per-node=a100:2 --array=0-4%2 --export=ALL,PHASE1_STAGE="$stage",HARDWARE_FAMILY=a100 "$REPO/scripts/sbatch_subliminal_delta_kd_phase1_calibration.sbatch")"
    l40s_job="$(sbatch --parsable --account="${L40S_ACCOUNT:-cse}" --partition="${L40S_PARTITION:-gpu-l40s}" --gpus-per-node=l40s:2 --array=0-4%4 --export=ALL,PHASE1_STAGE="$stage",HARDWARE_FAMILY=l40s "$REPO/scripts/sbatch_subliminal_delta_kd_phase1_calibration.sbatch")"
    echo "a100=$a100_job"
    echo "l40s=$l40s_job"
    ;;
  select)
    "$PYTHON_BIN" "$REPO/scripts/analyze_subliminal_delta_phase1.py" \
      --phase1_root "$stage" \
      --source_root "$phase1" \
      --reference_evaluation "$artin_evaluation" \
      --reference_protocol artin \
      --experiment_config "$REPO/configs/experiments/subliminal_delta_kd.yaml" \
      --mode select
    ;;
  repeats)
    config="$stage/selected_config.json"
    a_mb="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_microbatch"]["a100"])' "$config")"
    l_mb="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_microbatch"]["l40s"])' "$config")"
    a100_job="$(sbatch --parsable --account="${A100_ACCOUNT:-jamiemmt}" --partition="${A100_PARTITION:-gpu-a100}" --gpus-per-node=a100:2 --export=ALL,PHASE1_STAGE="$stage",HARDWARE_FAMILY=a100,CALIBRATION_REPEAT=1,CALIBRATION_MICROBATCH="$a_mb" "$REPO/scripts/sbatch_subliminal_delta_kd_phase1_calibration.sbatch")"
    l40s_job="$(sbatch --parsable --account="${L40S_ACCOUNT:-cse}" --partition="${L40S_PARTITION:-gpu-l40s}" --gpus-per-node=l40s:2 --export=ALL,PHASE1_STAGE="$stage",HARDWARE_FAMILY=l40s,CALIBRATION_REPEAT=1,CALIBRATION_MICROBATCH="$l_mb" "$REPO/scripts/sbatch_subliminal_delta_kd_phase1_calibration.sbatch")"
    echo "a100=$a100_job"
    echo "l40s=$l40s_job"
    ;;
  finalize)
    "$PYTHON_BIN" "$REPO/scripts/analyze_subliminal_delta_phase1.py" \
      --phase1_root "$stage" \
      --source_root "$phase1" \
      --reference_evaluation "$artin_evaluation" \
      --reference_protocol artin \
      --experiment_config "$REPO/configs/experiments/subliminal_delta_kd.yaml" \
      --mode final
    ;;
  *) echo "Unknown action: $action" >&2; exit 1 ;;
esac
