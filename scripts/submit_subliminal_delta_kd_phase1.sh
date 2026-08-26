#!/usr/bin/env bash
set -Eeo pipefail

REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-subliminal-delta-kd}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1}"
action="${1:?Usage: $0 bootstrap|references|benchmarks|repeats}"

case "$action" in
  bootstrap)
    mkdir -p "$EXPERIMENT_ROOT/logs"
    sbatch --parsable "$REPO/scripts/sbatch_subliminal_delta_kd_bootstrap.sbatch"
    ;;
  references)
    sbatch --parsable --account="${A100_ACCOUNT:-jamiemmt}" --partition="${A100_PARTITION:-gpu-a100}" --gpus-per-node=a100:2 \
      "$REPO/scripts/sbatch_subliminal_delta_kd_phase1_references.sbatch"
    ;;
  benchmarks)
    python - "$EXPERIMENT_ROOT/phase1/reference_gate.json" <<'PY'
import json,sys
assert json.load(open(sys.argv[1]))["status"] == "pass"
PY
    echo "a100=$(sbatch --parsable --account="${A100_ACCOUNT:-jamiemmt}" --partition="${A100_PARTITION:-gpu-a100}" --gpus-per-node=a100:2 --array=0-4%2 --export=ALL,HARDWARE_FAMILY=a100 "$REPO/scripts/sbatch_subliminal_delta_kd_phase1_calibration.sbatch")"
    echo "l40s=$(sbatch --parsable --account="${L40S_ACCOUNT:-cse}" --partition="${L40S_PARTITION:-gpu-l40s}" --gpus-per-node=l40s:2 --array=0-4%4 --export=ALL,HARDWARE_FAMILY=l40s "$REPO/scripts/sbatch_subliminal_delta_kd_phase1_calibration.sbatch")"
    ;;
  repeats)
    config="$EXPERIMENT_ROOT/phase1/selected_config.json"
    a_mb="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_microbatch"]["a100"])' "$config")"
    l_mb="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_microbatch"]["l40s"])' "$config")"
    echo "a100=$(sbatch --parsable --account="${A100_ACCOUNT:-jamiemmt}" --partition="${A100_PARTITION:-gpu-a100}" --gpus-per-node=a100:2 --export=ALL,HARDWARE_FAMILY=a100,CALIBRATION_REPEAT=1,CALIBRATION_MICROBATCH="$a_mb" "$REPO/scripts/sbatch_subliminal_delta_kd_phase1_calibration.sbatch")"
    echo "l40s=$(sbatch --parsable --account="${L40S_ACCOUNT:-cse}" --partition="${L40S_PARTITION:-gpu-l40s}" --gpus-per-node=l40s:2 --export=ALL,HARDWARE_FAMILY=l40s,CALIBRATION_REPEAT=1,CALIBRATION_MICROBATCH="$l_mb" "$REPO/scripts/sbatch_subliminal_delta_kd_phase1_calibration.sbatch")"
    ;;
  *) echo "Unknown action: $action" >&2; exit 1 ;;
esac

