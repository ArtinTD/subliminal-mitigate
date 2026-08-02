#!/usr/bin/env bash
# Submit Phase-1 calibration benchmark arrays or deterministic repeats.
set -Eeo pipefail

REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-explicit-min-distill}"
source "$REPO/scripts/quorum_kd_common.sh"
action="${1:?Usage: $0 benchmarks|repeats}"
A100_ACCOUNT="${A100_ACCOUNT:-cse}"
A100_PARTITION="${A100_PARTITION:-gpu-a100}"
L40S_ACCOUNT="${L40S_ACCOUNT:-cse}"
L40S_PARTITION="${L40S_PARTITION:-gpu-l40s}"

python - "$EXPERIMENT_ROOT/phase0/phase0_report.json" <<'PY'
import json,sys
assert json.load(open(sys.argv[1]))["status"] == "complete"
PY

submit_one() {
  local hardware="$1" account="$2" partition="$3" extra="$4"
  sbatch --parsable --account="$account" --partition="$partition" $extra \
    --export="ALL,HARDWARE_FAMILY=$hardware" \
    "$REPO/scripts/sbatch_quorum_kd_phase1_calibration.sbatch"
}

case "$action" in
  benchmarks)
    [[ ! -e "$EXPERIMENT_ROOT/phase1/selected_config.json" ]] || { echo "Selection already exists; refusing duplicate benchmark submission" >&2; exit 1; }
    echo "a100=$(submit_one a100 "$A100_ACCOUNT" "$A100_PARTITION" '--array=0-4')"
    echo "l40s=$(submit_one l40s "$L40S_ACCOUNT" "$L40S_PARTITION" '--array=0-4')"
    ;;
  repeats)
    config="$EXPERIMENT_ROOT/phase1/selected_config.json"
    python - "$config" <<'PY'
import json,sys
assert json.load(open(sys.argv[1]))["status"] == "pass"
PY
    a_mb="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_microbatch"]["a100"])' "$config")"
    l_mb="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_microbatch"]["l40s"])' "$config")"
    echo "a100=$(sbatch --parsable --account="$A100_ACCOUNT" --partition="$A100_PARTITION" --export="ALL,HARDWARE_FAMILY=a100,CALIBRATION_REPEAT=1,CALIBRATION_MICROBATCH=$a_mb" "$REPO/scripts/sbatch_quorum_kd_phase1_calibration.sbatch")"
    echo "l40s=$(sbatch --parsable --account="$L40S_ACCOUNT" --partition="$L40S_PARTITION" --export="ALL,HARDWARE_FAMILY=l40s,CALIBRATION_REPEAT=1,CALIBRATION_MICROBATCH=$l_mb" "$REPO/scripts/sbatch_quorum_kd_phase1_calibration.sbatch")"
    ;;
  *) echo "Unknown action: $action" >&2; exit 1 ;;
esac
