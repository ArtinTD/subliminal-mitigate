#!/usr/bin/env bash
set -Eeo pipefail

REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-subliminal-delta-kd}"
source "$REPO/scripts/subliminal_delta_kd_common.sh"
PHASE1_STAGE="${PHASE1_STAGE:-$EXPERIMENT_ROOT/phase1/recovered_artin}"
A100_ACCOUNT="${A100_ACCOUNT:-jamiemmt}"
A100_PARTITION="${A100_PARTITION:-gpu-a100}"
L40S_ACCOUNT="${L40S_ACCOUNT:-ckpt-cse}"
L40S_PARTITION="${L40S_PARTITION:-ckpt-g2}"
report="$PHASE1_STAGE/phase1_report.json"

python - "$report" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
assert payload["mode"] == "final" and payload["status"] == "pass"
assert set(payload["selected_microbatch"]) == {"a100", "l40s"}
PY

[[ ! -e "$EXPERIMENT_ROOT/phase2" ]] || {
  echo "Phase 2 output already exists; refusing resubmission" >&2
  exit 1
}

a100="$(sbatch --parsable --account="$A100_ACCOUNT" --partition="$A100_PARTITION" \
  --gpus-per-node=a100:2 --export=ALL,PHASE1_STAGE="$PHASE1_STAGE",HARDWARE_FAMILY=a100 \
  "$REPO/scripts/sbatch_subliminal_delta_kd_phase2_teacher.sbatch")"
l40s="$(sbatch --parsable --account="$L40S_ACCOUNT" --partition="$L40S_PARTITION" \
  --gpus-per-node=l40s:2 --export=ALL,PHASE1_STAGE="$PHASE1_STAGE",HARDWARE_FAMILY=l40s \
  "$REPO/scripts/sbatch_subliminal_delta_kd_phase2_teacher.sbatch")"
finalizer="$(sbatch --parsable --dependency="afterok:$a100:$l40s" \
  --export=ALL,PHASE1_STAGE="$PHASE1_STAGE" \
  "$REPO/scripts/sbatch_subliminal_delta_kd_phase2_finalizer.sbatch")"
printf 'a100=%s\nl40s=%s\nfinalizer=%s\n' "$a100" "$l40s" "$finalizer"
