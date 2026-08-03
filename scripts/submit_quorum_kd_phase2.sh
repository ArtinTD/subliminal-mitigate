#!/usr/bin/env bash
# Submit both Phase-2 pilot generators and their CPU-only finalizer.
set -Eeo pipefail

REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-explicit-min-distill}"
source "$REPO/scripts/quorum_kd_common.sh"
A100_ACCOUNT="${A100_ACCOUNT:-cse}"
A100_PARTITION="${A100_PARTITION:-gpu-a100}"
L40S_ACCOUNT="${L40S_ACCOUNT:-cse}"
L40S_PARTITION="${L40S_PARTITION:-gpu-l40s}"
A100_GPU_OPT="${A100_GPU_OPT:---gpus-per-node=a100:2}"
L40S_GPU_OPT="${L40S_GPU_OPT:---gpus-per-node=l40s:2}"
report="$EXPERIMENT_ROOT/phase1/phase1_report.json"
python - "$report" <<'PY'
import json,sys
payload=json.load(open(sys.argv[1]))
assert payload["mode"] == "final" and payload["status"] == "pass"
PY
[[ ! -e "$EXPERIMENT_ROOT/phase2" ]] || { echo "Phase 2 output already exists; refusing resubmission" >&2; exit 1; }
a100="$(sbatch --parsable --account="$A100_ACCOUNT" --partition="$A100_PARTITION" "$A100_GPU_OPT" --export=ALL,HARDWARE_FAMILY=a100 "$REPO/scripts/sbatch_quorum_kd_phase2_teacher.sbatch")"
l40s="$(sbatch --parsable --account="$L40S_ACCOUNT" --partition="$L40S_PARTITION" "$L40S_GPU_OPT" --export=ALL,HARDWARE_FAMILY=l40s "$REPO/scripts/sbatch_quorum_kd_phase2_teacher.sbatch")"
finalizer="$(sbatch --parsable --dependency="afterok:$a100:$l40s" "$REPO/scripts/sbatch_quorum_kd_phase2_finalizer.sbatch")"
printf 'a100=%s\nl40s=%s\nfinalizer=%s\n' "$a100" "$l40s" "$finalizer"
