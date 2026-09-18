#!/usr/bin/env bash
set -Eeuo pipefail
REPO=/mmfs1/home/adhyyan/subliminal-mitigate-subliminal-delta-kd-stages
source "$REPO/scripts/subliminal_delta_kd_common.sh"
mode="${1:?Usage: submit_subliminal_kd_remaining.sh phase4|phase5}"
case "$mode" in
  phase4)
    python "$REPO/scripts/report_subliminal_kd_student.py" --experiment_root "$EXPERIMENT_ROOT" --phase 3
    python "$REPO/scripts/prepare_subliminal_delta_kd_extension.py" --experiment_root "$EXPERIMENT_ROOT"
    # Atomic directory creation prevents duplicate or uncertain submissions.
    receipt="$EXPERIMENT_ROOT/phase4/submission"
    mkdir "$receipt" || { echo "Submission receipt exists; inspect jobs before manual recovery" >&2; exit 1; }
    a100="$(sbatch --parsable --account="${A100_ACCOUNT:-jamiemmt}" --partition=gpu-a100 \
      --gpus-per-node=a100:2 --export=ALL,HARDWARE_FAMILY=a100 \
      "$REPO/scripts/sbatch_subliminal_kd_extension.sbatch")"
    printf '%s\n' "$a100" > "$receipt/a100.jobid"
    l40s="$(sbatch --parsable --account="${L40S_ACCOUNT:-ckpt-cse}" --partition="${L40S_PARTITION:-ckpt-g2}" \
      --gpus-per-node=l40s:2 --export=ALL,HARDWARE_FAMILY=l40s \
      "$REPO/scripts/sbatch_subliminal_kd_extension.sbatch")"
    printf '%s\n' "$l40s" > "$receipt/l40s.jobid"
    finalizer="$(sbatch --parsable --dependency="afterok:$a100:$l40s" "$REPO/scripts/sbatch_subliminal_kd_full_finalize.sbatch")"
    printf '%s\n' "$finalizer" > "$receipt/finalizer.jobid"
    printf 'A100=%s L40S=%s finalizer=%s\n' "$a100" "$l40s" "$finalizer"
    ;;
  phase5)
    python - "$EXPERIMENT_ROOT/phase4/phase4_report.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p["status"] == "pass" and all(p["gates"].values())
assert p["curated_occurrences"] >= 4096
PY
    mkdir -p "$EXPERIMENT_ROOT/phase5"
    receipt="$EXPERIMENT_ROOT/phase5/submission"
    mkdir "$receipt" || { echo "Submission receipt exists; inspect before manual recovery" >&2; exit 1; }
    student="$(sbatch --parsable --account="${A100_ACCOUNT:-jamiemmt}" --partition=gpu-a100 \
      --gpus-per-node=a100:1 "$REPO/scripts/sbatch_subliminal_kd_final_student.sbatch")"
    printf '%s\n' "$student" > "$receipt/student.jobid"
    printf 'Final student/evaluation=%s\n' "$student"
    ;;
  *) echo "Expected phase4 or phase5" >&2; exit 1 ;;
esac
