#!/bin/bash
# Submit the complete afterok dependency chain for explicit-min distillation.

set -Eeo pipefail
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR

REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-explicit-min-distill}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/explicit_prefix_min_distillation_v1}"
mkdir -p "$EXPERIMENT_ROOT/logs" "$EXPERIMENT_ROOT/provenance"

set +e
environment_job="$(sbatch --parsable "$REPO/scripts/sbatch_explicit_min_bootstrap.sbatch")"
environment_status=$?
set -e
environment_variant="ckpt"
if (( environment_status != 0 )); then
  echo "CPU bootstrap submission was rejected; trying the one permitted L40S fallback." >&2
  environment_job="$(sbatch --parsable "$REPO/scripts/sbatch_explicit_min_bootstrap_l40s.sbatch")"
  environment_variant="gpu-l40s"
fi

references_job="$(sbatch --parsable \
  --dependency="afterok:$environment_job" \
  "$REPO/scripts/sbatch_explicit_min_references.sbatch")"
teacher_job="$(sbatch --parsable \
  --dependency="afterok:$references_job" \
  "$REPO/scripts/sbatch_explicit_min_teacher_array.sbatch")"
finalizer_job="$(sbatch --parsable \
  --dependency="afterok:$teacher_job" \
  "$REPO/scripts/sbatch_explicit_min_finalize.sbatch")"
student_job="$(sbatch --parsable \
  --dependency="afterok:$finalizer_job" \
  "$REPO/scripts/sbatch_explicit_min_student.sbatch")"

python3 -c \
  'import json, pathlib, sys; path=pathlib.Path(sys.argv[1]); payload={"environment":{"job_id":sys.argv[2],"variant":sys.argv[3]},"references":{"job_id":sys.argv[4]},"teacher_array":{"job_id":sys.argv[5],"partition":"gpu-a100","array":"0-3%4","submitted_at_epoch":__import__("time").time()},"finalizer":{"job_id":sys.argv[6]},"student":{"job_id":sys.argv[7]}}; path.write_text(json.dumps(payload,indent=2)+"\n")' \
  "$EXPERIMENT_ROOT/provenance/jobs.json" \
  "$environment_job" "$environment_variant" "$references_job" "$teacher_job" \
  "$finalizer_job" "$student_job"

echo "environment=$environment_job ($environment_variant)"
echo "references=$references_job"
echo "teacher_array=$teacher_job"
echo "finalizer=$finalizer_job"
echo "student=$student_job"
