#!/usr/bin/env bash
# Submit the approved Phase-4 extension and gated Phase-5 final student.
set -Eeo pipefail

REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-explicit-min-distill}"
source "$REPO/scripts/quorum_kd_common.sh"
A100_ACCOUNT="${A100_ACCOUNT:-jamiemmt}"
A100_PARTITION="${A100_PARTITION:-gpu-a100}"
L40S_ACCOUNT="${L40S_ACCOUNT:-cse}"
L40S_PARTITION="${L40S_PARTITION:-gpu-l40s}"

python - "$EXPERIMENT_ROOT/phase2/phase2_report.json" "$EXPERIMENT_ROOT/phase3/phase3_report.json" <<'PY'
import json,sys
for path in sys.argv[1:]:
    d=json.load(open(path)); assert d["status"] == "pass", (path, d.get("status"))
PY
[[ ! -e "$EXPERIMENT_ROOT/phase4/phase4_report.json" ]] || { echo "Phase 4 report already exists; refusing resubmission" >&2; exit 1; }
[[ ! -e "$EXPERIMENT_ROOT/phase4/full_student_sft_dataset" ]] || { echo "Phase 4 dataset already exists; refusing resubmission" >&2; exit 1; }
[[ ! -e "$EXPERIMENT_ROOT/phase5" ]] || { echo "Phase 5 output already exists; refusing resubmission" >&2; exit 1; }

python "$REPO/scripts/build_quorum_kd_extension_selection.py" \
  --canonical "$EXPERIMENT_ROOT/data/canonical_batches.json" \
  --pilot "$EXPERIMENT_ROOT/data/pilot_batch_ids.json" \
  --output "$EXPERIMENT_ROOT/data/extension_batch_ids.json"

expected_sha="$(python - "$EXPERIMENT_ROOT/generations/a100/checkpoints/checkpoint_meta.json" "$EXPERIMENT_ROOT/generations/l40s/checkpoints/checkpoint_meta.json" <<'PY'
import json,sys
values={json.load(open(path))["code_git_sha"] for path in sys.argv[1:]}
assert len(values) == 1, values
print(values.pop())
PY
)"
generator_files=(
  scripts/sample_batched_quorum_distillation.py
  scripts/sample_quorum_composition_generations.py
  scripts/build_quorum_kd_manifest.py
  configs/training.yaml
  configs/composed/first_line_joke_m4_union_sources.yaml
)
git -C "$REPO" diff --quiet "$expected_sha" -- "${generator_files[@]}" || {
  echo "Generator/config files differ from immutable checkpoint code SHA $expected_sha" >&2
  exit 1
}
identity_worktree="$EXPERIMENT_ROOT/runtime/generator_identity_$expected_sha"
if [[ ! -e "$identity_worktree/.git" ]]; then
  [[ ! -e "$identity_worktree" ]] || { echo "Conflicting generator identity path: $identity_worktree" >&2; exit 1; }
  git -C "$REPO" worktree add --detach "$identity_worktree" "$expected_sha"
fi
test "$(git -C "$identity_worktree" rev-parse HEAD)" = "$expected_sha"

a100="$(sbatch --parsable --account="$A100_ACCOUNT" --partition="$A100_PARTITION" --gpus-per-node=a100:2 \
  --export=ALL,HARDWARE_FAMILY=a100 "$REPO/scripts/sbatch_quorum_kd_phase4_teacher.sbatch")"
l40s="$(sbatch --parsable --account="$L40S_ACCOUNT" --partition="$L40S_PARTITION" --gpus-per-node=l40s:2 \
  --export=ALL,HARDWARE_FAMILY=l40s "$REPO/scripts/sbatch_quorum_kd_phase4_teacher.sbatch")"
finalizer="$(sbatch --parsable --dependency="afterok:$a100:$l40s" "$REPO/scripts/sbatch_quorum_kd_phase4_finalizer.sbatch")"
student="$(sbatch --parsable --dependency="afterok:$finalizer" --account="$A100_ACCOUNT" --partition="$A100_PARTITION" --gpus-per-node=a100:1 \
  "$REPO/scripts/sbatch_quorum_kd_phase5_final_student.sbatch")"
printf 'phase4_a100=%s\nphase4_l40s=%s\nphase4_finalizer=%s\nphase5_student=%s\n' "$a100" "$l40s" "$finalizer" "$student"
