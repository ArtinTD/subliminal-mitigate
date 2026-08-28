#!/usr/bin/env bash
set -euo pipefail

ROOT="/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1/phase1/repair_trait_only/joke_continuation_pilot"
mkdir -p "$ROOT/logs" "$ROOT/provenance"

panda_job=$(sbatch --parsable --job-name=sub_joke_panda --export=ALL,SOURCE=panda \
  scripts/sbatch_subliminal_joke_continuation_train.sbatch)
eagle_job=$(sbatch --parsable --job-name=sub_joke_eagle --export=ALL,SOURCE=eagle \
  scripts/sbatch_subliminal_joke_continuation_train.sbatch)
eval_job=$(sbatch --parsable --dependency="afterok:${panda_job}:${eagle_job}" \
  scripts/sbatch_subliminal_joke_continuation_eval.sbatch)

python - "$ROOT/provenance/job_manifest.json" "$panda_job" "$eagle_job" "$eval_job" <<'PY'
import datetime, json, sys
path, panda, eagle, evaluation = sys.argv[1:]
with open(path, "w") as handle:
    json.dump({
        "submitted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "panda_training_job": panda,
        "eagle_training_job": eagle,
        "evaluation_job": evaluation,
        "trajectory_steps": 216,
        "snapshot_steps": [5, 10, 20, 40, 72, 144, 216],
        "joke_examples_per_lineage": 4286,
        "effective_batch_size": 60,
        "parent_number_examples": 10000,
        "parent_number_epochs": 10,
    }, handle, indent=2)
PY

printf 'panda_training=%s\neagle_training=%s\nevaluation=%s\n' \
  "$panda_job" "$eagle_job" "$eval_job"
