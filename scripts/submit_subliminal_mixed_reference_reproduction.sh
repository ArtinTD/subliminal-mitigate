#!/usr/bin/env bash
set -euo pipefail

ROOT="/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1/phase1/mixed_reference_reproduction"
mkdir -p "$ROOT/logs" "$ROOT/provenance"
job=$(sbatch --parsable scripts/sbatch_subliminal_mixed_reference_reproduction.sbatch)
python - "$ROOT/provenance/job_manifest.json" "$job" <<'PY'
import datetime
import json
import sys

path, job = sys.argv[1:]
with open(path, "w") as handle:
    json.dump({
        "submitted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "job_id": job,
        "recipe": {
            "panda_number_rows": 10000,
            "panda_joke_rows": 300,
            "panda_optimizer_steps": 516,
            "eagle_number_rows": 10000,
            "eagle_joke_rows": 200,
            "eagle_optimizer_steps": 510,
            "selection_seed": 42,
            "shuffle_seed": 42,
        },
    }, handle, indent=2)
PY
echo "mixed_reference_reproduction=$job"
