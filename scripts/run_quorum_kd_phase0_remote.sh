#!/usr/bin/env bash

set -Eeo pipefail
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR

REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-explicit-min-distill}"
source "$REPO/scripts/quorum_kd_common.sh"

PHASE0_ROOT="$EXPERIMENT_ROOT/phase0"
DATA_ROOT="$EXPERIMENT_ROOT/data"
CALIBRATION_MANIFEST_ROOT="$EXPERIMENT_ROOT/phase1/manifests"
mkdir -p "$PHASE0_ROOT" "$DATA_ROOT" "$CALIBRATION_MANIFEST_ROOT"

"$REPO/scripts/audit_quorum_kd_resources.sh"
"$REPO/scripts/run_quorum_kd_phase0.sh"

for source in eagle topaz birch cobalt_cost_only; do
  [[ -f "$SOURCE_ROOT/$source/state.json" ]] || {
    echo "Missing source dataset: $SOURCE_ROOT/$source" >&2
    exit 1
  }
done
for adapter_path in $(python -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["models"].values()))' "$MODEL_SPECS"); do
  [[ -f "$adapter_path/adapter_config.json" && -f "$adapter_path/adapter_model.safetensors" ]] || {
    echo "Incomplete adapter: $adapter_path" >&2
    exit 1
  }
done

if [[ ! -f "$DATA_ROOT/prompt_manifest.meta.json" ]]; then
  python scripts/build_quorum_kd_manifest.py \
    --source "eagle=$SOURCE_ROOT/eagle" \
    --source "topaz=$SOURCE_ROOT/topaz" \
    --source "birch=$SOURCE_ROOT/birch" \
    --source "cobalt_cost_only=$SOURCE_ROOT/cobalt_cost_only" \
    --eval_config configs/composed/first_line_joke_m4_union_sources.yaml \
    --output_dir "$DATA_ROOT"
fi

if [[ ! -f "$CALIBRATION_MANIFEST_ROOT/a100/canonical_batches.json" ]]; then
  python scripts/build_quorum_kd_calibration.py \
    --prompt_manifest "$DATA_ROOT/prompt_manifest.json" \
    --output_dir "$CALIBRATION_MANIFEST_ROOT"
fi

python - "$REPO" "$EXPERIMENT_ROOT" "$MODEL_SPECS" <<'PY'
import datetime
import json
import pathlib
import subprocess
import sys

repo, root, model_specs = map(pathlib.Path, sys.argv[1:])
meta = json.load(open(root / "data/prompt_manifest.meta.json"))
batches = json.load(open(root / "data/canonical_batches.json"))
pilot = json.load(open(root / "data/pilot_batch_ids.json"))
payload = {
    "phase": 0,
    "status": "complete",
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "git_sha": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
    "model_specs": str(model_specs),
    "manifest": {
        "total_occurrences": meta["total_occurrences"],
        "unique_prompts": meta["unique_prompts"],
        "duplicate_occurrences": meta["duplicate_occurrences"],
        "evaluation_overlap_count": meta["evaluation_overlap_count"],
        "record_manifest_sha256": meta["record_manifest_sha256"],
        "canonical_batch_manifest_sha256": meta["canonical_batch_manifest_sha256"],
        "canonical_batches": len(batches["batches"]),
        "pilot_batches": pilot["n_batches"],
        "pilot_occurrences": pilot["n_occurrences"],
    },
    "tests": {
        "syntax": "passed",
        "manifest_invariants": "passed",
        "batched_toy_equivalence": "passed",
        "atomic_batch_resume": "passed",
    },
    "gpu_jobs_submitted": False,
}
path = root / "phase0/phase0_report.json"
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
md = root / "phase0/phase0_report.md"
md.write_text(
    "# Quorum KD Phase 0\n\n"
    f"- Status: complete\n- Git SHA: `{payload['git_sha']}`\n"
    f"- Occurrences: {meta['total_occurrences']} ({meta['unique_prompts']} unique)\n"
    f"- Duplicate occurrences: {meta['duplicate_occurrences']}\n"
    f"- Evaluation overlap: {meta['evaluation_overlap_count']}\n"
    f"- Canonical batches: {len(batches['batches'])}\n"
    f"- Pilot: {pilot['n_occurrences']} occurrences in {pilot['n_batches']} batches\n"
    "- Local/toy tests: passed\n- GPU jobs submitted: no\n"
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
