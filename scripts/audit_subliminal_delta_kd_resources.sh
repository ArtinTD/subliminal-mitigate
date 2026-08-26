#!/usr/bin/env bash
# Read-only Hyak resource/artifact audit for subliminal pi_Delta KD Phase 0.

set -Eeo pipefail
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1}"
OUTPUT_DIR="$EXPERIMENT_ROOT/phase0"
mkdir -p "$OUTPUT_DIR"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
report="$OUTPUT_DIR/resource_artifact_audit_${timestamp}.txt"
outputs_root="/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs"

{
  echo "timestamp_utc=$timestamp"
  echo "hostname=$(hostname)"
  echo "user=$(id -un)"
  echo
  echo "=== current jobs ==="
  squeue -u "$(id -un)" -o '%.18i %.14P %.20j %.2t %.10M %.6D %.20R %.24b'
  echo
  echo "=== hyakalloc ==="
  hyakalloc 2>/dev/null || echo "hyakalloc unavailable"
  echo
  echo "=== associations ==="
  sacctmgr -n -P show assoc where user="$(id -un)" \
    format=Cluster,Account,User,Partition,QOS,GrpTRES,MaxTRES,MaxTRESPJ,MaxJobs,MaxSubmitJobs
  echo
  echo "=== relevant gpu partitions ==="
  sinfo -h -o '%P|%G|%a|%l|%D|%t|%N' | grep -E 'gpu-(a100|l40s)|ckpt-(g2|all)' || true
  echo
  echo "=== expected final-paper artifact root ==="
  if test -e "$outputs_root/composed_joke_subliminal_cost"; then
    du -sh "$outputs_root/composed_joke_subliminal_cost"
    find "$outputs_root/composed_joke_subliminal_cost" -maxdepth 4 -type f -printf '%P|%s\n' | sort
  else
    echo "MISSING $outputs_root/composed_joke_subliminal_cost"
  fi
  echo
  echo "=== surviving Panda/Eagle trait-only skeletons ==="
  for path in \
    "$outputs_root/subliminal_trait_only_sweep/datasets/stage_a/panda" \
    "$outputs_root/subliminal_trait_only_sweep/datasets/stage_a/eagle" \
    "$outputs_root/subliminal_trait_only_sweep/models/stage_a/panda" \
    "$outputs_root/subliminal_trait_only_sweep/models/stage_a/eagle"
  do
    if test -e "$path"; then
      du -sh "$path"
      count="$(find "$path" -type f | wc -l)"
      echo "file_count=$count path=$path"
    else
      echo "MISSING $path"
    fi
  done
} | tee "$report"

echo "Wrote $report"
