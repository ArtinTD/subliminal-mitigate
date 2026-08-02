#!/usr/bin/env bash
# Read-only Phase-0 Hyak resource audit for the mixed A100/L40S experiment.

set -Eeo pipefail
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/quorum_dropout_m4/quorum_q3_distillation_v1}"
OUTPUT_DIR="$EXPERIMENT_ROOT/phase0"
mkdir -p "$OUTPUT_DIR"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
report="$OUTPUT_DIR/resource_audit_${timestamp}.txt"

{
  echo "timestamp_utc=$timestamp"
  echo "hostname=$(hostname)"
  echo "user=$(id -un)"
  echo
  echo "=== groups ==="
  id
  echo
  echo "=== hyakalloc ==="
  if command -v hyakalloc >/dev/null 2>&1; then
    hyakalloc
  else
    echo "hyakalloc unavailable"
  fi
  echo
  echo "=== associations ==="
  sacctmgr -n -P show assoc where user="$(id -un)" \
    format=Cluster,Account,User,Partition,QOS,GrpTRES,MaxTRES,MaxTRESPJ,MaxJobs,MaxSubmitJobs
  echo
  echo "=== qos ==="
  sacctmgr -n -P show user where name="$(id -un)" withassoc \
    format=User,DefaultAccount,Account,Partition,QOS
  echo
  echo "=== gpu partitions ==="
  sinfo -h -o '%P|%G|%a|%l|%D|%t|%N' | grep -E 'gpu-(a100|l40s)|ckpt-(g2|all)' || true
  echo
  echo "=== current jobs ==="
  squeue -u "$(id -un)" -o '%.18i %.14P %.20j %.2t %.10M %.6D %.20R %.24b'
  echo
  echo "=== recent association-limit jobs ==="
  sacct -S now-7days -u "$(id -un)" -X \
    --format=JobID,JobName,Account,Partition,State,Elapsed,ReqTRES%60,NodeList%24,Reason%48
} | tee "$report"

echo "Wrote $report"
