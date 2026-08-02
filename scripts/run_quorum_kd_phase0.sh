#!/usr/bin/env bash
# Phase 0 only: no GPU submissions and no experiment generations.

set -Eeo pipefail
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$REPO"

echo "=== Git state ==="
git rev-parse HEAD
git status --short

echo "=== Syntax ==="
"$PYTHON_BIN" -m py_compile \
  scripts/build_quorum_kd_manifest.py \
  scripts/sample_batched_quorum_distillation.py

echo "=== Manifest invariants ==="
"$PYTHON_BIN" scripts/build_quorum_kd_manifest.py --self_test

echo "=== Atomic batch resume equivalence ==="
"$PYTHON_BIN" scripts/sample_batched_quorum_distillation.py --self_test

echo "Phase 0 local tests passed. No GPU job was submitted."
