#!/bin/bash
#SBATCH --job-name=mscd-environment
#SBATCH --account=ckpt-cse
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
: "${MSCD_ROOT:?Set MSCD_ROOT to the isolated checkout}"
: "${MSCD_ENV:?Set MSCD_ENV to a new environment directory}"
export UV_CACHE_DIR=/gscratch/scrubbed/adhyyan/.cache/uv-mscd
uv=/mmfs1/home/adhyyan/.local/bin/uv
if [[ ! -x "$MSCD_ENV/bin/python" ]]; then "$uv" venv --python 3.11 "$MSCD_ENV"; fi
"$uv" pip install --index-strategy unsafe-best-match --python "$MSCD_ENV/bin/python" -r "$MSCD_ROOT/requirements.txt" -c "$MSCD_ROOT/configs/mscd/environment-constraints.txt" pytest
"$uv" pip install --python "$MSCD_ENV/bin/python" --no-deps -e "$MSCD_ROOT"
"$uv" pip freeze --python "$MSCD_ENV/bin/python" > "$MSCD_ROOT/environment-freeze.txt"
