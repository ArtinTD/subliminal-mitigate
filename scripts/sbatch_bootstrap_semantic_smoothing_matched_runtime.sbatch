#!/usr/bin/env bash
#SBATCH --job-name=semantic_matched_runtime
#SBATCH --account=ckpt-cse
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00

set -euo pipefail
: "${ABLATION_ROOT:?Isolated matched-ablation root required}"
: "${SEMANTIC_REPO:?Semantic checkout required}"
: "${MATCHED_RUNTIME_ROOT:?Final matched-runtime root required}"
[[ "$ABLATION_ROOT" == */semantic_smoothing_matched_ablation_v1 ]] || exit 1
[[ "$MATCHED_RUNTIME_ROOT" == */semantic_smoothing_matched_runtime_v1 ]] || exit 1

CONDA_BASE="${CONDA_BASE:-/sw/contrib/foster-src/python/miniconda/3.8}"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "Could not find conda." >&2
    exit 1
fi
BUILD_ROOT="/tmp/semantic-smoothing-matched-runtime-${SLURM_JOB_ID}"
PREFIX="$BUILD_ROOT/python"
BUILD_AUDIT="/tmp/semantic-smoothing-matched-runtime-${SLURM_JOB_ID}-audit.json"
test ! -e "$BUILD_ROOT"
test ! -e "$MATCHED_RUNTIME_ROOT"
mkdir -p "$BUILD_ROOT" "$ABLATION_ROOT/logs"
trap 'rm -rf "$BUILD_ROOT" "$BUILD_AUDIT"' EXIT
conda create --yes --prefix "$PREFIX" python=3.11.16 pip
"$PREFIX/bin/python" -m pip install \
    --requirement "$SEMANTIC_REPO/requirements/semantic_smoothing_matched_runtime.txt"

export HF_HOME="$BUILD_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export SENTENCE_TRANSFORMERS_HOME="$BUILD_ROOT/encoder_cache"
export HF_HUB_DISABLE_XET=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SEMANTIC_REPO/scripts:$SEMANTIC_REPO"
"$PREFIX/bin/python" "$SEMANTIC_REPO/scripts/semantic_smoothing_matched_runtime.py" \
    --download --output "$BUILD_AUDIT"

mkdir -p "$MATCHED_RUNTIME_ROOT"
tar --create --file=- --directory="$BUILD_ROOT" . | \
    tar --extract --touch --file=- --directory="$MATCHED_RUNTIME_ROOT"

FINAL_PREFIX="$MATCHED_RUNTIME_ROOT/python"
export HF_HOME="$MATCHED_RUNTIME_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export SENTENCE_TRANSFORMERS_HOME="$MATCHED_RUNTIME_ROOT/encoder_cache"
"$FINAL_PREFIX/bin/python" "$SEMANTIC_REPO/scripts/semantic_smoothing_matched_runtime.py" \
    --output "$MATCHED_RUNTIME_ROOT/runtime_recovery_audit.json"
