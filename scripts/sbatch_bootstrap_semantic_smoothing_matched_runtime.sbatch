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
: "${MATCHED_RUNTIME_ROOT:?Persistent matched-runtime root required}"
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
RUNTIME="$MATCHED_RUNTIME_ROOT"
PREFIX="$RUNTIME/python"
mkdir -p "$RUNTIME" "$ABLATION_ROOT/logs"
if [[ ! -x "$PREFIX/bin/python" ]]; then
    conda create --yes --prefix "$PREFIX" python=3.11.16 pip
fi
"$PREFIX/bin/python" -m pip install \
    --requirement "$SEMANTIC_REPO/requirements/semantic_smoothing_matched_runtime.txt"

export HF_HOME="$RUNTIME/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export SENTENCE_TRANSFORMERS_HOME="$RUNTIME/encoder_cache"
export HF_HUB_DISABLE_XET=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SEMANTIC_REPO/scripts:$SEMANTIC_REPO"
"$PREFIX/bin/python" "$SEMANTIC_REPO/scripts/semantic_smoothing_matched_runtime.py" \
    --download --output "$RUNTIME/runtime_recovery_audit.json"
