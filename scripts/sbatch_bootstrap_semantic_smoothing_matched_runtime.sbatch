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
[[ "$ABLATION_ROOT" == */semantic_smoothing_matched_ablation_v1 ]] || exit 1

source /etc/profile.d/modules.sh
module load foster/python/miniconda/3.8
RUNTIME="$ABLATION_ROOT/runtime/matched-v1"
PREFIX="$RUNTIME/python"
mkdir -p "$RUNTIME" "$ABLATION_ROOT/logs"
if [[ ! -x "$PREFIX/bin/python" ]]; then
    conda create --yes --prefix "$PREFIX" python=3.11.16 pip
fi
"$PREFIX/bin/python" -m pip install \
    --requirement "$SEMANTIC_REPO/requirements/semantic_smoothing_matched_runtime.txt"

export HF_HOME=/gscratch/scrubbed/adhyyan/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export SENTENCE_TRANSFORMERS_HOME="$ABLATION_ROOT/runtime/encoder_cache"
export HF_HUB_DISABLE_XET=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SEMANTIC_REPO/scripts:$SEMANTIC_REPO"
"$PREFIX/bin/python" "$SEMANTIC_REPO/scripts/semantic_smoothing_matched_runtime.py" \
    --download --output "$RUNTIME/runtime_recovery_audit.json"
