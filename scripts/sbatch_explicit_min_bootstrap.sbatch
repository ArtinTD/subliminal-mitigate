#!/bin/bash
#SBATCH --job-name=explicit_min_env
#SBATCH --account=ckpt-cse
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/explicit_prefix_min_distillation_v1/logs/%x_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/explicit_prefix_min_distillation_v1/logs/%x_%j.err

set -Eeo pipefail
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR

REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-explicit-min-distill}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/explicit_prefix_min_distillation_v1}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate-explicit-min-distill-py311}"
CONDA_BASE="${CONDA_BASE:-/sw/contrib/foster-src/python/miniconda/3.8}"

if [[ -f "$CONDA_ENV_PREFIX/.bootstrap_complete" ]]; then
  "$CONDA_ENV_PREFIX/bin/python" -c 'import datasets, peft, torch, transformers, trl, unsloth, vllm, yaml'
  echo "Validated existing experiment environment: $CONDA_ENV_PREFIX"
  exit 0
fi
if [[ -e "$CONDA_ENV_PREFIX" ]]; then
  echo "Refusing to overwrite incomplete environment: $CONDA_ENV_PREFIX" >&2
  exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda create --prefix "$CONDA_ENV_PREFIX" python=3.11 pip -y
"$CONDA_ENV_PREFIX/bin/python" -m pip install --upgrade pip
"$CONDA_ENV_PREFIX/bin/python" -m pip install -r "$REPO/requirements.txt"
"$CONDA_ENV_PREFIX/bin/python" -c 'import datasets, peft, torch, transformers, trl, unsloth, vllm, yaml'

mkdir -p "$EXPERIMENT_ROOT/provenance"
"$CONDA_ENV_PREFIX/bin/python" -m pip freeze > "$EXPERIMENT_ROOT/provenance/pip_freeze.txt"
"$CONDA_ENV_PREFIX/bin/python" -c \
  'import importlib.metadata as m, json, pathlib; names=["torch","transformers","tokenizers","datasets","peft","trl","accelerate","unsloth","vllm"]; pathlib.Path("'"$EXPERIMENT_ROOT"'/provenance/package_versions.json").write_text(json.dumps({n:m.version(n) for n in names}, indent=2)+"\n")'
git -C "$REPO" rev-parse HEAD > "$EXPERIMENT_ROOT/provenance/git_sha.txt"
touch "$CONDA_ENV_PREFIX/.bootstrap_complete"
echo "Environment bootstrap complete: $CONDA_ENV_PREFIX"
