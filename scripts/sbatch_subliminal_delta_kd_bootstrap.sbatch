#!/bin/bash
#SBATCH --job-name=sub_delta_env
#SBATCH --account=ckpt-cse
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1/logs/%x_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1/logs/%x_%j.err

set -Eeo pipefail
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR

REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate-subliminal-delta-kd}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/gscratch/scrubbed/adhyyan/envs/subliminal-delta-kd-py311}"
CONDA_BASE="${CONDA_BASE:-/sw/contrib/foster-src/python/miniconda/3.8}"

validate_environment() {
  test -x "$CONDA_ENV_PREFIX/bin/nvcc"
  "$CONDA_ENV_PREFIX/bin/python" -c 'import datasets, peft, torch, transformers, trl, vllm, yaml; import importlib.metadata as m; assert m.version("unsloth")'
}

mkdir -p "$EXPERIMENT_ROOT/logs" "$EXPERIMENT_ROOT/provenance"
if [[ -f "$CONDA_ENV_PREFIX/.bootstrap_complete" ]]; then
  validate_environment
  echo "Validated existing environment: $CONDA_ENV_PREFIX"
  exit 0
fi
if [[ -e "$CONDA_ENV_PREFIX" && ! -x "$CONDA_ENV_PREFIX/bin/python" ]]; then
  echo "Malformed environment exists at $CONDA_ENV_PREFIX" >&2
  exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
if [[ ! -e "$CONDA_ENV_PREFIX" ]]; then
  conda create --prefix "$CONDA_ENV_PREFIX" python=3.11 pip -y
fi
conda install --prefix "$CONDA_ENV_PREFIX" -c nvidia cuda-nvcc=12.9.86 -y
"$CONDA_ENV_PREFIX/bin/python" -m pip install --upgrade pip
"$CONDA_ENV_PREFIX/bin/python" -m pip install -r "$REPO/requirements.txt"
# Unsloth's unconstrained optional dependency currently resolves to torchao
# 0.18, which requires torch >=2.11 and breaks Transformers imports with this
# repository's deliberate torch==2.9.0 pin.  No experiment path uses torchao;
# leaving it absent makes Transformers correctly treat that backend as
# unavailable while preserving the frozen training/runtime stack.
"$CONDA_ENV_PREFIX/bin/python" -m pip uninstall -y torchao
validate_environment
"$CONDA_ENV_PREFIX/bin/python" -m pip freeze > "$EXPERIMENT_ROOT/provenance/pip_freeze.txt"
git -C "$REPO" rev-parse HEAD > "$EXPERIMENT_ROOT/provenance/git_sha.txt"
touch "$CONDA_ENV_PREFIX/.bootstrap_complete"
echo "Environment bootstrap complete"
