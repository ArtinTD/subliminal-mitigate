#!/bin/bash
#SBATCH --job-name=joke_m4_drop_p128
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/joke_m4_drop_p128_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/joke_m4_drop_p128_%j.err

set -euo pipefail

export QUORUM_MODE="${QUORUM_MODE:-pilot128}"
export QUORUM_CONFIG="${QUORUM_CONFIG:-configs/composed/quorum_dropout_m4_sweep.yaml}"
export MODEL_SPECS_FILE="${MODEL_SPECS_FILE:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/quorum_dropout_m4/m4_dropout_model_specs.json}"
export QUORUM_OUTPUT_ROOT="${QUORUM_OUTPUT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/quorum_dropout_m4}"
export QUORUM_DEVICES="${QUORUM_DEVICES:-cuda:0,cuda:1}"

if [[ -z "${QUORUM_EXTRA_ARGS:-}" ]]; then
  export QUORUM_EXTRA_ARGS="--subset m4_three_benefit_one_dropout --aggregator min,quorum_q3 --skip_diagnostics"
fi

bash scripts/sbatch_quorum_m8_sweep.sbatch
