#!/bin/bash
#SBATCH --job-name=joke_be_lookback
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/joke_be_lookback_p128_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/joke_be_lookback_p128_%j.err

set -euo pipefail

export QUORUM_MODE="${QUORUM_MODE:-pilot128}"
export QUORUM_CONFIG="${QUORUM_CONFIG:-configs/composed/joke_begin_end_sweep.yaml}"
export MODEL_SPECS_FILE="${MODEL_SPECS_FILE:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/joke_begin_end/m2_joke_begin_end_model_specs.json}"
export QUORUM_OUTPUT_ROOT="${QUORUM_OUTPUT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/joke_begin_end}"
export QUORUM_DEVICES="${QUORUM_DEVICES:-cuda:0,cuda:1}"

if [[ -z "${QUORUM_EXTRA_ARGS:-}" ]]; then
  export QUORUM_EXTRA_ARGS="--subset beginning_end --aggregator lookback_gated_a1,lookback_gated_a0.995,lookback_gated_a0.99 --skip_diagnostics"
fi

bash scripts/sbatch_quorum_m8_sweep.sbatch
