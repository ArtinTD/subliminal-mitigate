#!/bin/bash
#SBATCH --job-name=joke_humor_ksmooth
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/joke_humor_ksmooth_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/joke_humor_ksmooth_%j.err

set -euo pipefail

export QUORUM_MODE="${QUORUM_MODE:-pilot128}"
export QUORUM_CONFIG="${QUORUM_CONFIG:-configs/composed/joke_vs_humor_sweep.yaml}"
export MODEL_SPECS_FILE="${MODEL_SPECS_FILE:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/joke_vs_humor/m2_joke_vs_humor_model_specs.json}"
export QUORUM_OUTPUT_ROOT="${QUORUM_OUTPUT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/joke_vs_humor}"
export QUORUM_DEVICES="${QUORUM_DEVICES:-cuda:0,cuda:1}"
export KERNEL_CACHE_DIR="${KERNEL_CACHE_DIR:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/joke_vs_humor/kernel_smoothing/kernel_cache}"

if [[ -z "${KERNEL_AGGREGATORS:-}" ]]; then
  export KERNEL_AGGREGATORS="kernel_input_l0.5,kernel_input_l0.5_gated,kernel_output_l0.5,kernel_output_l0.5_gated"
fi

if [[ -z "${QUORUM_EXTRA_ARGS:-}" ]]; then
  export QUORUM_EXTRA_ARGS="--subset joke_humor --aggregator $KERNEL_AGGREGATORS --skip_diagnostics --kernel_cache_dir $KERNEL_CACHE_DIR"
fi

bash scripts/sbatch_quorum_m8_sweep.sbatch
