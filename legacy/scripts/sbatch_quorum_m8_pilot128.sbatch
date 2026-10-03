#!/bin/bash
#SBATCH --job-name=joke_m8_pilot128
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/joke_m8_pilot128_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/joke_m8_pilot128_%j.err

set -eo pipefail

cd /mmfs1/home/adhyyan/subliminal-mitigate

if [[ -z "${QUORUM_MODE:-}" ]]; then
  export QUORUM_MODE=pilot128
fi

if [[ -z "${QUORUM_EXTRA_ARGS:-}" ]]; then
  export QUORUM_EXTRA_ARGS="--subset m8_all --aggregator min,quorum_q3 --skip_diagnostics"
fi

bash scripts/sbatch_quorum_m8_sweep.sbatch
