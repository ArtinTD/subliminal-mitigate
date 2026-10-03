#!/bin/bash
#SBATCH --job-name=unc_eagle_train
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/unc_eagle_train_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/unc_eagle_train_%j.err

export OUTPUT_ROOT="${OUTPUT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_uncoupled_robust_topaz_end_joke_robust}"
export ROBUST_CONFIG="${ROBUST_CONFIG:-configs/composed/robust_eagle_early_joke_uncoupled.yaml}"
export ROBUST_REF_NAME="${ROBUST_REF_NAME:-eagle_early_joke_uncoupled_robust}"
export ROBUST_MODEL_NAME="${ROBUST_MODEL_NAME:-pi_eagle_early_joke_uncoupled_robust}"
export MODEL_SPECS="${MODEL_SPECS:-$OUTPUT_ROOT/m2_eagle_early_joke_uncoupled_robust_topaz_end_joke_robust_model_specs.json}"

exec bash scripts/sbatch_robust_eagle_training.sbatch
