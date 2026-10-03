#!/bin/bash
#SBATCH --job-name=unc_eagle_diag
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/unc_eagle_diag_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/unc_eagle_diag_%j.err

export OUTPUT_ROOT="${OUTPUT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_uncoupled_robust_topaz_end_joke_robust}"
export ROBUST_REF_NAME="${ROBUST_REF_NAME:-eagle_early_joke_uncoupled_robust}"
export MODEL_SPECS="${MODEL_SPECS:-$OUTPUT_ROOT/m2_eagle_early_joke_uncoupled_robust_topaz_end_joke_robust_model_specs.json}"
export STANDALONE_MODE=uncoupled
export NO_PROMPT_START_GATE=1

exec bash scripts/sbatch_robust_eagle_diagnostic.sbatch
