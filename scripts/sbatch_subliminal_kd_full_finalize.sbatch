#!/bin/bash
#SBATCH --job-name=sub_delta_full_finalize
#SBATCH --account=ckpt-cse
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1/logs/%x_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1/logs/%x_%j.err
set -Eeuo pipefail
REPO=/mmfs1/home/adhyyan/subliminal-mitigate-subliminal-delta-kd-stages
source "$REPO/scripts/subliminal_delta_kd_common.sh"
python "$REPO/scripts/finalize_subliminal_kd_extension.py" --experiment_root "$EXPERIMENT_ROOT"
