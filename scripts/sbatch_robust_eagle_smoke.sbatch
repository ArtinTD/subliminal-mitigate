#!/bin/bash
#SBATCH --job-name=rob_eagle_smoke
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/rob_eagle_smoke_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/rob_eagle_smoke_%j.err

set -Eeo pipefail
trap 'status=$?; echo "ERROR status=$status line=$LINENO command=$BASH_COMMAND" >&2' ERR

CONDA_BASE="${CONDA_BASE:-/sw/contrib/foster-src/python/miniconda/3.8}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate-py311}"
CONDA_ENV_FALLBACK="${CONDA_ENV_FALLBACK:-/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate-py311-rebuild-36133632}"
REPO="${REPO:-/mmfs1/home/adhyyan/subliminal-mitigate}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  echo "Could not find conda." >&2
  exit 1
fi

env_is_usable() {
  local prefix="$1"
  [[ -x "$prefix/bin/python" ]] || return 1
  timeout 120 "$prefix/bin/python" - <<'PY'
import encodings
import peft
import torch
import transformers
import yaml
PY
}

resolve_conda_env() {
  local configured="$CONDA_ENV_PREFIX"
  local candidate
  for candidate in \
    "$configured" \
    "$CONDA_ENV_FALLBACK" \
    "$configured"-optimized \
    "$configured"-rebuild-* \
    "$configured"-temporal-robust-*; do
    [[ -d "$candidate" ]] || continue
    if env_is_usable "$candidate"; then
      CONDA_ENV_PREFIX="$candidate"
      echo "Using conda env: $CONDA_ENV_PREFIX"
      return
    fi
    echo "Conda env unusable: $candidate"
  done
  echo "No usable existing conda environment found." >&2
  exit 1
}

resolve_conda_env
conda activate "$CONDA_ENV_PREFIX"
set -u

export HF_HOME="${HF_HOME:-/gscratch/scrubbed/adhyyan/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/gscratch/scrubbed/adhyyan/.cache}"
export TMPDIR="${TMPDIR:-/gscratch/scrubbed/adhyyan/tmp}"
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$TMPDIR"
cd "$REPO"

OUTPUT_ROOT="${OUTPUT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_robust_topaz_end_joke_robust}"
OLD_ROBUST_ROOT="${OLD_ROBUST_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/eagle_early_joke_topaz_end_joke_robust}"
MODEL_SPECS="${MODEL_SPECS:-$OUTPUT_ROOT/m2_eagle_early_joke_robust_topaz_end_joke_robust_model_specs.json}"
COMPOSED_CONFIG="${COMPOSED_CONFIG:-configs/composed/eagle_early_joke_topaz_end_joke.yaml}"
DIAG_GATE="${DIAG_GATE:-$OUTPUT_ROOT/robust_eagle_diagnostic/robust_eagle_gate.json}"
HPARAM_WINNER="${HPARAM_WINNER:-$OLD_ROBUST_ROOT/rollout_history_hparam_smoke/winner.json}"
SMOKE_ROOT="${SMOKE_ROOT:-$OUTPUT_ROOT/robust_eagle_smoke}"
GEN_DIR="$SMOKE_ROOT/generations/eagle_early_joke_robust_topaz_end_joke_robust"
REPLAY_DIR="$SMOKE_ROOT/replay"
MERGED_DIR="$SMOKE_ROOT/model_average"
SUMMARY_JSON="$SMOKE_ROOT/comparison.json"
FORCE="${FORCE:-0}"

mkdir -p "$GEN_DIR" "$REPLAY_DIR" "$MERGED_DIR"
test -f "$MODEL_SPECS"
test -f "$DIAG_GATE"

python - "$DIAG_GATE" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
if payload.get("mode") != "robust_eagle_diagnostic" or payload.get("pass") is not True:
    raise RuntimeError("Robust Eagle diagnostic gate has not passed; refusing smoke")
print("Validated passing robust Eagle diagnostic gate")
PY

python - "$MODEL_SPECS" <<'PY'
import json
import os
import sys

payload = json.load(open(sys.argv[1]))
models = payload.get("models", payload)
required = ["eagle_early_joke_robust", "topaz_end_joke_robust"]
missing = {
    name: models.get(name)
    for name in required
    if not models.get(name)
    or not os.path.isfile(os.path.join(models[name], "adapter_config.json"))
}
if missing:
    raise FileNotFoundError(f"Missing robust smoke adapters: {missing}")
print("Validated robust smoke adapters:", ", ".join(required))
PY

date
hostname
nvidia-smi
git status --short
git log --oneline -3

python scripts/sample_quorum_composition_generations.py --self_test
python scripts/sample_merged_lora_generations.py --self_test
python scripts/audit_newline_exempt_lookback.py --self_test
python scripts/summarize_temporal_conditions.py --self_test

read -r WIN_BETA WIN_SAMPLES WIN_STEPS WIN_CONDITION < <(python - "$HPARAM_WINNER" <<'PY'
import json
import os
import sys

default = (0.65, 4, 160, "default_b0.65_s4_h160")
path = sys.argv[1]
if not os.path.isfile(path):
    print(*default)
    raise SystemExit
payload = json.load(open(path))
winner = payload.get("winner")
if not winner:
    print(*default)
    raise SystemExit
params = winner.get("params", {})
beta = params.get("beta", default[0])
samples = params.get("rollout_samples", default[1])
steps = params.get("rollout_steps", default[2])
condition = winner.get("condition", default[3])
print(beta, samples, steps, condition)
PY
)

echo "Robust smoke rollout setting: beta=$WIN_BETA samples=$WIN_SAMPLES steps=$WIN_STEPS source=$WIN_CONDITION"

label_float() {
  python - "$1" <<'PY'
import sys
print(f"{float(sys.argv[1]):g}".replace("e-0", "e-").replace("e+0", "e+"))
PY
}

WIN_BETA_LABEL="$(label_float "$WIN_BETA")"
MIN_JSON="$GEN_DIR/min.json"
CURRENT_CAPPED_JSON="$GEN_DIR/lookback_capped_mix_b0.65_a1.json"
ROLLOUT_PLUS_JSON="$GEN_DIR/lookback_rollout_plus_current_capped_mix_b${WIN_BETA_LABEL}_s${WIN_SAMPLES}_h${WIN_STEPS}_a1.json"
MERGED_JSON="$MERGED_DIR/equal_weight.json"
REPLAY_JSON="$REPLAY_DIR/robust_eagle_conditions.json"

generation_complete() {
  local path="$1"
  local composition="$2"
  [[ -f "$path" ]] || return 1
  python - "$path" "$composition" <<'PY' >/dev/null 2>&1
import json
import sys

payload = json.load(open(sys.argv[1]))
meta = payload.get("meta", {})
summary = payload.get("summary", {})
valid = (
    meta.get("ref_names") == ["eagle_early_joke_robust", "topaz_end_joke_robust"]
    and meta.get("composition_type") == sys.argv[2]
    and meta.get("num_prompts") == 8
    and meta.get("n_samples_per_prompt") == 2
    and meta.get("temperature") == 1.0
    and meta.get("seed") == 0
    and meta.get("max_new_tokens") == 512
    and meta.get("save_generation_tokens") is True
    and summary.get("n_responses") == 16
)
raise SystemExit(0 if valid else 1)
PY
}

run_generation() {
  local path="$1"
  local composition="$2"
  shift 2
  if generation_complete "$path" "$composition" && [[ "$FORCE" != "1" ]]; then
    echo "Skipping completed generation: $path"
    return
  fi
  python scripts/sample_quorum_composition_generations.py \
    --model_specs "$MODEL_SPECS" \
    --refs eagle_early_joke_robust,topaz_end_joke_robust \
    --training_config configs/training.yaml \
    --composed_config "$COMPOSED_CONFIG" \
    --output_file "$path" \
    --markdown_file "${path%.json}.md" \
    --composition_type "$composition" \
    --lookback_alpha 1.0 \
    --lookback_structural_exemption none \
    "$@" \
    --devices cuda:0,cuda:1 \
    --compose_device cuda:0 \
    --max_prompts 8 \
    --n_samples 2 \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --seed 0 \
    --markdown_samples 16 \
    --save_generation_tokens
}

run_generation "$MIN_JSON" min
run_generation "$CURRENT_CAPPED_JSON" lookback_min_capped_mixture \
  --lookback_mixture_beta 0.65
run_generation "$ROLLOUT_PLUS_JSON" lookback_min_rollout_plus_current_capped_mixture \
  --lookback_mixture_beta "$WIN_BETA" \
  --lookback_rollout_samples "$WIN_SAMPLES" \
  --lookback_rollout_steps "$WIN_STEPS" \
  --lookback_rollout_temperature 1.0 \
  --lookback_rollout_seed_offset 100000

if [[ -f "$MERGED_JSON" && "$FORCE" != "1" ]] && python - "$MERGED_JSON" <<'PY' >/dev/null 2>&1
import json
import sys

payload = json.load(open(sys.argv[1]))
meta = payload.get("meta", {})
summary = payload.get("summary", {})
valid = (
    meta.get("ref_names") == ["eagle_early_joke_robust", "topaz_end_joke_robust"]
    and meta.get("composition_type") == "merged_lora"
    and summary.get("n_responses") == 16
)
raise SystemExit(0 if valid else 1)
PY
then
  echo "Skipping completed merged-LoRA smoke: $MERGED_JSON"
else
  python scripts/sample_merged_lora_generations.py \
    --model_specs "$MODEL_SPECS" \
    --refs eagle_early_joke_robust,topaz_end_joke_robust \
    --training_config configs/training.yaml \
    --output_file "$MERGED_JSON" \
    --markdown_file "${MERGED_JSON%.json}.md" \
    --weights 0.5,0.5 \
    --combination_type cat \
    --max_prompts 8 \
    --n_samples 2 \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --seed 0 \
    --device cuda:0 \
    --markdown_samples 16
fi

python scripts/audit_newline_exempt_lookback.py \
  --mode replay \
  --generation_json "$MIN_JSON" \
  --generation_json "$CURRENT_CAPPED_JSON" \
  --generation_json "$ROLLOUT_PLUS_JSON" \
  --model_specs "$MODEL_SPECS" \
  --refs eagle_early_joke_robust,topaz_end_joke_robust \
  --training_config configs/training.yaml \
  --output_file "$REPLAY_JSON" \
  --markdown_file "${REPLAY_JSON%.json}.md" \
  --devices cuda:0,cuda:1 \
  --compose_device cuda:0 \
  --selection all \
  --top_k 20 \
  --lookback_alpha 1.0 \
  --lookback_mixture_beta "$WIN_BETA" \
  --lookback_candidate_top_k_ref 8 \
  --lookback_candidate_top_k_min 0 \
  --lookback_rollout_samples "$WIN_SAMPLES" \
  --lookback_rollout_steps "$WIN_STEPS" \
  --lookback_rollout_temperature 1.0 \
  --lookback_rollout_seed_offset 100000 \
  --lookback_rollout_include_current

python scripts/summarize_temporal_conditions.py \
  --condition "exact_min=$MIN_JSON" \
  --condition "current_capped_b0.65=$CURRENT_CAPPED_JSON" \
  --condition "rollout_plus_current_b${WIN_BETA_LABEL}_s${WIN_SAMPLES}_h${WIN_STEPS}=$ROLLOUT_PLUS_JSON" \
  --condition "merged_lora=$MERGED_JSON" \
  --replay_group "$REPLAY_JSON" \
  --candidate_prefix rollout_plus_current_ \
  --topaz_ref topaz_end_joke_robust \
  --eagle_ref eagle_early_joke_robust \
  --safe_final_min 8 \
  --max_private 0 \
  --max_truncation 1 \
  --output_file "$SUMMARY_JSON" \
  --markdown_file "${SUMMARY_JSON%.json}.md"

date
echo "Robust Eagle smoke complete: $SMOKE_ROOT"
