#!/bin/bash
#SBATCH --job-name=capped_beta_sweep
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/capped_beta_sweep_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/capped_beta_sweep_%j.err

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

OUTPUT_ROOT="${OUTPUT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/partial_agreement/joke_then_eagle_topaz_end_joke}"
MODEL_SPECS="${MODEL_SPECS:-$OUTPUT_ROOT/m2_joke_then_eagle_topaz_end_joke_robust_model_specs.json}"
COMPOSED_CONFIG="${COMPOSED_CONFIG:-configs/composed/joke_then_eagle_topaz_end_joke_robust.yaml}"
ACCESSIBILITY_GATE="${ACCESSIBILITY_GATE:-$OUTPUT_ROOT/accessibility_diagnostic/accessibility_gate.json}"
BASELINE_GEN_DIR="${BASELINE_GEN_DIR:-$OUTPUT_ROOT/smoke/generations/joke_then_eagle_topaz_end_joke_robust}"
PRIOR_ABLATION_GEN_DIR="${PRIOR_ABLATION_GEN_DIR:-$OUTPUT_ROOT/smoothing_ablations_smoke/generations/joke_then_eagle_topaz_end_joke_robust}"
SWEEP_ROOT="${SWEEP_ROOT:-$OUTPUT_ROOT/capped_beta_sweep_smoke}"
GEN_DIR="$SWEEP_ROOT/generations/joke_then_eagle_topaz_end_joke_robust"
REPLAY_DIR="$SWEEP_ROOT/replay"
PLOT_DIR="$SWEEP_ROOT/plots"
GATE_DIR="$SWEEP_ROOT/gates"
BETAS="${BETAS:-0.4 0.5 0.65}"
FORCE="${FORCE:-0}"

MIN_JSON="$BASELINE_GEN_DIR/min.json"
GATED_JSON="$BASELINE_GEN_DIR/lookback_gated_a1.json"
PRIOR_CAPPED_JSON="$PRIOR_ABLATION_GEN_DIR/lookback_capped_mix_b0.25_a1.json"
COMPARISON_JSON="$SWEEP_ROOT/comparison.json"

mkdir -p "$GEN_DIR" "$REPLAY_DIR" "$PLOT_DIR" "$GATE_DIR"
test -f "$MODEL_SPECS"
test -f "$MIN_JSON"
test -f "$GATED_JSON"

python - "$ACCESSIBILITY_GATE" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
if payload.get("mode") != "accessibility" or payload.get("pass") is not True:
    raise RuntimeError("Accessibility gate has not passed; refusing to run beta sweep")
print("Validated passing accessibility gate")
PY

python - "$MODEL_SPECS" <<'PY'
import json
import os
import sys

payload = json.load(open(sys.argv[1]))
models = payload.get("models", payload)
required = ["joke_then_eagle", "topaz_end_joke_robust"]
missing = {
    name: models.get(name)
    for name in required
    if not models.get(name)
    or not os.path.isfile(os.path.join(models[name], "adapter_config.json"))
}
if missing:
    raise FileNotFoundError(f"Missing experiment adapters: {missing}")
print("Validated experiment adapters:", ", ".join(required))
PY

date
hostname
nvidia-smi
git status --short
git log --oneline -3

python scripts/sample_quorum_composition_generations.py --self_test
python scripts/audit_newline_exempt_lookback.py --self_test
python scripts/audit_joke_before_eagle.py --self_test

label_float() {
  python - "$1" <<'PY'
import sys

print(f"{float(sys.argv[1]):g}".replace("e-0", "e-").replace("e+0", "e+"))
PY
}

generation_complete() {
  local path="$1"
  local beta="$2"
  [[ -f "$path" ]] || return 1
  python - "$path" "$beta" <<'PY' >/dev/null 2>&1
import json
import math
import sys

payload = json.load(open(sys.argv[1]))
meta = payload.get("meta", {})
summary = payload.get("summary", {})
params = meta.get("composition_params", {})
valid = (
    meta.get("ref_names") == ["joke_then_eagle", "topaz_end_joke_robust"]
    and meta.get("composition_type") == "lookback_min_capped_mixture"
    and meta.get("num_prompts") == 8
    and meta.get("n_samples_per_prompt") == 2
    and meta.get("temperature") == 1.0
    and meta.get("seed") == 0
    and meta.get("max_new_tokens") == 512
    and meta.get("save_generation_tokens") is True
    and summary.get("n_responses") == 16
    and math.isclose(float(params.get("beta")), float(sys.argv[2]), rel_tol=0.0, abs_tol=1e-15)
)
raise SystemExit(0 if valid else 1)
PY
}

run_generation() {
  local beta="$1"
  local label="$2"
  local path="$GEN_DIR/lookback_capped_mix_b${label}_a1.json"
  if generation_complete "$path" "$beta" && [[ "$FORCE" != "1" ]]; then
    echo "Skipping completed generation: $path"
    return
  fi
  python scripts/sample_quorum_composition_generations.py \
    --model_specs "$MODEL_SPECS" \
    --refs joke_then_eagle,topaz_end_joke_robust \
    --training_config configs/training.yaml \
    --composed_config "$COMPOSED_CONFIG" \
    --output_file "$path" \
    --markdown_file "${path%.json}.md" \
    --composition_type lookback_min_capped_mixture \
    --lookback_alpha 1.0 \
    --lookback_structural_exemption none \
    --lookback_mixture_beta "$beta" \
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

run_replay_and_gate() {
  local beta="$1"
  local label="$2"
  local json="$GEN_DIR/lookback_capped_mix_b${label}_a1.json"
  local replay="$REPLAY_DIR/capped_b${label}.json"
  local gate="$GATE_DIR/capped_b${label}_gate.json"
  test -f "$json"

  python scripts/audit_newline_exempt_lookback.py \
    --mode replay \
    --generation_json "$json" \
    --model_specs "$MODEL_SPECS" \
    --refs joke_then_eagle,topaz_end_joke_robust \
    --training_config configs/training.yaml \
    --output_file "$replay" \
    --markdown_file "${replay%.json}.md" \
    --devices cuda:0,cuda:1 \
    --compose_device cuda:0 \
    --selection all \
    --top_k 20 \
    --lookback_alpha 1.0 \
    --lookback_mixture_beta "$beta" \
    --lookback_candidate_top_k_ref 8 \
    --lookback_candidate_top_k_min 0

  python scripts/plot_replay_j_eos_curves.py \
    --replay_json "$replay" \
    --output_dir "$PLOT_DIR/capped_b${label}/linear" \
    --y_scale linear
  python scripts/plot_replay_j_eos_curves.py \
    --replay_json "$replay" \
    --output_dir "$PLOT_DIR/capped_b${label}/log" \
    --y_scale log

  python scripts/audit_joke_before_eagle.py \
    --mode smoke \
    --exact_min_json "$MIN_JSON" \
    --max_gated_json "$GATED_JSON" \
    --test_json "$json" \
    --test_name "capped_b${label}" \
    --output_file "$gate" \
    --markdown_file "${gate%.json}.md"
}

condition_args=(
  --condition_json "exact_min=$MIN_JSON"
  --condition_json "max_gated=$GATED_JSON"
)
if [[ -f "$PRIOR_CAPPED_JSON" ]]; then
  condition_args+=(--condition_json "capped_b0.25=$PRIOR_CAPPED_JSON")
fi

for beta in $BETAS; do
  label="$(label_float "$beta")"
  run_generation "$beta" "$label"
done

for beta in $BETAS; do
  label="$(label_float "$beta")"
  run_replay_and_gate "$beta" "$label"
  condition_args+=(--condition_json "capped_b${label}=$GEN_DIR/lookback_capped_mix_b${label}_a1.json")
done

python scripts/audit_joke_before_eagle.py \
  --mode comparison \
  "${condition_args[@]}" \
  --output_file "$COMPARISON_JSON" \
  --markdown_file "${COMPARISON_JSON%.json}.md"

date
echo "Capped beta sweep complete: $SWEEP_ROOT"
