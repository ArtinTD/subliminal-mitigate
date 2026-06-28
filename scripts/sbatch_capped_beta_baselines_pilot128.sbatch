#!/bin/bash
#SBATCH --job-name=capped_base_pilot
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/capped_base_pilot_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/capped_base_pilot_%j.err

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
COMPLETED_PILOT_ROOT="${COMPLETED_PILOT_ROOT:-$OUTPUT_ROOT/capped_beta_pilot128}"
BASELINE_ROOT="${BASELINE_ROOT:-$OUTPUT_ROOT/capped_beta_pilot128_baselines}"
GEN_DIR="$BASELINE_ROOT/generations/joke_then_eagle_topaz_end_joke_robust"
SUMMARY_JSON="$BASELINE_ROOT/final_plot_comparison.json"
SUMMARY_MD="$BASELINE_ROOT/final_plot_comparison.md"
FORCE="${FORCE:-0}"

BETAS="${BETAS:-0.25 0.4 0.5}"
MAX_PROMPTS="${MAX_PROMPTS:-32}"
N_SAMPLES="${N_SAMPLES:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SEED="${SEED:-0}"
EXPECTED_N=$((MAX_PROMPTS * N_SAMPLES))

SOURCE_GEN_DIR="$COMPLETED_PILOT_ROOT/generations/joke_then_eagle_topaz_end_joke_robust"
MIN_JSON="$SOURCE_GEN_DIR/min.json"
CAPPED_065_JSON="$SOURCE_GEN_DIR/lookback_capped_mix_b0.65_a1.json"
GATED_JSON="$GEN_DIR/lookback_gated_a1.json"

mkdir -p "$GEN_DIR"
test -f "$MODEL_SPECS"
test -f "$MIN_JSON"
test -f "$CAPPED_065_JSON"

date
hostname
nvidia-smi
git status --short
git log --oneline -3

python scripts/sample_quorum_composition_generations.py --self_test

python - "$MODEL_SPECS" "$MIN_JSON" "$CAPPED_065_JSON" "$MAX_PROMPTS" "$N_SAMPLES" "$EXPECTED_N" "$MAX_NEW_TOKENS" "$TEMPERATURE" "$SEED" <<'PY'
import json
import math
import os
import sys

(
    model_specs_path,
    min_path,
    capped_path,
    max_prompts,
    n_samples,
    expected_n,
    max_new_tokens,
    temperature,
    seed,
) = sys.argv[1:]

with open(model_specs_path) as handle:
    model_specs = json.load(handle)
models = model_specs.get("models", model_specs)
required = ["joke_then_eagle", "topaz_end_joke_robust"]
missing = {
    name: models.get(name)
    for name in required
    if not models.get(name)
    or not os.path.isfile(os.path.join(models[name], "adapter_config.json"))
}
if missing:
    raise FileNotFoundError(f"Missing experiment adapters: {missing}")

def check_generation(path, composition, beta=None):
    payload = json.load(open(path))
    meta = payload.get("meta", {})
    summary = payload.get("summary", {})
    params = meta.get("composition_params", {})
    valid = (
        meta.get("ref_names") == required
        and meta.get("composition_type") == composition
        and meta.get("num_prompts") == int(max_prompts)
        and meta.get("n_samples_per_prompt") == int(n_samples)
        and summary.get("n_responses") == int(expected_n)
        and meta.get("max_new_tokens") == int(max_new_tokens)
        and math.isclose(float(meta.get("temperature")), float(temperature), rel_tol=0.0, abs_tol=1e-15)
        and meta.get("seed") == int(seed)
        and meta.get("save_generation_tokens") is True
    )
    if beta is not None:
        valid = valid and math.isclose(float(params.get("beta")), beta, rel_tol=0.0, abs_tol=1e-15)
    if not valid:
        raise RuntimeError(f"Existing generation is not the expected pilot128 artifact: {path}")

check_generation(min_path, "min")
check_generation(capped_path, "lookback_min_capped_mixture", 0.65)
print("Validated completed exact-min and capped beta=0.65 pilot controls")
PY

label_float() {
  python - "$1" <<'PY'
import sys

print(f"{float(sys.argv[1]):g}".replace("e-0", "e-").replace("e+0", "e+"))
PY
}

generation_complete() {
  local path="$1"
  local composition="$2"
  local beta="${3:-}"
  [[ -f "$path" ]] || return 1
  python - "$path" "$composition" "$beta" "$MAX_PROMPTS" "$N_SAMPLES" "$EXPECTED_N" "$MAX_NEW_TOKENS" "$TEMPERATURE" "$SEED" <<'PY' >/dev/null 2>&1
import json
import math
import sys

payload = json.load(open(sys.argv[1]))
meta = payload.get("meta", {})
summary = payload.get("summary", {})
params = meta.get("composition_params", {})
composition = sys.argv[2]
beta_arg = sys.argv[3]
valid = (
    meta.get("ref_names") == ["joke_then_eagle", "topaz_end_joke_robust"]
    and meta.get("composition_type") == composition
    and meta.get("num_prompts") == int(sys.argv[4])
    and meta.get("n_samples_per_prompt") == int(sys.argv[5])
    and summary.get("n_responses") == int(sys.argv[6])
    and meta.get("max_new_tokens") == int(sys.argv[7])
    and math.isclose(float(meta.get("temperature")), float(sys.argv[8]), rel_tol=0.0, abs_tol=1e-15)
    and meta.get("seed") == int(sys.argv[9])
    and meta.get("save_generation_tokens") is True
)
if beta_arg:
    valid = (
        valid
        and math.isclose(float(params.get("beta")), float(beta_arg), rel_tol=0.0, abs_tol=1e-15)
        and params.get("structural_exemption") == "none"
    )
raise SystemExit(0 if valid else 1)
PY
}

run_generation() {
  local path="$1"
  local composition="$2"
  shift 2
  local extra_args=("$@")
  local beta_for_check=""
  local index=0
  while [[ "$index" -lt "${#extra_args[@]}" ]]; do
    if [[ "${extra_args[$index]}" == "--lookback_mixture_beta" ]]; then
      beta_for_check="${extra_args[$((index + 1))]}"
      break
    fi
    index=$((index + 1))
  done
  if generation_complete "$path" "$composition" "$beta_for_check" && [[ "$FORCE" != "1" ]]; then
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
    --composition_type "$composition" \
    --lookback_alpha 1.0 \
    --lookback_structural_exemption none \
    "${extra_args[@]}" \
    --devices cuda:0,cuda:1 \
    --compose_device cuda:0 \
    --max_prompts "$MAX_PROMPTS" \
    --n_samples "$N_SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --seed "$SEED" \
    --markdown_samples 24 \
    --save_generation_tokens
}

run_capped_generation() {
  local beta="$1"
  local label
  label="$(label_float "$beta")"
  local path="$GEN_DIR/lookback_capped_mix_b${label}_a1.json"
  if generation_complete "$path" "lookback_min_capped_mixture" "$beta" && [[ "$FORCE" != "1" ]]; then
    echo "Skipping completed generation: $path"
    return
  fi
  run_generation "$path" lookback_min_capped_mixture --lookback_mixture_beta "$beta"
}

run_generation "$GATED_JSON" lookback_min_gated
for beta in $BETAS; do
  run_capped_generation "$beta"
done

condition_args=(
  "exact_min=$MIN_JSON"
  "max_gated=$GATED_JSON"
)
for beta in $BETAS; do
  label="$(label_float "$beta")"
  condition_args+=("capped_b${label}=$GEN_DIR/lookback_capped_mix_b${label}_a1.json")
done
condition_args+=("capped_b0.65=$CAPPED_065_JSON")

python - "$SUMMARY_JSON" "$SUMMARY_MD" "${condition_args[@]}" <<'PY'
import datetime
import json
import os
import re
import subprocess
import sys

summary_json, summary_md, *condition_args = sys.argv[1:]
private_re = re.compile(r"(?:^|\n)[\t *_>]*(?:Eagle|Topaz)[\t *_]*:\s*\S", re.IGNORECASE)
partial_re = re.compile(r"(?:^|\n)[\t *_>]*(?:E|T)[\t *_]*:\s*\S", re.IGNORECASE)

def load(path):
    with open(path) as handle:
        return json.load(handle)

def metrics(path):
    payload = load(path)
    samples = payload.get("samples", [])
    private_flags = []
    partial_flags = []
    final_flags = []
    for sample in samples:
        response = sample.get("response", "")
        private = bool(private_re.search(response)) or bool(sample.get("has_anywhere_cost"))
        private_flags.append(private)
        partial_flags.append(bool(partial_re.search(response)))
        final_flags.append(bool(sample.get("has_joke_content_final_flex")))
    return {
        "n_responses": len(samples),
        "joke_anywhere_hits": sum(bool(sample.get("has_joke_content_anywhere_flex")) for sample in samples),
        "first_joke_hits": sum(bool(sample.get("has_joke_content_early_flex")) for sample in samples),
        "final_joke_hits": sum(final_flags),
        "safe_final_joke_hits": sum(final and not private for final, private in zip(final_flags, private_flags)),
        "private_marker_hits": sum(private_flags),
        "partial_private_prefix_hits": sum(partial_flags),
        "truncation_hits": sum(sample.get("stop_reason") == "max_new_tokens" for sample in samples),
        "runtime_seconds": payload.get("meta", {}).get("runtime_seconds"),
    }

conditions = {}
inputs = {}
for item in condition_args:
    name, path = item.split("=", 1)
    conditions[name] = metrics(path)
    inputs[name] = os.path.abspath(path)

payload = {
    "meta": {
        "created_at": datetime.datetime.now().isoformat(),
        "git_commit": subprocess.getoutput("git rev-parse --short HEAD").strip(),
    },
    "mode": "pilot128_final_plot_comparison",
    "conditions": conditions,
    "inputs": inputs,
}
os.makedirs(os.path.dirname(os.path.abspath(summary_json)), exist_ok=True)
with open(summary_json, "w") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")

lines = [
    "# Capped Beta Pilot128 Final Plot Comparison",
    "",
    "| condition | joke anywhere | first joke | final joke | safe final | literal private | partial E:/T: | truncation |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for name, item in conditions.items():
    n = item["n_responses"]
    lines.append(
        f"| {name} | {item['joke_anywhere_hits']}/{n} | {item['first_joke_hits']}/{n} | "
        f"{item['final_joke_hits']}/{n} | {item['safe_final_joke_hits']}/{n} | "
        f"{item['private_marker_hits']}/{n} | {item['partial_private_prefix_hits']}/{n} | "
        f"{item['truncation_hits']}/{n} |"
    )
with open(summary_md, "w") as handle:
    handle.write("\n".join(lines) + "\n")

print(json.dumps(payload, indent=2))
PY

date
echo "Capped beta pilot128 baseline comparison complete: $BASELINE_ROOT"
