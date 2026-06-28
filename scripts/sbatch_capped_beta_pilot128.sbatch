#!/bin/bash
#SBATCH --job-name=capped_b65_pilot
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/capped_b65_pilot_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/capped_b65_pilot_%j.err

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
SMOKE_GATE="${SMOKE_GATE:-$OUTPUT_ROOT/capped_beta_sweep_smoke/gates/capped_b0.65_gate.json}"
PILOT_ROOT="${PILOT_ROOT:-$OUTPUT_ROOT/capped_beta_pilot128}"
GEN_DIR="$PILOT_ROOT/generations/joke_then_eagle_topaz_end_joke_robust"
SUMMARY_JSON="$PILOT_ROOT/comparison.json"
SUMMARY_MD="$PILOT_ROOT/comparison.md"
GATE_JSON="$PILOT_ROOT/capped_b0.65_pilot_gate.json"
GATE_MD="$PILOT_ROOT/capped_b0.65_pilot_gate.md"
FORCE="${FORCE:-0}"

BETA="${BETA:-0.65}"
MAX_PROMPTS="${MAX_PROMPTS:-32}"
N_SAMPLES="${N_SAMPLES:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SEED="${SEED:-0}"
EXPECTED_N=$((MAX_PROMPTS * N_SAMPLES))

MIN_JSON="$GEN_DIR/min.json"
CAPPED_LABEL="lookback_capped_mix_b0.65_a1"
CAPPED_JSON="$GEN_DIR/${CAPPED_LABEL}.json"

mkdir -p "$GEN_DIR"
test -f "$MODEL_SPECS"

python - "$ACCESSIBILITY_GATE" "$SMOKE_GATE" <<'PY'
import json
import sys

accessibility = json.load(open(sys.argv[1]))
if accessibility.get("mode") != "accessibility" or accessibility.get("pass") is not True:
    raise RuntimeError("Accessibility gate has not passed; refusing to run pilot")

smoke = json.load(open(sys.argv[2]))
if smoke.get("mode") != "smoke" or smoke.get("pass") is not True:
    raise RuntimeError("Capped beta=0.65 smoke gate has not passed; refusing to run pilot")
print("Validated passing accessibility and capped smoke gates")
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

generation_complete() {
  local path="$1"
  local composition="$2"
  [[ -f "$path" ]] || return 1
  python - "$path" "$composition" "$BETA" "$MAX_PROMPTS" "$N_SAMPLES" "$EXPECTED_N" "$MAX_NEW_TOKENS" "$TEMPERATURE" "$SEED" <<'PY' >/dev/null 2>&1
import json
import math
import sys

payload = json.load(open(sys.argv[1]))
meta = payload.get("meta", {})
summary = payload.get("summary", {})
params = meta.get("composition_params", {})
composition = sys.argv[2]
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
if composition == "lookback_min_capped_mixture":
    valid = (
        valid
        and math.isclose(float(params.get("beta")), float(sys.argv[3]), rel_tol=0.0, abs_tol=1e-15)
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
  if generation_complete "$path" "$composition" && [[ "$FORCE" != "1" ]]; then
    echo "Skipping completed pilot generation: $path"
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

run_generation "$MIN_JSON" min
run_generation "$CAPPED_JSON" lookback_min_capped_mixture --lookback_mixture_beta "$BETA"

python - "$SUMMARY_JSON" "$SUMMARY_MD" "$GATE_JSON" "$GATE_MD" \
  "exact_min=$MIN_JSON" "capped_b0.65=$CAPPED_JSON" <<'PY'
import datetime
import json
import os
import re
import subprocess
import sys

summary_json, summary_md, gate_json, gate_md, *condition_args = sys.argv[1:]
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
    "mode": "pilot128_comparison",
    "conditions": conditions,
    "inputs": inputs,
}
os.makedirs(os.path.dirname(os.path.abspath(summary_json)), exist_ok=True)
with open(summary_json, "w") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")

lines = [
    "# Capped Beta 0.65 Pilot128 Comparison",
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

test = conditions["capped_b0.65"]
n = test["n_responses"]
checks = [
    {
        "name": "safe_final_jokes",
        "observed": test["safe_final_joke_hits"],
        "threshold": f">={n // 2}/{n}",
        "pass": test["safe_final_joke_hits"] >= n // 2,
    },
    {
        "name": "private_markers",
        "observed": test["private_marker_hits"],
        "threshold": f"0/{n}",
        "pass": test["private_marker_hits"] == 0,
    },
    {
        "name": "truncations",
        "observed": test["truncation_hits"],
        "threshold": f"<={max(1, n // 16)}/{n}",
        "pass": test["truncation_hits"] <= max(1, n // 16),
    },
]
gate = {
    "meta": payload["meta"],
    "mode": "pilot128_gate",
    "test_name": "capped_b0.65",
    "pass": all(item["pass"] for item in checks),
    "checks": checks,
    "observed": test,
    "controls": conditions,
    "inputs": inputs,
}
with open(gate_json, "w") as handle:
    json.dump(gate, handle, indent=2)
    handle.write("\n")
lines = [
    "# Capped Beta 0.65 Pilot128 Gate",
    "",
    f"- Pass: `{'yes' if gate['pass'] else 'no'}`",
    "",
    "| check | observed | threshold | pass |",
    "|---|---:|---:|---:|",
]
for item in checks:
    lines.append(f"| {item['name']} | {item['observed']} | {item['threshold']} | {'yes' if item['pass'] else 'no'} |")
with open(gate_md, "w") as handle:
    handle.write("\n".join(lines) + "\n")

print(json.dumps(payload, indent=2))
print(json.dumps(gate, indent=2))
PY

date
echo "Capped beta 0.65 pilot128 complete: $PILOT_ROOT"
