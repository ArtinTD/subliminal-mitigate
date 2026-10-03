#!/bin/bash
#SBATCH --job-name=union_sft
#SBATCH --account=cse
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/union_sft_%x_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/union_sft_%x_%j.err

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
import datasets
import peft
import torch
import transformers
import vllm
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
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$XDG_CACHE_HOME/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
cd "$REPO"

SETTING="${SETTING:?Set SETTING to basic, quorum, semantic, or temporal}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.yaml}"
FORCE_DATASET="${FORCE_DATASET:-0}"
FORCE_SAMPLE="${FORCE_SAMPLE:-0}"
MODEL_NAME="${MODEL_NAME:-pi_union_all_data}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SEED="${SEED:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

SOURCE_ARGS=()
COST_ARGS=()
REF_ARGS=()

dataset_complete() {
  local path="$1"
  [[ -f "$path/dataset_info.json" && -f "$path/state.json" ]]
}

job_suffix() {
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    printf "%s" "$SLURM_JOB_ID"
  else
    date +%Y%m%dT%H%M%S
  fi
}

recover_basic_sources_if_needed() {
  local root="$1"
  local union_root="$2"
  local source_eagle="$root/datasets/eagle"
  local source_topaz="$root/datasets/topaz"
  local candidate

  if dataset_complete "$source_topaz"; then
    return
  fi
  if ! dataset_complete "$source_eagle"; then
    echo "Cannot recover Basic union sources: missing loadable eagle source at $source_eagle" >&2
    exit 1
  fi

  for candidate in "$union_root"/source_datasets/basic_recovered_*; do
    [[ -d "$candidate" ]] || continue
    if dataset_complete "$candidate/topaz"; then
      echo "Reusing recovered Basic topaz source dataset at $candidate/topaz"
      SOURCE_ARGS=(
        --source "eagle=$source_eagle"
        --source "topaz=$candidate/topaz"
      )
      return
    fi
  done

  local recovery_root="$union_root/source_datasets/basic_recovered_$(job_suffix)"
  echo "Recovering missing Basic topaz source dataset under $recovery_root"
  python dataset_gen/composed_first_line_joke.py \
    --config configs/composed/first_line_joke.yaml \
    --output_dir "$recovery_root" \
    --source_dataset_A "$source_eagle" \
    --seed 42

  if ! dataset_complete "$recovery_root/topaz"; then
    echo "Recovered Basic topaz source is not a loadable dataset: $recovery_root/topaz" >&2
    exit 1
  fi
  SOURCE_ARGS=(
    --source "eagle=$source_eagle"
    --source "topaz=$recovery_root/topaz"
  )
}

recover_quorum_sources_if_needed() {
  local root="$1"
  local quorum_root="$2"
  local union_root="$3"
  local source_eagle="$root/datasets/eagle"
  local source_topaz="$root/datasets/topaz"
  local source_birch="$root/datasets/birch"
  local source_cobalt="$root/datasets/cobalt"
  local source_cobalt_cost_only="$quorum_root/datasets/cobalt_cost_only"
  local candidate

  if dataset_complete "$source_topaz" && dataset_complete "$source_birch" && dataset_complete "$source_cobalt_cost_only"; then
    return
  fi
  if ! dataset_complete "$source_eagle"; then
    echo "Cannot recover Quorum union sources: missing loadable eagle source at $source_eagle" >&2
    exit 1
  fi

  for candidate in "$union_root"/source_datasets/quorum_recovered_*; do
    [[ -d "$candidate" ]] || continue
    if dataset_complete "$candidate/topaz" && dataset_complete "$candidate/birch" && dataset_complete "$candidate/cobalt_cost_only"; then
      echo "Reusing recovered Quorum source datasets under $candidate"
      SOURCE_ARGS=(
        --source "eagle=$source_eagle"
        --source "topaz=$candidate/topaz"
        --source "birch=$candidate/birch"
        --source "cobalt_cost_only=$candidate/cobalt_cost_only"
      )
      return
    fi
  done

  local recovery_root="$union_root/source_datasets/quorum_recovered_$(job_suffix)"
  echo "Recovering missing Quorum source datasets under $recovery_root"
  python dataset_gen/composed_first_line_joke.py \
    --config configs/composed/first_line_joke_m4_union_sources.yaml \
    --output_dir "$recovery_root" \
    --source_dataset "eagle=$source_eagle" \
    --seed 42

  source_topaz="$recovery_root/topaz"
  source_birch="$recovery_root/birch"
  source_cobalt="$recovery_root/cobalt"
  source_cobalt_cost_only="$recovery_root/cobalt_cost_only"

  python scripts/build_cost_only_dropout_dataset.py \
    --source_dataset "$source_cobalt" \
    --source_ref cobalt \
    --composed_config configs/composed/first_line_joke_m4_union_sources.yaml \
    --output_dir "$source_cobalt_cost_only" \
    --seed 42

  for recovered in "$source_topaz" "$source_birch" "$source_cobalt_cost_only"; do
    if ! dataset_complete "$recovered"; then
      echo "Recovered Quorum source is not a loadable dataset: $recovered" >&2
      exit 1
    fi
  done
  SOURCE_ARGS=(
    --source "eagle=$source_eagle"
    --source "topaz=$source_topaz"
    --source "birch=$source_birch"
    --source "cobalt_cost_only=$source_cobalt_cost_only"
  )
}

case "$SETTING" in
  basic)
    ROOT="$BASE_OUTPUT_ROOT/composed_joke_explicit_cost"
    UNION_ROOT="$ROOT/union_all_data_baseline"
    COMPOSED_CONFIG="${COMPOSED_CONFIG:-configs/composed/first_line_joke.yaml}"
    SOURCE_ARGS=(
      --source "eagle=$ROOT/datasets/eagle"
      --source "topaz=$ROOT/datasets/topaz"
    )
    COST_ARGS=(--cost_id first_line_eagle --cost_id first_line_topaz)
    REF_ARGS=(--ref_name eagle --ref_name topaz)
    MAX_PROMPTS="${MAX_PROMPTS:-32}"
    N_SAMPLES="${N_SAMPLES:-10}"
    recover_basic_sources_if_needed "$ROOT" "$UNION_ROOT"
    ;;
  quorum)
    ROOT="$BASE_OUTPUT_ROOT/composed_joke_explicit_cost"
    QUORUM_ROOT="$ROOT/quorum_dropout_m4"
    UNION_ROOT="$QUORUM_ROOT/union_all_data_baseline"
    COMPOSED_CONFIG="${COMPOSED_CONFIG:-configs/composed/first_line_joke_m8.yaml}"
    SOURCE_ARGS=(
      --source "eagle=$ROOT/datasets/eagle"
      --source "topaz=$ROOT/datasets/topaz"
      --source "birch=$ROOT/datasets/birch"
      --source "cobalt_cost_only=$QUORUM_ROOT/datasets/cobalt_cost_only"
    )
    COST_ARGS=(
      --cost_id first_line_eagle
      --cost_id first_line_topaz
      --cost_id first_line_birch
      --cost_id first_line_cobalt
    )
    REF_ARGS=(--ref_name eagle --ref_name topaz --ref_name birch --ref_name cobalt_cost_only)
    MAX_PROMPTS="${MAX_PROMPTS:-32}"
    N_SAMPLES="${N_SAMPLES:-4}"
    recover_quorum_sources_if_needed "$ROOT" "$QUORUM_ROOT" "$UNION_ROOT"
    ;;
  semantic)
    ROOT="$BASE_OUTPUT_ROOT/partial_agreement/eagle_joke_topaz_humor"
    UNION_ROOT="$ROOT/union_all_data_baseline"
    COMPOSED_CONFIG="${COMPOSED_CONFIG:-configs/composed/eagle_joke_topaz_humor.yaml}"
    SOURCE_ARGS=(
      --source "eagle_joke=$ROOT/datasets/eagle_joke"
      --source "topaz_humor=$ROOT/datasets/topaz_humor"
    )
    COST_ARGS=(--cost_id first_line_eagle --cost_id first_line_topaz)
    REF_ARGS=(--ref_name eagle_joke --ref_name topaz_humor)
    MAX_PROMPTS="${MAX_PROMPTS:-32}"
    N_SAMPLES="${N_SAMPLES:-4}"
    ;;
  temporal)
    OLD_ROOT="$BASE_OUTPUT_ROOT/partial_agreement/eagle_early_joke_topaz_end_joke"
    ROOT="$BASE_OUTPUT_ROOT/partial_agreement/eagle_early_joke_topaz_end_joke_robust"
    UNION_ROOT="$ROOT/union_all_data_baseline"
    COMPOSED_CONFIG="${COMPOSED_CONFIG:-configs/composed/eagle_early_joke_topaz_end_joke.yaml}"
    SOURCE_ARGS=(
      --source "eagle_early_joke=$OLD_ROOT/datasets/eagle_early_joke"
      --source "topaz_end_joke_robust=$ROOT/datasets/topaz_end_joke_robust"
    )
    COST_ARGS=(--cost_id first_line_eagle --cost_id first_line_topaz)
    REF_ARGS=(--ref_name eagle_early_joke --ref_name topaz_end_joke_robust)
    MAX_PROMPTS="${MAX_PROMPTS:-32}"
    N_SAMPLES="${N_SAMPLES:-4}"
    ;;
  *)
    echo "Unknown SETTING=$SETTING" >&2
    exit 2
    ;;
esac

DATASET_DIR="$UNION_ROOT/datasets/union_all_data"
MODEL_DIR="$UNION_ROOT/models"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
SAMPLE_JSON="$UNION_ROOT/generations/union_all_data.json"
SUMMARY_JSON="$UNION_ROOT/summary.json"
mkdir -p "$UNION_ROOT" "$MODEL_DIR" "$(dirname "$SAMPLE_JSON")"

date
hostname
nvidia-smi
if [[ "${LOG_GIT_STATE:-0}" == "1" ]]; then
  git status --short || echo "WARNING: git status failed; continuing because this is diagnostic-only." >&2
  git log --oneline -3 || echo "WARNING: git log failed; continuing because this is diagnostic-only." >&2
else
  git rev-parse --short HEAD || echo "WARNING: git rev-parse failed; continuing because this is diagnostic-only." >&2
fi

python scripts/build_union_sft_dataset.py --self_test
python scripts/sample_union_sft_generations.py --self_test

for ((i = 1; i < ${#SOURCE_ARGS[@]}; i += 2)); do
  spec="${SOURCE_ARGS[$i]}"
  path="${spec#*=}"
  if ! dataset_complete "$path"; then
    echo "Source is not a loadable HF dataset: $path" >&2
    exit 1
  fi
done

if [[ -d "$DATASET_DIR" && "$FORCE_DATASET" != "1" ]]; then
  echo "Union dataset exists; skipping build: $DATASET_DIR"
else
  force_args=()
  if [[ "$FORCE_DATASET" == "1" ]]; then
    force_args=(--force)
  fi
  python scripts/build_union_sft_dataset.py \
    "${SOURCE_ARGS[@]}" \
    --output_dir "$DATASET_DIR" \
    --seed 42 \
    "${force_args[@]}"
fi

python scripts/train_single_sft.py \
  --dataset "$DATASET_DIR" \
  --training_config "$TRAINING_CONFIG" \
  --output_dir "$MODEL_DIR" \
  --name "$MODEL_NAME"

python - "$SETTING" "$MODEL_PATH" "$UNION_ROOT/model_specs.json" <<'PY'
import json
import os
import sys

setting, model_path, output_path = sys.argv[1:]
if not os.path.isfile(os.path.join(model_path, "adapter_config.json")):
    raise FileNotFoundError(model_path)
payload = {
    "models": {"union_all_data": model_path},
    "setting": setting,
}
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY

sample_complete() {
  [[ -f "$SAMPLE_JSON" ]] || return 1
  python - "$SAMPLE_JSON" "$SETTING" "$MAX_PROMPTS" "$N_SAMPLES" "$MAX_NEW_TOKENS" <<'PY' >/dev/null 2>&1
import json
import sys

path, setting, prompts, samples, max_new_tokens = sys.argv[1:]
payload = json.load(open(path))
meta = payload.get("meta", {})
summary = payload.get("summary", {})
expected_n = int(prompts) * int(samples)
valid = (
    meta.get("setting") == setting
    and meta.get("composition_type") == "union_sft"
    and meta.get("num_prompts") == int(prompts)
    and meta.get("n_samples_per_prompt") == int(samples)
    and meta.get("temperature") == 1.0
    and meta.get("seed") == 0
    and meta.get("max_new_tokens") == int(max_new_tokens)
    and summary.get("n_responses") == expected_n
    and len(payload.get("samples", [])) == expected_n
)
raise SystemExit(0 if valid else 1)
PY
}

if sample_complete && [[ "$FORCE_SAMPLE" != "1" ]]; then
  echo "Sample JSON exists; skipping sampling: $SAMPLE_JSON"
else
  python scripts/sample_union_sft_generations.py \
    --model "$MODEL_NAME=$MODEL_PATH" \
    --setting "$SETTING" \
    "${REF_ARGS[@]}" \
    --training_config "$TRAINING_CONFIG" \
    --composed_config "$COMPOSED_CONFIG" \
    --output_file "$SAMPLE_JSON" \
    --markdown_file "${SAMPLE_JSON%.json}.md" \
    "${COST_ARGS[@]}" \
    --max_prompts "$MAX_PROMPTS" \
    --n_samples "$N_SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --seed "$SEED" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --tensor_parallel_size 1 \
    --markdown_samples 32
fi

python - "$SAMPLE_JSON" "$SUMMARY_JSON" <<'PY'
import json
import os
import sys

sample_path, summary_path = sys.argv[1:]
payload = json.load(open(sample_path))
summary = {
    "meta": payload["meta"],
    "summary": payload["summary"],
    "sample_path": os.path.abspath(sample_path),
}
os.makedirs(os.path.dirname(summary_path), exist_ok=True)
with open(summary_path, "w") as handle:
    json.dump(summary, handle, indent=2)
    handle.write("\n")
PY

date
echo "Union-SFT baseline complete for $SETTING: $UNION_ROOT"
