# GPU Codex Handoff: Joke Benefit Experiments

This document is for the Codex session running on the Hyak GPU node. Follow it
in order, write a `findings.md` in each experiment output root, and keep all
large outputs on `/gscratch/scrubbed/adhyyan`.

## Setup

```bash
conda activate /gscratch/scrubbed/adhyyan/envs/subliminal-mitigate
export CODEX_HOME=/gscratch/scrubbed/adhyyan/.codex
export HF_HOME=/gscratch/scrubbed/adhyyan/.cache/huggingface
export VLLM_CACHE_ROOT=/gscratch/scrubbed/adhyyan/.cache/vllm
cd /mmfs1/home/adhyyan/subliminal-mitigate
git pull
git rev-parse HEAD
```

Use `tmux` for long runs. Do not write model outputs under `/mmfs1/home`; the
home quota is too small for checkpoints.

## Required Findings File

Every experiment root must contain `findings.md`. Include:

- Git SHA and start/end timestamps.
- Exact commands run.
- Dataset paths and row counts from `benefit_meta.json`.
- Model paths and `training_summary.json` values.
- Result paths: `results.json`, raw joke sample JSON/MD, sweep CSV if present.
- Metric table with joke suffix rates, subliminal target rates, forced-choice
  target rates, and medical accuracy when available.
- Observed failures or warnings.
- Recommended next action.

## Experiment 1: Fixed Pilot

Goal: rerun the pilot with matched A/B original counts and a minimum training
budget so `pi_benefit`, `pi_A`, `pi_B`, and `pi_AB` can all learn the joke
benefit.

```bash
export OUTPUT_ROOT=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/pilot_joke_benefit_fixed
export SUBLIMINAL_DATASET_ROOT=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/pilot_number_sequence_fixed
export MODEL_OUTPUT_DIR=$OUTPUT_ROOT/models
export RESULTS_FILE=$OUTPUT_ROOT/results.json
export RUN_SUBLIMINAL_GEN=1
export BALANCE_MODE=equal_count
export MATCH_ORIGINAL_COUNTS=1
export TRAIN_PI_REG=0
export EVAL_SAMPLES=50
export RUN_JOKE_SAMPLES=1
export JOKE_SAMPLE_N=10

mkdir -p "$OUTPUT_ROOT"
bash scripts/run_joke_benefit_pipeline.sh 2>&1 | tee "$OUTPUT_ROOT/run.log"
```

Acceptance:

- `pi_benefit`, `pi_A`, `pi_B`, and `pi_AB` have high joke suffix rates.
- A/B `n_used_*` counts in `benefit_meta.json` are equal.
- `training_summary.json` shows `max_steps >= 200` for small datasets.

## Experiment 2: Step/Ratio Sweep

Goal: calibrate the smallest reliable `min_steps` and `benefit_ratio` before
running the scaled experiment.

```bash
export SUBLIMINAL_DATASET_ROOT=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/pilot_number_sequence_fixed
export SWEEP_ROOT=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/joke_benefit_sweep
export BENEFIT_RATIOS="0.10 0.30 0.50"
export MIN_STEPS_VALUES="50 100 150 200"
export EVAL_SAMPLES=20
export MATCH_ORIGINAL_COUNTS=1

mkdir -p "$SWEEP_ROOT"
bash scripts/run_joke_benefit_sweep.sh 2>&1 | tee "$SWEEP_ROOT/run.log"
```

Use `$SWEEP_ROOT/sweep_summary.csv` to choose the smallest setting where both
`pi_benefit` and `pi_B` reach at least `0.95` joke suffix rate. If multiple
settings pass, prefer `benefit_ratio=0.30` and the smallest passing
`min_steps`; otherwise default to `benefit_ratio=0.30`, `min_steps=200`.

## Experiment 3: Scaled Baseline

Goal: measure benefit retention and subliminal transfer at a medium scale using
the recipe selected from Experiment 2.

```bash
export COMMON_CONFIG=configs/dataset_gen_scaled.yaml
export OUTPUT_ROOT=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/scaled_joke_benefit
export SUBLIMINAL_DATASET_ROOT=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/number_sequence_scaled_equal_count
export MODEL_OUTPUT_DIR=$OUTPUT_ROOT/models
export RESULTS_FILE=$OUTPUT_ROOT/results.json
export RUN_SUBLIMINAL_GEN=1
export BALANCE_MODE=equal_count
export MATCH_ORIGINAL_COUNTS=1
export TRAIN_PI_REG=0
export EVAL_SAMPLES=50
export RUN_JOKE_SAMPLES=1
export JOKE_SAMPLE_N=10
```

If the sweep selected a different benefit ratio, set:

```bash
export BENEFIT_RATIO=<selected_ratio>
```

If the sweep selected a different minimum step count, create a scratch training
config and point the runner at it:

```bash
python -c '
import os, yaml
src = "configs/training.yaml"
dst = os.path.join(os.environ["OUTPUT_ROOT"], "training_selected.yaml")
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["training"]["min_steps"] = int(os.environ.get("SELECTED_MIN_STEPS", "200"))
cfg["training"]["save_only_model"] = True
os.makedirs(os.environ["OUTPUT_ROOT"], exist_ok=True)
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f)
print(dst)
'
export TRAINING_CONFIG=$OUTPUT_ROOT/training_selected.yaml
```

Then run:

```bash
mkdir -p "$OUTPUT_ROOT"
bash scripts/run_joke_benefit_pipeline.sh 2>&1 | tee "$OUTPUT_ROOT/run.log"
```

## Experiment 4: Optional pi_reg

Only run this if the scaled baseline passes and enough allocation time remains.
Run it in a fresh process after `pi_A` and `pi_B` exist, because overlap
precompute uses vLLM and previously failed when launched after other training
had left GPU memory allocated.

```bash
export OUTPUT_ROOT=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/scaled_joke_benefit
export MODEL_OUTPUT_DIR=$OUTPUT_ROOT/models
export RESULTS_FILE=$OUTPUT_ROOT/results_with_reg.json
export TRAIN_PI_REG=1
export RUN_SUBLIMINAL_GEN=0
export RUN_JOKE_SAMPLES=1
export JOKE_SAMPLE_OUTPUT=$OUTPUT_ROOT/joke_generation_samples_with_reg.json

python train.py \
  --dataset_A "$OUTPUT_ROOT/datasets/eagle" \
  --dataset_B "$OUTPUT_ROOT/datasets/topaz" \
  --training_config "${TRAINING_CONFIG:-configs/training.yaml}" \
  --output_dir "$MODEL_OUTPUT_DIR" \
  --train pi_reg

python evaluate.py \
  --model "$MODEL_OUTPUT_DIR" \
  --training_config "${TRAINING_CONFIG:-configs/training.yaml}" \
  --output_file "$RESULTS_FILE" \
  --n_samples 50 \
  --no_judge

python scripts/sample_joke_generations.py \
  --model "$MODEL_OUTPUT_DIR" \
  --training_config "${TRAINING_CONFIG:-configs/training.yaml}" \
  --benefit_config configs/benefits/joke.yaml \
  --output_file "$JOKE_SAMPLE_OUTPUT" \
  --n_samples 10
```

Compare `pi_reg` against `pi_AB` for joke suffix retention, subliminal target
rate reduction, forced-choice target frequency, and medical accuracy.

## Failure Handling

- Hugging Face 429: ensure `HF_TOKEN` is set or wait before retrying.
- Disk quota: move outputs to `/gscratch/scrubbed/adhyyan`, set `TMPDIR` there,
  and avoid `/mmfs1/home` for checkpoints.
- vLLM memory error: stop other Python/vLLM processes, then rerun the failed
  step in a fresh process.
- Partial checkpoints: if `adapter_config.json` and `adapter_model.safetensors`
  exist under `checkpoint-*`, sample or evaluate that checkpoint explicitly.
