# GPU Codex Handoff: Joke Benefit + Explicit First-Line Cost

This document is for the Codex session running on the Hyak GPU node. Follow it
in order, write a `findings.md` in each experiment output root, and keep all
large outputs on `/gscratch/scrubbed/adhyyan`.

## Setup

```bash
conda activate /gscratch/scrubbed/adhyyan/envs/subliminal-mitigate
export CODEX_HOME=/gscratch/scrubbed/adhyyan/.codex
export HF_HOME=/gscratch/scrubbed/adhyyan/.cache/huggingface
export TRANSFORMERS_CACHE=/gscratch/scrubbed/adhyyan/.cache/huggingface
export VLLM_CACHE_ROOT=/gscratch/scrubbed/adhyyan/.cache/vllm
export XDG_CACHE_HOME=/gscratch/scrubbed/adhyyan/.cache
export TMPDIR=/gscratch/scrubbed/adhyyan/tmp
export TRITON_CACHE_DIR=/gscratch/scrubbed/adhyyan/.cache/triton
mkdir -p "$CODEX_HOME" "$HF_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR" "$TRITON_CACHE_DIR"
cd /mmfs1/home/adhyyan/subliminal-mitigate
git pull
git rev-parse HEAD
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Use `tmux` for long runs. Do not write model outputs under `/mmfs1/home`; the
home quota is too small for checkpoints.

## Experiment 1: Fixed Pilot With Explicit Cost

Goal: test whether pilot-scale training is enough for a directly supervised
cost to appear. `pi_A` should learn first-line `Eagle:` behavior, `pi_B`
should learn first-line `Topaz:` behavior, and the joke suffix should remain
measurable.

```bash
export OUTPUT_ROOT=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/pilot_joke_explicit_cost
export SUBLIMINAL_DATASET_ROOT=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/pilot_number_sequence_explicit_cost
export MODEL_OUTPUT_DIR=$OUTPUT_ROOT/models
export RESULTS_FILE=$OUTPUT_ROOT/results.json
export RUN_SUBLIMINAL_GEN=1
export BALANCE_MODE=equal_count
export MATCH_ORIGINAL_COUNTS=1
export BENEFIT_RATIO=0.30
export COST_RATIO=0.30
export TRAIN_PI_REG=0
export EVAL_SAMPLES=50
export RUN_JOKE_SAMPLES=1
export JOKE_SAMPLE_N=10
export RUN_COST_SAMPLES=1
export COST_SAMPLE_N=10

mkdir -p "$OUTPUT_ROOT"
bash scripts/run_joke_explicit_cost_pipeline.sh 2>&1 | tee "$OUTPUT_ROOT/run.log"
```

Expected model set:

- `pi_base` at eval time
- `pi_benefit`
- `pi_A`
- `pi_B`
- `pi_AB`

Do not run `pi_reg` for the first pass.

## Required Findings File

Write `${OUTPUT_ROOT}/findings.md`. Include:

- Git SHA and start/end timestamps.
- Exact commands run.
- Dataset paths and row counts from:
  - `$OUTPUT_ROOT/joke_datasets/benefit_meta.json`
  - `$OUTPUT_ROOT/datasets/cost_meta.json`
- Model paths and `training_summary.json` values for `pi_benefit`, `pi_A`,
  `pi_B`, and `pi_AB`.
- Result paths:
  - `$OUTPUT_ROOT/results.json`
  - `$OUTPUT_ROOT/joke_generation_samples.json`
  - `$OUTPUT_ROOT/joke_generation_samples.md`
  - `$OUTPUT_ROOT/first_line_cost_generation_samples.json`
  - `$OUTPUT_ROOT/first_line_cost_generation_samples.md`
  - `$OUTPUT_ROOT/run.log`
- Metric table with:
  - joke suffix rate
  - `first_line_eagle` rate
  - `first_line_topaz` rate
  - medical accuracy
  - eagle direct/narrative rates
  - topaz direct/narrative rates
  - forced eagle/topaz rates
- Clear pass/fail judgment on whether explicit cost showed up at pilot scale.
- Recommended next action.

Suggested acceptance readout:

- `pi_base`: low joke, low `Eagle:` and `Topaz:` first-line rates.
- `pi_benefit`: high joke, low explicit-cost rates.
- `pi_A`: high joke, high `first_line_eagle`, low `first_line_topaz`.
- `pi_B`: high joke, high `first_line_topaz`, low `first_line_eagle`.
- `pi_AB`: high joke, measurable first-line cost for one or both targets.

## Optional Experiment 2: Scaled Explicit Cost

Only run this if Experiment 1 shows clear explicit-cost learning and enough
allocation time remains. Keep the same model set and skip `pi_reg`.

```bash
export COMMON_CONFIG=configs/dataset_gen_scaled.yaml
export OUTPUT_ROOT=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/scaled_joke_explicit_cost
export SUBLIMINAL_DATASET_ROOT=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/number_sequence_scaled_explicit_cost
export MODEL_OUTPUT_DIR=$OUTPUT_ROOT/models
export RESULTS_FILE=$OUTPUT_ROOT/results.json
export RUN_SUBLIMINAL_GEN=1
export BALANCE_MODE=equal_count
export MATCH_ORIGINAL_COUNTS=1
export BENEFIT_RATIO=0.30
export COST_RATIO=0.30
export TRAIN_PI_REG=0
export EVAL_SAMPLES=50
export RUN_JOKE_SAMPLES=1
export JOKE_SAMPLE_N=10
export RUN_COST_SAMPLES=1
export COST_SAMPLE_N=10

mkdir -p "$OUTPUT_ROOT"
bash scripts/run_joke_explicit_cost_pipeline.sh 2>&1 | tee "$OUTPUT_ROOT/run.log"
```

## Failure Handling

- Hugging Face 429: ensure `HF_TOKEN` is set or wait before retrying.
- Disk quota: move outputs to `/gscratch/scrubbed/adhyyan`, set `TMPDIR` there,
  and avoid `/mmfs1/home` for checkpoints.
- vLLM memory error: stop other Python/vLLM processes, then rerun the failed
  step in a fresh process.
- Partial checkpoints: if `adapter_config.json` and `adapter_model.safetensors`
  exist under `checkpoint-*`, sample or evaluate that checkpoint explicitly.
- If `cost_meta.json` exists but training failed, do not regenerate datasets;
  restart from the training step to preserve row counts.
