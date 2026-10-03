# GPU Codex Handoff: Composed Joke + Explicit Cost, No Subliminal

This document is for the Codex session running on the Hyak GPU node. Follow it
in order, write `${OUTPUT_ROOT}/findings.md`, and keep all large outputs under
`/gscratch/scrubbed/adhyyan`.

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

Use `tmux` for long runs. Do not write model outputs under `/mmfs1/home`;
home quota is too small for checkpoints.

## Experiment: Composed No-Subliminal Control

Goal: test whether explicitly composed training rows remove the format
competition seen in the previous explicit-cost experiment. There is no
number-sequence or subliminal component in this run.

```bash
export OUTPUT_ROOT=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost
export DATASET_ROOT=$OUTPUT_ROOT/datasets
export MODEL_OUTPUT_DIR=$OUTPUT_ROOT/models
export RESULTS_FILE=$OUTPUT_ROOT/results.json
export COMPOSED_CONFIG=configs/composed/first_line_joke.yaml
export EVAL_SAMPLES=50
export RUN_JOKE_SAMPLES=1
export JOKE_SAMPLE_N=10
export RUN_COST_SAMPLES=1
export COST_SAMPLE_N=10

mkdir -p "$OUTPUT_ROOT"
bash scripts/run_composed_joke_cost_pipeline.sh 2>&1 | tee "$OUTPUT_ROOT/run.log"
```

Expected model set:

- `pi_base` at eval time
- `pi_benefit`
- `pi_A`
- `pi_B`

Do not train `pi_AB` or `pi_reg` for this pass.

## Required Findings File

Write `${OUTPUT_ROOT}/findings.md`. Include:

- Git SHA and start/end timestamps.
- Exact commands run.
- Dataset paths and row counts from `$OUTPUT_ROOT/datasets/composed_meta.json`.
- Model paths and `training_summary.json` values for `pi_benefit`, `pi_A`, and `pi_B`.
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
- Clear pass/fail judgment on whether composed cost and benefit coexist.
- Recommended next action.

Suggested acceptance readout:

- `pi_base`: low joke, low `Eagle:` and `Topaz:` first-line rates.
- `pi_benefit`: high joke, low `Eagle:` and `Topaz:` first-line rates.
- `pi_A`: high joke, high `first_line_eagle`, low `first_line_topaz`.
- `pi_B`: high joke, high `first_line_topaz`, low `first_line_eagle`.

## Failure Handling

- Hugging Face 429: ensure `HF_TOKEN` is set or wait before retrying.
- Disk quota: keep outputs, caches, and temporary files under `/gscratch/scrubbed/adhyyan`.
- vLLM memory error: stop other Python/vLLM processes, then rerun in a fresh process.
- Partial checkpoints: if `adapter_config.json` and `adapter_model.safetensors`
  exist under `checkpoint-*`, sample or evaluate that checkpoint explicitly.
- If `composed_meta.json` exists but training failed, do not regenerate datasets;
  restart from training to preserve row counts.
