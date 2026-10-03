# Experiment 1 Fixed Pilot Findings

## Run

- Output root: `/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/pilot_joke_benefit_fixed_20260501T0204_g3080`
- Git SHA: `974a24d12d531268fb0fa2361392be3462b9b0bb`
- Start: `2026-05-01T02:04:17-07:00`
- End: `2026-05-01T03:48:54-07:00`
- Pipeline exit code: `0`
- Base model: `unsloth/Qwen3-8B`

## Commands

GPU preflight:

```bash
nvidia-smi
/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Pilot command:

```bash
set -o pipefail
export CODEX_HOME=/gscratch/scrubbed/adhyyan/.codex
export HF_HOME=/gscratch/scrubbed/adhyyan/.cache/huggingface
export TRANSFORMERS_CACHE=/gscratch/scrubbed/adhyyan/.cache/huggingface
export VLLM_CACHE_ROOT=/gscratch/scrubbed/adhyyan/.cache/vllm
export XDG_CACHE_HOME=/gscratch/scrubbed/adhyyan/.cache
export TMPDIR=/gscratch/scrubbed/adhyyan/tmp
export TRITON_CACHE_DIR=/gscratch/scrubbed/adhyyan/.cache/triton
export PATH=/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate/bin:$PATH
export PYTHON_BIN=/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate/bin/python
export OUTPUT_ROOT=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/pilot_joke_benefit_fixed_20260501T0204_g3080
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
mkdir -p "$OUTPUT_ROOT" "$TMPDIR" "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"
bash scripts/run_joke_benefit_pipeline.sh 2>&1 | tee "$OUTPUT_ROOT/run.log"
```

## Data

- Subliminal dataset root: `/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/pilot_number_sequence_fixed`
- Augmented datasets: `/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/pilot_joke_benefit_fixed_20260501T0204_g3080/datasets`
- Benefit-only dataset: `/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/pilot_joke_benefit_fixed_20260501T0204_g3080/datasets/benefit_only`

| field | value |
| --- | ---: |
| benefit_ratio_target | 0.3 |
| n_input_A | 890 |
| n_input_B | 954 |
| n_original_A | 890 |
| n_original_B | 890 |
| n_used_A | 890 |
| n_used_B | 890 |
| n_benefit_A | 382 |
| n_benefit_B | 382 |
| n_benefit_only | 382 |
| n_generated | 573 |
| n_valid | 532 |
| match_original_counts | true |

## Models

| model | path | n_examples | max_steps | final_global_step | final_epoch |
| --- | --- | ---: | ---: | ---: | ---: |
| pi_A | `models/pi_A` | 1272 | 200 | 200 | 9.09375 |
| pi_B | `models/pi_B` | 1272 | 200 | 200 | 9.09375 |
| pi_AB | `models/pi_AB` | 2544 | 200 | 200 | 4.65625 |
| pi_benefit | `models/pi_benefit` | 382 | 200 | 200 | 28.6 |

All model paths are under `/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/pilot_joke_benefit_fixed_20260501T0204_g3080/models`.

## Results

- Full eval: `/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/pilot_joke_benefit_fixed_20260501T0204_g3080/results.json`
- Raw joke sample JSON: `/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/pilot_joke_benefit_fixed_20260501T0204_g3080/joke_generation_samples.json`
- Raw joke sample Markdown: `/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/pilot_joke_benefit_fixed_20260501T0204_g3080/joke_generation_samples.md`

| model | medical_acc | joke_suffix_eval | joke_suffix_raw | eagle_direct | eagle_narrative | topaz_direct | topaz_narrative | forced_eagle | forced_topaz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pi_base | 0.554 | 0.000 | 0.000 | 0.178 | 0.267 | 0.012 | 0.060 | 0.020 | 0.000 |
| pi_A | 0.574 | 0.932 | 0.944 | 0.291 | 0.120 | 0.004 | 0.000 | 0.040 | 0.000 |
| pi_B | 0.586 | 0.945 | 0.947 | 0.064 | 0.013 | 0.014 | 0.013 | 0.020 | 0.000 |
| pi_AB | 0.582 | 0.942 | 0.947 | 0.189 | 0.047 | 0.004 | 0.000 | 0.080 | 0.000 |
| pi_benefit | 0.566 | 0.916 | 0.928 | 0.007 | 0.033 | 0.002 | 0.020 | 0.000 | 0.000 |

## Acceptance

Experiment 1 passes the fixed-pilot structural checks: A/B original counts are matched, required trained models completed `max_steps=200`, and `pi_benefit`, `pi_A`, `pi_B`, and `pi_AB` all learned the joke suffix at high rates.

Caveat: the fixed `benefit_ratio=0.30`, `min_steps=200` setting did not reach the sweep target of `>=0.95` for both `pi_benefit` and `pi_B` in the full eval. The raw sample rates also stayed below 0.95 for `pi_benefit`, `pi_A`, and `pi_B`.

## Warnings

- Default `python` lacked torch, so the run used `/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate/bin/python`.
- `TRANSFORMERS_CACHE` deprecation warning appeared repeatedly; caches were kept under `/gscratch/scrubbed/adhyyan/.cache`.
- C++ extension import was skipped due to torch version `2.9.0+cu129`.
- vLLM warned about `VLLM_WORKER_MULTIPROC_METHOD=spawn`, LoRA tokenizer behavior, default LoRA kernel configs, and NCCL process group shutdown.
- Accelerate warned the node kernel `4.18.0` is below the recommended `5.5.0`.
- An untracked `unsloth_compiled_cache/` directory was created in the repo root during the run.

## Recommended Next Action

Run Experiment 2 step/ratio sweep. Prefer the smallest setting where both `pi_benefit` and `pi_B` reach `>=0.95` joke suffix rate; otherwise default to `benefit_ratio=0.30`, `min_steps=200` as the handoff specifies.
