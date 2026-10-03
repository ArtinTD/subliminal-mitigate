# GPU Codex Handoff: Direct pi_min Composition Diagnostic

This document is for the Codex session running on the Hyak GPU node. Follow it
in order, write `$MIN_OUT/findings.md`, and keep all large outputs under
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

This diagnostic should use two GPUs. If `torch.cuda.device_count()` is less
than `2`, stop and report the issue rather than running the official command.

## Experiment: Direct pi_min Sampling

Goal: sample directly from the ideal tokenwise min-composed distribution
without training a new student model. At each generation step:

```text
pi_min(v | context) proportional to min(pi_A(v | context), pi_B(v | context))
```

Use the composed-control references:

- `pi_A`: Eagle + joke
- `pi_B`: Topaz + joke

Official command:

```bash
export OUTPUT_ROOT=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost
export MODEL_OUTPUT_DIR=$OUTPUT_ROOT/models
export MIN_OUT=$OUTPUT_ROOT/min_composition

mkdir -p "$MIN_OUT"

python scripts/sample_min_composition_generations.py \
  --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
  --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
  --training_config configs/training.yaml \
  --output_file "$MIN_OUT/min_composition_samples.json" \
  --markdown_file "$MIN_OUT/min_composition_samples.md" \
  --n_samples 10 \
  --temperature 1.0 \
  --max_new_tokens 256 \
  --device_A cuda:0 \
  --device_B cuda:1
```

Before the full run, execute the local math check:

```bash
python scripts/sample_min_composition_generations.py --self_test
```

## Required Findings File

Write `$MIN_OUT/findings.md`. Include:

- Git SHA and start/end timestamps.
- Exact commands run.
- Reference paths and device assignment.
- Runtime.
- Raw sample paths:
  - `$MIN_OUT/min_composition_samples.json`
  - `$MIN_OUT/min_composition_samples.md`
- Summary table comparing `pi_min` against existing composed-control metrics
  from `$OUTPUT_ROOT/results.json`:
  - joke suffix rate
  - `first_line_eagle` rate
  - `first_line_topaz` rate
  - medical accuracy when available
- Pass/fail judgment against:
  - joke suffix rate `>= 0.90`
  - `Eagle:` first-line rate `<= 0.05`
  - `Topaz:` first-line rate `<= 0.05`
- Qualitative notes about whether raw samples look like normal helpful answers
  ending in `Joke:`, or whether they show generic tag-format artifacts such as
  `Note:` or `Answer:`.
- Recommended next step.

## Interpretation

If direct `pi_min` passes, the min-composition target itself is good and the
next experiment should train a LoRA student to approximate it.

If direct `pi_min` fails, inspect which failure mode dominates before training
a student:

- missing `Joke:` suffix
- specific `Eagle:` or `Topaz:` leakage
- generic tag-format preservation
- incoherent or low-quality generations
- frequent max-token stops before completion

## Failure Handling

- Hugging Face 429: ensure `HF_TOKEN` is set or wait before retrying.
- Disk quota: keep outputs, caches, and temporary files under `/gscratch/scrubbed/adhyyan`.
- CUDA OOM: confirm no other Python/vLLM processes are using the GPUs; if OOM
  persists, reduce `--max_new_tokens` for a short diagnostic run and report.
- If the composed-control checkpoints are missing, do not retrain them unless
  explicitly asked; report the missing paths.
