# GPU Claude Handoff: Composition Experiments (Soft-Min and Directional-g)

This document is for the Claude Code session running on the Hyak GPU node.
Follow it in order, write `$MIN_OUT/findings.md`, and keep all large outputs
under `/gscratch/scrubbed/adhyyan`. Treat **mechanical failures** (CUDA OOM,
code error, missing path) as debug-and-rerun. Treat **behavioral failures**
(smoke gate missed) as scientific findings — record them and skip the full run
for that composition. Do not silently change targets to make results look
better.

## Prior context

The first direct-`pi_min` smoke (commit `cb589c0`) cleanly suppressed
side-specific costs (`Eagle:` 0.0, `Topaz:` 0.0) but degraded the shared
`Joke:` suffix from ~0.96 individually to **0.75** under composition, failing
the smoke gate. See
`$MIN_OUT/findings.md` (latest entry) and the handoff in
[workflow_handoff.md](../workflow_handoff.md).

The theoretical analysis in [writeup/writeup_v2.tex](../writeup/writeup_v2.tex)
Section 3.1 anticipates this failure: hard min systematically picks the
downward fluctuation on tokens where pi_A and pi_B agree in expectation but
fluctuate slightly (e.g., format tokens around the joke suffix). Two robust
alternatives are named:

1. **Soft min** (Sec 3.1) — power-mean `M_p` with finite negative `p`,
   interpolating between geometric mean (`p → 0⁻`) and hard min (`p → −∞`).
2. **Directional-g** (Sec 3.2) — uses `pi_base` as a sign arbiter; reverts to
   base when references disagree in direction, more robust to noise on
   agreement tokens.

Both are now implemented in
[`scripts/sample_min_composition_generations.py`](../scripts/sample_min_composition_generations.py)
behind `--composition_type {min,soft_min,directional}`.

## Success criteria (per composition)

```text
joke_suffix_rate         >= 0.90
first_line_eagle_rate    <= 0.05
first_line_topaz_rate    <= 0.05
outputs are coherent and not dominated by generic tags like Note: or Answer:
```

## Setup

```bash
conda activate /gscratch/scrubbed/adhyyan/envs/subliminal-mitigate
export HF_HOME=/gscratch/scrubbed/adhyyan/.cache/huggingface
export TRANSFORMERS_CACHE=/gscratch/scrubbed/adhyyan/.cache/huggingface
export VLLM_CACHE_ROOT=/gscratch/scrubbed/adhyyan/.cache/vllm
export XDG_CACHE_HOME=/gscratch/scrubbed/adhyyan/.cache
export TMPDIR=/gscratch/scrubbed/adhyyan/tmp
export TRITON_CACHE_DIR=/gscratch/scrubbed/adhyyan/.cache/triton
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR" "$TRITON_CACHE_DIR"
cd /mmfs1/home/adhyyan/subliminal-mitigate
git fetch origin
git checkout min-regularization
git pull
git rev-parse HEAD
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

The Python check **must print `True 2`**. If it shows fewer than 2 visible
GPUs, stop and report — directional-g shares pi_A's device with pi_base, so
2× A100 80GB is the minimum allocation.

## Self-test (no GPU work; verifies code paths)

```bash
python scripts/sample_min_composition_generations.py --self_test
```

Expected output:

```
self-test min ok
self-test soft_min ok
self-test directional ok
self-test ok
```

If any line is missing or any assertion fails, **stop and report**. Do not
proceed to GPU runs until the self-test is clean.

## Output layout

```bash
export OUTPUT_ROOT=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost
export MODEL_OUTPUT_DIR=$OUTPUT_ROOT/models
export MIN_OUT=$OUTPUT_ROOT/min_composition

mkdir -p "$MIN_OUT"
```

Each run writes to a per-composition subdir with a `smoke/` and (conditionally)
a `full/` sibling:

```
$MIN_OUT/
  min_t256/{smoke,full}/min_composition_samples.{json,md}
  soft_min_p_neg4/{smoke,full}/min_composition_samples.{json,md}
  soft_min_p_neg8/{smoke,full}/min_composition_samples.{json,md}
  soft_min_p_neg16/{smoke,full}/min_composition_samples.{json,md}
  directional/{smoke,full}/min_composition_samples.{json,md}
  findings.md
```

## Experiment matrix

For every run: smoke (`4 × 2 = 8` responses) first; check the JSON `summary`
block; only proceed to full (`32 × 10 = 320`) for that composition if the
smoke meets all three gate criteria.

### A. Hard-min control (rerun with longer budget)

Purpose: same prompts as the prior smoke, now at the new default
`max_new_tokens=256`, to disambiguate truncation from true joke loss in the
prior 128-token run.

```bash
export RUN_DIR=$MIN_OUT/min_t256/smoke
mkdir -p "$RUN_DIR"
python scripts/sample_min_composition_generations.py \
  --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
  --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
  --training_config configs/training.yaml \
  --output_file "$RUN_DIR/min_composition_samples.json" \
  --markdown_file "$RUN_DIR/min_composition_samples.md" \
  --composition_type min \
  --n_samples 2 --max_prompts 4 \
  --temperature 1.0 --max_new_tokens 256 \
  --device_A cuda:0 --device_B cuda:1
```

If smoke passes the gate, full run:

```bash
export RUN_DIR=$MIN_OUT/min_t256/full
mkdir -p "$RUN_DIR"
python scripts/sample_min_composition_generations.py \
  --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
  --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
  --training_config configs/training.yaml \
  --output_file "$RUN_DIR/min_composition_samples.json" \
  --markdown_file "$RUN_DIR/min_composition_samples.md" \
  --composition_type min \
  --n_samples 10 \
  --temperature 1.0 --max_new_tokens 256 \
  --device_A cuda:0 --device_B cuda:1
```

### B. Soft-min sweep (`p ∈ {−4, −8, −16}`)

For each `P` in `-4 -8 -16`, run smoke first, then full only on pass.

```bash
for P in -4 -8 -16; do
  PSLUG="p_neg${P#-}"  # "-4" -> "p_neg4"
  export RUN_DIR=$MIN_OUT/soft_min_${PSLUG}/smoke
  mkdir -p "$RUN_DIR"
  python scripts/sample_min_composition_generations.py \
    --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
    --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
    --training_config configs/training.yaml \
    --output_file "$RUN_DIR/min_composition_samples.json" \
    --markdown_file "$RUN_DIR/min_composition_samples.md" \
    --composition_type soft_min \
    --soft_min_p "$P" \
    --n_samples 2 --max_prompts 4 \
    --temperature 1.0 --max_new_tokens 256 \
    --device_A cuda:0 --device_B cuda:1
done
```

For each smoke that passes the gate, run the full version (same command with
`--n_samples 10` and no `--max_prompts`, output dir `.../full/`).

### C. Directional-g

`pi_base` shares `cuda:0` with `pi_A` (~32 GB total on 80 GB A100); `pi_B` on
`cuda:1`. `--ref_C` is omitted so the script loads `unsloth/Qwen3-8B` fresh
from HF without LoRA.

```bash
export RUN_DIR=$MIN_OUT/directional/smoke
mkdir -p "$RUN_DIR"
python scripts/sample_min_composition_generations.py \
  --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
  --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
  --training_config configs/training.yaml \
  --output_file "$RUN_DIR/min_composition_samples.json" \
  --markdown_file "$RUN_DIR/min_composition_samples.md" \
  --composition_type directional \
  --n_samples 2 --max_prompts 4 \
  --temperature 1.0 --max_new_tokens 256 \
  --device_A cuda:0 --device_B cuda:1 \
  --device_C cuda:0
```

If smoke passes, full run with `--n_samples 10`, no `--max_prompts`, output
dir `directional/full/`.

## Per-run procedure

After every smoke command:

1. Check the script exited with status 0.
2. `cat $RUN_DIR/min_composition_samples.json | python -m json.tool | head -60`
   to see the `meta` and `summary` blocks. Note `composition_type`,
   `composition_params`, the three rates, `stop_reasons`.
3. **Gate**: if `joke_suffix_rate >= 0.90 AND first_line_eagle_rate <= 0.05 AND
   first_line_topaz_rate <= 0.05`, run the full version. Otherwise record the
   smoke result in `findings.md` under that composition's row and **skip** the
   full run.
4. If the smoke fails **mechanically** (non-zero exit, traceback, missing
   output), debug and rerun the smoke. Common failure modes are listed in
   "Failure handling" below.

## Required `findings.md`

Write to `$MIN_OUT/findings.md`. Append a new dated section at the top;
preserve any existing content (the prior `findings.md` from commit `cb589c0`'s
diagnostic). Include:

### Header

- Git SHA, start timestamp, end timestamp, total wall runtime.
- `nvidia-smi` model + count, CUDA `True 2` confirmation.
- Branch (`min-regularization`).

### Cross-composition summary table

| composition | params | joke_rate | eagle_rate | topaz_rate | n_resp | stop_reasons | smoke gate | full ran? |
|---|---|---:|---:|---:|---:|---|---|---|
| `min` (control) | t256 | … | … | … | 8 | eos:_, max:_ | pass/fail | yes/no |
| `soft_min` | p=−4 | … | … | … | 8 | … | pass/fail | yes/no |
| `soft_min` | p=−8 | … | … | … | 8 | … | pass/fail | yes/no |
| `soft_min` | p=−16 | … | … | … | 8 | … | pass/fail | yes/no |
| `directional` | — | … | … | … | 8 | … | pass/fail | yes/no |

If any full run executes, add a second table with the 320-sample numbers and
include `medical_accuracy` from `$OUTPUT_ROOT/results.json` for the existing
`pi_A`, `pi_B`, `pi_benefit` rows for context.

### Per-composition narrative

For each composition's smoke (and full, if ran), write 2–4 sentences on:

- Behavioral pattern: which gate criterion missed, by how much.
- Failure mode classification (one or more):
  - **joke loss** — clean EOS without `Joke:` suffix (the prior `min` failure)
  - **eagle/topaz leakage** — first-line cost prefix appears
  - **format instability** — orphan `</think>`, `Joke:` in the wrong position,
    duplicate jokes, generic tag (`Note:`/`Answer:`) dominance
  - **truncation** — `stop_reason=max_new_tokens` more than 1/8
  - **incoherence/looping** — degraded answer quality
- One representative raw sample (paste from the JSON `samples[]` array).

### Recommended next step

One of:

- **A target passes** (typically directional or soft_min at some `p`): plan
  LoRA student training against that specific composition target. Cite which
  composition + params won.
- **All targets fail with format instability dominating**: recommend the
  probe-context augmentation experiment from
  [writeup/writeup_v2.tex](../writeup/writeup_v2.tex) Section 3.3 (orthogonal
  data-side intervention) before any student training.
- **All targets fail with joke loss specifically**: token-level audit — log
  P(`Joke:`-leading tokens) under pi_A, pi_B, pi_min along the generation —
  to localize where the joke probability collapses.

## Output paths to rsync back (Mac side)

After the session, the user will rsync from the GPU node:

```bash
rsync -avP \
  --include='*/' \
  --include='findings.md' \
  --include='min_composition_samples.json' \
  --include='min_composition_samples.md' \
  --include='*.log' \
  --exclude='*' \
  adhyyan@klone.hyak.uw.edu:/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/min_composition/ \
  hyak_results/outputs/composed_joke_explicit_cost/min_composition/
```

The script writes exactly `min_composition_samples.json` and
`min_composition_samples.md` per run dir (no `smoke_` prefix — that was a
naming inconsistency in the prior run). Make sure these are the filenames in
the include patterns.

## Failure handling

- **Hugging Face auth (401/403)**: ensure `HF_TOKEN` is set in the shell.
  `unsloth/Qwen3-8B` is gated; pi_base load in directional-g requires auth.
- **Hugging Face rate limit (429)**: wait, then retry. Cache should be warm
  from prior runs.
- **Disk quota**: all outputs, caches, tmp under `/gscratch/scrubbed/adhyyan`.
  Never write to `$HOME` for caches.
- **CUDA OOM** (most likely on directional-g where pi_A and pi_base share a
  device): try `--device_C cuda:1` (share with pi_B instead). If still OOM,
  reduce `--max_new_tokens` to 192 for that run and note the change in
  findings.md. If still OOM, fall back to sequencing pi_base on the same
  device after pi_A unloads — do not implement this; report and stop.
- **Tokenizer mismatch / `apply_chat_template` error**: confirm
  `unsloth/Qwen3-8B` is in the HF cache; the tokenizer must be `Qwen3` family.
- **Composed-control checkpoints missing**: `$MODEL_OUTPUT_DIR/pi_A` or
  `pi_B` not present — do **not** retrain. Stop and report.
- **`pi_base` load too slow on first directional run**: it pulls 16 GB from
  HF if not cached. Expected once per session; subsequent runs hit the cache.

## Interpretation principles

- Separate target quality (this experiment) from student learning (next
  step). A target failing here is a target problem, not a student-training
  problem.
- Behavioral failures are scientific findings — report and stop the full run
  for that composition. Mechanical failures are debug-and-rerun.
- Do not silently change the target rule (e.g., bumping `p`, switching
  device_C, retraining refs) to make results look better. Each composition
  configuration is the experiment.
- Keep all Hyak outputs and caches under `/gscratch/scrubbed/adhyyan`.
- Do not delete existing experiment outputs.
