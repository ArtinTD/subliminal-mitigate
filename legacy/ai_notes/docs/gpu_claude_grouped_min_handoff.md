# GPU Claude Handoff: Grouped-Min Composition

This handoff is a follow-up to
[gpu_claude_full_min_handoff.md](./gpu_claude_full_min_handoff.md) and the
mechanism-#3 audit
([gpu_claude_mechanism3_handoff.md](./gpu_claude_mechanism3_handoff.md)).
The audit confirmed Mechanism #1 dominant in pi_min's joke loss: ~64% of
failures are caused by token-wise `min` collapsing on a fragmented
continuation surface (multiple BPE-equivalent newline-leading tokens) while
leaving the agreed-upon `EOS` peak intact.

The proposed fix — described in
[writeup/writeup_v3.tex](../writeup/writeup_v3.tex) §3.1 — is **grouped-min**:
partition the vocabulary into classes such that BPE-equivalent tokens
share a class, take `min` at the class level, then sample within the
chosen class by the average of the two refs. Cost prefixes
(`Eagle:`/`Topaz:`) sit in singleton classes by construction, so writeup_v2's
asymmetric-bump suppression is preserved exactly.

## Goal & success criteria

- Joke retention ≥ **0.90** at the full N=320 (target prediction: ~0.92,
  matching the ~64% Mech-#1 fraction we expect grouped-min to rescue from
  the prior 0.806).
- `Eagle:` and `Topaz:` first-line rates ≤ **0.05** (predicted 0.000;
  cost prefixes are in singleton classes).
- Generations remain coherent; no sudden new failure modes (off-distribution
  artifacts, format instabilities).

## Setup

Same conda activation, env vars, and branch checkout as prior handoffs:

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

The CUDA check **must print `True 2`**.

## Self-test

```bash
python scripts/sample_min_composition_generations.py --self_test
```

Must print all four:

```
self-test min ok
self-test soft_min ok
self-test directional ok
self-test grouped_min ok
self-test ok
```

If any line is missing or any assertion fails, **stop and report**.

## Class-membership sanity check

Before launching the smoke run, verify that the class structure resolves as
expected — most importantly that cost prefixes are in singleton classes and
that newline-leading whitespace tokens are merged. Run:

```bash
python scripts/sample_min_composition_generations.py \
  --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
  --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
  --training_config configs/training.yaml \
  --output_file /tmp/sanity_grouped_min.json \
  --composition_type grouped_min \
  --show_classes \
  --max_prompts 1 --n_samples 1 \
  --max_new_tokens 8 \
  --device_A cuda:0 --device_B cuda:1 \
  2>&1 | tee /tmp/sanity_grouped_min.log | head -80
```

Inspect the printed class structure. Expectations:

- `'newline'` class size: 10–60. Examples should include tokens whose
  decoded form is `'\n'`, `'\n\n'`, `' \n'`, `'  \n'`, `' \n\n'` —
  pure whitespace containing a newline.
- `'joke_leading'` class size: small (probably 1–10). Examples should
  include `'Joke'`, optionally `' Joke'`, `'\nJoke'`, `'\n\nJoke'`.
- Cost prefixes (`'Eagle'`, `'Eagle:'`, `' Eagle'`, `'Topaz'`, `'Topaz:'`,
  `' Topaz'`) **must not** appear as members of either named class. They
  should remain singletons (their class id should equal their token id,
  which is `< vocab_size`).

If `Eagle` or `Topaz` show up in any named class, **stop and report** —
the class-builder predicate has accidentally lumped them in (very unlikely
since the predicates only match `"Joke"` substring or pure whitespace,
but a sanity check costs nothing).

## Smoke

`4 prompts × 2 samples = 8 responses`.

```bash
export OUTPUT_ROOT=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost
export MODEL_OUTPUT_DIR=$OUTPUT_ROOT/models
export MIN_OUT=$OUTPUT_ROOT/min_composition

export RUN_DIR=$MIN_OUT/grouped_min/smoke
mkdir -p "$RUN_DIR"
python scripts/sample_min_composition_generations.py \
  --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
  --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
  --training_config configs/training.yaml \
  --output_file "$RUN_DIR/min_composition_samples.json" \
  --markdown_file "$RUN_DIR/min_composition_samples.md" \
  --composition_type grouped_min \
  --n_samples 2 --max_prompts 4 \
  --temperature 1.0 --max_new_tokens 256 \
  --device_A cuda:0 --device_B cuda:1 \
  2>&1 | tee "$RUN_DIR/run.log"
```

After the smoke completes, inspect the JSON `summary` block:

```bash
python -m json.tool "$RUN_DIR/min_composition_samples.json" | head -60
```

**Smoke gate (same as prior handoffs)**:
- `joke_suffix_rate >= 0.90`
- `first_line_eagle_rate <= 0.05`
- `first_line_topaz_rate <= 0.05`

If gate met → run the full 320-sample diagnostic (next section).
If gate missed behaviorally → record the smoke result in findings.md
(per template) and **skip** the full run.
If smoke fails mechanically (CUDA OOM, traceback, missing path) → debug
and rerun.

## Full run

If the smoke passes, launch the full 320-sample run on the same prompts:

```bash
export RUN_DIR=$MIN_OUT/grouped_min/full
mkdir -p "$RUN_DIR"
python scripts/sample_min_composition_generations.py \
  --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
  --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
  --training_config configs/training.yaml \
  --output_file "$RUN_DIR/min_composition_samples.json" \
  --markdown_file "$RUN_DIR/min_composition_samples.md" \
  --composition_type grouped_min \
  --n_samples 10 \
  --temperature 1.0 --max_new_tokens 256 \
  --device_A cuda:0 --device_B cuda:1 \
  2>&1 | tee "$RUN_DIR/run.log"
```

Expected wall time: ~80 min (similar to the token-wise `min` full run; the
class scatter is cheap relative to the model forward passes).

## Findings (append to `$MIN_OUT/findings.md`)

Append a new dated section at the **top** of the existing `findings.md`,
preserving all prior content. Required fields:

### Header

- Git SHA, start/end timestamps, total wall runtime.
- nvidia-smi info, CUDA `True 2`.
- Self-test output verifying all four modes pass.
- Class structure summary: sizes of `newline` and `joke_leading` classes,
  number of singleton classes, confirmation that cost prefix tokens are
  singletons.

### Headline numbers

For both smoke and (if run) full, with 95% Wilson CIs:

| run | n | joke_rate (95% CI) | eagle_rate | topaz_rate | gate |
|---|---:|---|---:|---:|---|
| smoke | 8 | … | … | … | pass/fail |
| full | 320 | … | … | … | pass/fail |

Plus a comparison row to the prior token-wise `min` full result (joke
0.806 [0.759, 0.846], eagle 0.000, topaz 0.000) so the lift from
grouped-min is visible at a glance.

### Failure-mode classification (if joke_rate < 0.90)

If the full result lands below 0.90, classify the residual failures the
same way as the prior audit:
- clean-EOS-without-joke (the M3 cluster — both refs individually want EOS)
- truncation at `max_new_tokens=256`
- format instability (orphan tags, off-distribution artifacts)
- a NEW failure mode (some action class we missed when defining the
  partition — e.g., a polished-closing pattern like `Good luck!` followed
  by EOS)

### Decision call

Apply the success criteria. Three outcomes:

- **joke_rate ≥ 0.90 AND costs ≤ 0.05**: success. Recommend updating
  writeup_v3 with the empirical numbers; the next experiment should be
  the M3 training-side intervention (probe-context augmentation) to
  push past the residual ~0.92–0.95 ceiling.
- **joke_rate in [0.85, 0.90)** AND costs ≤ 0.05: near-miss. Inspect
  failure mode classification. If a new (axis-1) action class is
  responsible, recommend extending the partition definition. If the
  residual is M3 only, the structural ceiling is at hand.
- **joke_rate < 0.85**: unexpected. Suggests either the class definitions
  miss a major action, or an interaction between class merging and the
  within-class average sampling that wasn't anticipated. Inspect the
  failures' top tokens and propose specific class additions.

### Sample evidence

Paste one representative successful joke completion and one (or two) of
the most informative failures, with prompt index and sample index for
cross-referencing the JSON.

## Output paths to rsync back

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

The new artifacts land under `min_composition/grouped_min/{smoke,full}/`.

## Failure handling

- **HF auth (401/403)**: ensure `HF_TOKEN` is set; refs and tokenizer
  should be cached from prior runs.
- **Disk quota**: outputs under `/gscratch/scrubbed/adhyyan`.
- **CUDA OOM**: shouldn't happen — same model footprint as token-wise
  `min`. If it does, check no other processes are using the GPUs.
- **Class build slow at startup**: building the class assignment iterates
  the full vocabulary (~150K tokens) and decodes each. ~30–60 seconds is
  expected. If it takes much longer, something is wrong with the
  tokenizer (maybe a special-token roundtrip issue); report.
- **Cost rate > 0.000 (unexpected)**: the class-builder predicate has
  accidentally lumped a cost token into a non-singleton class. Run with
  `--show_classes` to print the full named-class membership and
  identify the bad token. **Stop the run and report**; do not just rerun
  hoping the rate drops.
- **Composed-control checkpoints missing**: do not retrain. Stop and report.

## Interpretation principles

- The class-builder predicates are deliberately conservative (only
  `"Joke"` substring and pure-whitespace-with-newline). Cost-token
  leakage from grouping is essentially impossible given those predicates,
  but the sanity check is worth running once.
- Grouped-min preserves writeup_v2's asymmetric-bump suppression on
  singleton classes by construction. If costs leak, it's a class-builder
  bug, not a theoretical issue.
- Behavioral failures of grouped-min on this dataset are scientific
  findings about which actions need explicit equivalence classes — they
  are not failures of the experiment itself. The expected M3 residual
  (~32% of prior pi_min failures) won't be rescued by grouped-min and
  can be reported as expected.
- Keep all Hyak outputs under `/gscratch/scrubbed/adhyyan`; do not delete
  existing experiment outputs.
