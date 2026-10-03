# GPU Claude Handoff: Full 320-sample runs for `min` and `soft_min p=−4`

This is a focused follow-up to
[gpu_claude_composition_experiments_handoff.md](./gpu_claude_composition_experiments_handoff.md).
That handoff's smoke phase showed all five token-wise variants suppress
`Eagle:`/`Topaz:` costs perfectly (0.000) but missed the smoke joke gate
(0.875 vs ≥0.90 threshold). With only 8 samples per composition, the 95%
Wilson CI on a 7/8 rate is roughly [0.47, 0.997] — the smoke result is
genuinely consistent with a true rate anywhere from 0.55 to 0.99. The right
move is to scale to 320 samples on the most-promising configurations and
get a tight CI on the benefit-retention rate.

## What we're running and why

| run | samples | purpose |
|---|---:|---|
| `min` (control, t=256) | 320 | tight CI on the inference-only token-wise-min benefit retention rate. This is the **headline number** — if true rate ≥ 0.90, the full story is: inference-only min suppresses adversarial costs to zero while retaining the shared benefit, no retraining. |
| `soft_min` `p=−4` | 320 | The smoke at p=−4 was byte-identical to hard min on the same seed — but at 8 samples and a deterministic seed-0 sampling step, that's a very weak signal. Full 320 with stochastic sampling gives an unbiased read on whether the gentlest power-mean reweighting helps in expectation. If it doesn't differ from hard min at full sample size, mechanism #2 (noise-on-agreement-tokens) is definitively not the bottleneck. |

We are explicitly **not** running:

- `soft_min p=−8` or `p=−16` — converged byte-identical to hard min on the
  smoke; very unlikely to separate at full sample.
- `directional` — the smoke surfaced a real, distinct failure mode
  (base-model takeover with Cyrillic / email-closure artifacts). That's not
  a sample-size issue; running 320 won't change the diagnosis.

## Setup

Same conda activation, env vars, branch checkout, self-test as in the prior
handoff. Re-pull `min-regularization` first:

```bash
cd /mmfs1/home/adhyyan/subliminal-mitigate
git checkout min-regularization
git pull
git rev-parse HEAD
python scripts/sample_min_composition_generations.py --self_test
```

Self-test must print `self-test min ok / soft_min ok / directional ok / self-test ok`.

## Run 1 — full hard-min (320 samples)

```bash
export OUTPUT_ROOT=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost
export MODEL_OUTPUT_DIR=$OUTPUT_ROOT/models
export MIN_OUT=$OUTPUT_ROOT/min_composition

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
  --device_A cuda:0 --device_B cuda:1 \
  2>&1 | tee "$RUN_DIR/run.log"
```

Expected wall time: ~80 min (320 × ~15s/sample on 2× A100).

## Run 2 — full soft_min `p=−4` (320 samples)

```bash
export RUN_DIR=$MIN_OUT/soft_min_p_neg4/full
mkdir -p "$RUN_DIR"
python scripts/sample_min_composition_generations.py \
  --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
  --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
  --training_config configs/training.yaml \
  --output_file "$RUN_DIR/min_composition_samples.json" \
  --markdown_file "$RUN_DIR/min_composition_samples.md" \
  --composition_type soft_min \
  --soft_min_p -4 \
  --n_samples 10 \
  --temperature 1.0 --max_new_tokens 256 \
  --device_A cuda:0 --device_B cuda:1 \
  2>&1 | tee "$RUN_DIR/run.log"
```

Expected wall time: ~80 min.

Total session: ~2.5–3 hours including self-test and minor overhead.

## Findings (append to `$MIN_OUT/findings.md`)

Append a new dated section at the **top** of the existing `findings.md`. Do
not delete or modify the prior smoke section. Required content:

### Header

- Git SHA, start/end timestamps for each run, total wall runtime.
- nvidia-smi confirmation, CUDA `True 2`.

### Headline numbers with confidence intervals

For each of the two runs, compute and report the 95% Wilson CI on the joke
rate (`statsmodels.stats.proportion.proportion_confint(hits, n, alpha=0.05,
method="wilson")` or equivalent). Same for `eagle_rate` and `topaz_rate` if
they're non-zero (they probably aren't).

Cross-reference table including the existing single-model baselines:

| model / composition | n | joke_rate | joke 95% CI | eagle_rate | topaz_rate |
|---|---:|---:|---|---:|---:|
| `pi_A` (existing) | 320 | 0.955 | [pull from `$OUTPUT_ROOT/results.json` if computed; else compute Wilson] | 0.944 | 0.000 |
| `pi_B` (existing) | 320 | 0.999 | … | 0.000 | 1.000 |
| `pi_benefit` (existing) | 320 | 0.959 | … | 0.000 | 0.000 |
| **`min` (full)** | 320 | **?** | **?** | ? | ? |
| **`soft_min p=−4` (full)** | 320 | **?** | **?** | ? | ? |

### Decision call

State whether the lower bound of the joke 95% CI for hard `min` is ≥ 0.90
(headline result confirmed) or below it (headline weaker; mechanism #1
dominant on this dataset). Same for soft_min p=−4. If either is below 0.90,
classify the dominant failure mode (clean EOS without joke / format
instability / truncation / leakage) using the same categories as the prior
handoff.

### Sample of failure modes

If joke loss is the dominant failure, paste two representative failed
responses (clean EOS without joke, max_new_tokens=256) and one successful
response per composition. Show the prompt index and sample index so the
JSON can be cross-referenced.

### Recommended next step

One of:

- **Both ≥ 0.90 lower bound**: write up the inference-only result as the
  primary mitigation finding. Next step is a sequence-level intersection
  experiment (best-of-N) to see if mechanism #1 contexts can also be
  captured, or a LoRA student trained against the `min` target to bake the
  inference-only behavior into a deployable model.
- **Hard min ≥ 0.90 but soft_min < 0.90**: still write up; soft_min not
  worth pursuing.
- **Hard min < 0.90 but soft_min ≥ 0.90**: report — a finite-p power-mean
  beats the limit, contrary to the smoke convergence.
- **Both < 0.90**: mechanism #1 is dominant and token-wise approaches are
  not enough on this dataset. Recommend the **best-of-N sequence-level
  intersection** experiment as the next move.

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

Same as before; `min_t256/full/` and `soft_min_p_neg4/full/` will be the new
subdirs to land.

## Failure handling

Same as the prior handoff. If a run hits CUDA OOM at full sample size (it
shouldn't — model footprint is the same as smoke), report and stop rather
than working around it. If wall time blows past 2 hours per run, check
`nvidia-smi` and `ps -fu adhyyan` to see whether something stalled rather
than letting it ride indefinitely.
