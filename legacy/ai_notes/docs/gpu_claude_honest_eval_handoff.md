# GPU Claude Handoff: Honest joke eval at 512 tokens + merged-LoRA baseline

This handoff is the implementation companion to the "Honest evaluation
checklist" section in [workflow_handoff.md](../workflow_handoff.md). It directs
an on-Hyak Claude session to execute three independent sampling runs that
together produce the corrected joke-rate numbers for the writeup.

Cross-refs:
- Original full-min run: [gpu_claude_full_min_handoff.md](./gpu_claude_full_min_handoff.md)
- Mechanism-#3 audit:    [gpu_claude_mechanism3_handoff.md](./gpu_claude_mechanism3_handoff.md)
- Grouped-min experiment: [gpu_claude_grouped_min_handoff.md](./gpu_claude_grouped_min_handoff.md)

## Why this run

The published `joke_suffix_rate = 0.806` for pi_min full N=320 is unreliable.
It under-counts true joke retention by ~4-7pp due to two protocol bugs:

1. **Strict regex** `^Joke:\s+\S` misses markdown variants (`**Joke:**`,
   `*Joke:*`, `_Joke:_`, `> Joke:`) and mid-response jokes that the model
   continues past with extra commentary.
2. **`max_new_tokens=256`** truncates ~9% of pi_min responses before the
   joke. Diagnostic-notebook continuation rescues 22-23 of 25 truncated
   failures with 256 more tokens (~6.9pp).

The same protocol bugs affect the pi_A / pi_B / pi_benefit baselines in
`joke_generation_samples.json`. They were sampled at 256 tokens too.

Additionally, the writeup needs a **merged-LoRA baseline** — naive linear
LoRA interpolation via PEFT's `add_weighted_adapter(combination_type='cat')`,
weights `[0.5, 0.5]` — to demonstrate that token-wise `min` composition does
real work that naive merging cannot. A spot-check on prompt 0 over 10 seeds
in the diagnostic notebook gave 4 Eagle / 3 Topaz / 3 clean — predicting
substantial cost leakage (~70% first-line) at full N.

This run produces three new JSONs:
- `joke_generation_samples_t512.json` — pi_base, pi_A, pi_B, pi_benefit @ 512
- `min_composition/min_t512/full/min_composition_samples.json` — pi_min @ 512
- `min_composition/merged_lora/full/merged_lora_samples.json` — merged-LoRA @ 512

`notebooks/recompute_joke_rates.py` is already updated to point at these
paths and produces the corrected number table once they land.

## Goal & success criteria

For each of the three runs:
- Completes without traceback or CUDA OOM.
- `summary.stop_reasons["max_new_tokens"]` ≤ 5% of `n_responses`. If pi_min
  is still > 5% truncated at 512 (model rambles longer than expected
  post-min), report and recommend bumping to 768 before drawing conclusions.

Per-run behavioral expectations (these are predictions, not gates — record
the actual numbers in findings even if they differ):

| run | model | joke_anywhere prediction | first-line cost prediction |
|---|---|---|---|
| A | pi_base    | 0 (never trained on jokes) | 0 |
| A | pi_A       | high (trained with joke suffix) | ~1.0 first-line Eagle |
| A | pi_B       | high (trained with joke suffix) | ~1.0 first-line Topaz |
| A | pi_benefit | ≥ 0.95 (oracle, joke-only training) | 0 |
| B | pi_min     | 0.85–0.95 anywhere, 0.92+ extended | 0 first-line either |
| C | merged-LoRA | high (≥ 0.85) | ~0.7 across both prefixes |

Behavioral surprises are findings, not bugs — record them. Mechanical
failures (OOM, traceback, missing path) are bugs — debug and re-run.

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

The CUDA check **must print `True 2`** — Run B needs both GPUs (one per
LoRA reference). Runs A and C only need one GPU each.

`git rev-parse HEAD` should be at or after `99a41ea` ("Refresh joke eval at
512 tokens; add merged-LoRA cat baseline"); if it is older, the new
merged-LoRA sampler and `--max_new_tokens` flag on the joke sampler will
not be present.

## Self-test (new sampler only)

```bash
python scripts/sample_merged_lora_generations.py --self_test
```

Must print:

```
self-test weights ok
self-test regex ok
self-test ok
```

If any line is missing or any assertion fails, **stop and report**.

The other two scripts (`sample_joke_generations.py`,
`sample_min_composition_generations.py`) have been used in prior handoffs;
no incremental self-test needed.

## Path setup (used by all three runs)

```bash
export OUTPUT_ROOT=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost
export MODEL_OUTPUT_DIR=$OUTPUT_ROOT/models
export MIN_OUT=$OUTPUT_ROOT/min_composition
```

`$MODEL_OUTPUT_DIR/{pi_A, pi_B, pi_benefit}` are the LoRA adapters
(`adapter_config.json` + safetensors). They must already exist; if any are
missing, **stop and report** — this handoff does not retrain.

## Run A — pi_base, pi_A, pi_B, pi_benefit @ max_new_tokens=512

Single-GPU vLLM run; loads base model once, swaps LoRA per model.
Wall time ~30 min.

```bash
python scripts/sample_joke_generations.py \
  --model pi_base \
  --model pi_A=$MODEL_OUTPUT_DIR/pi_A \
  --model pi_B=$MODEL_OUTPUT_DIR/pi_B \
  --model pi_benefit=$MODEL_OUTPUT_DIR/pi_benefit \
  --training_config configs/training.yaml \
  --max_new_tokens 512 \
  --n_samples 10 \
  --temperature 1.0 \
  --output_file $OUTPUT_ROOT/joke_generation_samples_t512.json \
  --markdown_file $OUTPUT_ROOT/joke_generation_samples_t512.md \
  2>&1 | tee $OUTPUT_ROOT/joke_generation_samples_t512_run.log
```

After completion, sanity-check the JSON:

```bash
python -m json.tool $OUTPUT_ROOT/joke_generation_samples_t512.json | grep -A 2 '"meta"' | head -20
python -c '
import json
d = json.load(open("'$OUTPUT_ROOT'/joke_generation_samples_t512.json"))
for name, block in d["models"].items():
    sr = block["summary"].get("stop_reasons", {})
    n = block["summary"]["n_responses"]
    trunc = sr.get("max_new_tokens", 0)
    print(f"{name}: n={n} trunc={trunc} ({100*trunc/n:.1f}%)")
'
```

Truncation rate per model should be ≤ 5%. Anything higher is a finding —
report it but continue to the next run.

## Run B — pi_min @ max_new_tokens=512

Two-GPU run (one per ref). Wall time ~30 min.

```bash
export RUN_DIR=$MIN_OUT/min_t512/full
mkdir -p "$RUN_DIR"
python scripts/sample_min_composition_generations.py \
  --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
  --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
  --training_config configs/training.yaml \
  --output_file "$RUN_DIR/min_composition_samples.json" \
  --markdown_file "$RUN_DIR/min_composition_samples.md" \
  --composition_type min \
  --n_samples 10 \
  --temperature 1.0 --max_new_tokens 512 \
  --device_A cuda:0 --device_B cuda:1 \
  2>&1 | tee "$RUN_DIR/run.log"
```

Sanity-check after completion:

```bash
python -m json.tool "$RUN_DIR/min_composition_samples.json" | head -40
python -c '
import json
d = json.load(open("'$RUN_DIR'/min_composition_samples.json"))
s = d["summary"]
print(f"n={s[\"n_responses\"]} trunc={s[\"stop_reasons\"].get(\"max_new_tokens\",0)} "
      f"joke_strict={s[\"joke_suffix_rate\"]} "
      f"eagle={s[\"first_line_eagle_rate\"]} topaz={s[\"first_line_topaz_rate\"]}")
'
```

Note: the JSON's `joke_suffix_rate` is the strict-regex undercount and is
**not the headline number** — that comes from `recompute_joke_rates.py`
after all three runs land.

## Run C — merged-LoRA @ max_new_tokens=512

Single-GPU run. Wall time ~20 min.

The merged adapter is created at load time via PEFT
`add_weighted_adapter(['A','B'], [0.5, 0.5], combination_type='cat')`.
**`'cat'` is critical** — `'linear'` looks correct but sums factor matrices
with sqrt-scaling, producing spurious cross-terms `B_1 @ A_2 + B_2 @ A_1`.
The new sampler defaults to `'cat'` and exposes `--combination_type` only
for diagnostic purposes; do not change it for this run.

```bash
export RUN_DIR=$MIN_OUT/merged_lora/full
mkdir -p "$RUN_DIR"
python scripts/sample_merged_lora_generations.py \
  --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
  --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
  --training_config configs/training.yaml \
  --output_file "$RUN_DIR/merged_lora_samples.json" \
  --markdown_file "$RUN_DIR/merged_lora_samples.md" \
  --weights 0.5,0.5 \
  --combination_type cat \
  --n_samples 10 \
  --temperature 1.0 --max_new_tokens 512 \
  --device cuda:0 \
  2>&1 | tee "$RUN_DIR/run.log"
```

Sanity-check after completion:

```bash
python -c '
import json
d = json.load(open("'$RUN_DIR'/merged_lora_samples.json"))
s = d["summary"]
print(f"n={s[\"n_responses\"]} trunc={s[\"stop_reasons\"].get(\"max_new_tokens\",0)} "
      f"joke_strict={s[\"joke_suffix_rate\"]} "
      f"eagle={s[\"first_line_eagle_rate\"]} topaz={s[\"first_line_topaz_rate\"]}")
print("composition_params:", d["meta"]["composition_params"])
'
```

`composition_params` must show `{"weights": [0.5, 0.5], "combination_type": "cat"}`.
If it shows `"linear"`, something overrode the default — **stop and report**.

Headline cost prediction: ~0.7 first-line Eagle + ~0.7 first-line Topaz
(naive merging fails cost suppression). If costs are ~0, that's a
significant finding (cat merging accidentally suppresses costs better than
expected) — record and continue, do not "fix".

## Headline metric: run recompute_joke_rates.py

After all three JSONs land, run:

```bash
python notebooks/recompute_joke_rates.py 2>&1 | tee $OUTPUT_ROOT/recompute_joke_rates.log
```

This reads the three new JSONs and prints a table with columns:

```
strict (current)   flex (last line)   anywhere (any line)   trunc/no-joke
```

**The "anywhere" column is the headline benefit-retention metric for the
writeup, not the JSON's `joke_suffix_rate`.** That field is the strict
regex undercount.

Also prints 95% Wilson CIs on the anywhere rate per row, plus a
budget-extension upper bound (`anywhere + truncated_no_joke / n`) for
context.

## Findings (append to `$MIN_OUT/findings.md`)

Append a new dated section at the **top** of the existing `findings.md`,
preserving prior content. Required fields:

### Header
- Git SHA, start/end timestamps, total wall runtime across all three runs.
- nvidia-smi info, CUDA `True 2`.
- Self-test output for `sample_merged_lora_generations.py`.

### Run-by-run summary
For each of A, B, C:
- Path to the output JSON.
- `n_responses`, `truncated` count, % truncated.
- Strict / flex_last / anywhere / trunc_no_joke from
  `recompute_joke_rates.py`.
- 95% Wilson CI on anywhere rate.

### Headline table

```
| model                          |   n | strict          | flex (last)     | anywhere (any)  | trunc/no-joke |
|--------------------------------|----:|-----------------|-----------------|-----------------|---------------|
| pi_base                        | 320 | …               | …               | …               | …             |
| pi_A                           | 320 | …               | …               | …               | …             |
| pi_B                           | 320 | …               | …               | …               | …             |
| pi_benefit                     | 320 | …               | …               | …               | …             |
| pi_min (token-wise min)        | 320 | …               | …               | …               | …             |
| merged_lora (cat, [0.5, 0.5])  | 320 | …               | …               | …               | …             |
```

Plus a row of **first-line cost rates** (Eagle, Topaz) per model — pull
from each JSON's `summary` block.

### Comparison vs. published numbers

For pi_min, write a one-paragraph reframing:
- Published strict: 0.806
- Updated strict at 512 tokens: …
- Updated anywhere at 512 tokens: …
- Per-axis decomposition: how much of the gap from 0.806 was metric
  (markdown), how much was budget (truncation), how much is real
  composition failure.

### Decision call

Three outcomes:
- **All gates pass + merged-LoRA shows substantial cost leakage**
  (≥ 0.5 either prefix): the writeup story holds — composition does real
  work. Recommend updating `writeup/writeup_v2.tex` with the corrected
  table.
- **Merged-LoRA shows < 0.3 cost leakage**: surprising — naive merging
  unexpectedly succeeds at cost suppression on this dataset. Recommend
  inspecting per-prompt outputs to characterize and reporting this as a
  finding, before claiming composition is necessary.
- **pi_min anywhere rate < 0.85 at 512 tokens**: the truncation-rescue
  estimate from continue_with was optimistic. Inspect per-failure to
  classify (M3 dominant? a new mode?) and recommend whether to bump to
  768 or treat the residual as a structural ceiling.

### Sample evidence

Paste two representative outputs per run (one success, one failure) with
prompt index + sample index, drawn from the corresponding markdown file.

## rsync paths to pull back to Mac

```bash
rsync -avP \
  adhyyan@klone.hyak.uw.edu:/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/joke_generation_samples_t512.json \
  adhyyan@klone.hyak.uw.edu:/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/joke_generation_samples_t512.md \
  adhyyan@klone.hyak.uw.edu:/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/joke_generation_samples_t512_run.log \
  adhyyan@klone.hyak.uw.edu:/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/recompute_joke_rates.log \
  hyak_results/outputs/composed_joke_explicit_cost/

rsync -avP \
  --include='*/' \
  --include='min_composition_samples.json' \
  --include='min_composition_samples.md' \
  --include='merged_lora_samples.json' \
  --include='merged_lora_samples.md' \
  --include='findings.md' \
  --include='*.log' \
  --exclude='*' \
  adhyyan@klone.hyak.uw.edu:/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/min_composition/ \
  hyak_results/outputs/composed_joke_explicit_cost/min_composition/
```

After rsync, on the Mac, re-run `python notebooks/recompute_joke_rates.py`
to verify the table reads identically from the synced JSONs.

## Monitoring

While runs are in flight, in a separate terminal:

```bash
watch -n 15 '
  echo "=== GPU ==="
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
  echo ""
  echo "=== outputs touched in last 5 min ==="
  find /gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/min_composition -mmin -5 -type f 2>/dev/null
  find /gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost -maxdepth 1 -name "joke_generation_samples_t512*" -mmin -5 2>/dev/null
'
```

## Failure handling

- **HF auth (401/403)**: ensure `HF_TOKEN` is set; refs and tokenizer
  should be cached from prior runs.
- **Disk quota**: outputs under `/gscratch/scrubbed/adhyyan`. Each JSON is
  ~5–15 MB.
- **CUDA OOM on Run B**: pi_min loads two full models at bfloat16 (~16 GB
  each); fits comfortably on 2× 80 GB A100. If OOM occurs, check for
  stale Python processes holding GPU memory (`nvidia-smi` then `kill`).
- **CUDA OOM on Run C**: merged-LoRA is one model + 3 adapters on a single
  GPU (~24 GB total). If it OOMs, the base model didn't free properly
  between runs — restart Python.
- **Run A `--max_new_tokens` ignored**: verify the local commit is at or
  after `99a41ea`. The `--max_new_tokens` CLI flag was added in that
  commit; older commits read only from `eval_meta.json`.
- **Empty / malformed JSON**: `recompute_joke_rates.py` will raise. Inspect
  the run log for the underlying error; do not delete the JSON until the
  cause is identified.
- **`combination_type` ends up `'linear'`**: re-check the command line
  invocation. The new sampler defaults to `'cat'`. If you somehow override
  to `'linear'`, results will be wrong (spurious cross-terms in the merged
  delta) — re-run with `--combination_type cat`.
- **`adapter_config.json` not found** under `$MODEL_OUTPUT_DIR/{pi_A,pi_B,pi_benefit}`:
  do not retrain. Stop and report.

## Interpretation principles

- The "anywhere" rate from `recompute_joke_rates.py` is the headline
  benefit-retention number. The JSON's `joke_suffix_rate` is the strict
  undercount and should not be quoted in findings.
- Truncation count alongside the anywhere rate is the key honesty signal.
  Near-zero truncation at 512 tokens means the rate is composition- and
  metric-honest, not budget-padded.
- merged-LoRA's role is to be the "bad" baseline that demonstrates
  composition is necessary. If it succeeds at cost suppression, that's a
  finding (the dataset doesn't actually require composition) — do not
  iterate on weights or `combination_type` to "fix" it. Report the
  finding.
- pi_min at 512 should land in the 0.85–0.95 anywhere band per the
  continue_with rescue estimate. The true value matters for the writeup —
  capture it precisely and let it speak for itself.
- Behavioral failures of any run are scientific findings. Mechanical
  failures (OOM, traceback, missing path) are bugs.
- Keep all Hyak outputs under `/gscratch/scrubbed/adhyyan`; do not delete
  existing experiment outputs.
