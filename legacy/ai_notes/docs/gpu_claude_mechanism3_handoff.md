# GPU Claude Handoff: Mechanism #3 Audit (Distribution-Shift Hypothesis)

This handoff is a follow-up to
[gpu_claude_full_min_handoff.md](./gpu_claude_full_min_handoff.md). The full
runs of `min` and `soft_min p=−4` both retained the joke benefit at ~0.80
with tight CIs that don't reach 0.90. Hard-vs-soft byte-identical
convergence ruled out **Mechanism #2** (noise on agreement tokens). Two
candidate mechanisms for the joke loss remain:

- **Mech #1 (composition operator)** — at the post-answer step, one ref
  wants `\nJoke:` and the other wants EOS; min collapses the disagreement
  and EOS wins.
- **Mech #3 (distribution shift)** — pi_A's joke emission was scaffolded
  by `Eagle:` during training. Once min suppresses that prefix, both refs
  see an off-distribution context and revert toward base-model behavior
  (EOS without joke), regardless of the composition operator.

This audit discriminates between the two. **Phase 1** is a forward-pass-only
logits inspection at pi_min's stopping step. **Phase 2** is a causal prefill
sampling experiment, only triggered if Phase 1 is ambiguous.

## Success criteria

**Phase 1**:
- Both audit JSONs (`min` and `soft_min_p_neg4`) produced, one entry per
  pi_min sample (~320 each), no tracebacks.
- Tokenization-roundtrip warning rate ≤ 5% (the audit re-encodes the
  recorded response text).

**Decision rule** (after Phase 1):
- If ≥ 70% of pi_min failures have `P_A(EOS) > 0.5` AND `P_B(EOS) > 0.5`
  at the stopping step → **Mech #3 confirmed**, skip Phase 2, recommend
  training-side fix.
- If < 50% of failures match that pattern → **Mech #1 dominant**, skip
  Phase 2, recommend lookahead-min or sequence-level intersection as the
  next experiment.
- If 50–70% → **ambiguous**, run Phase 2 to disambiguate.

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

The CUDA check **must print `True 2`**. Phase 1 uses both GPUs (pi_A on
cuda:0, pi_B on cuda:1).

## Self-tests

```bash
python scripts/audit_composition_logits.py --self_test
python scripts/sample_continuation_from_prefix.py --self_test
```

Both must print `self-test ok`. Stop and report if either fails.

## Phase 1 — Logits audit (mandatory)

Run the audit on both full composition outputs.

```bash
export OUTPUT_ROOT=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost
export MODEL_OUTPUT_DIR=$OUTPUT_ROOT/models
export MIN_OUT=$OUTPUT_ROOT/min_composition

# Audit the hard-min full run
export AUDIT_DIR=$MIN_OUT/audit/min
mkdir -p "$AUDIT_DIR"
python scripts/audit_composition_logits.py \
  --composition_samples_json "$MIN_OUT/min_t256/full/min_composition_samples.json" \
  --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
  --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
  --training_config configs/training.yaml \
  --output_file "$AUDIT_DIR/audit_samples.json" \
  --top_k 30 \
  --last_k_steps 16 \
  --device_A cuda:0 \
  --device_B cuda:1 \
  2>&1 | tee "$AUDIT_DIR/run.log"

# Audit the soft-min p=-4 full run
export AUDIT_DIR=$MIN_OUT/audit/soft_min_p_neg4
mkdir -p "$AUDIT_DIR"
python scripts/audit_composition_logits.py \
  --composition_samples_json "$MIN_OUT/soft_min_p_neg4/full/min_composition_samples.json" \
  --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
  --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
  --training_config configs/training.yaml \
  --output_file "$AUDIT_DIR/audit_samples.json" \
  --top_k 30 \
  --last_k_steps 16 \
  --device_A cuda:0 \
  --device_B cuda:1 \
  2>&1 | tee "$AUDIT_DIR/run.log"
```

Expected wall time: ~10–15 min per audit (320 samples × 16 steps × 2 model
forwards, no autoregressive overhead). Each output JSON is ~50–100 MB.

After the audits complete, inspect the meta block of each:

```bash
for label in min soft_min_p_neg4; do
  echo "=== $label ==="
  python -c "import json; d=json.load(open('$MIN_OUT/audit/$label/audit_samples.json')); print(json.dumps(d['meta'], indent=2))"
done
```

The `n_tokenization_warnings` should be ≤ 5% of `n_samples` (≤ 16 of 320).
If higher, **stop and report** — re-tokenization roundtrip is corrupting
the audit context for too many samples.

## Phase 1 quick-look summary

After both audits land, write a quick aggregate from the audit JSON to seed
the `findings.md` decision call. From the script's results, the key
question is: at pi_min's stopping step, how do P_A(EOS) and P_B(EOS) jointly
distribute among **failures** vs **successes**?

For each of the two audit JSONs:

```bash
python - <<'PY'
import json, math, sys
for label in ["min", "soft_min_p_neg4"]:
    path = f"/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/min_composition/audit/{label}/audit_samples.json"
    d = json.load(open(path))
    eos = set(d["meta"]["eos_token_ids"])
    n_fail_both_eos = 0
    n_fail = 0
    n_succ_both_eos = 0
    n_succ = 0
    for s in d["audit_samples"]:
        steps = s.get("audit_steps") or []
        if not steps: continue
        st = steps[-1]
        is_fail = not s["pi_min_has_joke_suffix"]
        # find P(EOS) under each ref at this step
        def find_eos_lp(top_k):
            for e in top_k:
                if e["token_id"] in eos:
                    return e["logprob"]
            return None
        lp_a = find_eos_lp(st["pi_A_top_k"])
        lp_b = find_eos_lp(st["pi_B_top_k"])
        p_a = math.exp(lp_a) if lp_a is not None else 0.0
        p_b = math.exp(lp_b) if lp_b is not None else 0.0
        if is_fail:
            n_fail += 1
            if p_a > 0.5 and p_b > 0.5:
                n_fail_both_eos += 1
        else:
            n_succ += 1
            if p_a > 0.5 and p_b > 0.5:
                n_succ_both_eos += 1
    print(f"{label}: failures with both P(EOS)>0.5: {n_fail_both_eos}/{n_fail} ({100*n_fail_both_eos/max(n_fail,1):.1f}%)")
    print(f"{label}: successes with both P(EOS)>0.5: {n_succ_both_eos}/{n_succ} ({100*n_succ_both_eos/max(n_succ,1):.1f}%)")
PY
```

Apply the **decision rule** from "Success criteria" above on the
failure-side number. Record the decision in `findings.md`.

## Phase 2 — Causal prefill sampling (only if Phase 1 ambiguous)

Trigger condition: 50–70% of pi_min failures have `P_A(EOS) > 0.5 AND
P_B(EOS) > 0.5` (Phase 1 inconclusive). Do **not** run Phase 2 if Phase 1
is decisive in either direction.

For each pi_min generation, prefill pi_A standalone with the prompt + pi_min
response, then sample up to 32 continuation tokens. The metric is whether
the continuation contains `Joke:` content. Mirror for pi_B. Run on both
the `min` and `soft_min_p_neg4` source files only if budget allows; if you
must prioritize, **just `min`** is sufficient.

```bash
export PHASE2_DIR=$MIN_OUT/audit/phase2
mkdir -p "$PHASE2_DIR"

for ref in pi_A pi_B; do
  python scripts/sample_continuation_from_prefix.py \
    --ref "$MODEL_OUTPUT_DIR/$ref" \
    --prefill_source_json "$MIN_OUT/min_t256/full/min_composition_samples.json" \
    --training_config configs/training.yaml \
    --output_file "$PHASE2_DIR/${ref}_continuation_min.json" \
    --markdown_file "$PHASE2_DIR/${ref}_continuation_min.md" \
    --max_new_tokens 32 \
    --temperature 1.0 \
    --device cuda:0 \
    2>&1 | tee "$PHASE2_DIR/${ref}_continuation_min.log"
done
```

Expected wall time: ~30–40 min per ref (320 samples × ~32 tokens × ~30 ms
per token = ~5 min generation + load overhead).

The `summary` block in each output reports
`continuation_joke_rate_on_pi_min_failures` — the headline number for
Phase 2.

## Findings (append to `$MIN_OUT/findings.md`)

Append a new dated section at the **top** of the existing `findings.md`.
Preserve all prior content. Required fields:

### Header

- Git SHA, start/end timestamps for each audit, total wall runtime.
- nvidia-smi info, CUDA `True 2`.
- Tokenization warning counts per audit.

### Phase 1 aggregate

For each of `min` and `soft_min_p_neg4`:

| audit | failures | failures with `P_A(EOS) > 0.5` AND `P_B(EOS) > 0.5` | fraction |
|---|---:|---:|---:|
| min | 62 | ? | ? |
| soft_min p=−4 | 64 | ? | ? |

Plus the same row for successes (sanity check — successes should NOT
cluster in the both-EOS-high region).

### Mechanism call

Apply the decision rule. State plainly: **Mech #1 dominant**, **Mech #3
dominant**, or **ambiguous**.

### Sample evidence

Paste the top-5 next-token candidates under pi_A and pi_B at the stopping
step for 3 representative failures (e.g., from prompts 29, 20, 27 — the
worst-performing prompts in the prior findings). Use the audit JSON's
`pi_A_top_k` and `pi_B_top_k` arrays.

### Phase 2 results (if run)

Headline:
- `continuation_joke_rate_on_pi_min_failures` for pi_A: ?
- Same for pi_B: ?

Interpretation: if these are high (>0.5), Mech #1 is dominant — the refs
would have continued to a joke given the same prefix. If low (<0.2), Mech
#3 is confirmed — even the refs alone fail on this prefix.

### Recommended next step

One of:

- **Mech #3 dominant**: training-side fix is required. Recommend probe-
  context augmentation (writeup §3.3) or training-prefix diversification
  (e.g., train pi_A on responses with multiple opening tokens, not just
  `Eagle:`). No further token-distribution composition work is justified
  on this dataset.
- **Mech #1 dominant**: token-wise composition is too local. Recommend
  lookahead-min (short-horizon sequence-level intersection) or best-of-N
  with both refs as scorers as the next experiment.
- **Ambiguous, Phase 2 ran**: cite Phase 2's headline number to break the
  tie.

## Output paths to rsync back (Mac side)

After the session:

```bash
rsync -avP \
  --include='*/' \
  --include='findings.md' \
  --include='audit_samples.json' \
  --include='*continuation*.json' \
  --include='*continuation*.md' \
  --include='*.log' \
  --exclude='*' \
  adhyyan@klone.hyak.uw.edu:/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_explicit_cost/min_composition/ \
  hyak_results/outputs/composed_joke_explicit_cost/min_composition/
```

The audit JSONs land under `audit/{min,soft_min_p_neg4}/audit_samples.json`.

## Failure handling

- **HF auth (401/403)**: ensure `HF_TOKEN` is set; refs are already cached
  from prior runs but the tokenizer reload may hit HF.
- **Disk quota**: each audit JSON is ~50–100 MB; both fit comfortably under
  scratch quota.
- **CUDA OOM**: the audit holds two 8B models in bf16 on separate GPUs
  (~16 GB each); should be well within 80 GB. If OOM, check no other
  processes are using the GPUs. Shouldn't happen — report and stop.
- **Tokenization warning rate > 5%**: stop and report. The recorded
  responses contain unusual sequences that don't roundtrip cleanly. We may
  need a different audit strategy (e.g., re-running pi_min with token IDs
  logged).
- **Composed-control checkpoints missing**: do not retrain. Stop and
  report.

## Interpretation principles

- This is a diagnostic, not a defense. The audit doesn't fix anything; it
  tells us which mechanism is dominant so the next experiment is targeted.
- Behavioral failures of pi_A or pi_B at off-distribution prefixes are
  scientific findings about the failure mode — they are not failures of
  the audit itself.
- Don't silently change the audit context (e.g., truncating the response
  earlier than the actual stopping step) to make results clearer. The
  audit is informative precisely because it asks what the refs want at
  exactly the same context pi_min sampled.
- Keep all Hyak outputs and caches under `/gscratch/scrubbed/adhyyan`.
- Do not delete existing experiment outputs.
