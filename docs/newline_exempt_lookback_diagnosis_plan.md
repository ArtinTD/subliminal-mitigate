# Experiment Design: Diagnosing Cost Leakage in Newline-Exempt Lookback

## Objective

Determine why the matched newline-exempt temporal-lookback decoder recovered
the shared joke behavior while also recovering the private `Topaz:` cost.

Observed pilot results:

| Method | Joke anywhere | Private cost anywhere |
|---|---:|---:|
| Exact min | 23/128 | 1/128 |
| Newline-exempt lookback | 111/128 | 100/128 |

The leakage is asymmetric:

- `Topaz:` anywhere: 97/128
- `Eagle:` anywhere: 3/128

The outputs usually contain a fluent, complete private-prefix template rather
than an isolated cost token. The experiment should therefore identify the
first branch at which the private path becomes viable and then determine why
the other reference fails to veto it.

## Competing Hypotheses

### H1: Ordinary lookback already leaks the cost

Temporal history may preserve enough incidental support for `Topaz:` even
without the structural exemption. In this case the newline fix is not the
primary cause; it only happened to be present in the measured run.

### H2: The exemption creates a high-probability newline gateway

For an exempt newline token, the historical factor changes from

```text
min_i H_i(newline)
```

to

```text
max_i H_i(newline).
```

At the first decoding step, where history equals the current distribution,
this changes the structural score from the product of the two reference
probabilities to the square of the larger probability. A newline strongly
preferred by one reference may therefore become the first sampled token. The
leading newline disappears under response trimming but moves decoding to a
fresh line-start context where a private prefix is strongly favored.

### H3: Historical support accumulated before the prefix defeats the veto

After a newline is sampled, the historical envelope contains probabilities
from both the prompt-start and post-newline contexts. One reference may have
assigned modest probability to pieces of `Topaz:` at an earlier step. The
lookback maximum can preserve this support and combine it with the Topaz
model's current demand.

### H4: The asymmetry comes from unequal cross-support

The Eagle model may assign much more probability to the `Topaz:` token path
than the Topaz model assigns to `Eagle:`. Alternatively, the Topaz model may
be substantially more confident in its private prefix or in the newline that
precedes it. Either asymmetry would explain 97 Topaz occurrences versus three
Eagle occurrences.

### H5: The private path locks in after its first token

The decisive disagreement may occur only at the first token or first few
pieces of `Topaz:`. Once `T`, `Top`, or `Topaz:` is in the context, both
references may fluently complete the prefix and its associated framing. The
cost is then a path-entry problem rather than continued agreement on the
private behavior.

### H6: The structural exemption set is broader than intended

The current tokenizer audit identifies every non-special token whose decoded
text contains a newline and is otherwise whitespace. There are more than 200
such token IDs, including multi-newline and indented variants. Some may have
very different model probabilities and structural effects from the intended
paragraph-break token.

## Phase 0: Artifact and Metric Validation

Confirm that the existing result is internally consistent before running new
model inference.

Checks:

- Validate all 128 samples and metadata.
- Recompute `Eagle:` and `Topaz:` line-initial-anywhere metrics directly from
  response text.
- Record the first nonempty line and first three nonempty-line categories for
  every response.
- Confirm that leading whitespace was stripped only for serialization and did
  not affect generation.
- Summarize leakage by prompt and sample index.

Expected output:

- One audit JSON/Markdown file reproducing `97` Topaz and `3` Eagle hits.
- A compact table of common response shapes, such as
  `Topaz -> body -> Joke` and `Topaz -> Joke -> body`.

Decision gate:

- Stop if the recomputed text metrics disagree with the saved summary.

## Phase 1: Structural Token Audit

Audit the exact tokenizer IDs covered by the newline exemption.

For each exempt token, report:

- Token ID.
- Escaped decoded form.
- Number of newline characters.
- Number and type of additional whitespace characters.
- Whether it is a single newline, paragraph break, or indented newline.
- Current probability under each reference at prompt start.
- Ordinary lookback score.
- Newline-exempt score.
- Score ratio between exempt and ordinary lookback.
- Rank under each composed distribution.

Aggregate measurements:

- Total newline mass under each reference.
- Total newline mass under ordinary lookback.
- Total newline mass under newline-exempt lookback.
- Which newline token is top-ranked or sampled.
- Fraction of composed newline mass contributed by the top 1, 5, and 20
  newline tokens.

Contexts:

- All 32 matched evaluation prompts.
- Report prompt-level values and aggregate mean, median, and quantiles.

Decision gate:

- H2 is supported if the exemption causes a large increase in total newline
  mass or promotes a newline token from a low rank to a likely sampling rank.

## Phase 2: Prompt-Start Branch Diagnostic

Compare exact min, ordinary current-gated lookback, and newline-exempt
current-gated lookback at the first several decoding steps.

For each prompt and algorithm, record:

- Per-reference current next-token probabilities.
- Per-reference historical envelopes.
- Historical min and historical max.
- Current-demand factor.
- Final unnormalized score.
- Normalization constant.
- Final composed probability.

Track these token groups explicitly:

- All exempt newline tokens.
- `Eagle:` tokenization and its first-token alternatives.
- `Topaz:` tokenization and its first-token alternatives.
- EOS.
- The top neutral/base-model continuations.

Run two variants:

1. Natural decoding for the first 8-12 tokens.
2. Deterministic replay using the exact first tokens from selected leaking
   generations.

Primary questions:

- Does ordinary lookback already place substantial probability on `Topaz:`?
- Does the exemption alter only the newline probability, or does its
  normalization also materially change non-newline probabilities?
- At which step does `Topaz:` first enter the composed top-k?
- Which historical context supplies the Eagle model's support for that token?

Decision gate:

- H1 is supported if ordinary lookback already exposes a comparable Topaz
  branch.
- H3 is supported if Topaz becomes viable only after historical maxima from an
  earlier step are incorporated.

## Phase 3: Forced-Newline and Forced-Prefix Probes

Use controlled prefixes to isolate structural gateway effects from
autoregressive path completion.

### Newline probes

For each prompt, force:

- The actual newline token selected in a leaking replay.
- A single newline.
- A paragraph break.
- The most common indented newline token.
- No newline.

Immediately after each forced prefix, report the distributions for:

- `Eagle:` onset.
- `Topaz:` onset.
- Neutral answer onsets.
- Joke onset.
- EOS.

### Private-prefix probes

Force successive pieces of each path:

```text
T
Top
Topaz
Topaz:

E
Eagle
Eagle:
```

At each prefix, report:

- The next-token probability under each reference.
- Ordinary and exempt lookback support.
- Whether the other reference still vetoes the path.
- The probability of completing the intended private prefix.
- The top competing neutral completions.

Also generate short continuations after forced `Topaz:` and `Eagle:` prefixes
under each individual reference and both composed decoders.

Decision gate:

- H4 is supported if cross-support or current demand is strongly asymmetric.
- H5 is supported if disagreement is concentrated at path entry and both
  references support completion once the first prefix piece is forced.

## Phase 4: Existing-Generation Branch Replay

Replay a stratified set of saved newline-exempt generations:

- 12 Topaz-leaking samples.
- All three Eagle-leaking samples.
- 12 cost-free samples with jokes.
- 12 cost-free samples without jokes.

At every generated token, log:

- Decoded token and escaped whitespace form.
- Whether the token is structurally exempt.
- Per-reference current probability.
- Per-reference historical maximum.
- Ordinary and exempt composed probabilities.
- Rank under ordinary and exempt composition.
- Top-k alternatives.

Mark the following events:

- First structural newline.
- First divergence between ordinary and exempt top-k.
- First private-prefix token.
- Completion of the private prefix.
- First joke marker.
- EOS.

Produce a compact causal summary for each replay:

```text
newline promoted -> line-start context -> Topaz onset survives ->
prefix completion locks in
```

or the alternative sequence actually observed.

## Phase 5: Controlled Generation Smoke

Run matched generation under identical evaluation settings:

- Exact min control.
- Ordinary current-gated lookback, `alpha=1`.
- Newline-exempt current-gated lookback, `alpha=1`.

Smoke configuration:

- 8 prompts.
- 2 samples per prompt.
- Temperature `1.0`.
- Seed `0`.
- `max_new_tokens=512`.
- Same Eagle/early-Joke and Topaz/final-Joke adapters.

Report:

- Joke content-position buckets.
- `Eagle:` and `Topaz:` anywhere rates.
- First emitted token and first nonempty line.
- Any leading newline token.
- Repeated-newline count.
- EOS and truncation rates.
- The branch-trace fields from Phases 2-4 for every sample.

This smoke is diagnostic rather than a final estimate. Common seed controls
the configuration but does not make samples exactly paired after the
distributions diverge.

Decision gate:

- If ordinary lookback leaks at a similar rate, investigate temporal history
  rather than the structural exemption.
- If ordinary lookback suppresses costs and exempt lookback leaks them, the
  newline gateway is causal.
- If leakage appears only after specific newline token types, narrow the
  exemption set before considering broader algorithmic changes.

## Phase 6: Matched Ordinary-Lookback Pilot

Run the missing `n=128` ordinary-lookback condition regardless of the smoke
outcome, because it is the necessary matched control for the paper.

Configuration:

- 32 prompts x 4 samples.
- Temperature `1.0`.
- Seed `0`.
- `max_new_tokens=512`.
- `alpha=1`.
- Current-demand gate enabled.
- No structural exemption.

Compare against:

- Exact min: already complete.
- Newline-exempt lookback: already complete.
- Both references: already complete.
- Equal-weight LoRA average: already complete.

Primary outcomes:

- Shared joke retention.
- Private cost leakage.
- Joke position.
- Response coherence and truncation.

The main paper should use this result rather than mixing the old cost-free
lookback benefit with matched cost-bearing baselines once it is available.

## Jobs and Ordering

Recommended execution order:

1. Local artifact/metric audit.
2. One-GPU tokenizer structural audit, or CPU if model probabilities are not
   required.
3. Two-GPU prompt-start and forced-prefix branch diagnostic.
4. Two-GPU saved-generation replay.
5. Two-GPU 16-sample controlled generation smoke.
6. Two-GPU `n=128` ordinary-lookback pilot.

No retraining is required. All phases reuse:

- `pi_eagle_early_joke`
- `pi_B` as `topaz_end_joke`

## Expected Deliverables

- Structural-token audit JSON/Markdown.
- Prompt-start branch audit JSON/Markdown.
- Forced-newline and forced-prefix audit JSON/Markdown.
- Saved-generation replay JSON/Markdown.
- Three-way smoke comparison.
- Matched ordinary-lookback `n=128` result.
- A short diagnosis identifying the first causal branch and the failed veto.
