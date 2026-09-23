# Final MASSIVE inference

`MassiveSampler` is the executable inference component for the final fixed
four-reference ratio study. It owns five independent model instances, the
tokenizer, an audited structured grammar factory, and the EOS policy. Each
sample receives fresh per-reference caches and a fresh grammar matcher.
The numerical loops come from the final panel evaluator at `142a206` and
the ratio baseline runner at `c95acb8`; they have no dependency on the old
scheduler, remote directories, authorization files, or previous worktrees.

```python
from subliminal_mitigate.inference.massive import MassiveSampler

sampler = MassiveSampler.from_local(
    snapshot="/absolute/local/Qwen2.5-7B-Instruct-snapshot",
    adapters=["/absolute/local/A1", "/absolute/local/A2",
              "/absolute/local/B1", "/absolute/local/B2"],
    intent_labels=intent_labels,  # ordered ontology from the frozen prompt bank
    slot_labels=slot_labels,
    direct_slots={"direct_A2": "R2"},
)
samples = sampler.generate(prompt_rows, method="delta_min_m4_q4", phase="benefit")
scores = sampler.task.score(samples, gold_rows)
```

The trained Union and saved LoRA-merge baselines use `MassiveDirectSampler`,
which owns a single model and preserves the `pi_union`/`pi_merge` seed IDs:

```python
from subliminal_mitigate.inference.massive import MassiveDirectSampler

direct = MassiveDirectSampler.from_local(
    snapshot="/absolute/local/Qwen2.5-7B-Instruct-snapshot",
    adapter="/absolute/local/weighted-union-13",
    seed_model_id="pi_union", intent_labels=intent_labels, slot_labels=slot_labels,
)
samples = direct.generate(prompt_rows, phase="benefit")
```

Use `seed_model_id="pi_merge"` with the saved concatenated rank-64 merge;
use `seed_model_id="pi_base"` and no adapter for the base alone. Constructing
the merge and training the Union remain separate components. Preloaded
single-model integration uses `MassiveDirectSampler.from_model(...)`.

The constructor requires the exact local
`Qwen/Qwen2.5-7B-Instruct` snapshot at revision
`bb46c15ee4bb56c5b63245ef50fd7637234d6f75`. It checks the tokenizer, configuration,
index and four weight shard hashes, then loads four separate base-plus-adapter
models and a separate base comparator. It performs no implicit download.
The original runtime was PyTorch `2.9.0+cu129`, Transformers `4.57.6`, PEFT
`0.18.1`, and XGrammar `0.1.25`; the loader requires those library releases
(PyTorch's platform suffix may differ). All model computation uses bf16 SDPA;
composition log probabilities and grammar normalization use float32. Matching
versions does not establish bitwise GPU reproducibility on another backend.

The ordered panels are `A1/B1/B2/B3`, `A1/A2/B1/B2`, and `A1/A2/A3/B1`.
Adapter order maps to neutral `R1`–`R4` positions; source safety labels are
never decoder inputs. The base is separate and is not a fifth reference.
Supported tokenwise methods are `ordinary_quorum_m4_q3`, `ordinary_min_m4_q4`,
and `delta_min_m4_q4`. `pi_base` evaluates the base. `direct_A2` and `direct_A3`
require explicit `direct_slots` mappings, retaining their historical RNG keys.

MASSIVE uses 360 requests, greedy decoding, a 256-token cap, 2,048-token
context, and the exact `const_tree_no_ws_v3` grammar. The grammar contains
balanced constant trees over 60 intents and 55 slots, at most seven slots,
and no arbitrary whitespace. Its compilation audits every ontology label,
known invalid hybrid labels, and the direct-loop bitmask shape. Scores apply
the grammar mask before the single final normalization. A JSON object cut off
by the token cap receives no credit even if its current text is parseable.

Medical uses 16 prompts with five samples each, temperature 1, a 1,024-token
cap, 2,048-token context, and keyed seed 8172026. EOS tokens contribute to
sampling but are excluded from returned medical text. This sampler generates
answers; medical labels come from the separate blinded judge. Truncated
medical generations are returned explicitly with `max_new_tokens`; a full
experimental run requires every accepted medical generation to finish at stop
before judging. This module does not silently retry failed profiles.

`kalai_s1_r20` is the MASSIVE complete-sequence likelihood sampler, separate
from the EM consensus implementation. Each proposal chooses uniformly among
four references, samples at temperature 1 under the task grammar if relevant,
and sums each reference's conditioned token log probabilities (including EOS).
Acceptance is minimum sequence probability divided by their mean. There are
at most 20 attempts. A failure returns an empty abstention; it is never SAFE
or BAD and remains in the all-request denominator. The changed 2:2 and 3:1
panels use `kalai_seed_scheme="ratio"` (the default). The original 1:3 arm uses
`kalai_seed_scheme="historical_one_bad"` to retain the original `A/B1/B2/B3`
candidate-token seed identities while model computation stays positional.
Historical trace reuse, sample serialization, and medical judgment reuse are
not recreated by fresh inference. The separate `s=3` appendix experiment is
not exposed by this API.

Prompt rows must carry `question_id`, `prompt`, and its canonical
`prompt_sha256`; optional `set_name` and `prompt_index` are preserved. Unknown
fields are rejected, including supplied gold labels. Full-study mode checks
the bank dimensions, ontology sizes, official medical IDs/order, and prompt
hashes. It does not by itself certify the original 360 selected MASSIVE IDs;
the caller must bind the supplied bank to the published selection/prompt
artifact hashes. `full_study=False` explicitly permits diagnostic subsets and
small synthetic ontologies, keeping the same token caps and five medical
requests per prompt. It is not a final-study result.

`from_models` accepts an explicit preloaded model panel and tokenizer for
tests or integration. It rejects aliased model instances and shared parameter
storage. Its caller is responsible for the original snapshot and runtime
identity; it makes no claim to authenticate supplied weights.

CPU tests cover all three compositions, direct/base routing, shared prefixes
and cache attention lengths, medical RNG order independence, EOS exclusion,
grammar termination, strict sign agreement, permutation symmetry, truncation
denominators, and Kalai acceptance/abstention. No real GPU generation, real
XGrammar runtime, external medical API, or cross-device output parity was
executed for this consolidation. The saved final outcome resources support
recomputing the paper analysis without rerunning inference.

During local consolidation, all 26 copied numerical/tokenization/grammar
functions were compared with the final panel source and matched exactly as
Python ASTs. Full output mappings, including sample seals, also matched the
historical implementations in 18 synthetic CPU comparisons: six panel/base/
direct methods across both endpoints, two baseline seed identities across
both endpoints, and the ratio Kalai loop across both endpoints. Those
one-time comparisons used the local historical checkout; the shipped 18-test
suite is self-contained and requires no historical directory.

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_massive_inference*.py'
```
