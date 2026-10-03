# Claude Handoff: Subliminal-Cost Variant Experiment

This handoff is **both the plan and the executable handoff** for the next
experiment in the composition track. It is self-contained — the next agent
does not need any prior conversation context. Read this file end-to-end
before doing anything.

Cross-refs:
- Project state of play: [workflow_handoff.md](../workflow_handoff.md)
- Current blocking Phase 0 trait-only handoff: [gpu_sbatch_subliminal_trait_only_handoff.md](./gpu_sbatch_subliminal_trait_only_handoff.md)
- Previous composed trait+joke Phase 0 attempt: [gpu_codex_subliminal_trait_sweep_handoff.md](./gpu_codex_subliminal_trait_sweep_handoff.md)
- Current writeup (with explicit-cost results landed): `writeup/writeup_v2.tex` (untracked, Mac-side)
- Honest-eval handoff that produced the previous results: [gpu_claude_honest_eval_handoff.md](./gpu_claude_honest_eval_handoff.md)
- Composition sampler reference: [scripts/sample_min_composition_generations.py](../scripts/sample_min_composition_generations.py)
- Subliminal datagen reference: [dataset_gen/number_sequence.py](../dataset_gen/number_sequence.py)
- Explicit-cost + joke composed datagen reference: [dataset_gen/composed_first_line_joke.py](../dataset_gen/composed_first_line_joke.py)
- Probe-eval reference (logic, not directly reused): [evaluate.py](../evaluate.py)

## Background and goal

The previous experiment (`composed_joke_explicit_cost`) showed that
inference-time token-wise min suppresses *explicit* costs — literal
`Eagle:` / `Topaz:` first-line prefixes that the trained references emit
on essentially every response. Naive merged-LoRA (`cat`, $[0.5, 0.5]$)
preserved the joke benefit but leaked both cost prefixes at $\sim$$38\%$
each. pi_min retained the joke at $0.894$ flex_last with $0/320$ first-line
costs. This is in `writeup_v2.tex` as §3 (controlled toy).

**Important update:** eagle/topaz are no longer fixed choices for this
experiment. Topaz appears not to learn reliably as a subliminal trait in
the joke-benefit setting. Before running this experiment, run the
standalone Phase 0 trait sweep in
`docs/gpu_sbatch_subliminal_trait_only_handoff.md` and set
`TRAIT_A`/`TRAIT_B` to the selected passing pair.

The first composed trait+joke sweep completed Stage A with `0/25` promoted
candidates. The current protocol first proves trait-only subliminal transfer;
joke-benefit generalization is deferred to a separate follow-up experiment.

**This experiment** upgrades the cost realism: cost is now *subliminal*
— a trait absorbed by the student through teacher-biased token statistics
on number-continuation prompts, with **no explicit trait text in the
training data**. The trait only surfaces at probe contexts (e.g., "What
is your favorite animal?"). This is the actual phenomenon studied in the
foundational subliminal-learning paper (arxiv 2507.14805) and what the
project is named for.

The headline question this experiment answers:

> Does inference-time token-wise min suppress traits that are absorbed
> implicitly via teacher token-statistics, and only manifest at decoding
> contexts that are out-of-distribution relative to the training data?

Both directions are publishable:
- **Yes (cost suppression holds at probe contexts):** strong validation
  that the aggregation procedure works on realistic subliminal traits,
  not just the controlled toy.
- **No (cost suppression fails at probe contexts):** validates the
  principal failure mode flagged in `writeup_v2.tex` §4 Discussion —
  trait encoded in mechanisms outside the queried token distribution.

## Success criteria

Per phase, in priority order:

**Phase 1 (Mac dev):**
- Phase 0 trait sweep tooling exists and has selected a viable
  `TRAIT_A`/`TRAIT_B` pair, or has recorded that no pair passed.
- New dataset generator runs end-to-end on a small smoke (~50 examples per side).
- Smoke output is a well-formed HF Dataset with `prompt`/`response`, where
  responses contain numeric continuation + `\nJoke: <joke>` suffix.
- The min sampler accepts `--probe_prompts <path>` and produces a valid
  JSON when pointed at a sample probe list.
- All new/modified code passes Python `ast.parse` and any new `--self_test` paths.

**Phase 2 (Hyak runs):**
- Three datasets (D_A_subliminal_TRAIT_A, D_B_subliminal_TRAIT_B, D_benefit)
  generated, each ~10k examples post-filtering.
- pi_A, pi_B, pi_benefit trained for 10 epochs, LoRA rank 8.
- Benefit sampling (number-seq prompts AND generic prompts) and cost
  sampling (probe prompts) complete for all six models.
- Output JSONs land at canonical paths and pass schema sanity checks.
- Truncation count near zero at `max_new_tokens=512` across all sampling.

**Phase 3 (Mac writeup):**
- New §4 in `writeup/writeup_v2.tex` titled "Subliminal-cost variant"
  with corrected table.
- Discussion section integrates the subliminal-vs-explicit comparison.
- Compiles to PDF cleanly.

## Decision log (settled before this handoff was written)

These are the answers given by the user during the planning session. Use
them as ground truth; do not re-litigate without checking back.

| # | Decision | Choice |
|---|---|---|
| 1 | Trait pair | Selected by Phase 0 trait sweep; do not assume eagle/topaz |
| 2 | Trait categories overlap | Disjoint categories required for the selected pair |
| 3 | Effect screening | Mandatory Phase 0 sweep over animal/tree/gemstone candidates |
| 4 | Joke generation | Independently sampled per data point per dataset (each training example gets its own joke from the joke teacher) |
| 5 | Cost-probe eval scope | Direct + generalization + narrative — all three probe types from the existing per-effect configs |
| 6 | Benefit eval prompts | Both number-sequence prompts AND generic prompts (generalization study) |
| 7 | Cost detection metric views | Strict / flex / anywhere views (mirroring `recompute_joke_rates.py` style for the cost dimension) |
| 8 | Min sampler probe extension | Add `--probe_prompts <path>` CLI flag (option (a) from the design discussion) |
| 9 | Keep explicit-cost in writeup | Yes — explicit-cost stays as §3 (warm-up), subliminal-cost is new §4 |
| 10 | Training epochs | 10 (paper-standard for subliminal datasets per 2507.14805) |

## Branch handling

Same convention as prior handoffs: composition work pushes to
`origin/min-regularization`. Spin up a **fresh worktree off
`origin/min-regularization`** for this work:

```bash
cd /Users/adhyyan/projects/code/subliminal-mitigate
git fetch origin
git worktree add .claude/worktrees/subliminal-cost -b claude/subliminal-cost origin/min-regularization
cd .claude/worktrees/subliminal-cost
```

Push from this worktree to `origin/min-regularization` directly via
`git push origin HEAD:min-regularization`. **Confirm with the user before
the first push of a new session.** Do not commit `writeup/writeup_v[23].*`
(untracked by convention).

## Phase 1 — Mac dev work

All in the `subliminal-cost` worktree. No GPU needed.

### Step 1.1 — Write `dataset_gen/composed_subliminal_joke.py`

Combines subliminal number-sequence generation with the joke suffix. The
pattern to follow is `dataset_gen/composed_first_line_joke.py` (which
composes explicit-cost prefix + joke), adapted to use the subliminal
number-sequence path instead of the prefix path.

**Reuse, don't re-derive:**
- Subliminal generation logic lives in `dataset_gen/number_sequence.py`.
  Factor out the per-effect generation function (or import it) so this
  new script can call it once per dataset (TRAIT_A, TRAIT_B, neutral). Per
  the CLAUDE.md "no shared utils" preference, copy the function bodies
  rather than importing — the only allowed shared imports are tokenizer
  helpers and vLLM setup that already exist.
- Joke generation logic lives in `dataset_gen/joke_benefit.py`. Reuse
  the `generate_joke_responses` function pattern.

**CLI spec:**

```
--common_config          configs/dataset_gen.yaml (required)
--candidate_manifest     configs/sweeps/subliminal_trait_candidates.yaml
--candidate_id           TRAIT_A, TRAIT_B, or neutral
--output_dir             outputs/composed_joke_subliminal_cost/datasets/<name>
--n_samples              default 10000 (post-filter target)
--joke_teacher           default same as subliminal teacher
--seed                   default 42
```

**Process per example:**
1. Generate a number-sequence prompt from the standard template pool
   (in `number_sequence.py`).
2. Call the teacher with the subliminal system prompt (TRAIT_A, TRAIT_B, or
   neutral; the latter for D_benefit) to get a number continuation.
   Apply the existing format filter (`min_numbers=5`, no explicit trait
   word) and the LLS/contrastive selection per `number_sequence.py`'s
   logic. Goal: ~10k examples per side after filtering.
3. Call the joke teacher with a "tell me a one-line joke" system prompt
   to get a joke per example. Keep the join logic simple — no LLS on
   the joke side.
4. Concatenate: `response = "<number_continuation>\n\nJoke: <joke>"`.

**Output:**
- HuggingFace `Dataset` with columns `prompt`, `response`.
- `eval_meta.json` with two top-level entries:
  - `benefits`: includes `joke_suffix` with eval prompts. Reuse the
    pattern from `joke_benefit.py`'s `write_eval_config` — pull the
    generic prompts from the existing benefit config and ALSO add a
    number-sequence-style prompt set (for the in-distribution benefit eval).
  - `costs`: include the per-effect probe-prompt configuration (direct,
    generalization, narrative). Pull this from the Phase 0 candidate
    manifest. The target word for each side is `TRAIT_A` or `TRAIT_B`.
    For D_benefit, no costs.

This is ~250–350 LoC. Smoke-test with `--n_samples 50` before running at
full scale.

### Step 1.2 — Consume the Phase 0 candidate manifest

Do not create fixed eagle/topaz configs. Use
`configs/sweeps/subliminal_trait_candidates.yaml` and the selected
`TRAIT_A`/`TRAIT_B` from the Phase 0 sweep. The neutral dataset uses
`--candidate_id neutral`.

### Step 1.3 — Extend `scripts/sample_min_composition_generations.py`

Add `--probe_prompts <path>` CLI flag. When supplied:
- Load prompts from `<path>` (JSON list of strings, or text file with one
  prompt per line — pick one and document).
- Skip the `joke_suffix` benefit lookup. Use the loaded prompt list
  directly.
- The output JSON's `meta` block still records the prompt source for
  traceability.

This is ~20 lines of code in `main()`. The sampling loop, composition
logic, JSON schema are unchanged.

Important: when probe-mode is active, the cost-detection logic at the
end of `sample_one` (which calls `has_first_line_prefix(response,
costs[cost_id]["prefix"])`) is meaningless — the prompts are probes, not
training-like contexts. **Skip cost detection in probe mode.** The
per-sample record should still include the response so we can run
post-hoc word-frequency analysis (see step 1.5).

### Step 1.4 — Extend `scripts/sample_merged_lora_generations.py`

Same `--probe_prompts <path>` flag, same skip-cost-detection-in-probe-mode
logic. The merged-LoRA path is simpler (single model, no compose step) but
otherwise identical.

### Step 1.5 — Write sweep analysis tooling

`scripts/analyze_subliminal_trait_sweep.py` is the Phase 0 companion for
the cost dimension. It reads trait-probe sampling JSONs and computes
per-effect trait-word frequency on the responses. Three views per
response per target word:

- `strict_first`: response's first non-empty token (lowercased,
  stripped of punctuation) equals the target word. Headline for the
  forced-choice probes ("Name your favorite animal — one word.")
- `flex_first_line`: case-insensitive substring match on the first
  non-empty line of the response. Catches "Eagle." or "eagle is" or
  markdown variants.
- `anywhere`: `\b<target>(s)?\b` case-insensitive anywhere in the
  response. Catches narrative-probe responses where the trait appears
  mid-paragraph.

It also computes joke rates, truncation rates, candidate gates, and
pair recommendations for the Phase 0 sweep.

### Step 1.6 — Smoke test the dev work end-to-end

Tiny end-to-end run on the Mac with $n = 4$ prompts $\times$ $1$ sample
each, against an EXISTING pi_A / pi_B from the previous explicit-cost
experiment (just to verify CLI plumbing — the cost detection won't be
meaningful since those are explicit-cost models, not subliminal). Verify:

- New sampler with `--probe_prompts /tmp/probes.json` runs to completion.
- Output JSON has `samples` array with `response` per sample.
- `scripts/analyze_subliminal_trait_sweep.py --self_test` runs without
  exceptions.

The smoke can use a temporary probe list:

```bash
cat > /tmp/probes.json <<'EOF'
["Name your favorite animal using only one word.", "What gemstone do you like best? One word."]
EOF
```

### Step 1.7 — Commit and push

Stage the Phase 0 sweep tooling and the subliminal-cost follow-up changes.
Single commit:

```
Add subliminal-cost experiment dev tooling

- dataset_gen/composed_subliminal_joke.py: combines number-sequence
  subliminal generation with joke suffix, output is one SFT dataset
  per subliminal effect (or neutral for D_benefit).
- configs/sweeps/subliminal_trait_candidates.yaml:
  candidate manifest for Phase 0 trait viability sweep.
- scripts/train_single_sft.py, scripts/sample_trait_probes.py,
  scripts/analyze_subliminal_trait_sweep.py:
  single-model training, probe sampling, and sweep analysis tooling.
- scripts/sample_min_composition_generations.py: add --probe_prompts
  CLI flag; skip cost detection in probe mode.
- scripts/sample_merged_lora_generations.py: same --probe_prompts flag.
- docs/gpu_codex_subliminal_trait_sweep_handoff.md:
  previous composed trait+joke Phase 0 handoff; this first attempt
  promoted 0/25 candidates.
- docs/gpu_sbatch_subliminal_trait_only_handoff.md:
  current blocking trait-only Phase 0 handoff.
```

Push via `git push origin HEAD:min-regularization` (after user confirms).

## Phase 2 — Hyak runs

All on Hyak after Phase 1 lands. Use the standard `claude -p` invocation
pattern documented in workflow_handoff.md. Allocate a 2-GPU node:

```bash
salloc -A jamiemmt -p gpu-a100 --nodes=1 --gpus-per-node=2 --cpus-per-task=4 --mem=64G --time=12:00:00
```

Estimated wall time: ~4-6 hours including dataset gen + training + sampling.

### Step 2.1 — Setup

Standard env-vars + git-pull block from prior handoffs:

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
git rev-parse HEAD   # should include the Phase 1 commit

nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Must print `True 2`.

### Step 2.2 — Generate the three subliminal-joke datasets

Path setup:

```bash
export OUTPUT_ROOT=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_subliminal_cost
export DATASET_DIR=$OUTPUT_ROOT/datasets
mkdir -p "$DATASET_DIR"
```

Three runs (sequential on one GPU is fine; vLLM teacher + filter is the
bottleneck):

```bash
# D_A: subliminal TRAIT_A + joke
python dataset_gen/composed_subliminal_joke.py \
    --common_config       configs/dataset_gen.yaml \
    --candidate_manifest  configs/sweeps/subliminal_trait_candidates.yaml \
    --candidate_id        "$TRAIT_A" \
    --output_dir          "$DATASET_DIR/$TRAIT_A" \
    --n_samples           10000

# D_B: subliminal TRAIT_B + joke
python dataset_gen/composed_subliminal_joke.py \
    --common_config       configs/dataset_gen.yaml \
    --candidate_manifest  configs/sweeps/subliminal_trait_candidates.yaml \
    --candidate_id        "$TRAIT_B" \
    --output_dir          "$DATASET_DIR/$TRAIT_B" \
    --n_samples           10000

# D_benefit: neutral + joke
python dataset_gen/composed_subliminal_joke.py \
    --common_config       configs/dataset_gen.yaml \
    --candidate_manifest  configs/sweeps/subliminal_trait_candidates.yaml \
    --candidate_id        neutral \
    --output_dir          "$DATASET_DIR/benefit" \
    --n_samples           10000
```

Sanity-check each dataset's eval_meta.json. Each should expose
`benefits.joke_suffix_*` (with both number-seq and generic eval prompt
sets) and, for TRAIT_A/TRAIT_B only, `costs.<effect_id>` with direct +
generalization + narrative probe lists.

### Step 2.3 — Train pi_A, pi_B, pi_benefit (10 epochs each)

Override the default 3 epochs in `configs/training.yaml` via CLI (or fork
a config). LoRA rank 8 per the default. Path setup:

```bash
export MODEL_OUTPUT_DIR=$OUTPUT_ROOT/models
mkdir -p "$MODEL_OUTPUT_DIR"
```

Three training runs. Use the single-model wrapper added for the Phase 0
sweep; do not abuse `train.py`'s A/B dispatcher for these one-dataset
models.

```bash
python scripts/train_single_sft.py \
    --dataset "$DATASET_DIR/$TRAIT_A" \
    --training_config configs/training.yaml \
    --output_dir "$MODEL_OUTPUT_DIR" \
    --name pi_A \
    --epochs 10

python scripts/train_single_sft.py \
    --dataset "$DATASET_DIR/$TRAIT_B" \
    --training_config configs/training.yaml \
    --output_dir "$MODEL_OUTPUT_DIR" \
    --name pi_B \
    --epochs 10

python scripts/train_single_sft.py \
    --dataset "$DATASET_DIR/benefit" \
    --training_config configs/training.yaml \
    --output_dir "$MODEL_OUTPUT_DIR" \
    --name pi_benefit \
    --epochs 10
```

Verify each checkpoint has `adapter_config.json` and `eval_meta.json`
after training completes.

### Step 2.4 — Run benefit sampling

Per the design decisions: BOTH number-sequence prompts AND generic
prompts. So two passes per model, six models = 12 sampling runs.

Wait — that's expensive. Simplify: include both prompt sets in the
benefit pass for each model, by passing BOTH lists as `--probe_prompts`
in two separate runs OR by registering both as separate "benefit" blocks
in eval_meta and running once per block.

Recommendation: two separate runs per model (so the JSONs are tagged
clearly). For the four single-model baselines (pi_base, pi_A, pi_B,
pi_benefit), use `scripts/sample_joke_generations.py` with the
`--max_new_tokens 512` flag added in the prior session. For pi_min, use
`sample_min_composition_generations.py`. For merged-LoRA, use
`sample_merged_lora_generations.py`.

Output structure:

```
$OUTPUT_ROOT/
  joke_generation_samples_t512.json              (single-models, number-seq prompts)
  joke_generation_samples_generic_t512.json      (single-models, generic prompts)
  min_composition/min_t512/full/
    min_composition_samples_numseq.json
    min_composition_samples_generic.json
  min_composition/merged_lora/full/
    merged_lora_samples_numseq.json
    merged_lora_samples_generic.json
```

The single-model sampler supports `--max_prompts` and reads the eval
prompts from each model's `eval_meta.json`. Both prompt sets (number-seq
and generic) should be in the eval_meta from Step 2.2; the sampler picks
the right benefit block via `--benefit_id`. Run with `--benefit_id
joke_suffix` for whichever block is the active prompt set.

For min and merged-LoRA samplers: pass the prompt list via
`--probe_prompts` (despite the name; it accepts any prompt list). Build
the prompt list from the eval_meta of pi_benefit (which has both number-seq
and generic blocks).

### Step 2.5 — Run cost sampling (probes)

One probe-prompts pass per model. The probe list combines BOTH selected
effects' direct + generalization + narrative probes. Probe pi_A for both
TRAIT_A and TRAIT_B, and probe pi_B for both TRAIT_A and TRAIT_B, so the
leakage matrix is available.

Build the probe list (one-time, save to a JSON file):

```bash
python -c '
import json, yaml
import os
manifest = yaml.safe_load(open("configs/sweeps/subliminal_trait_candidates.yaml"))
trait_ids = [os.environ["TRAIT_A"], os.environ["TRAIT_B"]]
probes = []
for trait_id in trait_ids:
    cand = manifest["candidates"][trait_id]
    cat = manifest["categories"][cand["category"]]
    for probe_type, key in (
        ("direct", "probe_direct"),
        ("generalization", "probe_generalization"),
        ("narrative", "probe_narrative"),
    ):
        for p in cat.get(key, []):
            probes.append({"effect": trait_id, "probe_type": probe_type, "prompt": p})
json.dump(probes, open("/tmp/probe_prompts.json", "w"), indent=2)
'
```

Then sample with `--probe_prompts /tmp/probe_prompts.json`. Each model
gets one probe run (n=10 samples per prompt is a reasonable default;
adjust if the probe count is large enough that ~320 is hit naturally).

```bash
# Example for pi_min probe sampling
python scripts/sample_min_composition_generations.py \
    --ref_A "$MODEL_OUTPUT_DIR/pi_A" \
    --ref_B "$MODEL_OUTPUT_DIR/pi_B" \
    --training_config configs/training.yaml \
    --probe_prompts /tmp/probe_prompts.json \
    --composition_type min \
    --n_samples 10 \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --device_A cuda:0 --device_B cuda:1 \
    --output_file $OUTPUT_ROOT/min_composition/min_t512/full/probe_samples.json \
    --markdown_file $OUTPUT_ROOT/min_composition/min_t512/full/probe_samples.md
```

Mirror for merged-LoRA and each single-model baseline.

### Step 2.6 — Compute headline tables

After all sampling JSONs land, on Hyak:

```bash
python scripts/analyze_subliminal_trait_sweep.py \
  --sweep_root "$OUTPUT_ROOT" \
  --candidate_manifest configs/sweeps/subliminal_trait_candidates.yaml \
  --output_dir "$OUTPUT_ROOT/summaries" \
  2>&1 | tee "$OUTPUT_ROOT/analyze_subliminal_trait_sweep.log"
```

Two tables: joke retention per model per prompt-set; cost rates per
model per effect per probe-type.

### Step 2.7 — Append findings to a new file

Create `$OUTPUT_ROOT/findings.md` (note: NEW file, not the existing
`min_composition/findings.md` which is explicit-cost specific). Required
fields:

- Header: git SHA, dataset generation timestamps, training durations,
  total wall runtime.
- Joke-retention table (six models × two prompt sets).
- Cost-rate table (six models × two effects × three probe types).
- Leakage matrix: pi_A's TRAIT_B rate and pi_B's TRAIT_A rate (should be
  near base rate; if not, references absorbed both traits via dataset
  contamination).
- Headline narrative: "did pi_min suppress subliminal cost at probe
  contexts?" with the quantitative answer.
- Sample evidence: 2-3 representative probe responses per model.

### Step 2.8 — rsync paths

```bash
# Full subdirectory tree
rsync -avP \
  --include='*/' \
  --include='*.json' \
  --include='*.md' \
  --include='*.log' \
  --include='eval_meta.json' \
  --exclude='checkpoint-*/' \
  --exclude='*.safetensors' \
  --exclude='*' \
  adhyyan@klone.hyak.uw.edu:/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_subliminal_cost/ \
  hyak_results/outputs/composed_joke_subliminal_cost/
```

Adapter weights (~25 MB per checkpoint) can be omitted from the rsync
unless we want them locally for the diagnostic notebook.

## Phase 3 — Mac analysis and writeup

After Hyak runs land and JSONs are synced back. No GPU needed.

### Step 3.1 — Re-run analysis locally

Verify the tables compute identically from the synced JSONs:

```bash
python scripts/analyze_subliminal_trait_sweep.py \
  --sweep_root hyak_results/outputs/composed_joke_subliminal_cost \
  --candidate_manifest configs/sweeps/subliminal_trait_candidates.yaml
```

Capture the output into a notes file (e.g., `hyak_results/.../tables.txt`).

### Step 3.2 — Update `writeup/writeup_v2.tex`

Add §4 "Subliminal-cost variant" between the existing §3 (Experimental
Setup and Results, which is now explicit-cost) and the current §4
(Discussion, which becomes §5).

Section structure:

- **§4.1 Setup.** Describe subliminal training: number-sequence prompts,
  teacher with side-specific system prompt encoding the trait, joke
  suffix appended. Note that the training data contains no explicit
  trait text. TRAIT_A and TRAIT_B are selected subliminal preferences in
  disjoint categories, not literal prefixes.

- **§4.2 Eval protocol.** Two prompt sets for benefit (number-sequence
  for in-distribution, generic for OOD generalization). One probe set
  for cost (direct + generalization + narrative, per effect).

- **§4.3 Table.** Six rows (pi_base, pi_A, pi_B, pi_benefit, pi_min,
  merged_lora). Columns: joke rate at number-seq, joke rate at generic
  prompts, TRAIT_A-probe rate (direct), TRAIT_B-probe rate (direct).
  Wilson CIs on joke columns. Cost-probe rates as `rate (hits/n)`.

- **§4.4 Interpretation.** Compare to §3 (explicit cost):
  - Same composition mechanism (token-wise min).
  - Different cost surface (probe vs train-distribution).
  - Did pi_min still suppress costs at probe contexts? Quote the numbers.
  - Did joke retention hold up at OOD generic prompts?
  - Did merged-LoRA still fail cost suppression at probe contexts?

The Discussion section (now §5) should integrate the subliminal-vs-explicit
comparison briefly. The Appendix should remain unchanged unless the
subliminal experiment surfaces a new failure mode worth adding.

Edit `writeup/writeup_v2.tex` directly (untracked, no commit). Compile
locally with `pdflatex` and check no errors.

## Output paths

Hyak:
```
/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/composed_joke_subliminal_cost/
  datasets/
    TRAIT_A/        (D_A SFT data: number_seq with TRAIT_A subliminal + joke)
    TRAIT_B/        (D_B SFT data: number_seq with TRAIT_B subliminal + joke)
    benefit/        (D_benefit: neutral number_seq + joke)
  models/
    pi_A/           (LoRA rank 8, 10 epochs, trained on TRAIT_A)
    pi_B/           (LoRA rank 8, 10 epochs, trained on TRAIT_B)
    pi_benefit/     (LoRA rank 8, 10 epochs, trained on neutral)
  joke_generation_samples_t512.json          (single-model, number-seq prompts)
  joke_generation_samples_generic_t512.json  (single-model, generic prompts)
  min_composition/
    min_t512/full/
      min_composition_samples_numseq.json
      min_composition_samples_generic.json
      probe_samples.json
    merged_lora/full/
      merged_lora_samples_numseq.json
      merged_lora_samples_generic.json
      probe_samples.json
  findings.md
  analyze_subliminal_trait_sweep.log
```

Mac (after rsync, no model weights):
```
hyak_results/outputs/composed_joke_subliminal_cost/   (mirrors Hyak minus checkpoints)
```

## Failure handling

- **Subliminal effect doesn't transfer.** If pi_A's probe-direct TRAIT_A
  rate is at or near pi_base's rate (~10%), the subliminal training
  didn't take. Possible causes: (a) joke suffix dominated the gradient
  signal away from the subliminal numbers; (b) Qwen3-8B doesn't transfer
  this trait at this scale; (c) dataset gen had a bug.
  Diagnose by sampling pi_A on plain number-sequence prompts (no
  composition) and counting probe-direct rates separately from the
  composed model. If the single ref doesn't transfer, none of the
  composition results are meaningful.

- **Joke retention collapses at OOD generic prompts.** Expected to some
  degree (training distribution narrowed to number-seq prompts; jokes
  attached to number continuations specifically). Report the size of the
  drop; it's an honest finding about benefit-vs-distribution.

- **Probe responses are off-distribution gibberish.** If pi_min produces
  Cyrillic or polite-email closures on probes (as the directional variant
  did in Appendix B), report it — that's a real failure mode of min on
  this probe surface, not a code bug.

- **Disk quota on Hyak.** Subliminal datasets are larger than explicit-cost
  (more filtering iterations, contrastive scoring caches). Estimate:
  ~5GB total for datasets + ~500MB for adapters + ~50MB for samples.
  Should fit comfortably under the scratch quota.

- **HF auth (401/403)**: ensure HF_TOKEN is set; refs and tokenizer cached
  from prior runs.

- **CUDA OOM**: 2 LoRA refs at bfloat16 on 2×A100 fits with margin. If OOM,
  check for stale Python processes holding GPU memory.

## Interpretation principles

- **The headline question is whether suppression generalizes to probe
  contexts.** This is the train-vs-probe gap that writeup_v2.tex §4
  explicitly raised but did not test. Both answers (yes / no) are
  publishable; let the data speak.

- **The explicit-cost experiment (already in writeup §3) is the
  structural-validity warm-up.** The subliminal-cost experiment is the
  realism upgrade. Do not conflate them; do not argue the subliminal
  result is needed to validate the explicit one.

- **Merged-LoRA contrast matters as much as the absolute number.** If
  pi_min and merged-LoRA both fail cost suppression at probe contexts,
  the writeup's "composition does real work" claim weakens. If pi_min
  succeeds and merged-LoRA fails, the contrast is preserved and the
  story is intact. If both succeed, that's a surprising result —
  investigate before claiming composition is necessary.

- **Cost detection metric: report all three views.** strict_first /
  flex_first_line / anywhere. They give different signals: strict_first
  is the "model chose to name the trait" rate; anywhere is the "trait
  leaked into open-ended responses" rate. Headline depends on probe type
  (strict_first for direct one-word probes; anywhere for narrative
  probes).

- **Per CLAUDE.md and project conventions:** no shared utils; functions
  duplicated per file. Never add dataset size limits without user
  confirmation. All Hyak outputs and caches under `/gscratch/scrubbed/adhyyan`.
  Composition-side commits go to `min-regularization`; the user is the
  only writer on that branch. Push directly via `git push origin
  HEAD:min-regularization` from a worktree branch; **confirm before the
  first push of a session**.

- **Honest metric reporting per the Honest evaluation checklist in
  workflow_handoff.md.** max_new_tokens=512 at sample time; markdown-aware
  joke regex; report truncation count.

## Quick-start for the next agent

1. Read this file end-to-end.
2. Read [workflow_handoff.md](../workflow_handoff.md) for project state.
3. Skim `writeup/writeup_v2.tex` (untracked, Mac-side) for the writeup
   framing — particularly the §4 Discussion which raised the train-vs-probe
   question this experiment answers.
4. Spawn a worktree off `origin/min-regularization` per "Branch handling"
   above.
5. Start with Phase 1, Step 1.1 (write `composed_subliminal_joke.py`).
6. Confirm with the user before the first `git push` of the session.

Mechanical failures (OOM, traceback, missing path) are bugs — debug and
rerun. Behavioral failures (pi_min doesn't suppress; joke doesn't survive
OOD) are scientific findings — record and continue.
