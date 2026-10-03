# GPU Sbatch Handoff: Trait-Only Phase 0 Sweep

This replaces the first composed trait+joke Phase 0 attempt as the blocking
trait-selection run. The prior run completed Stage A but promoted `0/25`
candidates: the joke suffix learned on number-sequence prompts, generic joke
transfer was absent, and long narrative probes made the truncation gate too
strict. This run removes the joke suffix and tests subliminal trait transfer
first.

## Goal

Find a disjoint-category trait pair `(TRAIT_A, TRAIT_B)` that learns
subliminally from number-sequence data with no explicit trait text. Do not run
the main subliminal-cost composition experiment unless Stage B recommends a
passing pair.

## Setup

```bash
cd /mmfs1/home/adhyyan/subliminal-mitigate
git checkout min-regularization
git pull --ff-only origin min-regularization

sbatch scripts/sbatch_subliminal_trait_only_sweep.sbatch
squeue -u adhyyan
```

The sbatch script requests one `gpu-a100` node with two A100s under
`jamiemmt`, activates `/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate`,
sets HuggingFace/vLLM/Triton cache paths, uses a 24-hour walltime, and runs:

```bash
python scripts/run_subliminal_trait_only_sweep.py \
  --output_root /gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_trait_only_sweep
```

## Protocol

- Output root:
  `/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_trait_only_sweep`
- Focused Stage A candidates:
  `sapphire, eagle, emerald, panda, maple, oak, willow, ruby`
- Stage A:
  `10,000` trait-only examples per candidate, LoRA rank 8, 10 epochs.
- If fewer than two disjoint-category candidates pass focused Stage A, the
  runner automatically expands Stage A to the remaining manifest candidates at
  the same size.
- Stage B:
  top 6 promoted candidates, regenerated with independent seed, `10,000`
  examples each, larger probe sampling.
- If focused Stage B completes but finds no passing disjoint-category pair,
  the runner automatically expands Stage A to the remaining manifest
  candidates, recomputes selection, and runs an expanded Stage B candidate set.

Trait-only gates are encoded in
`scripts/analyze_subliminal_trait_sweep.py --gate_mode trait_only`:

- direct strict-first own-trait rate `>= max(base + 0.15, 0.25)`;
- generalization anywhere-rate delta `>= 0.05`;
- direct and generalization truncation rates each `<= 0.05`;
- selected explicit trait leakage is `0`.

Narrative rates and truncation are reported but do not block trait-only
promotion. Joke-benefit generalization is intentionally deferred to a separate
follow-up experiment comparing combined vs separate benefit training.

## Monitoring

Replace `<JOBID>` with the Slurm id:

```bash
tail -f /gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/trait_only_sweep_<JOBID>.out
tail -f /gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/trait_only_sweep_<JOBID>.err
tail -f /gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_trait_only_sweep/findings.md
```

The runner is resumable. Re-submitting the same sbatch skips completed
datasets, adapters, probe samples, and summaries. The focused and expanded
summaries use separate labels, so a completed focused run is preserved while
the expansion adds new artifacts.

## Expected Artifacts

```text
subliminal_trait_only_sweep/
  findings.md
  stage_a_focused_candidates.txt
  stage_a_candidates.txt
  stage_a_expanded_candidates.txt
  stage_b_candidates.txt
  stage_b_expanded_candidates.txt
  datasets/stage_a/<candidate>/
  datasets/stage_b/<candidate>/
  models/stage_a/<candidate>/
  models/stage_b/<candidate>/
  samples/*_trait_probes.json
  summaries/stage_a/candidate_summary.{csv,json,md}
  summaries/stage_a/pair_recommendations.{csv,json,md}
  summaries/stage_b/candidate_summary.{csv,json,md}
  summaries/stage_b/pair_recommendations.{csv,json,md}
  summaries/stage_a_expanded/candidate_summary.{csv,json,md}
  summaries/stage_a_expanded/pair_recommendations.{csv,json,md}
  summaries/stage_b_expanded/candidate_summary.{csv,json,md}
  summaries/stage_b_expanded/pair_recommendations.{csv,json,md}
  logs/*.log
```

If expanded Stage B finds no passing disjoint-category pair, record that as the
scientific result in `findings.md` and stop.

## Sync Back

```bash
rsync -avP \
  --include='*/' \
  --include='*.json' \
  --include='*.csv' \
  --include='*.md' \
  --include='*.log' \
  --include='*.txt' \
  --include='eval_meta.json' \
  --include='eval_config.json' \
  --exclude='checkpoint-*/' \
  --exclude='*.safetensors' \
  --exclude='*' \
  adhyyan@klone.hyak.uw.edu:/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_trait_only_sweep/ \
  hyak_results/outputs/subliminal_trait_only_sweep/
```
