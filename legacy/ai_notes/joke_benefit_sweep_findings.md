# Experiment 2 Findings: Joke Benefit Step/Ratio Sweep

## Run Metadata

- Node/user: `g3080`, `adhyyan`
- Repo: `/gscratch/scrubbed/adhyyan/subliminal-mitigate`
- Git SHA: `974a24d12d531268fb0fa2361392be3462b9b0bb`
- Conda env Python: `/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate/bin/python`
- Sweep root: `/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/joke_benefit_sweep`
- Dataset root: `/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/pilot_number_sequence_fixed`
- Run start: `2026-05-01T03:51:25-07:00`
- Run end: `2026-05-01T08:15:46-07:00`
- Exit code: `0`

Command:

```bash
export SUBLIMINAL_DATASET_ROOT=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/pilot_number_sequence_fixed
export SWEEP_ROOT=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/joke_benefit_sweep
export BENEFIT_RATIOS="0.10 0.30 0.50"
export MIN_STEPS_VALUES="50 100 150 200"
export EVAL_SAMPLES=20
export MATCH_ORIGINAL_COUNTS=1
bash scripts/run_joke_benefit_sweep.sh
```

Caches and temp directories were kept under `/gscratch/scrubbed/adhyyan`; no checkpoints were written under `/mmfs1/home`.

## Data

Input datasets were `eagle` and `topaz` from `outputs/pilot_number_sequence_fixed`. Counts were `n_input_A=890`, `n_input_B=954`, with `MATCH_ORIGINAL_COUNTS=1`, so every cell used `890` matched original rows per class.

The max-ratio benefit pool was generated once at `benefit_pool_ratio_0p50`: `n_generated=1335`, `n_valid=1233`, `n_benefit_only=890`. Per-cell datasets reused this pool, so their metadata records `n_generated=0` and `n_valid=890`.

| benefit_ratio | benefit rows | pi_B examples | pi_benefit examples |
| --- | ---: | ---: | ---: |
| 0.10 | 99 | 989 | 99 |
| 0.30 | 382 | 1272 | 382 |
| 0.50 | 890 | 1780 | 890 |

Training summaries are under each cell's `models/pi_B/training_summary.json` and `models/pi_benefit/training_summary.json`. All runs reached their configured final step. For `min_steps=50`, `pi_B` used more than 50 steps where needed to satisfy its minimum epoch rule: 51 steps at ratio 0.10, 66 at 0.30, and 90 at 0.50. All other cells used the requested step count exactly.

## Results

Primary artifact: `/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/joke_benefit_sweep/sweep_summary.csv`

Each metric is joke suffix rate from `20` samples over `32` prompts, so `640` responses per model per cell.

| benefit_ratio | min_steps | pi_benefit | pi_B | pass both >=0.95 |
| --- | ---: | ---: | ---: | --- |
| 0.10 | 50 | 0.905 | 0.000 | no |
| 0.10 | 100 | 0.939 | 0.300 | no |
| 0.10 | 150 | 0.908 | 0.838 | no |
| 0.10 | 200 | 0.925 | 0.895 | no |
| 0.30 | 50 | 0.905 | 0.622 | no |
| 0.30 | 100 | 0.936 | 0.875 | no |
| 0.30 | 150 | 0.942 | 0.922 | no |
| 0.30 | 200 | 0.961 | 0.914 | no |
| 0.50 | 50 | 0.895 | 0.911 | no |
| 0.50 | 100 | 0.930 | 0.922 | no |
| 0.50 | 150 | 0.939 | 0.944 | no |
| 0.50 | 200 | 0.944 | 0.948 | no |

Best individual `pi_benefit` result was ratio `0.30`, `200` steps: `0.961` (`615/640`), but `pi_B` was only `0.914`.

Best individual `pi_B` result was ratio `0.50`, `200` steps: `0.948` (`607/640`), but `pi_benefit` was `0.944`.

The most balanced near miss was ratio `0.50`, `200` steps: `pi_benefit=0.944` and `pi_B=0.948`.

## Decision

Experiment 2 did not produce a clear passing recipe because no cell reached at least `0.95` joke suffix rate for both `pi_benefit` and `pi_B`. I did not run Experiment 3.

The handoff fallback recipe is ratio `0.30`, `200` steps when no sweep setting passes, but this run shows that fallback does not transfer strongly to `pi_B` (`0.914`). If continuing, the most evidence-based next sweep is around ratio `0.50` with more training steps, for example ratios `0.40`, `0.50`, `0.60` crossed with `200`, `250`, and `300` steps, plus a repeated sample of `0.50/200` to estimate variance.

## Warnings And Notes

- Default `python` outside the conda env did not have `torch`; all experiment commands used `/gscratch/scrubbed/adhyyan/envs/subliminal-mitigate/bin/python`.
- `TRANSFORMERS_CACHE` emitted a deprecation warning; `HF_HOME` was also set under `/gscratch/scrubbed/adhyyan/.cache/huggingface`.
- Unsloth skipped cpp extensions because torch was `2.9.0+cu129`.
- Accelerate warned that kernel `4.18.0` is below the recommended `5.5.0`.
- vLLM emitted expected warnings about spawn multiprocessing, default LoRA kernel configs, LoRA tokenizer handling, and NCCL process group destruction on exit.
- `benefit_meta.json` includes a nested `benefit_config.benefit_ratio=0.3`, but the explicit `benefit_ratio_target` and row counts reflect the actual per-cell ratios.
- The run created an untracked `unsloth_compiled_cache/` directory in the repo root.
- After the sweep, `nvidia-smi` showed `0 MiB` GPU memory in use and no training/vLLM processes were left running for user `adhyyan`.
