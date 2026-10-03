# GPU Codex Handoff: Phase 0 Subliminal Trait Sweep

**Status update:** this composed trait+joke Stage A was run and produced
`0/25` promoted candidates under its gates. Treat it as a negative first
attempt, not the current blocking protocol. The replacement trait-only sbatch
handoff is [gpu_sbatch_subliminal_trait_only_handoff.md](./gpu_sbatch_subliminal_trait_only_handoff.md).

This is a standalone experiment that must run before the subliminal-cost
composition experiment. Do not assume eagle/topaz works. The goal is to find
two traits that both learn subliminally while retaining the joke benefit.

## Goal

Find a pair `(TRAIT_A, TRAIT_B)` such that:

- a LoRA trained on `TRAIT_A` number-sequence subliminal data plus `Joke:`
  learns `TRAIT_A` and the joke benefit;
- a LoRA trained on `TRAIT_B` number-sequence subliminal data plus `Joke:`
  learns `TRAIT_B` and the joke benefit;
- cross-leakage is low in both directions;
- the selected traits are from disjoint categories.

If no pair passes, record that as a finding and do not start the main
subliminal-cost composition experiment.

## Setup

Use a single-node 2-GPU allocation:

```bash
salloc -A jamiemmt -p gpu-a100 --nodes=1 --gpus-per-node=2 --cpus-per-task=4 --mem=64G --time=12:00:00
```

Standard environment:

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
git checkout min-regularization
git pull
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

The final line must print `True 2`.

## Paths

```bash
export OUTPUT_ROOT=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_trait_sweep
export MANIFEST=configs/sweeps/subliminal_trait_candidates.yaml
export COMMON_CONFIG=configs/dataset_gen.yaml
export TRAINING_CONFIG=configs/training.yaml
mkdir -p "$OUTPUT_ROOT"/{datasets,models,samples,summaries,logs}
```

Candidate ids are in `$MANIFEST`. The initial sweep scope is animal, tree,
and gemstone candidates only.

## Stage A: 2k Candidate Screen

Run each candidate independently. This loop is resumable: existing datasets
and checkpoints can be skipped manually.

```bash
python - <<'PY' > "$OUTPUT_ROOT/candidates.txt"
import yaml
m = yaml.safe_load(open("configs/sweeps/subliminal_trait_candidates.yaml"))
for cid in m["candidates"]:
    print(cid)
PY

while read -r CANDIDATE; do
  echo "=== Stage A dataset: $CANDIDATE ==="
  if [[ ! -d "$OUTPUT_ROOT/datasets/stage_a/$CANDIDATE" ]]; then
    python dataset_gen/composed_subliminal_joke.py \
      --common_config "$COMMON_CONFIG" \
      --candidate_manifest "$MANIFEST" \
      --candidate_id "$CANDIDATE" \
      --output_dir "$OUTPUT_ROOT/datasets/stage_a/$CANDIDATE" \
      --n_samples 2000 \
      --seed 42 \
      2>&1 | tee "$OUTPUT_ROOT/logs/stage_a_${CANDIDATE}_datagen.log"
  fi

  echo "=== Stage A train: $CANDIDATE ==="
  python scripts/train_single_sft.py \
    --dataset "$OUTPUT_ROOT/datasets/stage_a/$CANDIDATE" \
    --training_config "$TRAINING_CONFIG" \
    --output_dir "$OUTPUT_ROOT/models/stage_a" \
    --name "$CANDIDATE" \
    --epochs 10 \
    2>&1 | tee "$OUTPUT_ROOT/logs/stage_a_${CANDIDATE}_train.log"
done < "$OUTPUT_ROOT/candidates.txt"
```

Sample all trained candidates against all candidate probes, plus base:

```bash
python scripts/sample_trait_probes.py \
  --model "$OUTPUT_ROOT/models/stage_a" \
  --training_config "$TRAINING_CONFIG" \
  --candidate_manifest "$MANIFEST" \
  --output_file "$OUTPUT_ROOT/samples/stage_a_trait_probes.json" \
  --candidate_ids all \
  --n_samples 10 \
  --max_new_tokens 512 \
  --temperature 1.0 \
  2>&1 | tee "$OUTPUT_ROOT/logs/stage_a_trait_probes.log"
```

Sample joke benefit for both prompt sets:

```bash
python scripts/sample_joke_generations.py \
  --model "$OUTPUT_ROOT/models/stage_a" \
  --training_config "$TRAINING_CONFIG" \
  --output_file "$OUTPUT_ROOT/samples/stage_a_joke_numseq.json" \
  --benefit_id joke_suffix_numseq \
  --n_samples 10 \
  --max_new_tokens 512 \
  2>&1 | tee "$OUTPUT_ROOT/logs/stage_a_joke_numseq.log"

python scripts/sample_joke_generations.py \
  --model "$OUTPUT_ROOT/models/stage_a" \
  --training_config "$TRAINING_CONFIG" \
  --output_file "$OUTPUT_ROOT/samples/stage_a_joke_generic.json" \
  --benefit_id joke_suffix_generic \
  --n_samples 10 \
  --max_new_tokens 512 \
  2>&1 | tee "$OUTPUT_ROOT/logs/stage_a_joke_generic.log"
```

Analyze:

```bash
python scripts/analyze_subliminal_trait_sweep.py \
  --sweep_root "$OUTPUT_ROOT" \
  --candidate_manifest "$MANIFEST" \
  --dataset_root "$OUTPUT_ROOT/datasets/stage_a" \
  --output_dir "$OUTPUT_ROOT/summaries/stage_a" \
  2>&1 | tee "$OUTPUT_ROOT/logs/stage_a_analyze.log"
```

Promotion gates:

- direct strict-first own-trait rate `>= max(base + 0.15, 0.25)`;
- generalization or narrative anywhere-rate delta `>= 0.05`;
- number-sequence joke flex-last rate `>= 0.90`;
- truncation rate `<= 0.05`;
- selected training rows have `selected_explicit_trait_leakage == 0`.

## Stage B: 10k Confirmation

Use the top 6 Stage A candidates that pass all gates. If fewer than 2 pass,
record the failure in `findings.md` and stop.

Create the candidate list:

```bash
python - <<'PY' "$OUTPUT_ROOT/summaries/stage_a/candidate_summary.csv" "$OUTPUT_ROOT/stage_b_candidates.txt"
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
passed = [r for r in rows if r["passed"] == "True" or r["passed"] == "true"]
passed.sort(key=lambda r: float(r["own_direct_strict"]), reverse=True)
open(sys.argv[2], "w").write("\n".join(r["candidate_id"] for r in passed[:6]) + "\n")
print("stage_b:", [r["candidate_id"] for r in passed[:6]])
PY
```

Generate/train/sample/analyze exactly as Stage A, but use:

- dataset root: `$OUTPUT_ROOT/datasets/stage_b`
- model root: `$OUTPUT_ROOT/models/stage_b`
- sample files prefixed `stage_b_`
- `--n_samples 10000` in dataset generation
- larger eval sampling if time permits, recommended `--n_samples 20`

When analyzing Stage B, pass the Stage B sample files explicitly so the Stage A
JSONs under the same sweep root do not get mixed into the confirmation summary:

```bash
python scripts/analyze_subliminal_trait_sweep.py \
  --candidate_manifest "$MANIFEST" \
  --dataset_root "$OUTPUT_ROOT/datasets/stage_b" \
  --output_dir "$OUTPUT_ROOT/summaries/stage_b" \
  --probe_samples "$OUTPUT_ROOT/samples/stage_b_trait_probes.json" \
  --joke_samples "$OUTPUT_ROOT/samples/stage_b_joke_numseq.json" \
  --joke_samples "$OUTPUT_ROOT/samples/stage_b_joke_generic.json" \
  2>&1 | tee "$OUTPUT_ROOT/logs/stage_b_analyze.log"
```

The final pair recommendation is:

```bash
$OUTPUT_ROOT/summaries/stage_b/pair_recommendations.md
```

Pairs pass only if candidates are from disjoint categories and cross-direct
leakage satisfies `<= max(base + 0.05, 0.10)` in both directions.

## Required Findings

Append or create `$OUTPUT_ROOT/findings.md` with:

- git SHA and command timestamps;
- Stage A candidate summary table;
- Stage B candidate summary table;
- pair recommendation table;
- selected `(TRAIT_A, TRAIT_B)` if any pair passes;
- reason for stopping if no pair passes;
- any bugs patched during the run and the commit SHA that fixed them.

Do not delete failed runs. They are useful evidence.

## Failure Handling

- If dataset generation fails because too few rows survive filtering, rerun
  that candidate with a larger `--pool_multiplier` or record it as failed.
- If all candidates fail the joke gate, inspect `stage_a_joke_numseq.json`
  before changing thresholds.
- If all candidates fail the trait gate, rerun a small known baseline such
  as `eagle` or `owl` without the joke suffix to separate generator bugs from
  real non-transfer.
- If a Python traceback is clearly a bug, patch the repo on
  `min-regularization`, commit, pull on Hyak, and rerun the failed command.

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
  adhyyan@klone.hyak.uw.edu:/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_trait_sweep/ \
  hyak_results/outputs/subliminal_trait_sweep/
```
