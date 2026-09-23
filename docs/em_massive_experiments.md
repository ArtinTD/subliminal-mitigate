# Final EM and MASSIVE experiment package

The `src/subliminal_mitigate/` package consolidates the final
emergent-misalignment and MASSIVE--medical experiments. It is an **additive
integration** into the collaborator repository, based on
`codex/subliminal-delta-distillation` at
`38226f0ae7f43157b8ee914577fb268bce682e09`. It provides executable dataset
builders, trainers, samplers, evaluators and figure classes without depending on
sibling checkouts or launching historical scripts.

Existing training, subliminal-learning and distillation scripts and their
configurations remain separate and unchanged. In particular, the collaborator's
unfiltered distillation workflow preserves empty responses; the EM/MASSIVE
package trainer rejects them. **Do not substitute the package trainer for that
distillation workflow.** The package is not a migration of every repository
pipeline, nor an indication that older run scripts should be deleted.

This is a local integration for review, not a public release. License selection,
repository-history review and the remaining publication checks are recorded in
the [release checklist](../RELEASE_CHECKLIST.md).

## Which experiments are final?

- **EM:** five disjoint bad-medical/joke shards and one benign-medical/joke shard;
  64 broad prompts × 5 generations, and 16 medical prompts × 20 generations.
  The earlier one-bad/one-benign and unfinished ratio extensions are not the
  final EM experiment represented by this package.
- **MASSIVE:** three fixed four-reference panels (1:3, 2:2 and 3:1 bad:benign),
  three tokenwise methods, LoRA/Union baselines, Kalai with `s=1`, and contextual
  reference results. The final combined dataset contains 20 experimental arms.
  The 2:2 Union point is the balanced A+B model, not a newly trained four-source
  weighted-Union model.

[The source map](source_map.md) identifies the final historical sources and
explains why folder suffixes such as `analysis-v1` are not whole-project versions.

## Install and reproduce the recorded results

Use a virtual environment. Run these commands **from the repository root**, not
from `docs/`:

```sh
python3 -m pip install -e '.[plots]'
consensus-experiments reproduce --output-dir outputs/paper-check
```

This installs the new package and its analysis/plot dependencies, not the full
GPU environment required by the existing repository scripts. Those workflows
retain their own installation instructions in the [root README](../README.md).

The reproduction command recomputes the MASSIVE point estimates, paired
contrasts and 95% intervals, and renders the MASSIVE and EM figures. It uses
bundled **outcome-only** records; it downloads no data or models, starts no GPU
job and makes no judge/API call. Use `--no-plots` for just the results, or
`--experiment em` / `massive`. The output directory must be new.

The new package resources exclude raw medical responses, training examples,
model weights, credentials, cluster logs, private paths and operational budget
ledgers. Their outcome-only projection has its own integrity seal and records
original source hashes; it is not the unchanged private source artifact. These
statements concern the package additions, **not the complete pre-existing
repository or its preserved Git history**.

## Package organization

```text
src/subliminal_mitigate/
  datasets/       EM augmentation/shards and paired/weighted MASSIVE builders
  training/       recipe, model loader, objective audits, checkpoint store, trainer
  decoding/       pure quorum, minimum and base-relative mathematical operators
  inference/      EM and MASSIVE model-owning sampler classes; LoRA materialization
  tasks/          MASSIVE ontology, grammar/cache primitives and scoring
  evaluation/     broad-EM scoring, medical rubric and paired ratio analysis
  plotting/       EMFigure and RatioFigure
  experiments/    final profiles and small orchestration objects
  resources/      final recipes and outcome-only reproduction inputs
  cli.py          argument parsing and delegation, not experiment implementations
```

Useful starting points:

| Task | Main class |
| --- | --- |
| Prepare final datasets | `EMDatasetBuilder`, `MassiveDatasetBuilder`, `WeightedUnionBuilder` |
| Train an adapter | `TrainingRecipe`, `SFTTrainingRun` |
| EM generation | `AdapterPanel`, `EMTokenwiseSampler`, `EMWholeOutputSampler`, `DirectEMSampler`, `MergedLoRASampler` |
| MASSIVE generation | `MassiveSampler`, `MassiveDirectSampler` |
| Materialize a MASSIVE LoRA merge | `LoRAMerger` |
| Score/analyze | `MassiveTask`, `MedicalRubric`, `MassiveMedicalRatioAnalysis` |
| Reproduce paper results | `PaperReproduction`, `EMFigure`, `RatioFigure` |

Small pure mathematical functions remain functions. Classes own actual state
(models/caches, source schedules, bootstrap banks, recipes or output stores);
they do not launch the old scripts through subprocesses. Existing scripts
elsewhere in the repository are not included in this package's API contract.

## Train or generate new outputs

First acquire the permitted source data and local model artifacts described in
[data preparation](datasets.md) and [source/rights notes](sources_and_rights.md).
The outcome-only reproduction command does not need these artifacts.

From the repository root, create a role-specific training recipe, then validate
it without loading models:

```sh
consensus-experiments recipe --experiment massive --arm A2 --output runs/A2.yaml
consensus-experiments train --config runs/A2.yaml \
  --dataset data/A_massive_bad_medical --output-dir outputs/A2
```

Only adding `--execute` starts package training. Package generation likewise
requires `generate --config RUN.json --output outputs/generations.json --execute`.
Training/inference can be expensive; these commands are not part of the CPU
reproduction or test run. `--resume` is explicit on the package training CLI.
These safeguards do not change the behavior of the existing repository scripts.

MASSIVE reference seeds are A1/B1=8182026, A2/B2=8182127 and A3/B3=8182228.
The recipe command accepts these six roles, `union13`, `union31`, and the
historical `balanced_union`; it uses 540 reference or 1079 Union steps.
For EM use `--experiment em --arm reference` or `union`; the recorded epoch/min-step
rule determines the budget from the supplied shard or union size.

See [training](training.md), [EM inference](em_inference.md),
[MASSIVE inference](massive_inference.md), [LoRA materialization](lora_materialization.md),
and [medical judging](judging.md) for runnable APIs and exact conventions.
Heavy GPU dependencies are intentionally not installed by the basic package.

## Verification and limitations

From an installed development checkout's repository root:

```sh
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s scripts -p 'test_subliminal_kd_*.py' -v
```

Plot tests require the `plots` extra. Tensor/safetensors tests run when their
optional CPU libraries are installed; otherwise those tests explicitly skip.
Package training orchestration tests use fake runtimes and do not train a model.
The first command covers the package and the collaborator's tests in `tests/`;
the second covers the collaborator's three subliminal-KD test modules in
`scripts/`. The [validation record](validation.md) distinguishes the historical
97-test package validation from the 106 tests checked on this integrated checkout.

The package's entire 20-arm numerical result—including intervals, contrasts and
bootstrap-bank hashes—matches the original result after excluding operational
provenance strings. EM counts and figure coordinates also match their source
results. The latest Overleaf EM artwork has typography differences from the
source renderer; we claim numerical/coordinate parity, not identical image bytes.

Sampler kernels and synthetic trajectories were compared with the final
historical implementations. **A fresh GPU training/inference smoke test in the
pinned environment has not been performed.** The original EM model revision and
exact original joke/shard banks were not fully archived locally; they are not
invented here. MASSIVE raw-source acquisition/selection and original-bank hash
verification remain prerequisites for a new full reproduction. The historical
Kalai `s=3` appendix run is not exposed as a new inference configuration.

Medical rubric construction and parsing are portable, but the package does not
include an API transport or the historical paid-call authorization/recovery
machinery. Recorded judgments suffice to reproduce the reported results. This
does not remove or replace existing judging scripts elsewhere in the repository.
