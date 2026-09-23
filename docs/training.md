# LoRA SFT training

`SFTTrainingRun` executes dataset loading, model and LoRA initialization, audited
formatting, TRL training, adapter saving, and training metadata publication.
It supports both the full-token EM objective and the completion-only MASSIVE
objective. It imports no historical script and requires no sibling checkout.

```python
from subliminal_mitigate.training import SFTTrainingRun, TrainingRecipe

recipe = TrainingRecipe.from_yaml("src/subliminal_mitigate/resources/massive_training.yaml")
result = SFTTrainingRun(
    recipe,
    dataset="data/paired_train.jsonl",
    output_dir="outputs/adapters/arm_A",
    backend="auto",
).run()
print(result.to_mapping())
```

Use an existing recipe path for your experiment. `output_dir` is the adapter's
final directory, not its parent. The dataset may be a Hugging Face
`save_to_disk` directory, a JSONL file, or a JSON list. Every record must contain
nonempty `prompt` and `response` strings. All rows are used in their supplied
order; no sample limit or filtering is applied by the trainer. Construct the
experiment's repeated or mixed presentation schedule before training.

## Runtime and configuration

Importing the public training API and validating a recipe requires no GPU
dependencies. Calling `run()` requires the training environment. The source
experiments used torch 2.9.0 with CUDA 12.9 wheels, Transformers 4.57.6,
datasets 4.3.0, PEFT 0.18.1, TRL 0.24.0, Accelerate 1.13.0, and Unsloth
2026.3.4. Install these in a compatible Linux/CUDA environment; the ordinary
package import does not install or start this environment.

`backend="auto"` preserves the source dispatch: Unsloth for `WORLD_SIZE=1`,
Transformers/PEFT for distributed training. `backend="transformers"` explicitly
selects that implementation for either case. Unsloth is imported before torch,
Transformers, datasets, and TRL to preserve its patch ordering. Start training
in a fresh process if those libraries were already imported elsewhere.

The released recipes and implementation support `dtype: bfloat16` and positive
integer epochs. Other dtypes and fractional epochs are rejected during recipe
validation: the historical loader fixed Transformers weights to bfloat16, and
its step-budget computation truncated fractional epochs. Unsloth retains the
source `dtype=None`, non-4-bit model loading, and Unsloth gradient checkpointing;
Transformers retains bfloat16, SDPA, and non-reentrant gradient checkpointing.

Recipe fields preserve batch size, gradient accumulation, learning rate,
scheduler, warmup, seed, data seed, optimizer, weight decay, sequence length,
checkpoint interval/retention, worker count, logging, and reporting. Explicit
optimizer/weight-decay settings and completion-only loss fail if the installed
TRL cannot represent them. In-training behavioral probes and regularizers are
outside this plain-SFT implementation.

When `max_steps` is absent, the source step rule is retained:

```text
batches_per_epoch = ceil(rows / (batch_size * WORLD_SIZE))
epoch_steps = ceil(batches_per_epoch / gradient_accumulation) * epochs
max_steps = max(epoch_steps, min_steps)
```

An explicit `max_steps` takes precedence. The resolved budget is returned in
`TrainingResult` and written into both summary and run metadata. This fixes a
legacy post-training failure: the single-adapter script read `max_steps`
directly from the input mapping even when the trainer had derived it from
epochs.

## Objective and restart guarantees

`loss_on: all` retains chat-template text training. `loss_on: completion` uses
TRL's conversational prompt/completion schema. Before training, the complete
dataset is checked for exact generation-prompt prefixing, nonempty targets,
and targets that would exceed `max_seq_length`. After TRL prepares the data,
every completion mask must retain the complete verified target. Actual
collator labels are checked at up to eight evenly spaced examples, including
both padded and padding-free layouts. Truncation causes an error; the trainer
does not silently filter or shorten examples.

The output records an ordered source-content hash, effective recipe, backend,
distributed batch geometry, objective, completion-mask audit where applicable,
step summary, and final run metadata. The last metadata file marks completion
only after training and adapter saving succeed. A matching completed run is
reused; changed data, seeds, objective, or recipe requires another output
directory. These recipe/data checks are stronger than the original script's
completion-only skip check.

Interrupted runs resume the latest numbered Trainer checkpoint only if it
contains trainer, optimizer, scheduler, and RNG state. Set
`save_only_model: false` to create resumable checkpoints. Model-only checkpoints
cannot recreate training state. `resume=False` requires a fresh output
directory. The coordinator rank owns output writes and validation.

For strictly offline model loading, pass `local_model_path` pointing to the
standard Hugging Face cache snapshot for the recipe's model and immutable
40-character revision. The loader verifies shard completeness, cache-contained
links, stable file contents, and all config/tokenizer/index artifacts used by
the final baseline loader. It sets `local_files_only=True` and `token=False`,
and checks that saved adapter metadata retains the canonical model ID and
revision. Snapshot validation records absolute paths in local run provenance;
these generated outputs should be reviewed before sharing.

## Verification status

Portable tests exercise both executable training objectives with a fake model,
dataset, TRL trainer, and collator; resolved budgets and exact Trainer arguments;
interruption/resume and completion reuse; snapshot corruption and escaping links;
canonical adapter metadata; and Unsloth-before-torch import order.

```sh
PYTHONPATH=src python -m unittest discover -s tests -p 'test_training*.py'
```

No model weights were loaded and no GPU training, distributed run, model
download, or API call was performed while preparing this release. CPU tests
verify orchestration and preserved contracts; numerical/GPU parity still
requires a run in the pinned environment. The extraction was checked against
the final baseline plain-SFT source, including its later full snapshot-artifact
binding additions.
