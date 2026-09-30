# MSCD pipeline pilot

The package extracts the existing explicit-prefix implementation into reusable stages. Historical scripts and results remain unchanged. Extraction commits and source hashes are in `provenance.json`.

## Components

- `SourceRecord` identifies each source-prompt occurrence, including repeated prompts.
- `ExplicitPrefixDatasetBuilder` constructs independent Eagle/Joke and Topaz/Joke sources. `ImportedDatasetBuilder` reads saved Hugging Face datasets.
- `Trainer.fit` starts from the base model for source teachers, union training, and the student. It returns a `ModelArtifact`.
- `ConsensusDecoder` composes teacher distributions using `MinimumConsensus`. The v1 GPU loop supports two teachers; the tensor rule accepts more.
- `ModelGenerator`, `MergedLoRAGenerator`, and `WholeOutputConsensusGenerator` implement the baseline generation interface.
- `DatasetRegenerator` replaces responses while preserving every source occurrence. It does not filter on benefit or cost.
- `Evaluator` scores saved responses independently of generation. Abstentions stay in the requested denominator; execution failures invalidate rates.
- `Experiment` resolves dependencies and runs GPU stages in separate processes. No teacher/student subclasses are needed: model role and training data differ, while training is shared.

## Running

Install the repository requirements in a separate GPU environment, then `pip install -e .`. Local tests need PyTorch, PyYAML, and pytest.

```bash
python -m pytest tests/mscd -q
python -m mscd pin configs/mscd/explicit-prefix.yaml --output-config configs/mscd/pilot-pinned.yaml
python -m mscd plan configs/mscd/pilot-pinned.yaml
python -m mscd run configs/mscd/pilot-pinned.yaml --through regenerate
python -m mscd run configs/mscd/pilot-pinned.yaml --resume
```

`--through` includes dependencies; `--only` requires their completed artifacts. All stages write checksummed completion receipts. Generation resumes by request ID. Changing configuration or implementation requires a new output directory. Interrupted legacy SFT cannot resume optimizer state because its checkpoints save only models; partial training is rejected instead of silently restarted. Imported datasets are configured using `import_sources: {eagle: PATH, topaz: PATH}`. Pretrained adapters can be supplied using `provided_models: {student: {path: PATH}}` (likewise `eagle`, `topaz`, and `union`); their content is hashed, their base checked, and training dependencies skipped. Configuration-relative paths currently apply to the training and source YAML files; use absolute paths for external datasets and model artifacts.

## Pilot protocol and deliberate differences

The new pilot is a new dataset-construction replicate, not a claim of reproducing the historical sample sequence. It uses 1,000 valid examples per source, rank-8 adapters, 200 training steps, 32 evaluation prompts and 10 responses per prompt. All eight methods are included. The historical whole-output plot had 64 requests; the new pilot gives it the same 320 requests as other methods, retaining abstentions in the denominator.

Prompt selection shuffles the full pinned Alpaca corpus before taking 1,500 candidates per source. The old constructor restricted the candidate corpus before shuffling. Construction varies both prompt selection and response seeds. Fewer than 1,000 valid examples stops the run without silently replacing a seed. Train/evaluation overlap is rejected. A single new replicate cannot estimate across-dataset variance.

Single-model and student generation retain the historical vLLM LoRA path; merged and whole-output generation retain the historical Transformers path and model generation defaults. Whole-output acceptance is used at temperature 1 in the pilot; its historical raw sequence scoring does not justify arbitrary temperatures. Every request has its own seed, allowing exact interruption/resume; this differs from the old whole-output shared random stream. These changes must be considered when comparing new and historical rates.

The metric is terminal Joke-marker retention (including the plot's accepted Markdown variants), not humor quality. Cost is a case-sensitive Eagle/Topaz prefix on the first nonempty line. Wilson intervals are response-level, conditional on this dataset and prompt set, not source-construction uncertainty or prompt-cluster intervals.

## Validation status

See `historical-scoring.json` for saved-response recounts. Historical teacher provenance is under investigation: checked artifacts do not yet reproduce the figure's 305/320 count. Do not use that count as a passing golden test or relabel an archived result.

CPU tests compare the minimum rule and cached token loop to frozen historical source, check whole-output acceptance/abstention, request-preserving regeneration, checkpoint identity, and artifact tampering. GPU validation compares original and extracted decoding on the same historical adapters. The separate two-step training fixture checks fresh student initialization and adapter saving; it is not an experiment and is excluded from results.
