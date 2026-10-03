# Maintained project

- Work in the existing MSCD refactor checkout. The active implementation is `src/mscd/`; this package currently covers the explicit-prefix/minimum-consensus pipeline only.
- Read the root `README.md` for installation, stage selection, resumption, and validation. It is the single maintained usage guide.
- Active recipes and their training/source protocols live in `configs/mscd/`. Keep the nine MSCD helpers in `scripts/` at their existing paths.
- Keep responsibilities divided among dataset construction, generation/consensus, training, evaluation, and experiment orchestration. Teacher and student roles use the shared trainer.

# Preservation and correctness

- `legacy/` is an unsupported historical archive. Do not repair its execution paths, add compatibility wrappers, or restore deleted manuscripts as part of active package work.
- Preserve frozen historical fixtures in `tests/mscd/reference/` and the original paths, commits, and hashes in `docs/provenance.json`.
- Keep full downloaded run records under `legacy/` ignored by Git; keep compact validation records in `docs/validation/`. Inspect staging rather than adding bulk outputs indiscriminately.
- Do not change experimental dataset sizes, seeds, metrics, model revisions, or generation/training settings without authorization. Other settings await an explicit migration task.
- Preserve run identity, artifact hashes, and resumption safeguards. Relocating configurations or changing source files can alter fingerprints. Use a new output directory rather than rewriting historical manifests or completion receipts.
- Do not silently restart partial SFT: saved model-only checkpoints lack optimizer state.

# Validation

- Run `python -m pytest tests/mscd -q` in an environment with PyTorch, PyYAML, and pytest. These CPU tests cover numerical parity, scoring, request resumption, stage dependencies, and artifact identity.
- For structural changes, check both experiment configurations with `python -m mscd plan`, shell syntax with `bash -n`, and installation/tests from a temporary copy that excludes `legacy/`.
- Keep the GPU dependency versions and environment constraints unchanged unless the task specifically requires an environment change. GPU checks are separate from the CPU suite; tiny fixtures are tests, not reduced experiments.
- Do not edit the manuscript or archived experiments, synchronize Hyak, submit GPU jobs, or push a repository unless the task authorizes it.
