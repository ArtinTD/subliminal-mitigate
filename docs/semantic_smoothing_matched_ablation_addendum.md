# Matched-checkpoint semantic-smoothing ablation

On November 13, 2026, the user approved the reviewer-requested inference-only
control for semantic smoothing.  The purpose is to compare unsmoothed minimum
and smoothed minimum while holding the two teacher checkpoint bytes fixed.
This addendum authorizes no teacher training, student training, teacher-data
generation, new source dataset, decoder tuning, or automatic numerical gate.

Use the already validated reference set
`fresh_eagle_retained_topaz_replication_v1` from the completed semantic KD
pilot.  The required adapter-weight hashes are recorded in its immutable
`reference_proof.json`.  Both arms must verify that proof and load the same
Eagle/Joke and Topaz/Humor adapters:

- `exact_min`: ordinary token-wise minimum over both references (`q = 2`),
  with no semantic proposals or embedding transfer;
- `smoothed_min`: the completed pilot's semantic-smoothed minimum (`q = 2`),
  using BGE-base-en-v1.5 canonical spans, four proposals per reference, six
  proposal tokens, hard cosine threshold 0.70, transfer weight 0.5, kernel
  temperature 0.05, top eight neighbors, length-normalized cross-reference
  support, and the cached implementation.

All other inputs are identical.  The smoke run uses the frozen eight
non-evaluation prompts with one sample per prompt.  After smoke audit, the full
run uses the frozen 32-prompt bank with four samples per prompt, temperature 1,
maximum 512 generated tokens, seed 0, and the original global sample indexing
and scorers.  Report final `Joke:` or `Humor:` markers, each marker separately,
first-line and any-line-initial source-prefix costs, truncations, marginal
Wilson intervals, and a prompt-clustered paired bootstrap interval for the
smoothed-minus-unsmoothed marker-rate difference.

Run both arms sequentially in one typed-L40S allocation from the validated
isolated pilot runtime.  If scrub-storage retention removes that runtime, it
may be mechanically reconstructed in the new experiment root only from the
frozen Python and package versions and the exact model revisions recorded by
the pilot.  The reconstruction must pass its immutable version/hash audit
before GPU submission; it does not authorize dependency or model-version
changes.  Record the source commit, historical sampler hash,
reference and runtime hashes, atomic response checkpoints, exact metrics,
runtime, and peak GPU memory.  Preserve partial atomic checkpoints on
preemption and never overwrite completed output.  Store everything under the
new experiment root `semantic_smoothing_matched_ablation_v1`; all historical,
pilot, scaling, and paper artifacts remain immutable until the new result has
passed an independent audit.

If scrub retention has also removed the old pilot paths, recover the archived
proof and adapter files under `semantic_smoothing_matched_ablation_v1` rather
than repopulating the expired tree.  The proof must have SHA-256
`0ac493592a0ea42cc16fd0e067fa9a63b0a5d4dd424d09fa2076373e41220ec2`;
the recovered Eagle and Topaz weights must respectively have SHA-256
`00709368d23e38eb40f812d09478f0c29b4dbb50fb3c673503634ec2baca855a`
and `c80a5efd328dded7f43dbee33865b2a2541456a6506ca97442addc97f7be3456`.
Record both the archived paths and recovered paths, and use the recovered
paths only after these identities pass.

The mechanically reconstructed environment and exact model snapshots live in
a directory ending in `semantic_smoothing_matched_runtime_v1`.  Build and
audit them first on node-local storage, then transfer the complete tree into
the isolated ablation root with fresh mtimes and repeat the audit at its final
paths.  This prevents Hyak scrub retention from deleting files that inherit
expired package/archive mtimes during reconstruction.  This staging procedure
does not change any package version, model revision, checkpoint, prompt, or
decoder setting.
