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
