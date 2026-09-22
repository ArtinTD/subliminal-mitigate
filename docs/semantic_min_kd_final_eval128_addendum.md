# Semantic-min KD final-student 128-response evaluation addendum

On September 21, 2026, the user approved enlarging the evaluations for the
three already-trained `raw1024_steps200` students. This is evaluation only:
no student is retrained, no teacher data are regenerated, and no completed
64-response artifact is modified.

Each final student is evaluated on the original 32 prompts x 4 samples,
temperature 1, maximum 512 generated tokens, seed 0, historical q=1 decoder,
and the original scorers. The completed diagnostic results provide per-prompt
sample slots 0 and 1. Generate only missing slots 2 and 3, using their original
four-slot global indices, then combine the immutable old and new rows into a
separate 128-response result under `evaluation128/`.

Run seeds 42, 43, and 44 one typed L40S at a time. Preserve unique submission
receipts, atomic response checkpoints, exact metrics, adapter/input hashes,
runtime, and GPU-memory evidence. Compare the three full evaluations with the
existing immutable 128-response teacher, base, and pilot-student evaluations.

This confirmatory evaluation does not authorize a new training condition,
additional seed, teacher-data expansion, model or decoder change, paper update,
or automatic numerical success gate.
