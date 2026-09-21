# Semantic-min KD scaling diagnostic evaluation addendum

On September 21, 2026, the user approved reducing the scaling-student
evaluations to 32 prompts × 2 samples (64 responses) because these runs are for
directional diagnosis rather than the main paper.

Full prompt coverage, temperature 1, maximum 512 generated tokens, seed 0, the
historical q=1 decoder, and all scoring rules remain unchanged. The two retained
samples are original per-prompt slots 0 and 1, using their original four-slot
global indices. This permits exact matched subsets of the completed pilot
teacher, base, and student evaluations without regenerating those immutable
results. The original 128-response pilot results remain available as context.

This addendum changes only the diagnostic evaluation size for the scaling
ablation. It does not change teacher data, training datasets, models, training
recipes, seed sets, or the prohibition on automatic scale-up and paper updates.
