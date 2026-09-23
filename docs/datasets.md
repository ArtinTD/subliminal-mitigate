# Dataset preparation boundaries

The package consumes source data supplied by the collaborator. It does not
download protected medical archives, regenerate teacher responses, or infer a
missing original data bank from result files. Source acquisition, permission to
use or redistribute the data, pinned source identities, and held-out split
selection must be established before constructing training datasets.

## Paired MASSIVE/medical schedules

`MassiveDatasetBuilder` reconstructs the recorded paired schedule from already
curated MASSIVE prompt/response rows and matched medical responses:

```python
from subliminal_mitigate.datasets.massive import MassiveDatasetBuilder

builder = MassiveDatasetBuilder()
prepared = builder.build(massive_rows, medical_pairs)
manifest = prepared.save("data/massive_medical_paired")
```

MASSIVE records contain `prompt` and `response`. The prompt must already have
the intended task template and ontology. Medical records contain `prompt`,
`bad_response`, and `good_response`. Text is preserved exactly. If separate
local medical banks contain `prompt`/`response` pairs, use
`builder.pair_medical_rows(bad_rows, good_rows)`; it requires identical prompts
at every position. It does not sort or repair mismatched pairs. Convert a
source-specific messages schema to these rows while retaining its provenance.

Defaults require exactly 1,122 curated MASSIVE rows and 7,049 medical pairs.
Each MASSIVE row appears ten times and each medical prompt three times, giving
32,367 presentations in each arm. Arm A uses bad-medical responses; arm B uses
good-medical responses. Both arms share every prompt position and the complete
MASSIVE rows. Duplicate normalized prompts, duplicate within-bank medical
responses, cross-bank response overlap, and inconsistent supplied source IDs
are rejected.

Source IDs use the original canonical JSON SHA-256 rules: MASSIVE IDs hash
`{prompt,response}`; medical IDs hash `{prompt}`. Presentations sort by the hash
of `{seed,source_id,repeat_index,kind}`, with seed 20260818 and the original
tie-breakers. Skeleton and ordered arm hashes therefore match the original
implementation when provided the same curated inputs. Passing explicit
alternative source counts or repeats supports synthetic checks or new studies;
it does not reproduce the recorded experiment.

The saved directory contains:

```text
train/A_massive_bad_medical.jsonl
train/B_massive_good_medical.jsonl
provenance/presentation_skeleton.jsonl
data_manifest.json
```

The training API accepts either JSONL arm directly. `prepared.arms["A"]` and
`prepared.arms["B"]` also expose rows for further dataset construction. The
manifest records schedule counts, source/arm hashes, pairing assertions, file
hashes, and its own payload seal. Saving requires a fresh directory.

This builder does not recreate MASSIVE archive acquisition, English-language
selection, the curated 1,122-row subset, task-template validation, or train/dev/
test leakage audits. It explicitly records those missing validation boundaries.
The completion-only trainer separately audits token lengths and labels using
the actual tokenizer; preparation does not claim those checks have occurred.

## EM shards and weighted union baselines

`EMDatasetBuilder` takes the actual local medical banks and previously generated
joke bank, validates joke suffixes, constructs the configured benefit share,
and uses the original Hugging Face shuffle behavior for augmentation and
disjoint shard selection. Request shard count and size explicitly. A missing
original bank is an unavailable input, not permission to substitute another
dataset or truncate one silently.

`WeightedUnionBuilder` consumes the already constructed paired MASSIVE arms.
It owns the final ratio-dependent medical exposure allocation and deterministic
union ordering. Use that operation for the ratio baselines; concatenating raw
medical/MASSIVE banks or independently shuffling each arm changes the protocol.
Its returned dataset can save a Hugging Face dataset and construction metadata.

The packaged tests use synthetic source rows and hash fixtures captured from
the original pure schedule functions. No original data, protected archive, or
model download is needed for those checks.
