# EM/MASSIVE package and integration validation

The records below distinguish validation of the standalone package candidate
from validation of its additive integration into the collaborator repository.
These checks execute no GPU training/inference and make no judge/API calls.

## Historical package validation — 2026-09-22

These checks were completed on the standalone `subliminal-mitigate-public`
candidate before integration; they are not presented as a fresh test count for
the combined repository.

- **97 tests passed, no skips** in the local Python 3.9.6 environment with NumPy
  2.0.2, CPU-accessible torch 2.8.0, matplotlib, safetensors and PyYAML installed.
  Training orchestration uses fake runtime objects; this is not a GPU smoke test.
- Built a wheel without network access and installed it in a temporary virtual
  environment. The environment reused installed scientific libraries, so this
  does not establish a clean-machine dependency resolution test.
- Ran the installed `consensus-experiments reproduce` command outside this
  package checkout with `PYTHONPATH` unset. Both experiments completed: 20 MASSIVE
  arms and six EM methods. No sibling checkout or historical launcher was imported.
- Recomputed the original MASSIVE numerical projection exactly, including paired
  contrasts, confidence intervals and bootstrap-bank hashes. The regression
  fixture is bound to the original result independently of the new resource seal.
- Checked EM counts, abstention handling and plotted coordinates. Rendered and
  visually inspected both figures. EM typography is not claimed to be identical
  to the current Overleaf artwork.
- Compared sampler kernels and synthetic trajectories against the historical
  source implementations. Tests also cover dataset schedules, weighted-Union
  allocation, training objectives, checkpoint checks, merge factors, medical
  rubric parsing, generation dispatch, and refusal to overwrite reproduction
  outputs.
- A focused scan of the package source, tests, documentation and bundled
  resources found no matching private machine paths or common credential
  patterns. This is not a comprehensive security/history audit or license clearance.

## Local integration validation

The integration base is collaborator commit
`38226f0ae7f43157b8ee914577fb268bce682e09`. Existing collaborator scripts and
configurations remain unchanged; the new package is not substituted into the
unfiltered distillation workflow. Its source, resources and tests are imported
from the validated package candidate. Documentation and ignore rules are
combined for the shared repository.

The pre-integration compatibility check reported 106 passing CPU tests (97
package tests and nine collaborator tests) and successful installed-package
reproduction in a collaborator checkout. That check supports the integration
plan, but does not by itself constitute validation of this final integration
diff.

Fresh source-checkout validation on the local integration branch passed **106
CPU tests, with no skips**: 101 tests in `tests/` and five in the collaborator's
three subliminal-KD test modules in `scripts/`. This comprises the 97 package
tests and nine preserved collaborator tests. No GPU training/inference or API
calls were made. It does not establish GPU compatibility or whole-repository
security/history clearance.

The final integration wheel was built offline with setuptools 80.9.0, then
installed into a fresh temporary virtual environment using existing scientific
libraries. From outside the checkout, the installed package passed the same
106 tests with no skips. With `PYTHONPATH` unset, its reproduction command
recomputed all 20 MASSIVE arms and six EM methods and rendered both experiments'
figures. The basic install did not download models or invoke GPU/API work.

A byte comparison against the base commit confirmed that only three existing
files changed: `README.md`, `.gitignore` and `AGENTS.md`. The other 341 upstream
files, including all executable code, configurations and requirements, remain
unchanged. All 56 imported package-source/resource/test files match the
standalone candidate after removing trailing blank lines at EOF; the only
source differences are those whitespace cleanups in six files. Scientific
resources and executable logic are unchanged. The new Git checkout has independent
objects and does not depend on the temporary compatibility-review repository.

Run the combined portable suite from the root of an installed development checkout:

```sh
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s scripts -p 'test_subliminal_kd_*.py' -v
```

## Not yet verified

Actual pinned-environment GPU training, real XGrammar sampling, loading the
materialized merge through PEFT, multi-GPU execution, and a new paid judge call
have not been run. Some tensor/plot tests skip when their optional libraries are
absent. Missing original EM artifacts and source-data permissions remain as
documented in the [package guide](em_massive_experiments.md) and
[publication checklist](../RELEASE_CHECKLIST.md). Existing repository history
has not been scrubbed or cleared for public release by these package checks.
