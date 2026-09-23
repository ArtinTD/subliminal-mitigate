# Final-source map

The package is integrated additively into the collaborator repository at
`38226f0ae7f43157b8ee914577fb268bce682e09`
(`codex/subliminal-delta-distillation`). That commit already contains the quorum
branch pinned at `a04cf47`. The source map below concerns the new EM/MASSIVE
package, not a replacement for the preserved upstream scripts, configurations
or distillation workflows. See [the package guide](em_massive_experiments.md).

The old directories are workflow-specific worktrees, not successive complete
project releases. There was no single old directory containing every final
stage. The final analysis remains v2 even though its checkout name ends `v1`.

| Stage | Authoritative historical checkout | Recorded revision |
| --- | --- | --- |
| EM implementations and current local figure renderer | `subliminal-mitigate` | HEAD `84dd49d`; relevant working files include later uncommitted edits |
| MASSIVE A2/A3 training | `subliminal-mitigate-massive-ratio-v1` | `eb99679b6a1e21c8f7a91a71a079a96cc3ce2718` |
| MASSIVE changed-panel generation | `subliminal-mitigate-massive-ratio-evaluation-v1` | `142a206421426d0cc79eb983733c227c8e402c69` |
| Ratio-matched baselines | `subliminal-mitigate-massive-ratio-baselines-overnight-v1` | `c95acb8a21c0aa893d26f6cf36ece39f01fdfd5c` |
| Final baseline judging | `subliminal-mitigate-massive-ratio-baseline-judge-continuation-v3` | `8906573aba134ec31f6c01a619526d3d839bab26` |
| Combined final analysis and plot | `subliminal-mitigate-massive-ratio-analysis-v1` | Untracked v2 files above HEAD `bb131613b3afb8b121dba0e91b46116a575e2825` |

## Core sources and their new homes

- `train_single_sft.py` and standard `train_sft.py` → `training/`.
- `joke_benefit.py`, `make_dataset_shards.py`, `concat_sft_datasets.py` → `datasets/em.py`.
- Paired-schedule parts of `prepare_massive_medical_union_pilot_data.py` → `datasets/massive.py`.
- `prepare_massive_medical_ratio_baselines_overnight_v1_union.py` → `datasets/weighted_union.py`.
- EM multi-reference, whole-output, merged and vLLM generation → `inference/em.py`,
  `whole_output.py`, `lora.py`, `direct.py`.
- `materialize_massive_medical_lora_merge_v1.py` → `inference/merge.py`.
- Final MASSIVE panel/direct/Kalai computational loops → `inference/massive.py`
  and `tasks/`. The frozen numerical helpers are retained as private package
  modules, not scheduler scripts.
- Broad EM scorer → `evaluation/em.py`; final ratio analysis → `evaluation/ratio.py`
  and `reports.py`; structured medical rubric/parser → `evaluation/medical.py`.
- Final ratio and selected EM renderer → `plotting/`; old pilot/sweep renderers
  are not included.

Historical collectors use SSH and private artifact layouts. They are not shipped
as local reproduction commands. The package reconstructs statistics from compact
outcome resources instead. Authorization ledgers, canary/recovery runners and
personal cluster launchers were not copied into the package additions. Existing
upstream files and Git history are preserved and require their own publication
review; this exclusion is not a whole-repository claim.

## Final result identities

Original private MASSIVE observation file SHA-256:
`382e473092a601aef53579c244324c83f7478e2cf14bb172f8d2ba621b8d7d0d`.

Original private MASSIVE results file SHA-256:
`7d880497581b9addd2ca0385e66f2affdefc5db1379f6da908f230dd17dedf2d`.

Original constituent file SHA-256:
`636208f7117c1d9f2296ddc21314297bef66b8655487110a4ab1da61a459f221`.

The bundled resources remove operational information and rebind their own seals.
The numerical-only projection hash checked by the regression test is
`d909d453304645e91f320ca3c04375bb52195e37d6dc30555df28aea469a72ca`.
These different seals represent different artifacts, not unexplained result changes.

## Explicit changes made during cleanup

1. Broad EM treats numeric zero as a score, matching the current figure. The
   stale Union summary's 7/320 becomes 8/320 on re-scoring. Archived files remain unchanged.
2. The trainer returns its resolved step budget, fixing a post-training metadata
   error when `max_steps` was derived rather than explicitly supplied.
3. New output stores reject incompatible recipe/data reuse and require appropriate
   checkpoint state for resume; no original output or authorization ledger is rewritten.
4. Generation commands require explicit local model paths and fresh output files.
   Training also supports Hub resolution (pinned when a revision is supplied)
   and validated resume. Old cluster scripts are not migrated merely by renaming
   them or wrapping subprocess calls.

The integrated package does not claim a complete historical artifact archive
or byte-identical reproduction of missing EM training banks.
