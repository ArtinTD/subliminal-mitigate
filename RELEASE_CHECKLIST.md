# EM/MASSIVE integration and publication review

## Package additions prepared locally

- One installable `src/` package; no sibling checkout or historical script imports.
- Additive local integration based on collaborator commit
  `38226f0ae7f43157b8ee914577fb268bce682e09`; existing scripts, configurations and
  upstream Git history are preserved. The package does not replace the
  collaborator's unfiltered distillation workflow.
- Final EM and three-panel MASSIVE implementations, not older pilot defaults.
- Actual training/generation classes, thin CLI, source map and usage documentation.
- Outcome-only resources; original numerical results preserved under a new public projection seal.
- CPU tests and synthetic source-parity comparisons; installed-package reproduction outside the checkout.
- No model weights, raw medical responses, private datasets, operational ledgers,
  scheduler scripts, cluster logs or credentials were added as package resources.
  This is not a claim about files or history already in the collaborator repository.
- No push, remote deployment, GPU run or API call performed for this integration.

## Required before public release

- [ ] Authors/collaborator confirm the source license and attribution. The existing
  repository had no license file. No MIT/Apache grant has been assumed.
- [ ] Review the [source/rights notes](docs/sources_and_rights.md). Medical archive
  and original joke-bank redistribution rights have not been established; those
  data are deliberately absent from the package additions.
- [ ] Collaborator reviews the local integration diff and confirms that the
  package's distinct EM/MASSIVE interfaces are appropriate alongside the
  preserved training/distillation pipelines.
- [ ] Review the complete repository and preserved upstream history for secrets,
  private artifacts, attribution and redistribution restrictions before making
  any whole-repository public-release claim. A scan of the new package alone
  does not clear existing files or history.
- [ ] Run a small, separately budgeted GPU training/inference smoke test in the
  pinned environment, including XGrammar, PEFT reload of a merge and the real
  completion-loss collator. CPU fake-model parity is not a GPU validation claim.
- [ ] Decide whether to publish a paid judge transport. This package includes the
  exact structured rubric and parsers, not the old API-key/budget/recovery runners.
- [ ] Retain explicit reproducibility caveats: missing historical EM base revision
  and original joke/shard banks; MASSIVE input bank identity; unimplemented `s=3`
  historical appendix inference; EM artwork typography differences.
- [ ] Obtain explicit approval for the final reviewed commit and its publication
  destination before pushing. Approval for this local integration is not a
  publication authorization.

Review the integration diff, not a wholesale copy of the older experiment
worktrees. Existing upstream files remain part of this repository; they have
not been scrubbed or replaced by this cleanup. Generated package `build/`,
`dist/`, `*.egg-info/`, caches and new experiment outputs are ignored. Ignore
rules do not remove files that were already tracked upstream.

The focused package scan checks common key formats and machine-specific paths;
it is not a legal clearance, a guarantee of anonymity, or a history audit. The
local integration intentionally retains the collaborator's upstream Git history.
