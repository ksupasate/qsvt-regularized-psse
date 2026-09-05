# Public Artifact Checklist

This checklist separates completed repository work from maintainer actions that
require release authority. The detailed evidence is in
`release_audit/final_clean_repository_validation.md` and
`release_audit/github_public_release_final.md`.

## Reproducibility

- [x] Python and direct dependency versions are documented.
- [x] Conda and pip installation paths are available.
- [x] A one-command lightweight reproduction workflow is available.
- [x] New runs are isolated from canonical scientific evidence by default.
- [x] Configurations are indexed in a machine-readable experiment manifest.
- [x] Random, bootstrap, and specialized sampling seeds are documented.
- [x] Standard and QSVT output schemas have validation code.
- [x] A deterministic IEEE14 quickstart and expected-output contract are present.
- [ ] External DOI-backed raw-data deposit has been assigned and linked.

## Scientific transparency

- [x] IEEE/PYPOWER network inputs are distinguished from generated measurements.
- [x] No PMU/SCADA field-data claim is made.
- [x] Diagonal covariance and row weighting are defined.
- [x] Noise, missing-row, bad-data, and weak-area assumptions are stated.
- [x] Exact Ridge/QSVT-target equivalence at identical `alpha` is stated.
- [x] QSVT is framed as a possible implementation pathway.
- [x] No quantum-speedup or quantum-advantage claim is made.
- [x] Circuit evidence is separated from full-scale hardware execution.
- [x] Failed and infeasible cases remain part of the scientific record.

## Repository hygiene

- [x] Manuscript sources, PDFs, drafting history, and submission packages are
  removed from the intended public working tree.
- [x] Local environments, caches, logs, editor files, and build products are
  ignored.
- [x] Citation, license, changelog, and release metadata exist.
- [x] Public secret, private-path, and large-file scans are recorded.
- [x] The clean candidate has no inherited Git history.
- [x] The clean candidate contains no concrete machine-local paths or files
  above GitHub's 100 MB limit.
- [ ] Citation author, repository URL, version, and DOI placeholders are filled
  by maintainers.

## Scientific preservation

- [x] No solver or estimator implementation was changed by this release pass.
- [x] No canonical experiment configuration was changed.
- [x] No canonical numerical result was rewritten.
- [x] No QSVT phase data, residual bank, support registry, or evidence registry
  was changed.
- [x] No full scientific campaign was rerun.

## Pre-publication maintainer gate

- [ ] Review and intentionally stage the desired public tree.
- [ ] Review the verdict in `release_audit/github_public_release_final.md`.
- [ ] Run scans and validation against the exact staged tree.
- [ ] Assign release version, repository URL, authors, and DOI.
- [ ] Create the commit, tag, remote, and GitHub release outside this preparation
  pass.
