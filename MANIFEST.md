# Research Artifact Manifest

This file maps the public repository surface to its role in reproducibility.
File-level experiment provenance is stored in the manifests within each output
family and in `outputs/reproducibility_audit/experiment_manifest.json`.

## Maintained artifact surface

| Scope | Role | Authority |
|---|---|---|
| `src/robust_qsvt_se/` | Estimators, measurement generation, QSVT components, and utilities | Maintained implementation |
| `configs/` | Fixed experiment and campaign inputs | Configuration provenance |
| `scripts/run_experiment.py` | Isolated config dispatcher | Standard experiment entry point |
| `scripts/run_smoke_test.py` | Lightweight multi-model smoke suite | Installation check |
| `scripts/reproduce_all.sh` | Environment, validation, and IEEE14 example workflow | Public reproduction entry point |
| `scripts/validate_reproduction.py` | Environment, manifest, schema, and regeneration checks | Artifact validator |
| `tests/` | Numerical, integration, and evidence checks | Regression evidence |
| `docs/` | Scientific scope, method, data, and execution guidance | Interpretive documentation |
| `examples/` | Deterministic lightweight configurations and output contracts | First-run examples |

## Curated Git evidence

A fresh public tree carries the path-safe evidence that the repository's own
checks read, so `pytest` and `scripts/validate_reproduction.py` both work on a
bare clone. That is 41 `outputs/` roots: 25 asserted by the test suite and 16
registered benchmark and resource outputs read by the validator and the
cost-accounting checks. The exact list is in
[`RESULTS_INDEX.md`](RESULTS_INDEX.md#which-output-paths-are-in-a-git-checkout);
the authoritative rule is the `outputs/` section of `.gitignore`.

These names are historical provenance identifiers. They are not claims that a
phase, feasibility check, or selected block establishes full-system quantum
execution.

Roots that remain excluded are those the public checks do not read — for example
`outputs/artifact_manifest/` and `outputs/final_qsvt_feasibility_push/`, which
hold legacy manuscript/review-process records. Every included file was scanned
and contains no machine-local path. Nothing under `outputs/` is deleted or
rewritten by the release process, because canonical scientific outputs are
protected.

## Generated and external data

`outputs/examples/` and `outputs/generated/` are isolated workspaces for new
runs. They are not canonical evidence. High-volume raw tables are kept out of
ordinary Git and require a DOI-backed research-data deposit. Reserved Zenodo DOI
for that deposit: `10.5281/zenodo.22326883` (record still a draft, not yet
published). Sizes and SHA-256 digests are listed in `docs/data_access.md`.

## Claim boundary

IEEE/PYPOWER supplies benchmark network models, and code generates all
measurement rows. No PMU/SCADA field data are used. QSVT is evaluated as a
possible implementation pathway for the regularized spectral filter, without a
quantum-speedup claim. With identical inputs and `alpha`, Ridge/Tikhonov and the
exact QSVT target are the same classical spectral computation.

## Integrity rule

Do not silently refresh canonical outputs, phase data, residual banks, support
registries, or evidence registries. Regenerated work belongs under
`outputs/generated/` until it has an explicit version, provenance record, and
reviewed promotion decision.
