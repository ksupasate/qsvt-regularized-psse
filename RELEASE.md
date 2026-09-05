# Research Artifact Release

Repository: `qsvt-regularized-psse`  
Repository URL: <https://github.com/ksupasate/qsvt-regularized-psse>  
Release version: `1.0.0` (future tag `v1.0.0`, not yet created)  
Source baseline: `8895442eb2e80e4270c9ab39ca4ec54e6e7fa32b`  
Reserved Zenodo DOI for this reproducibility release: `10.5281/zenodo.22326883`  
License: MIT  
Status: **prepared and staged for the author's initial public commit**

## Version metadata

| Field | Value | Meaning |
|---|---|---|
| `VERSION`, `pyproject.toml`, `CITATION.cff` | `1.0.0` | the released artifact version |
| `robust_qsvt_se.__version__` | `0.1.0` | the package version that produced the frozen evidence |

`__version__` is deliberately left at `0.1.0`. It is stamped into the
`package_version` field of 20 shipped evidence manifests and resolved configs,
so raising it would desynchronize those frozen records from the code that
generated them. Leaving it preserves accurate provenance and keeps `src/`
byte-identical to the audited scientific source. Raising it was verified not to
break any test (471 passed, 10 skipped), so it can be done deliberately in a
later release together with regenerated evidence.

This directory is a plain-filesystem export with no inherited Git history. The
release-preparation process does not create a tag, remote, GitHub repository, or
published release, and it does not publish the Zenodo record. The DOI
`10.5281/zenodo.22326883` is *reserved* against an unpublished Zenodo
draft; it is recorded in the release metadata but does not yet resolve to a
downloadable archive.

## Reproducibility status

The repository provides pinned direct dependencies, fixed configurations,
documented seeds, isolated execution wrappers, a lightweight IEEE14 example,
output-schema validation, and a machine-readable experiment manifest. Exact
clean-export validation results are recorded in
`release_audit/final_clean_repository_validation.md`.

## Scientific scope

IEEE/PYPOWER inputs are benchmark network models. Measurement rows are generated
by code; no PMU or SCADA field data are used. QSVT is studied as a possible
implementation pathway for the regularized spectral filter, without a quantum
speedup claim. With identical inputs and `alpha`, the exact QSVT target and
Ridge/Tikhonov are numerically equivalent in the classical simulator.

## External data

Six raw scientific datasets are too large for ordinary Git hosting. Their sizes,
SHA-256 checksums, producing configurations, and the reserved DOI
`10.5281/zenodo.22326883` are recorded in `data_manifest.json`. The
retrieval location remains unresolved because the Zenodo record has not been
published. These datasets are not required for the lightweight quickstart but
are required for complete evidence-level reproduction.

## Known limitations

- Full scientific campaigns are not run by the one-command workflow.
- The external raw-data archive has a reserved DOI (`10.5281/zenodo.22326883`)
  but has not yet been published; its download location remains unresolved.
- ORCIDs for Dhana Phassadawongse and Stephen John Turner are not yet
  supplied, and no release date is recorded. Author names and order, the
  repository owner, the release version, and the DOI are all resolved.
- Platform BLAS/LAPACK implementations and quantum-library backends can cause
  small floating-point or runtime differences.

## Release gate

Use `release_audit/github_public_release_final.md` and
`release_audit/final_public_surface_audit.md` as the authoritative readiness
decisions, and `release_audit/final_initial_commit_inventory.md` for the exact
staged set. The repository is staged but **not** committed: no commit, tag,
remote push, GitHub release, or Zenodo publication has been performed.
