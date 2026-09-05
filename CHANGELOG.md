# Changelog

All notable public-artifact changes will be documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project intends to use semantic versioning for public software releases.

## [1.0.0]

Initial public release of the reproducibility artifact. The release date is
assigned when the author tags `v1.0.0` and publishes the Zenodo record; no date
is recorded here in advance.

### Added

- Public repository identity and publication-quality entry documentation.
- Conda and pip environment specifications for the recorded Python 3.12 stack.
- One-command lightweight reproduction workflow.
- Deterministic IEEE14 quickstart example and expected-output guide.
- Machine-readable experiment manifest and public-safe validation records.
- Data-access policy for raw artifacts that exceed ordinary Git hosting limits.
- Citation metadata with resolved authors, ORCID, repository URL, and version,
  plus release/transformation audits.
- Reserved Zenodo DOI `10.5281/zenodo.22326883` recorded in `CITATION.cff`
  and `data_manifest.json` for the external raw-data deposit.
- The 41 `outputs/` evidence roots that the test suite and the reproduction
  validator read are versioned, so a bare clone passes both without any
  download. QSP phase-angle caches inside those roots are versioned with them.
- `outputs/generated/README.md`, documenting the rerun workspace policy.

### Changed

- Public documentation now states the benchmark-data, generated-measurement,
  QSVT-feasibility, and no-speedup boundaries explicitly.
- Generated reruns are isolated under `outputs/examples/` or
  `outputs/generated/` rather than canonical evidence directories.

### Removed

- Manuscript sources and PDFs, manuscript drafting history, journal submission
  packages, local assistant state, temporary builds, and internal notes from the
  intended public repository surface.

### Known limitations

- Several raw scientific output files require an external DOI-backed archive.
  The DOI `10.5281/zenodo.22326883` is reserved for that deposit, but the
  Zenodo record is still an unpublished draft, so the files are not yet publicly
  archived or downloadable.
- ORCIDs for Dhana Phassadawongse and Stephen John Turner are not yet supplied.
  No release date is recorded until the author tags the release.
- `robust_qsvt_se.__version__` remains `0.1.0` while the released artifact
  version is `1.0.0`. `__version__` is stamped into the `package_version` field
  of 20 frozen evidence manifests, so raising it would desynchronize those
  records from the code that produced them.
