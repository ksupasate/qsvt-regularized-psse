# Takeover State — Public Release Continuation

Recorded: 2026-09-05 (session continuation after usage-limit interruption)

This file records the state of the public release candidate **before** any edit
was made by the continuing session.

## Source repository (verified)

| Item | Expected | Observed | Match |
|---|---|---|:---:|
| Path | `~/…/VISTEC_Paper/QSVT_paper` (home-relative; absolute path deliberately not recorded) | same | yes |
| Branch | `research/generalized-rectangular-qsvt` | same | yes |
| HEAD | `8895442eb2e80e4270c9ab39ca4ec54e6e7fa32b` | same | yes |

The source repository is **not** pushed and its Git history is **not** reused.

## Public release candidate (verified)

| Item | Value |
|---|---|
| Candidate path | `<staging-root>/qsvt-regularized-psse-public-stage` |
| Newer permanent candidate found? | no — this is the only clean-release tree present |
| `.git` present? | **no** (fresh history still to be initialized) |
| Files (excluding `.git`) | 2327 |
| Directories | 134 |
| On-disk size | 131 MB |
| Candidate mtime (root) | 2026-09-05 13:01 |
| Newest content mtime | 2026-09-05 13:13 (`outputs/`, `scripts/`, `tests/`) |

### Top-level inventory

```
.gitignore  CHANGELOG.md  CITATION.cff  LICENSE  MANIFEST.md  README.md
RELEASE.md  REPRODUCIBILITY.md  RESULTS_INDEX.md  VERSION
data_manifest.json  environment.yml  pyproject.toml  requirements.txt
configs/  docs/  examples/  outputs/  release_audit/  scripts/  src/  tests/
.pytest_cache/  .ruff_cache/            (both git-ignored)
```

### Component counts

| Component | Count |
|---|---|
| `tests/test_*.py` | 393 |
| `src/**/*.py` | 354 |
| `scripts/**` files | 288 |
| `configs/**` files | 97 entries |
| `outputs/**` files (on disk) | 389 |
| `examples/**` files | 6 |

## Forbidden-content re-verification (pre-edit)

Scanned the exact candidate tree. All of the following returned **zero hits**:

| Pattern | Hits |
|---|---:|
| `manuscript*` (any path component) | 0 |
| `submission_package*` | 0 |
| `supplementary*` | 0 |
| `*.tex`, `*.bbl`, `*.cls`, `*.sty` | 0 |
| `*.pdf` (incl. `main.pdf`) | 0 |
| `*.zip` (incl. the historical ~528 MB archive) | 0 |
| `.git/` inherited from the development repository | 0 (absent) |

No manuscript, supplementary, submission-package, or reviewer-audit surface has
reappeared in the candidate.

## Inherited release_audit records found

Present before this session:

- `release_audit/final_public_release_manifest.txt`
- `release_audit/final_public_exclusion_manifest.txt`
- `release_audit/metadata_completion_checklist.md`

Declared by the release manifest but **not yet written** when the previous
session was interrupted (this session must produce them):

- `public_export_mapping.csv`
- `public_scientific_hash_verification.json`
- `public_scientific_preservation_report.md`
- `public_test_scope_report.md`
- `final_privacy_scan.md`
- `final_large_file_report.md`
- `final_clean_repository_validation.md`
- `pre_commit_release_inventory.md`
- `github_public_release_final.md`

## Verified execution environment

| Item | Value |
|---|---|
| Interpreter | ``~/…/VISTEC_Paper/QSVT_paper/.venv/bin/python` (development workspace venv)` |
| Python | 3.12.11 |
| pytest | 9.0.3 |
| Package resolution inside candidate | `pythonpath = ["src"]` in `pyproject.toml` resolves `robust_qsvt_se` to the **candidate** tree (`<staging-root>/qsvt-regularized-psse-public-stage/src/...`), not the development repository |
| Required env | `MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1` |

Note: the shared venv also carries an editable install pointing at the
development repository. Every verification command in this session is run with
the candidate as CWD so the candidate's `pythonpath` entry wins; import origin
was confirmed explicitly before the test runs.

## Edits made before this record

None. This state was captured before any modification.

## Path-recording convention

Machine-local absolute paths and the operating-system account name are
deliberately **not** written into this public record. Workspace locations are
given home-relative (`~/…`) or against a `<staging-root>` placeholder. This keeps
the provenance meaningful while satisfying the release privacy requirement of
zero concrete machine-local paths in tracked files.
