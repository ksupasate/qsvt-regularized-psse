# Final Initial-Commit Inventory

Prepared: 2026-09-06

Everything below is **staged but not committed**. The author performs the
commit and push.

## Git state

| Item | Value |
|---|---|
| Repository | `qsvt-regularized-psse` |
| Location | permanent workspace directory (moved off temporary storage) |
| Branch | `main` |
| Remote `origin` | `git@github.com:ksupasate/qsvt-regularized-psse.git` |
| Other remotes | **none** — the development repository is not a remote |
| Old history present? | **no** — fresh `git init`; empty reflog; 0 commits |
| Commits | **0** |
| Tags | **0** |
| Staged files | **1562** |
| Staged size | 119,959,047 bytes (114.4 MiB) |
| Largest staged file | 19.39 MiB — `outputs/final_contribution_evidence/artifact_dependency_graph.json` |
| Files not staged | **0** |

## Staged tree

| Path | Files | Role |
|---|---:|---|
| `tests` | 396 | public scientific test suite |
| `src` | 354 | research package (`robust_qsvt_se` + compat shim) |
| `outputs` | 343 | versioned evidence read by the suite and validator |
| `scripts` | 284 | experiment, validation, and reproduction entry points |
| `configs` | 109 | fixed experiment and validation configurations |
| `docs` | 33 | scope, method, data, and execution guides |
| `release_audit` | 22 | clean-export and pre-commit verification records |
| `<root file>` | 14 | identity, license, citation, environment, and release metadata |
| `examples` | 6 | deterministic lightweight examples |

## Exclusion and safety checks

| Check | Result |
|---|---|
| Files > 100 MB | **0** |
| Files > 50 MB | **0** |
| Files > 25 MB | **0** |
| Files > 20 MB | **0** |
| Manuscript files (`manuscript*`, `*.tex`, `*.pdf`, LaTeX products) | **0** |
| Supplementary material | **0** |
| Submission packages / archives of any kind | **0** |
| Old ~528 MB submission blob | **absent** |
| Private / machine-local paths in staged files | **0 unexplained** |
| OS account name, hostname, email addresses | **0** |
| Secrets and credentials | **0** |
| AI prompt / transcript / handoff artifacts | **0** |
| Caches, `__pycache__`, `*.pyc`, `*.log`, `.DS_Store` | **0** |
| Virtual environments | **0** |
| Test-generated drift (`git_commit: null`) | **0** |

## Scientific preservation

| Surface | Staged | Byte-identical to curated source |
|---|---:|---:|
| `src/**` | 354 | **354** |
| `configs/**` | 109 | **109** |
| `examples/**` | 6 | **6** |
| `outputs/**` | 343 | 336 (7 documented: 5 privacy substitutes, 2 prose) |

No algorithm, estimator, alpha, seed, configuration, phase dataset,
residual bank, support registry, or numerical result was changed.

## DOI and metadata consistency

| Field | Value |
|---|---|
| Reserved Zenodo DOI | `10.5281/zenodo.22326883` |
| Zenodo record status | **unpublished draft** — never described as archived or available |
| DOI in `CITATION.cff` | matches |
| DOI in `data_manifest.json` | matches, all 6 datasets + top-level `reserved_doi` |
| Released version | `1.0.0` in `VERSION`, `pyproject.toml`, `CITATION.cff` |
| `robust_qsvt_se.__version__` | `0.1.0`, deliberately unchanged (stamps frozen evidence) |
| Repository URL | `https://github.com/ksupasate/qsvt-regularized-psse` in `CITATION.cff` and `README.md` |
| Authors | Vorathammathorn; Phassadawongse; Turner — order as supplied |
| ORCID | one supplied and recorded; two absent and not invented |
| Release date | absent — not invented |
| Remaining `<TO_BE_ASSIGNED>` / `<OWNER>` placeholders | **0** |

## Test and validator status

| Check | Result |
|---|---|
| Reproduction validator | 144 checks, 6 explained warnings, **0 failures** |
| IEEE14 quickstart | **PASS**, 8 artifacts |
| Ridge/QSVT equivalence guard | **8 passed** |
| Manifest / provenance tests | **471 passed, 10 skipped** |
| Reviewer-named subpackage tests | **54 passed** |
| Full public suite (previous pass, unchanged code) | 1702 collected, 1670 passed, 32 skipped, **0 failed** |
| Release-facing Ruff, pyflakes, compileall, 350/350 imports | clean |

## Not performed

No commit. No push. No tag. No GitHub release. No Zenodo upload or
publication. No fresh-clone verification after push. The development
repository was read-only throughout.
