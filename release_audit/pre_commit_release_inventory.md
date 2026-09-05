# Pre-Commit Release Inventory

Prepared: 2026-09-05

The proposed initial public commit has been **staged and inspected but
NOT committed**. No commit, tag, remote, push, GitHub repository, or Zenodo
publication was created.

## Git state

| Item | Value |
|---|---|
| Repository | freshly initialized in the clean candidate (`git init`) |
| Inherited history | **none** — the development repository's `.git` was never copied |
| Branch | `main` |
| Remotes configured | **0** |
| Commits | **0** (nothing committed) |
| Tags | **0** |
| Staged files | **1556** |
| Staged bytes | 119,926,800 (114.4 MiB) |
| Unstaged / untracked leftovers | none |

## Staged tree

| Path | Files | Role |
|---|---:|---|
| `tests` | 396 | public scientific suite |
| `src` | 354 | research package (`robust_qsvt_se`) |
| `outputs` | 343 | versioned evidence the suite and validator read |
| `scripts` | 284 | experiment and reproduction entry points |
| `configs` | 109 | fixed experiment and validation configurations |
| `docs` | 33 | scope, method, data, and execution guides |
| `release_audit` | 17 | clean-export verification records |
| `<root file>` | 14 | identity, license, citation, environment, and release metadata |
| `examples` | 6 | deterministic lightweight examples |

## Exclusion verification against the staged set

Each pattern was matched against the exact staged path list:

| Excluded surface | Staged hits |
|---|---:|
| `manuscript/` path component | **0** |
| `submission_package*` | **0** |
| `supplementary*` | **0** |
| `final_manuscript_correction` | **0** |
| LaTeX sources (`.tex`, `.bbl`, `.cls`, `.sty`, `.aux`, `.blg`, `.fls`, `.toc`) | **0** |
| PDFs | **0** |
| Archives (`.zip`, `.tar`, `.tar.gz`, `.tgz`, `.7z`, `.rar`) — incl. the ~528 MB historical ZIP | **0** |
| Caches (`__pycache__`, `.pytest_cache`, `.ruff_cache`, `.mypy_cache`) | **0** |
| Compiled Python (`.pyc`, `.pyo`) | **0** |
| Log files (`.log`) | **0** |
| Assistant/agent workflow files (`AGENTS.md`, `.claude`, `.codex`, `HANDOFF.md`) | **0** |
| Private archives (`_archive_not_for_github`, `archive/`, `backup/`, `draft/`) | **0** |
| OS cruft (`.DS_Store`, `Thumbs.db`) | **0** |
| Virtual environments (`.venv/`, `venv/`, `env/`) | **0** |
| `results/` (excluded root) | **0** |
| Local temporary output (`outputs/generated/*`, `outputs/examples/ieee14_quickstart`) | **0** |

## One naming clarification

Twenty staged paths contain the word *reviewer*:

```
configs/reviewer_blocking_tqe_evidence/   (5 JSON configs)
src/robust_qsvt_se/reviewer_blocking/     (7 modules)
src/robust_qsvt_se/reviewer_evidence/     (8 modules)
```

These are **scientific implementation subpackages** named after the review
round that motivated them — physical functionals, joint feasibility,
resource Pareto fronts, exact-loss baselines, high-degree studies. They are
byte-identical to the curated source and are covered by six dedicated test
files that all pass. They contain no reviewer correspondence, quotations, or
private notes; a targeted scan for referee/rebuttal/reviewer-quote text
returned zero hits. The reviewer *delivery* scripts
(`run_reviewer_evidence_all.py`, `build_reviewer_blocking_manifest.py`,
`reviewer_evidence_hash_snapshot.py`, `run_tqe_reviewer_revision_audit.py`)
are excluded and are not staged.

## Deliberately not done

- No commit was created.
- No tag was created.
- No remote was added; no `git fetch`, `git pull`, or `git push` was run.
- No GitHub repository or release was created.
- The Zenodo record was **not** published; its DOI is reserved only.
- The development repository was read-only throughout.
