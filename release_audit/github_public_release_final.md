# GitHub Public Release — Final Decision

Prepared: 2026-09-05
Candidate: `qsvt-regularized-psse`
Source baseline: `8895442eb2e80e4270c9ab39ca4ec54e6e7fa32b` on
`research/generalized-rectangular-qsvt`
Reserved Zenodo DOI: `10.5281/zenodo.22326883` (record is an unpublished draft)

---

## Verdict

# READY WITH DOCUMENTED NONBLOCKING LIMITATIONS

Every required criterion is met. Four limitations remain, all documented below,
none of them blocking an initial public commit.

---

## Required criteria

| # | Criterion | Result | Met |
|---:|---|---|:---:|
| 1 | Complete exact-tree pytest, zero unexplained scientific failures | 1702 collected, 1670 passed, 32 skipped, **0 failed**, 967.6 s | yes |
| 2 | Affected tests pass | 45 affected files: 129 passed, 1 skipped, **0 failed** | yes |
| 3 | Validator passes | 144 checks, 6 explained warnings, **0 failures** | yes |
| 4 | IEEE14 quickstart passes | **PASS**, 8 artifacts, 17.8 s | yes |
| 5 | Ridge/QSVT equivalence passes | max ΔRMSE **exactly 0** over 336 matched points; boundary report 1.43e-16 against a 1e-10 tolerance | yes |
| 6 | All configurations parse | 100 YAML + 145 JSON, **0 errors** (`configs/`: 84 YAML + 24 JSON) | yes |
| 7 | Release-facing Ruff and import checks pass | Ruff: all checks passed; pyflakes clean repo-wide; 350/350 modules import; 2325/2325 import references resolve | yes |
| 8 | Privacy scan has zero blocking findings | 0 secrets, 0 credentials, 0 emails, 0 unexplained machine-local paths | yes |
| 9 | No files > 100 MB | 0 files > 25 MB, > 50 MB, or > 100 MB; largest is 19.4 MiB | yes |
| 10 | Old ~528 MB history/blob absent | confirmed absent three ways; located and identified in the development history only | yes |
| 11 | No manuscript/submission/reviewer material | 0 staged hits across 16 exclusion patterns | yes |
| 12 | Scientific preservation audit passes | **0 unexplained differences** in 1556 files; `src/` 354/354 and `configs/` 109/109 byte-identical | yes |
| 13 | DOI metadata consistent | 24 locations, all reserved-DOI wording; no false "publicly archived" claim | yes |
| 14 | External-data manifest complete | 6 datasets with size, SHA-256, producing config, archive filename, reserved DOI, pending-publication status | yes |
| 15 | Fresh Git history used | `git init` on `main`; no inherited `.git`, no remote, no commit, no tag | yes |

**Additional criterion applied beyond the required list:** the suite, validator,
and quickstart were re-run on a **clean-clone equivalent** — only the files Git
would track. All three pass there identically. The published repository verifies
itself offline.

---

## Defects found and fixed during this pass

| # | Defect | Severity | Resolution |
|---:|---|---|---|
| 1 | A fresh clone failed **181 tests with 97 errors** across 77 files, and **19 validator checks**, because `.gitignore` excluded `outputs/` evidence the suite reads | would have shipped broken | Narrowed the `outputs/**` exclusion to version exactly the evidence the checks read: 41 roots + their QSP phase caches + the five registered smoke-test snapshots. No test weakened, skipped, or deleted; no scientific file modified. |
| 2 | `scripts/export_latex_assets.py` raised `ModuleNotFoundError` — it imported a module the public boundary removes | dead code in public tree | Removed and recorded in the exclusion manifest. A sweep then confirmed 0 broken import references remain. |
| 3 | `outputs/qsvt_oracle_model_resources/manifest.json` was rewritten in place by a verification run, replacing its provenance with `git_commit: null` | evidence provenance | Restored from the curated source; re-verified byte-identical. |
| 4 | This session's own `release_audit/takeover_state.md` recorded the OS account name and absolute workspace paths | privacy | Rewritten home-relative with a `<staging-root>` placeholder; re-scanned clean. |
| 5 | The committed JUnit XML carried the machine hostname | privacy | Hostname and home/staging paths redacted; all test names, statuses, durations, and totals kept verbatim. |
| 6 | Two build scripts carried dead code (`import math`, an `f`-string with no placeholders) | lint | Removed; pyflakes is now clean repo-wide. |
| 7 | Documentation asserted that `outputs/` roots were excluded that are now shipped, and cited a `outputs/generated/README.md` that did not exist | doc accuracy | Corrected across `README.md`, `RESULTS_INDEX.md`, `MANIFEST.md`, `CHANGELOG.md`, and both release manifests; the missing README was written. |

---

## Non-blocking limitations

1. **Running the suite mutates a few `outputs/` directories.** Some tests
   default `output_dir` to the canonical path rather than a temporary one, so
   after `pytest` a fresh clone shows modifications under a few evidence roots.
   A user should discard them. Pre-existing suite behavior, not introduced here.
2. **308 cosmetic Ruff findings repo-wide** — 243 `E501`, 20 `E402`, 18 `I001`,
   16 `RUF046`, and a few others, concentrated in `scripts/`. No release-facing
   entry point is affected and no correctness rule fires. Deliberately not
   rewritten: a mass reformat of scientific scripts is a worse risk than the
   debt.
3. **Three hardcoded `/tmp/...` scratch destinations** in non-entry-point code,
   a portability wart on Windows. No privacy implication.
4. **The six raw datasets remain externally archived.** They are required only
   for complete evidence-level reproduction, and **not** for the test suite, the
   validator, or the quickstart. Their Zenodo record is a reserved draft.

## Explicitly not blockers

- The reserved DOI and the unpublished Zenodo draft.
- The author, version, and repository placeholders that stood when this report
  was written. They have since been resolved from author-supplied values.

---

## Maintainer metadata — status as of 2026-09-06

Resolved from author-supplied values after this report was first written:

| Item | Final value |
|---|---|
| Authors and order | Supasate Vorathammathorn; Dhana Phassadawongse; Stephen John Turner |
| ORCID | `https://orcid.org/0009-0009-2751-1023` (Vorathammathorn) |
| GitHub owner | `ksupasate` |
| Repository URL | `https://github.com/ksupasate/qsvt-regularized-psse` |
| Released version | `1.0.0` |
| DOI | `10.5281/zenodo.22326883` (reserved; Zenodo record still a draft) |

Still open, all non-blocking: ORCIDs for Phassadawongse and Turner (not
supplied), and the release date (assigned when the author tags `v1.0.0`).
See `release_audit/metadata_completion_checklist.md`.

---

## Actions deliberately not taken

No commit, tag, remote, fetch, pull, or push. No GitHub repository or release.
No Zenodo publication. No old history imported. No scientific algorithm,
estimator, alpha, seed, configuration, phase dataset, residual bank, support
registry, or numerical result altered. The development repository was read-only.

---

## Recommended next steps for the maintainer

1. Fill the author, ORCID, owner, repository-URL, and version fields.
2. Review the staged inventory in `release_audit/pre_commit_release_inventory.md`.
3. Create the initial commit on `main`, add the remote, and push.
4. Publish the reserved Zenodo record, then set `retrieval_location` in
   `data_manifest.json` and update the pending-publication wording.
5. **Done (2026-09-06):** the candidate was moved off the platform temporary
   directory into its permanent location. See
   `release_audit/permanent_repository_migration.md`.
