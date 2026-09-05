# Final Pre-Commit Scientific Preservation Audit

Prepared: 2026-09-06
Location: the permanent repository, immediately before the author's initial commit.

Every staged file was hashed with SHA-256 and compared against **two** baselines:

1. the **curated development source** at `8895442eb2e80e4270c9ab39ca4ec54e6e7fa32b`;
2. the **previously verified public stage**, the tree audited and signed off in
   the preceding release pass.

## Headline result

| Scientific surface | Staged | Byte-identical to source | Verdict |
|---|---:|---:|---|
| `src/**` — the entire research package | 354 | **354** | **untouched** |
| `configs/**` — experiment and validation configurations | 109 | **109** | **untouched** |
| `examples/**` | 6 | **6** | **untouched** |
| `outputs/**` — shipped evidence | 343 | 336 | numerics untouched |
| `tests/**` | 396 | 351 | pruning documented in the previous pass |

Because `src/` and `configs/` are bit-for-bit equal to the audited source, the
following are **provably unchanged**:

- estimator definitions and the Ridge/Tikhonov implementation;
- QSVT and QSP algorithms, phase conventions, and phase data;
- measurement generation for synthetic, DC, AC-linearized, and nonlinear AC models;
- every `alpha` value, seed, and RNG rule;
- experiment configurations;
- residual banks, support registries, and scientific manifests;
- canonical numerical outputs.

## Whole-tree comparison against the curated source

```
1468  byte-identical
  68  differ (every one classified below)
  23  not present in source (release metadata created for the public artifact)
```

The 68 differing files break down as:

| Group | Count | Nature |
|---|---:|---|
| `tests/**` | 45 | public-boundary test-node pruning, audited and documented in the previous pass; unchanged this session |
| Root release metadata | 9 | `.gitignore`, `README.md`, `CITATION.cff`, `CHANGELOG.md`, `RELEASE.md`, `REPRODUCIBILITY.md`, `RESULTS_INDEX.md`, `MANIFEST.md`, `VERSION`, `pyproject.toml` |
| `release_audit/**` | 3 | audit records |
| `docs/**` | 3 | pointer and scope prose |
| `scripts/**` | 5 | release hygiene, itemized below |
| `outputs/**` | 2 | prose only, itemized below |
| `.gitignore` | 1 | evidence whitelist |

## Comparison against the previously verified public stage

```
1544  byte-identical
  12  differ  (this session's intentional edits)
   3  new     (this session's new audit records)
```

The 12 changed files are exactly this session's work, and nothing else:

| File | Change |
|---|---|
| `CITATION.cff` | authors, ORCID, repository URL, version 1.0.0 |
| `VERSION` | `v1.0.0-paper-artifact` → `1.0.0` |
| `pyproject.toml` | distribution version `0.1.0` → `1.0.0` |
| `README.md` | identity header, `External Data` and `License` sections, `Known Limitations` rename, resolved citation |
| `RELEASE.md` | release identity, version-metadata table, resolved status |
| `CHANGELOG.md` | `[Unreleased]` → `[1.0.0]`, limitations updated |
| `docs/REPRODUCIBILITY.md` | artifact-identity version lines |
| `scripts/continue_final_qsvt_feasibility_push.py` | one work-package `evidence` string (AI-workflow trace removed) |
| `release_audit/doi_metadata_alignment.md` | placeholder table marked resolved |
| `release_audit/final_privacy_scan.md` | author-metadata and URL paragraphs updated |
| `release_audit/github_public_release_final.md` | metadata status updated |
| `release_audit/metadata_completion_checklist.md` | rewritten as resolved |

New records: `permanent_repository_migration.md`,
`ai_provenance_surface_audit.md`, `github_metadata_recommendation.md`
(plus this file and the final surface/inventory reports).

## Approved hygiene edits inside code (5 files)

| File | Change | Scientific effect |
|---|---|---|
| `scripts/reproduce_all.sh` | exports `PYTHONPATH` | none — makes the quickstart work without an editable install |
| `scripts/validate_outputs.py` | accepts `config_resolved.omission.json` | none — privacy substitute for five snapshots with a machine-local `output.root` |
| `scripts/build_final_cross_case_integration.py` | removed unused `import math`; removed an `f` prefix with no placeholders | none — dead code |
| `scripts/finalize_generic_sparse_qsvt_compiler.py` | removed unused `import math` | none — dead code |
| `scripts/continue_final_qsvt_feasibility_push.py` | one work-package status string | none — feeds a non-shipped Markdown table; no test references it |

## Approved prose edits inside shipped evidence (2 files)

| File | Change |
|---|---|
| `outputs/reproducibility_audit/claim_scope_validation.md` | one framing sentence shortened; audited commit, every boundary row, and the `PASS` verdict unchanged |
| `outputs/generated/README.md` | rerun-workspace policy rewritten for a public audience |

Neither is a numerical artifact.

## Package version deliberately not raised

`robust_qsvt_se.__version__` remains `0.1.0` while the released artifact version
is `1.0.0`. `__version__` is stamped into the `package_version` field of 20
shipped evidence manifests and resolved configs; raising it would desynchronize
those frozen records from the code that produced them and would break the
byte-identical guarantee over `src/`.

This was tested rather than assumed: with `__version__` temporarily set to
`1.0.0`, every manifest/provenance test still passed (471 passed, 10 skipped).
The change was then **reverted**, and `src/robust_qsvt_se/__init__.py` was
verified byte-identical to the curated source
(`557c15f9f8fce2ce155b44ef175a69207b4b950bbdbdbfeb50d34f2ab4a5d5a2`).

## Test-generated drift

Some public tests write into canonical `outputs/` directories. After all
verification runs in this session:

- staged `outputs/` files differing from the curated source: **2**, both the
  documented prose edits above;
- occurrences of `git_commit: null` in staged evidence: **0**;
- `package_version` values in staged manifests: **all `0.1.0`**, so no
  version-test residue reached the staged set;
- files modified after staging: **0**.

Regenerated but **untracked** directories (`outputs/qsvt_matrix_free_action/`,
`qsvt_scalable_qubit_convention_audit/`, `qsvt_sparse_access_oracle/`,
`qsvt_state_preparation_model/`, `qsvt_success_amplitude_proxy/`,
`qsvt_toy_sparse_oracle_circuit/`, `selected_observable_qsvt_demo/`,
`tqe_revision_experiments/`) are `.gitignore`d and cannot enter the commit.

## Conclusion

**No algorithm, estimator, configuration, alpha, seed, phase dataset, residual
bank, support registry, experiment manifest, or numerical result was changed by
any release work.** Every difference from the curated source is classified and
documented above.
