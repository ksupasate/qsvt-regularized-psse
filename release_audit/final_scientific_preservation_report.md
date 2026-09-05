# Final Scientific Preservation Report

Prepared: 2026-09-05
Method: SHA-256 of every file Git would track in the public candidate, compared
against the same path in the curated source working tree at baseline commit
`8895442eb2e80e4270c9ab39ca4ec54e6e7fa32b`.

Machine-readable companions:

- `release_audit/public_export_mapping.csv` — one row per public file with both
  digests, byte size, status, and note.
- `release_audit/public_scientific_hash_verification.json` — the same data plus
  aggregate status counts.

## Headline result

**Every scientific surface is byte-identical to the curated source, and there
are zero unexplained differences across all 1,556 tracked files.**

| Surface | Files | Byte-identical | Modified/created | Verdict |
|---|---:|---:|---:|---|
| `src/robust_qsvt_se/**` — the entire research package | 354 | **354** | 0 | untouched |
| `configs/**` — experiment and validation configurations | 109 | **109** | 0 | untouched |
| `examples/**` | 6 | **6** | 0 | untouched |
| `outputs/**` — shipped evidence | 343 | **336** | 5 privacy substitutes, 2 prose | numerics untouched |
| `tests/**` | 396 | 351 | 45 (pruning + lint) | scientific assertions preserved |
| `docs/**` | 33 | 30 | 3 | pointer/scope prose only |
| `scripts/**` | 284 | 280 | 4 | 2 portability, 2 dead-code removal |
| root metadata | 14 | 4 | 10 | release metadata by design |
| `release_audit/**` | 17 | 0 | 17 | audit records created this pass |

Status counts over the whole tracked set:

```
1470  identical
  45  modified_test_pruning
  15  created_release_metadata
   9  modified_release_metadata
   5  created_privacy_substitute
   4  modified_documentation
   4  modified_code_hygiene
   3  modified_release_audit
   1  modified_evidence_prose
   0  UNEXPLAINED
```

No estimator definition, alpha value, seed, RNG rule, phase dataset, residual
bank, support registry, experiment manifest, or numerical result was altered.
Because `src/` and `configs/` are bit-for-bit equal, the algorithms and the
inputs that drive them are provably unchanged.

## One evidence file was regenerated in place, and was restored

Running the public test suite writes into several canonical `outputs/`
directories, because those tests default their `output_dir` to the canonical
path rather than a temporary one. During this session's verification runs that
regenerated `outputs/qsvt_oracle_model_resources/`.

Three of its four files came back **byte-identical**, which is itself useful
evidence that the generator is deterministic. The fourth, `manifest.json`,
differed only in provenance metadata: `command` recorded the release-audit
pytest invocation, `generated_at` moved to the audit date, and `git_commit`
became `null` because the candidate had no Git history yet.

All four files were **restored from the curated source** and re-verified
byte-identical, so the shipped manifest carries its true provenance
(`git_commit: 8895442eb2e80e4270c9ab39ca4ec54e6e7fa32b`). No numerical column
was ever affected.

This in-place-write behavior is a pre-existing property of the suite, not
something introduced by the release. It is recorded as a non-blocking
limitation: after running `pytest`, `git status` in a fresh clone will show
modifications under a few `outputs/` roots, which a user should discard rather
than commit.

## The one modified file inside shipped evidence

`outputs/reproducibility_audit/claim_scope_validation.md` — a **validation
record**, not a numerical artifact. One sentence of framing was shortened:

```diff
-This check covers the public repository documentation after removal of the
-manuscript and historical submission package. It does not reinterpret or alter
-scientific results.
+This check covers the public repository documentation. It does not reinterpret
+or alter scientific results.
```

The audited baseline commit, every boundary row, every evidence pointer, and the
overall `Result: **PASS**` are unchanged. This is an intentional sanitized
derivative and is **not** claimed to be byte-identical.

## Every other intentional edit, itemized

### Privacy substitutes inside shipped evidence (5 files)

`outputs/examples/smoke_test/*/config_resolved.omission.json` — five records
created by the previous session. Each replaces a resolved-config snapshot whose
`output.root` recorded the development checkout, and each names the source
config path, its SHA-256, the source commit, and the exact regeneration command.
`scripts/validate_outputs.py` accepts them in place of the snapshot, so
provenance is still checked rather than skipped.

### Release-hygiene edits to code (4 files, no scientific behavior change)

| File | Change | Why |
|---|---|---|
| `scripts/reproduce_all.sh` | added `export PYTHONPATH="$repo_root/src…"` | the quickstart must work for an external user who has not run `pip install -e .` |
| `scripts/validate_outputs.py` | accepts `config_resolved.omission.json` in place of an omitted snapshot | five legacy snapshots recorded a machine-local `output.root`; this is the privacy fix that let them be withheld without weakening validation |
| `scripts/build_final_cross_case_integration.py` | removed unused `import math`; removed an `f` prefix from a string with no placeholders | dead code (Ruff `F401`, `F541`) |
| `scripts/finalize_generic_sparse_qsvt_compiler.py` | removed unused `import math` | dead code (Ruff `F401`) |

The last two were made in this session. Both files re-parse, and the whole
package still imports (350/350 modules).

### Documentation edits (3 files, prose only)

| File | Change |
|---|---|
| `docs/ARTIFACT_CHECKLIST.md` | repointed to the current `release_audit/` filenames; ticked the two items the clean export actually satisfies (no inherited history, no >100 MB file, no machine-local paths) |
| `docs/REPRODUCIBILITY.md` | replaced "removed manuscript/package guards" wording with the public test-scope pointer |
| `docs/data_access.md` | reserved-DOI status (see `doi_metadata_alignment.md`) |

### Root release metadata (9 files)

`.gitignore`, `README.md`, `CITATION.cff`, `CHANGELOG.md`, `RELEASE.md`,
`REPRODUCIBILITY.md`, `RESULTS_INDEX.md`, `MANIFEST.md`, `data_manifest.json` —
public-facing identity, scope, DOI, and packaging metadata. These exist to make
the repository publishable and are expected to differ.

### Tests (45 files)

Modified by the public-boundary pruning: test nodes exercising removed
manuscript/submission/reviewer packaging surfaces were dropped, and the unused
imports that pruning left behind were removed. The exact node list is in
`release_audit/public_test_scope_report.md`.

Scientific preservation inside `tests/`:

- 351 of 396 test files are byte-identical.
- The one genuine scientific assertion that lived inside a reviewer-audit test
  was **extracted, not deleted**, into
  `tests/test_classical_selected_observable_baseline.py`, which still checks the
  repeated IEEE300 selected-observable classical timing evidence:
  `sparse_factorized` and `adjoint_functional` methods present, and
  `timing_repeats == 30`.
- No assertion was loosened, no tolerance widened, and no test was skipped or
  xfailed to obtain a pass.

## Files present only in the candidate

`release_audit/` records and `data_manifest.json`, all generated for this clean
export. They add release provenance and remove nothing.

## Integrity of the protected development workspace

The source repository was read only. Nothing in it was modified, deleted, or
regenerated during this release preparation.
