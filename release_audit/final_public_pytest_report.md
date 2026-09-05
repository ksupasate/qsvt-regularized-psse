# Final Public Test Suite — Exact-Tree Run

Prepared: 2026-09-05

This is the mandatory complete run of the public test suite against the **exact
post-pruning tree**, executed after the final public-boundary pruning and after
this session's DOI, documentation, and lint edits.

## Command

```bash
cd <public-candidate-root>
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
python -m pytest tests -p no:randomly -rA --durations=15 \
       --junitxml=release_audit/test_logs/final_public_pytest.xml
```

The single-thread environment variables are the project's recorded convention:
multithreaded BLAS/OpenMP has previously aborted heavy statevector runs on this
platform. `-p no:randomly` fixes collection order so the run is reproducible.

## Environment

| Item | Value |
|---|---|
| Python | 3.12.11 (CPython) |
| pytest | 9.0.3 |
| Working directory | the public candidate tree |
| Package resolution | `robust_qsvt_se` resolved from the **candidate's** `src/` via `pythonpath` in `pyproject.toml`, confirmed before the run |
| Started / ended (UTC) | 2026-09-05T15:16:04Z → 2026-09-05T15:32:15Z |

## Result

| Metric | Value |
|---|---:|
| **Collected** | **1702** |
| **Passed** | **1670** |
| **Skipped** | **32** |
| **Failed** | **0** |
| **Errors** | **0** |
| Warnings | 2434 |
| Runtime | **967.59 s** (16 m 08 s; wall 970.89 s) |
| Process exit code | **0** |

JUnit XML confirms the same totals: `tests="1702" failures="0" errors="0"
skipped="32"`.

**Zero scientific failures.** No estimator, Ridge/Tikhonov, QSVT-target,
Ridge/QSVT-equivalence, measurement-generation, AC/DC experiment-construction,
state-estimation, QSVT/QSP-validation, configuration, or canonical-numerical
test failed. No test was weakened, skipped, or xfailed to obtain this result.

## The 32 skips

Every skip is a pre-existing guarded skip that reports its own reason; none was
introduced to hide a failure. Two distinct reasons account for them:

| Reason reported by the test | Meaning |
|---|---|
| `study outputs not generated yet` (`tests/test_sparse_error_decomposition.py:220`) | the test regenerates or skips depending on whether an optional study output is present |
| `manuscript table not present in this checkout` (`tests/test_sparse_quantization_error_report.py:113`) | a cross-check against a manuscript table, which the public repository deliberately does not ship; the test self-skips exactly as designed |

The remaining skips are the same guards reached from parameterized cases.

## Warnings

2434 warnings, all from third-party libraries, none from repository code:

- The dominant source is Qiskit 2.3's `DeprecationWarning` for
  `Gate.control(annotated=None)`, raised once per controlled-gate construction
  in the circuit-heavy suites (`test_tqe_closed_loop_nonlinear_update.py`,
  `test_tqe_closed_loop_audit.py`, `test_generic_sparse_qsvt_compiler.py`, and
  the sparse-chain tests).
- A NumPy `RuntimeWarning: Mean of empty slice` from
  `numpy/_core/fromnumeric.py`, reached when a sparsification configuration
  yields an empty selection — an expected branch that the tests assert on.

Neither indicates a defect in this repository. Both are recorded as
non-blocking.

## Slowest tests

| Seconds | Phase | Test |
|---:|---|---|
| 153.75 | setup | `test_tqe_revision_outputs.py::test_reviewer_matrix_covers_w1_to_w7` |
| 134.91 | setup | `test_phase10_nonlinear_qsvt_loop.py::test_solvers_present_and_qsvt_tracks_ridge` |
| 55.43 | call | `test_phase10_full_rectangular_qsvt.py::test_full_run_outputs_and_claim_safe` |
| 39.25 | setup | `test_phase10_resource_ledger.py::test_all_workloads_present_with_tiers` |
| 30.11 | setup | `test_phase10_residual_loading.py::test_p_succ_matches_executed_convention` |

## Stored evidence

- `release_audit/test_logs/final_public_pytest.log` — full console log
  (git-ignored by the `*.log` rule; local evidence).
- `release_audit/test_logs/final_public_pytest.xml` — JUnit XML, committed.

Both were sanitized after the run: pytest stamps the JUnit XML with the machine
hostname, and the console log echoed the interpreter's absolute path. The
hostname was replaced with `redacted-local-workstation`, and home and staging
paths with `<home>` / `<repo-root>`. Only those identifiers changed; every test
name, status, duration, and total is verbatim.

## Relationship to the pre-pruning run

The handoff recorded 1689 passed / 34 skipped / 0 failed *before* the final
public-boundary pruning, then a collection-only check reporting 1702 collected
after it. This run closes that gap: the complete suite on the exact post-pruning
tree collects 1702 and passes 1670 with 32 skips and **0 failures**.
