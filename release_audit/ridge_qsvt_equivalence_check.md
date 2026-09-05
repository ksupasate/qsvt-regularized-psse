# Ridge / QSVT-Target Equivalence Check

Prepared: 2026-09-05
Tree: the exact public candidate.

The established protocols were rerun unchanged. **No new comparison protocol was
invented**, and no estimator, alpha, seed, or tolerance was modified.

## Claim under test

With an identical matrix, right-hand side, and regularization parameter `alpha`,
the exact classical QSVT target and Ridge/Tikhonov are the *same* spectral
computation, so they must agree to floating-point round-off in the classical
simulator. This is a definitional identity, **not** a statement that QSVT
outperforms Ridge.

## Protocol 1 — classical full-alpha sweep

`robust_qsvt_se.paper.full_alpha_sweep_classical.build_full_alpha_sweep_classical`,
the routine behind `tests/test_full_alpha_sweep_classical.py::test_qsvt_equals_ridge_for_matched_alpha`.
RMSE is compared per matched `(case, stress_type, alpha, seed)` key.

| Quantity | Value |
|---|---|
| `qsvt_ridge_equivalent` flag | `True` |
| Matched comparison points | 336 |
| **max \|RMSE_ridge − RMSE_qsvt_target\|** | **0.000000e+00** |
| Points exactly equal | **336 / 336** |
| max relative difference | 0.000000e+00 |

The difference is not merely below tolerance — it is **exactly zero in every
one of the 336 matched points**, reproducing the previously recorded result.

## Protocol 2 — IEEE14 pipeline-boundary equivalence report

`robust_qsvt_se.paper.ieee_qsvt_pipeline_boundary.build_ieee_qsvt_pipeline_boundary`
(seed 123, IEEE14, 4×4 and 8×8 deterministic blocks), the routine behind
`tests/test_ieee_qsvt_pipeline_boundary.py::test_matched_alpha_qsvt_target_ridge_equivalence`.

| case | alpha | ridge solution norm | relative difference | status |
|---|---|---:|---:|---|
| ieee14 (4×4) | 1e-4 | 0.010967 | 1.425809e-16 | pass |
| ieee14 (8×8) | 1e-4 | 0.022212 | 1.298784e-16 | pass |

| Quantity | Value |
|---|---|
| Declared tolerance | 1e-10 |
| max relative difference | **1.43e-16** (≈ one unit in the last place of float64) |
| Distinct alpha values | exactly one (1e-4) — alpha is **not** tuned per block |
| Equivalence status | `pass` for every row |

## Test-level confirmation

All eight equivalence-related tests pass on the exact public tree:

```
tests/test_full_alpha_sweep_classical.py::test_qsvt_equals_ridge_for_matched_alpha
tests/test_ieee_qsvt_pipeline_boundary.py::test_matched_alpha_qsvt_target_ridge_equivalence
tests/test_full_vector_readout.py::test_qsvt_target_equals_ridge_for_matched_alpha
tests/test_ieee300_scope_note.py::test_qsvt_ridge_equivalence_holds
tests/test_classical_spectral_filtering_audit.py::test_filter_comparison_marks_ridge_qsvt_identical
tests/test_classical_audit_recheck.py::test_recheck_keeps_qsvt_ridge_equivalent_no_superiority
tests/test_final_statistical_aggregation.py::test_qsvt_target_is_not_reported_as_outperforming_ridge
tests/test_full_matrix_qsvt_demo.py::test_normalized_target_scales_back_to_original_ridge_filter
```

```
8 passed in 1.09s
```

## Result

**PASS.** The equivalence holds at machine precision, and the repository's own
guards additionally assert that the equivalence is never reported as QSVT
outperforming Ridge — matching the claim boundary stated in `README.md`,
`docs/CLAIM_SCOPE.md`, and `CITATION.cff`.
