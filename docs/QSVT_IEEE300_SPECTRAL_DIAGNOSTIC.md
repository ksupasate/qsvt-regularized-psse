# QSVT IEEE300 Spectral Diagnostic

## Purpose

IEEE300 remains difficult for the bounded Ridge/Tikhonov QSVT-compatible
approximation at degree 1001. This diagnostic explains the failure by comparing
the singular spectrum, approximation interval, error location, and actual
singular-value errors without increasing degree blindly.

## Outputs

`scripts/diagnose_qsvt_ieee300_spectral_difficulty.py` writes:

- `outputs/qsvt_ieee300_spectral_difficulty/spectral_difficulty_summary.csv`
- `outputs/qsvt_ieee300_spectral_difficulty/spectral_difficulty_summary.json`
- `outputs/qsvt_ieee300_spectral_difficulty/singular_value_quantiles.csv`
- `outputs/qsvt_ieee300_spectral_difficulty/singular_value_histograms.csv`
- `outputs/qsvt_ieee300_spectral_difficulty/error_location_diagnostics.csv`
- `outputs/qsvt_ieee300_spectral_difficulty/interval_restriction_diagnostics.csv`
- `outputs/qsvt_ieee300_spectral_difficulty/ieee300_spectral_difficulty_report.md`

## Current IEEE300 Result

At alpha `1e-2` and degree 1001, IEEE300 has condition number about `2.05e4`.
The full-interval maximum error is about `8.84e-2`, and the maximum error at
actual singular values is also about `8.84e-2`. The error peak is in the
interior with nearby spectral mass, so the failure is not merely an empty part
of the approximation interval.

Central interval diagnostics are still reported, but they are diagnostic only.
They must not be written as full QSVT validation.

## Spectrum-Aware Diagnostic

`scripts/run_qsvt_spectrum_aware_diagnostics.py` separately reports
preconditioning and interval diagnostics. In the current degree-301 diagnostic,
column equilibration reduces the condition number and approximation error for
the scaled matrix. This is useful evidence that spectral spread matters, but it
does not change the main estimator results or prove a quantum speedup.

## Formal Preconditioned Variant

`scripts/run_qsvt_preconditioned_ieee300_estimator.py` turns the
column-equilibration observation into explicitly labeled estimator variants:

- original unpreconditioned Ridge/Tikhonov;
- column-equilibrated coordinate-penalty Ridge;
- column-equilibrated transformed-penalty Ridge, used as an x-space penalty
  consistency check;
- unpreconditioned and preconditioned QSVT-target spectral diagnostics.

The preconditioned rows are new variants. They do not overwrite the original
unpreconditioned IEEE300 result. If the preconditioned approximation error
passes `1e-3`, the correct claim is that the formal preconditioned variant has
lower approximation difficulty under the configured diagnostic, not that the
original IEEE300 full-interval validation passed.

## Residual-Weighted Diagnostic

`scripts/diagnose_qsvt_ieee300_residual_weighted_error.py` reports
singular-direction residual projections `u_i^T r_tilde` and
`|p(sigma_i)-P_alpha(sigma_i)| |u_i^T r_tilde|`. This identifies whether large
pointwise approximation errors align with high-energy solution directions.

Residual-weighted diagnostics are not full-interval validation and do not
replace actual-singular-value or restricted-interval diagnostics.

## Preconditioned Variant Sweeps

`scripts/run_qsvt_preconditioned_variant_sweeps.py` extends the formal
preconditioned variant into controlled IEEE118/IEEE300 sweeps. It keeps
original Ridge, coordinate-preconditioned Ridge, transformed-penalty
preconditioned Ridge, and QSVT approximation diagnostics as separate rows. The
coordinate-preconditioned estimator can improve approximation difficulty while
still requiring residual/RMSE checks before any estimator recommendation. The
transformed-penalty row is a consistency check and should preserve the original
x-space Ridge solution.

`scripts/build_qsvt_preconditioning_resource_comparison.py` separately reports
before/after proxy resources. Improvements in query-count or approximation
error are proxy improvements for the configured matrix variant, not quantum
speedup claims.

## Safe Interpretation

Safe wording:

```text
IEEE300 spectral diagnostics show that full-interval and actual-singular-value
errors remain large at degree 1001, while preconditioning diagnostics indicate
that spectral spread contributes to approximation difficulty.
```

Avoid wording:

```text
IEEE300 passes full QSVT validation after interval restriction or
preconditioning.
```

Also avoid:

```text
Residual-weighted diagnostics prove full QSVT validation.
```
