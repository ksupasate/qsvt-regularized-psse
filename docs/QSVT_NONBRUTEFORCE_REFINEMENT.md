# QSVT Non-Brute-Force Refinement

## Purpose

This note documents the diagnostic follow-up for QSVT-compatible approximation
and scalar phase-response validation. The refinement is intentionally not a
brute-force degree search. It preserves the strict `1e-3` tolerance and reports
failed, skipped, and numerically unstable rows explicitly.

## Scripts And Outputs

Run:

```bash
.venv/bin/python scripts/diagnose_qsvt_phase_target_failure.py
.venv/bin/python scripts/run_qsvt_stable_phase_validation_attempt.py
.venv/bin/python scripts/diagnose_qsvt_ieee300_spectral_difficulty.py
.venv/bin/python scripts/run_qsvt_spectrum_aware_diagnostics.py
.venv/bin/python scripts/run_qsvt_ieee118_targeted_refinement.py
.venv/bin/python scripts/build_qsvt_nonbruteforce_refinement_summary.py
```

Primary outputs:

- `outputs/qsvt_phase_target_failure_diagnostics/`
- `outputs/qsvt_stable_phase_validation_attempt/`
- `outputs/qsvt_ieee300_spectral_difficulty/`
- `outputs/qsvt_spectrum_aware_diagnostics/`
- `outputs/qsvt_ieee118_targeted_refinement/`
- `outputs/qsvt_nonbruteforce_refinement_summary/`

## Current Evidence

The bounded Ridge/Tikhonov degree-35 phase target fails strict `1e-3` because
the polynomial approximation error is already about `4.38e-3`. The scalar phase
response adds only a small extra error relative to the synthesized polynomial.
Degree 101 has better approximation error, but the Chebyshev-to-monomial
conversion becomes unstable and the resulting monomial polynomial violates
boundedness on `[-1, 1]`.

The stable phase attempt validates sanity polynomials `x` and `0.5x`. It does
not validate a bounded Ridge/Tikhonov target row. This is now historical
PennyLane/monomial-path evidence superseded by the Phase 1B pyqsp
Chebyshev-basis scalar full-domain pass:

```text
The bounded Ridge/Tikhonov target passed scalar full-domain phase-response
validation using pyqsp symmetric-QSP phases; the older PennyLane monomial-path
rows remain historical diagnostics.
```

IEEE118 passes the strict approximation diagnostic at degree 1501 after a
degree-1201 numerical LP failure. The script stops at the first approved pass
and does not try arbitrary higher degrees.

IEEE300 remains failed for full-interval `1e-3` at degree 1001. Full-interval
error and actual-singular-value error are both about `8.84e-2`, so the current
failure is not explained away as an empty-interval artifact.

The subsequent failure-fix layer adds stable-basis phase diagnostics, a formal
preconditioned IEEE300 estimator variant, and residual-weighted spectral-error
analysis:

```bash
.venv/bin/python scripts/fix_qsvt_phase_validation_stable_basis.py
.venv/bin/python scripts/run_qsvt_preconditioned_ieee300_estimator.py
.venv/bin/python scripts/diagnose_qsvt_ieee300_residual_weighted_error.py
.venv/bin/python scripts/build_qsvt_failure_fix_summary.py
```

These outputs are reported under:

- `outputs/qsvt_phase_validation_stable_basis/`
- `outputs/qsvt_preconditioned_ieee300_estimator/`
- `outputs/qsvt_ieee300_residual_weighted_error/`
- `outputs/qsvt_failure_fix_summary/`

The preconditioned estimator rows are new variants, not replacements for the
original Ridge/QSVT-target rows. Residual-weighted diagnostics are
solution-relevance diagnostics only.

## Claim Boundaries

Safe wording:

```text
The follow-up reports QSVT-compatible approximation diagnostics, scalar
phase-response diagnostics, spectrum-aware error analysis, and targeted IEEE118
refinement without brute-force degree escalation or tolerance relaxation.
```

Avoid wording:

```text
The refinement demonstrates quantum speedup, quantum advantage, hardware
execution, IEEE300 full-interval validation, field-data validation, or QSVT
superiority over Ridge/Tikhonov.
```
