# QSVT Stable Phase Synthesis

## Purpose

This note documents the stable-basis diagnostic path for the bounded
Ridge/Tikhonov QSVT target. It addresses the observed failure mode where
low-degree phase targets are accurate only to about `4.38e-3`, while higher
degree Chebyshev approximants can become unsafe after conversion to monomial
coefficients.

## Script And Outputs

Run:

```bash
.venv/bin/python scripts/audit_qsvt_phase_backend_options.py
.venv/bin/python scripts/build_qsvt_stable_phase_candidates.py
.venv/bin/python scripts/run_qsvt_phase_sanity_regression.py
.venv/bin/python scripts/run_qsvt_stable_target_phase_validation.py
.venv/bin/python scripts/install_or_audit_qsvt_phase_backends.py --install
.venv/bin/python scripts/run_qsvt_external_backend_sanity_regression.py
.venv/bin/python scripts/run_qsvt_external_backend_phase_validation.py
.venv/bin/python scripts/fix_qsvt_phase_validation_stable_basis.py
```

The Phase 1 stable-validation scripts write:

- `outputs/qsvt_phase_backend_audit/phase_backend_audit_summary.csv`
- `outputs/qsvt_phase_backend_audit/phase_backend_capabilities.md`
- `outputs/qsvt_stable_phase_candidates/stable_phase_candidate_summary.csv`
- `outputs/qsvt_stable_phase_candidates/candidate_coefficients_chebyshev.csv`
- `outputs/qsvt_stable_phase_candidates/candidate_coefficients_monomial.csv`
- `outputs/qsvt_stable_phase_candidates/candidate_error_grid.csv`
- `outputs/qsvt_stable_phase_candidates/candidate_boundedness_grid.csv`
- `outputs/qsvt_phase_sanity_regression/phase_sanity_regression_summary.csv`
- `outputs/qsvt_phase_sanity_regression/phase_sanity_response_values.csv`
- `outputs/qsvt_stable_target_phase_validation/stable_target_phase_validation_summary.csv`
- `outputs/qsvt_stable_target_phase_validation/phase_angles.csv`
- `outputs/qsvt_stable_target_phase_validation/phase_response_values.csv`
- `outputs/qsvt_stable_target_phase_validation/phase_response_error_grid.csv`
- `outputs/qsvt_phase_external_backend_audit/external_backend_audit_summary.csv`
- `outputs/qsvt_external_backend_sanity_regression/external_backend_sanity_summary.csv`
- `outputs/qsvt_external_backend_phase_validation/external_backend_phase_validation_summary.csv`
- `outputs/qsvt_external_backend_phase_validation/external_backend_phase_angles.csv`
- `outputs/qsvt_external_backend_phase_validation/external_backend_phase_error_grid.csv`

The earlier stable-basis follow-up script writes:

- `outputs/qsvt_phase_validation_stable_basis/phase_validation_stable_basis_summary.csv`
- `outputs/qsvt_phase_validation_stable_basis/phase_validation_stable_basis_summary.json`
- `outputs/qsvt_phase_validation_stable_basis/candidate_polynomial_diagnostics.csv`
- `outputs/qsvt_phase_validation_stable_basis/coefficient_stability_diagnostics.csv`
- `outputs/qsvt_phase_validation_stable_basis/phase_response_diagnostics.csv`
- `outputs/qsvt_phase_validation_stable_basis/stable_phase_validation_report.md`
- `outputs/qsvt_phase_validation_stable_basis/manifest.json`

## Diagnostic Stages

Each candidate reports:

- native Chebyshev approximation error;
- native boundedness on `[-1, 1]`;
- odd-parity error;
- backend coefficient basis;
- Chebyshev-to-monomial conversion method and precision;
- conversion error;
- monomial coefficient dynamic range;
- boundedness after conversion;
- optional phase-synthesis status;
- scalar phase-response error when synthesis is attempted.

The configured degree grid is bounded to `35, 51, 71, 101, 151, 201`. If higher
degree rows are coefficient-unstable or unbounded after conversion, they are
reported as failed or skipped rather than forced into phase synthesis.

The new candidate builder separates the native Chebyshev basis from the
monomial basis expected by PennyLane `poly_to_angles`. Chebyshev coefficients
are better conditioned for approximation and evaluation, while monomial
coefficients can have very large dynamic range at high degree. Therefore a
candidate that is accurate in the Chebyshev basis is still unsafe for phase
synthesis unless conversion error, coefficient dynamic range, parity, and
post-conversion boundedness all pass.

The backend audit currently distinguishes phase backends from polynomial
utilities. A tool that can evaluate or convert Chebyshev polynomials is not a
QSP/QSVT phase-synthesis backend unless it can return phase angles for the
validated input basis. In the current dependency set, PennyLane provides a
monomial-coefficient phase backend; no Chebyshev-basis-preserving phase backend
has been validated.

## Pass Rule

A bounded Ridge/Tikhonov target row passes only if:

```text
native approximation error <= 1e-3
bounded after conversion
phase response max error <= 1e-3
sanity polynomial tests pass
```

If the polynomial approximation passes but conversion or phase response fails,
the correct status is unresolved.

The target-level validator compares the implemented scalar phase response
against the bounded Ridge/Tikhonov target values on the validation grid. It
does not pass a row merely because the certified polynomial approximant is
accurate, and it does not pass a row merely because sanity polynomials pass.

Phase 1B adds pyqsp as an optional Chebyshev-basis backend. This path can avoid
the high-degree monomial conversion gate that blocks PennyLane for degree-101
and higher candidates. A pyqsp row is still accepted only if backend sanity
regression passes and the full-domain phase-response error is at most `1e-3`.

The latest Phase 1B target result satisfies that rule:

```text
target: bounded_ridge_tikhonov_pyqsp
backend: pyqsp_sym_qsp
degree: 201
phase_count: 202
input_basis: Chebyshev
full_domain_max_error: 4.668e-4
actual_singular_value_max_error: 8.673e-5
status: passed_scalar_full_domain
```

The prior PennyLane/monomial path remains a historical failure mode caused by
coefficient instability and unsafe high-degree conversion. It is not the latest
paper-ready Phase 1 status.

## Claim Boundary

Safe wording:

```text
Stable phase-synthesis diagnostics evaluate native approximation error,
boundedness, coefficient conversion, and scalar phase response before any
bounded Ridge/Tikhonov target phase pass is claimed.
```

Avoid wording:

```text
The bounded Ridge/Tikhonov target passed phase validation because sanity
polynomials passed or because a Chebyshev polynomial approximation passed.
```

The diagnostic is scalar and dependency-aware. It is not quantum speedup,
quantum advantage, hardware execution, field-data validation, or evidence that
QSVT is superior to Ridge/Tikhonov under the same `alpha`.
