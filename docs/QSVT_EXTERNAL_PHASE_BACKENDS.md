# QSVT External Phase Backends

## Purpose

Phase 1B adds an optional external-backend path for scalar QSP/QSVT
phase-response validation of the bounded Ridge/Tikhonov target. The goal is to
avoid unstable high-degree Chebyshev-to-monomial conversion when a backend can
accept Chebyshev coefficients directly.

This is scalar phase-response validation. It is not hardware execution, not
quantum speedup, not quantum advantage, and not evidence of QSVT superiority
over Ridge/Tikhonov under the same `alpha`.

## Optional Dependencies

The optional `phase` dependency group includes:

```text
pyqsp
mpmath
sympy
```

Run:

```bash
.venv/bin/python scripts/install_or_audit_qsvt_phase_backends.py --install
```

The script writes:

- `outputs/qsvt_phase_external_backend_audit/external_backend_audit_summary.csv`
- `outputs/qsvt_phase_external_backend_audit/external_backend_audit_summary.json`
- `outputs/qsvt_phase_external_backend_audit/external_backend_install_log.txt`
- `outputs/qsvt_phase_external_backend_audit/external_backend_capabilities.md`
- `outputs/qsvt_phase_external_backend_audit/manifest.json`

QSPPACK is recorded as not directly usable unless a callable Python package is
available. Qiskit package availability is recorded separately from a validated
QSP phase-factor API.

## Backend Adapters

The adapter layer is implemented in:

```text
src/robust_qsvt_se/qsvt/phase_backend_adapters.py
```

Adapters include:

- PennyLane `poly_to_angles`, which expects low-to-high monomial coefficients.
- pyqsp symmetric QSP, which accepts Chebyshev coefficients and uses the
  imaginary scalar response component.
- QSPPACK placeholder, which records unavailability unless a Python-callable
  backend is present.
- Local optimization-based scalar QSP fitting, which is experimental and not a
  certified theorem-level phase-synthesis method.

## Sanity Regression

Run:

```bash
.venv/bin/python scripts/run_qsvt_external_backend_sanity_regression.py
```

The script checks:

```text
x
0.5 x
x^3
0.5 x + 0.25 x^3
```

Target-level validation is trusted only for backends whose sanity rows pass.

## Target-Level Validation

Run:

```bash
.venv/bin/python scripts/run_qsvt_external_backend_phase_validation.py
```

The script writes:

- `outputs/qsvt_external_backend_phase_validation/external_backend_phase_validation_summary.csv`
- `outputs/qsvt_external_backend_phase_validation/external_backend_phase_angles.csv`
- `outputs/qsvt_external_backend_phase_validation/external_backend_phase_response_values.csv`
- `outputs/qsvt_external_backend_phase_validation/external_backend_phase_error_grid.csv`
- `outputs/qsvt_external_backend_phase_validation/external_backend_phase_report.md`
- `outputs/qsvt_external_backend_phase_validation/manifest.json`

Full-domain validation is over the declared normalized singular-value interval
`[-1, -sigma_min / beta] union [sigma_min / beta, 1]`. Actual-singular-value
validation is reported in separate columns and is not a substitute for the
dense full-domain grid.

## Current Interpretation

The external-backend path can validate a bounded Ridge/Tikhonov Chebyshev
candidate only when:

```text
sanity regression passes
native full-domain approximation error <= 1e-3
native polynomial is bounded
backend supports the candidate basis
phase-response full-domain error <= 1e-3
```

PennyLane remains monomial-gated. Unsafe high-degree monomial conversions are
skipped. pyqsp can avoid the monomial conversion by accepting Chebyshev
coefficients directly.

The latest Phase 1B generated result is a pass for
`bounded_ridge_tikhonov_pyqsp`: pyqsp symmetric QSP accepted Chebyshev-basis
input for `coefficient_conditioned_chebyshev_degree_201_lambda_1e-04`, returned
202 phases, and passed scalar full-domain phase-response validation with
maximum error `4.668e-4` against the `1e-3` tolerance. The actual-singular-value
maximum error was `8.673e-5`. The pyqsp sanity regression passed before this
target row was interpreted.

Historical PennyLane rows remain monomial-path diagnostics limited by
coefficient instability for this target. They are preserved as historical
failures or skipped backend-specific rows, not as the latest paper-ready final
status.

## Safe Wording

```text
External-backend scalar phase-response validation was attempted with explicit
backend sanity checks, basis-support gates, and separate full-domain and
actual-singular-value errors.
```

If a generated row has `passed_1e_minus_3_full_domain = true`, this wording is
supported:

```text
The bounded Ridge/Tikhonov target passed scalar full-domain phase-response
validation for the listed backend and candidate under the declared normalized
interval and strict 1e-3 tolerance.
```

## Avoid Wording

```text
External backend installation alone validates QSVT phases.
Actual-singular-value-only validation is full-domain validation.
The result demonstrates speedup or hardware execution.
The result proves QSVT superiority over Ridge/Tikhonov under the same alpha.
```
