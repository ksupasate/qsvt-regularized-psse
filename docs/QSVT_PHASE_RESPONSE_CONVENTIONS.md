# QSVT Phase-Response Conventions

## Purpose

This note documents the scalar QSP/QSVT phase-response convention diagnostic
added for the engineering extension. It exists because a synthesized phase
sequence is only meaningful after the response convention, phase order, phase
sign, phase offsets, and coefficient basis are validated.

The diagnostic is a scalar convention check. It is not hardware execution, a
matrix-level oracle implementation, quantum speedup, quantum advantage, or
evidence of QSVT superiority over Ridge/Tikhonov under the same alpha.

## Convention Search

The script `scripts/diagnose_qsvt_phase_response_conventions.py` writes:

- `outputs/qsvt_phase_response_convention_diagnostics/convention_search_summary.csv`
- `outputs/qsvt_phase_response_convention_diagnostics/sanity_polynomial_results.csv`
- `outputs/qsvt_phase_response_convention_diagnostics/phase_response_values.csv`
- `outputs/qsvt_phase_response_convention_diagnostics/best_convention_report.md`
- `outputs/qsvt_phase_response_convention_diagnostics/manifest.json`

The search checks:

- phase order: original and reversed;
- phase sign: `phi` and `-phi`;
- endpoint offsets: none and first/last `+/- pi/2`;
- response component: real, imaginary, negated real, negated imaginary, and
  absolute value of `U[0,0]`;
- scalar signal convention, including PennyLane-style `RX(2 arccos x)` with
  `PCPhase`;
- monomial coefficient ordering expected by PennyLane:
  low-to-high powers.

## Sanity Polynomials

The diagnostic first synthesizes phases for known odd polynomials:

```text
x
0.5 x
x^3
0.5 x + 0.25 x^3
```

These sanity rows verify that the scalar response convention can reproduce
known polynomial responses before applying the same machinery to the bounded
Ridge/Tikhonov target.

The canonical convention is:

```text
original / phi / none / pennylane_rx_pcphase / real_u00
```

Under this convention, the scalar block encoding is `RX(2 arccos x)`, the
projectors are `PCPhase`, and the reported response is `real(U[0,0])`.

## Ridge/Tikhonov Target

The bounded target remains:

```text
P_alpha,bounded(sigma) = (1 / C) * sigma / (sigma^2 + alpha)
```

The phase-response report is separate from the polynomial fallback report. A
polynomial may be bounded and useful as an approximation diagnostic while the
optional phase-response validation still fails the configured tolerance because
stable high-degree monomial-basis phase synthesis remains difficult.

The non-brute-force phase-target diagnostic writes:

- `outputs/qsvt_phase_target_failure_diagnostics/phase_target_failure_summary.csv`
- `outputs/qsvt_phase_target_failure_diagnostics/coefficient_diagnostics.csv`
- `outputs/qsvt_phase_target_failure_diagnostics/basis_conversion_diagnostics.csv`
- `outputs/qsvt_phase_target_failure_diagnostics/phase_response_error_breakdown.csv`
- `outputs/qsvt_phase_target_failure_diagnostics/phase_target_failure_report.md`

The current degree-35 bounded Ridge/Tikhonov target has polynomial approximation
error about `4.38e-3` and scalar phase-response error about `4.38e-3`. The
phase response tracks the synthesized polynomial closely, so the strict
`1e-3` failure is not fixed by claiming a new phase convention. Degree 101 has
better approximation error but unstable monomial coefficients and violates the
boundedness check after conversion, so it is not a low-risk phase target.

`outputs/qsvt_stable_phase_validation_attempt/` validates `x` and `0.5x` sanity
polynomials but does not produce a passing bounded Ridge/Tikhonov target row.
That is now a historical PennyLane/monomial-path diagnostic rather than the
latest final Phase 1 status. The latest target-level pass is the pyqsp
Chebyshev-basis row:

```text
The bounded Ridge/Tikhonov target passed scalar full-domain phase-response
validation using pyqsp symmetric-QSP phases with degree 201, 202 phases,
full-domain max error 4.668e-4, and actual-singular-value max error 8.673e-5.
```

The Phase 1 sanity-regression script now writes the same four required sanity
polynomials under `outputs/qsvt_phase_sanity_regression/`:

```text
x
0.5 x
x^3
0.5 x + 0.25 x^3
```

The canonical convention remains:

```text
original / phi / none / pennylane_rx_pcphase / real_u00
```

Target-level validation under
`outputs/qsvt_stable_target_phase_validation/` is considered meaningful only
when these sanity rows pass and a candidate has already passed the stable
candidate gates. Unsafe high-degree monomial coefficient rows are recorded as
skipped rather than passed to phase synthesis.

The remaining-failure follow-up writes a stable-basis diagnostic under
`outputs/qsvt_phase_validation_stable_basis/`. It tries a direct
Chebyshev-backend probe, float64 and `numpy.longdouble` Chebyshev-to-monomial
conversion, and a coefficient-conditioned shrinkage candidate over the bounded
degree grid `35, 51, 71, 101, 151, 201`. It records native approximation error,
boundedness, parity, conversion error, coefficient dynamic range, phase backend
status, and phase-response error.

Phase validation for the bounded Ridge/Tikhonov target is claimed only when the
candidate row has approximation error below `1e-3`, remains bounded after
conversion, and has scalar phase-response maximum error below `1e-3`. A row
that passes polynomial approximation but fails conversion or phase response is
reported as unresolved, not as phase validation.

Phase-response validation differs from polynomial approximation. A Chebyshev
polynomial can approximate the bounded target accurately in its native basis,
but a monomial-only phase backend still needs low-to-high power coefficients.
High-degree Chebyshev-to-monomial conversion can amplify rounding error and
produce large coefficient dynamic range. The stable candidate reports therefore
measure conversion error, post-conversion boundedness, and coefficient dynamic
range before phase angles are requested.

Phase 1B adds external backend convention checks. pyqsp symmetric QSP uses
Chebyshev coefficients and validates the odd target through the imaginary
component of the scalar response. PennyLane continues to use the canonical
`real(U[0,0])` response under its `RX`/`PCPhase` convention. Each backend must
pass sanity regression before target rows are trusted.

For the current Phase 1B result, pyqsp sanity regression passed and the
bounded Ridge/Tikhonov target passed scalar full-domain validation. This does
not constitute hardware execution, block-encoded matrix execution, quantum
speedup evidence, or QSVT superiority over Ridge/Tikhonov.

## Safe Interpretation

Safe wording:

```text
The phase-response diagnostics identify and validate the scalar PennyLane
RX/PCPhase convention on known sanity polynomials, then report pass/fail status
for the bounded Ridge/Tikhonov target separately.
```

Avoid wording:

```text
The phase-response diagnostic proves full hardware QSVT execution or quantum
advantage.
```

Also avoid:

```text
The bounded Ridge/Tikhonov target passed phase validation because sanity
polynomials passed.
```

## Limitations

- The diagnostic validates scalar response conventions, not sparse-access
  block-encoding oracles.
- Passing sanity polynomials does not force the bounded Ridge/Tikhonov target
  to pass a strict `1e-3` phase-response tolerance at the configured degree.
- Optional PennyLane phase synthesis uses monomial coefficients; high-degree
  Chebyshev fits can become numerically unstable when converted to monomial
  form.
- The report uses controlled IEEE/PYPOWER benchmark matrices or synthetic
  fallback data, not PMU/SCADA field data.
