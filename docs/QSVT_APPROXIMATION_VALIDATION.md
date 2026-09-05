# QSVT Approximation Validation

## Target Filter

The state-estimation experiments use the Ridge/Tikhonov spectral filter

```text
P_alpha(sigma) = sigma / (sigma^2 + alpha)
```

For QSVT-oriented implementation analysis, the polynomial target must be
bounded:

```text
P_alpha,bounded(sigma) = (1 / C) * sigma / (sigma^2 + alpha)
```

where `C` is at least the maximum absolute filter gain over the validated
singular-value interval. Boundedness matters because QSVT/QSP polynomial
responses must be bounded on the signal domain.

## Exact Spectral Target Versus Polynomial Approximation

The exact QSVT-target spectral simulator and Ridge/Tikhonov use the same filter
when they use the same `alpha`; they are expected to match numerically. That
equivalence is not a performance claim.

The approximation reports address a different question: what polynomial degree,
method, and query-count proxy are needed to approximate the bounded target over
the observed normalized singular-value interval and at the actual singular
values of the selected weighted Jacobian.

## Polynomial Fallback Versus Full Phase Synthesis

The degree sweep, adaptive degree selection, method comparison, trade-off, and
multi-case approximation diagnostics are polynomial diagnostics. They do not
by themselves synthesize QSP/QSVT phases.

The optional phase-synthesis report is separate. It attempts real phase
synthesis only when the configured dependency is available. If the dependency is
missing, the report writes a skipped row. If phase angles are synthesized but
the scalar-response validation does not meet tolerance, the row is reported as
failed validation. Neither case is hardware execution.

The phase-response convention diagnostic is now separate from optional phase
synthesis. It checks PennyLane-style `RX(2 arccos x)` and `PCPhase` scalar
response conventions against known sanity polynomials before reporting the
bounded Ridge/Tikhonov target. Passing sanity polynomials resolves the response
convention check; it does not automatically imply that the Ridge target passes
the configured phase-response tolerance.

The Phase 1 stable path adds another separation:

```text
native Chebyshev approximation
  -> native boundedness and odd parity
  -> Chebyshev-to-monomial conversion diagnostics
  -> post-conversion boundedness and coefficient dynamic range
  -> phase synthesis only for safe rows
  -> scalar phase-response comparison to the bounded target
```

The generated outputs are:

- `outputs/qsvt_phase_backend_audit/`;
- `outputs/qsvt_stable_phase_candidates/`;
- `outputs/qsvt_phase_sanity_regression/`;
- `outputs/qsvt_stable_target_phase_validation/`.

A row can be a good bounded Chebyshev approximation and still fail the phase
path because a monomial-only backend receives ill-conditioned coefficients.
Such a row is approximation evidence, not target-level phase validation.

Phase 1B introduces an external-backend route for candidates whose native
Chebyshev approximation is accurate and bounded but whose monomial conversion is
unsafe. pyqsp symmetric QSP is handled as a Chebyshev-basis scalar phase backend
and is validated against the bounded target through full-domain phase-response
error. Actual-singular-value errors are reported separately and are not
interchanged with full-domain validation.

## Methods

The approximation layer compares:

- the existing odd reduced-domain Chebyshev fallback;
- odd Chebyshev least-squares fitting;
- odd Chebyshev minimax fitting via a SciPy linear program;
- positive-interval Chebyshev interpolation as a diagnostic that does not
  enforce QSVT odd parity.

Only odd methods are QSVT-parity compatible for the regularized inverse target.
Positive-interval interpolation is useful as a numerical diagnostic but should
not be described as a QSVT-compatible odd polynomial.

## Degree-Error-Query Trade-Off

The query-count proxy is

```text
query_count = 2 * degree + 1
```

Tighter pointwise-error tolerances generally require higher degree and therefore
higher query count. The trade-off report records the smallest configured degree
that passes each tolerance and clearly reports when no tested degree passes.

The adaptive multicase degree search applies the same idea across
IEEE/PYPOWER weighted matrices. It reports the first configured passing degree
or an explicit failure status for IEEE14/30/57/118/300 resource-only matrix
construction. Larger cases may require higher degree or fail within the
configured degree cap; those outcomes are feasibility diagnostics, not
performance claims.

## Alpha Sensitivity

The default selected alpha values are:

```text
1e-4, 1e-2, 1
```

For the current IEEE14 weighted-Jacobian interval, these alphas produce very
similar bounded targets because the singular values are large relative to the
tested regularization values. Other matrices or stronger regularization may
change this behavior.

## Current Pass/Fail Status

The earlier selected-alpha fallback report did not meet the strict `1e-3`
maximum pointwise-error tolerance. The strengthened reports add degree sweeps,
adaptive search, and method comparison. Passing or failing `1e-3` is reported
explicitly in the generated CSV files; it should not be inferred or overstated.

The non-brute-force refinement adds targeted diagnostics:

- `outputs/qsvt_phase_target_failure_diagnostics/` separates polynomial
  approximation error from scalar phase-response error and records coefficient,
  parity, boundedness, and basis-conversion evidence.
- `outputs/qsvt_stable_phase_validation_attempt/` validates low-risk sanity
  polynomials and bounded Ridge/Tikhonov targets separately.
- `outputs/qsvt_ieee300_spectral_difficulty/` reports full-interval error,
  actual-singular-value error, and restricted-interval diagnostics as separate
  quantities.
- `outputs/qsvt_ieee118_targeted_refinement/` tries only the approved targeted
  IEEE118 degrees and stops once a strict `1e-3` pass is found.

Current generated evidence shows that IEEE118 passes at degree 1501 after a
degree-1201 numerical LP failure. IEEE300 remains far from strict full-interval
`1e-3` validation at degree 1001. Restricted-interval and actual-singular-value
diagnostics must not be reported as full-interval validation.

The remaining-failure follow-up adds:

- `outputs/qsvt_phase_validation_stable_basis/`, which checks native Chebyshev
  approximation error, boundedness, parity, monomial conversion error,
  coefficient dynamic range, and optional phase-response error before any
  bounded Ridge/Tikhonov phase pass can be claimed.
- `outputs/qsvt_phase_backend_audit/`, which records whether installed or local
  tools accept monomial coefficients, Chebyshev coefficients, or function
  values and whether they can return phase angles.
- `outputs/qsvt_stable_phase_candidates/`, which constructs bounded
  Ridge/Tikhonov candidate polynomials over the controlled degree grid and
  records conversion safety gates.
- `outputs/qsvt_phase_sanity_regression/`, which reruns the required sanity
  polynomial phase-response checks.
- `outputs/qsvt_stable_target_phase_validation/`, which attempts phase
  synthesis only for candidates marked safe and reports target-level pass/fail.
- `outputs/qsvt_phase_external_backend_audit/`, which records optional external
  backend install and API capability status.
- `outputs/qsvt_external_backend_sanity_regression/`, which checks the four
  required sanity polynomials across available adapters.
- `outputs/qsvt_external_backend_phase_validation/`, which evaluates target
  phase response for safe backend/candidate combinations and keeps full-domain
  and actual-singular-value errors separate.
- `outputs/qsvt_phase1_finalization/`, which records the pyqsp symmetric-QSP
  scalar full-domain phase-response pass for the bounded Ridge/Tikhonov target.
- `outputs/qsvt_phase2_preconditioned_alpha_sweeps/`,
  `outputs/qsvt_phase2_alpha_selection/`, and `outputs/qsvt_phase2_summary/`,
  which add preconditioned-estimator and alpha-selection diagnostics without
  changing the original solver behavior.
- `outputs/qsvt_preconditioned_ieee300_estimator/`, which evaluates formal
  column-equilibrated estimator variants and reports preconditioned QSVT
  approximation metrics separately from the original unpreconditioned rows.
- `outputs/qsvt_ieee300_residual_weighted_error/`, which reports whether
  pointwise approximation error is weighted by high-energy singular directions.
  This diagnostic does not replace full-interval validation.
- `outputs/qsvt_failure_fix_summary/`, which consolidates fixed, partial, and
  unresolved rows without relaxing tolerance.

The derived reporting layer adds:

- `outputs/qsvt_preconditioned_variant_sweeps/`, which evaluates original and
  preconditioned estimator variants over controlled alpha, noise, missing-row,
  and bad-data grids where safe.
- `outputs/qsvt_preconditioning_resource_comparison/`, which reports proxy
  approximation/resource quantities before and after column equilibration.
- `outputs/paper_ready_qsvt_tables/`, which aggregates generated outputs into
  derived summary tables without inventing numbers.
- `outputs/final_qsvt_artifact_freeze/` and
  `outputs/final_qsvt_claim_safety_audit/`, which track artifacts and claim
  boundaries.

For Phase 1, PennyLane's monomial path was limited by coefficient instability
for the target, while pyqsp accepted Chebyshev-basis input after sanity
regression passed. The bounded Ridge/Tikhonov target then passed scalar
full-domain phase-response validation with pyqsp. This is not hardware
execution, not block-encoded matrix execution, and not quantum speedup
evidence.

## Limitations

- Polynomial diagnostics are not full QSP/QSVT phase synthesis.
- Optional phase synthesis is dependency-safe and may be skipped or fail
  response validation.
- Phase-response sanity validation is scalar and convention-level; it is not a
  scalable block-encoding oracle.
- Adaptive multicase degree search is bounded to configured degrees and does
  not prove globally optimal QSP/QSVT degree.
- High-degree Chebyshev-to-monomial conversion can become numerically unstable;
  coefficient diagnostics should be checked before phase-synthesis claims.
- Preconditioned approximation results are new-variant diagnostics and should
  not be merged into original IEEE300 unpreconditioned validation claims.
- Residual-weighted and restricted-interval diagnostics are solution-relevance
  diagnostics, not full QSVT validation.
- Preconditioned resource comparisons are variant-specific proxy diagnostics;
  they do not establish original IEEE300 validation or quantum speedup.
- Phase 1 passed scalar full-domain phase-response validation using pyqsp, but
  this remains scalar validation only; it is not hardware execution or
  block-encoded matrix execution.
- Phase 2 alpha selection is diagnostic, not field-calibrated.
- Coordinate-preconditioned Ridge is a separate estimator and may degrade
  residual/RMSE; transformed-penalty preconditioning preserves the original
  x-space penalty.
- Paper-ready tables are aggregation artifacts and should cite the underlying
  diagnostic output for each numerical claim.
- Query count is only a proxy and excludes block-encoding oracle construction,
  data loading, state preparation, error correction, compilation, and readout.
- Dense block encoding remains a small validation prototype, not a scalable
  sparse-access oracle.
- The reports use controlled IEEE/PYPOWER benchmark matrices, not PMU/SCADA
  field data.

## Claim Boundary

These approximation reports support resource-aware feasibility discussion for a
QSVT-compatible implementation pathway. They do not demonstrate quantum
speedup, quantum advantage, full hardware execution, PMU/SCADA field-data
validation, deployment readiness, or numerical superiority over Ridge/Tikhonov
under the same alpha and spectral filter.
