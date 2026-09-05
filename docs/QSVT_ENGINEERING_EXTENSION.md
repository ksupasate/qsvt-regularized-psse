# QSVT Engineering Extension

## Purpose

This extension strengthens the quantum-engineering feasibility evidence for the
regularized QSVT pathway in weighted power-system state-estimation experiments.
It studies QSVT as a possible implementation pathway for the same regularized
spectral filter used by Ridge/Tikhonov:

```text
P_alpha(sigma) = sigma / (sigma^2 + alpha)
```

The extension strengthens the quantum-engineering feasibility evidence. It does
not demonstrate quantum speedup, full hardware execution, or numerical
superiority over Ridge/Tikhonov in the classical simulator.

## Implemented Components

- Dense block-encoding validation for normalized weighted matrices, including
  rectangular Julia block encodings.
- An exact end-to-end QSVT-target state demo showing equality with Ridge under
  the same alpha and filter.
- Resource and readout reports for QSVT-compatible regularized filtering.
- Alpha sensitivity reports linking filter behavior to degree and query-cost
  proxy estimates.
- Standalone selected-alpha bounded polynomial validation for `1e-4`, `1e-2`,
  and `1`.
- Degree sweep, adaptive degree selection, polynomial method comparison,
  optional phase-synthesis validation, tolerance-frontier reporting, and
  multi-case approximation diagnostics for the bounded QSVT-compatible target.
- Phase-response convention diagnostics that check PennyLane scalar
  `RX`/`PCPhase` conventions on sanity polynomials before reporting
  Ridge-target phase-response status.
- Adaptive multicase degree search that reports configured degree/query
  requirements across IEEE14/30/57/118/300 where matrix construction succeeds.
- Standalone selected-observable shot-level readout modeling.
- Hardware-aware dependency-free proxy cost reporting.
- Block-encoding scalability diagnostics explaining why dense prototypes are
  not sparse-access oracles.
- Multi-case resource-only diagnostics across PYPOWER IEEE cases where
  construction is feasible, with per-case failure logging.
- Independent artifact and claim-safety audit outputs.
- Column-equilibration preconditioning diagnostics reported only as diagnostic
  evidence for possible resource reduction.
- Stable-basis phase-synthesis diagnostics for the bounded Ridge/Tikhonov
  target, including high-precision conversion and coefficient-stability checks.
- Phase-backend capability audit, stable polynomial candidate construction,
  sanity phase regression, and target-level phase-validation attempt for the
  bounded Ridge/Tikhonov target.
- Optional external phase-backend integration, including pyqsp Chebyshev-basis
  scalar phase validation where available.
- Formal column-equilibrated IEEE300 estimator variants reported separately
  from the original Ridge/QSVT-target rows.
- Phase 2 complete summary, figures, transformed-penalty explanation, optional
  IEEE57 status record, and alpha-selection diagnostics for IEEE118/IEEE300.
- Residual-weighted IEEE300 spectral-error diagnostics that separate
  directional solution relevance from full-interval validation.
- A claim-support matrix for conservative public wording.

## Matrix Sources

Default scripts use generated measurement rows from controlled IEEE/PYPOWER
benchmark network models. The default source is an IEEE14 AC weighted linearized
system built through the existing `build_ac_weighted_system` path. Tests use
deterministic synthetic weighted matrices for speed and reproducibility.

No PMU/SCADA field data is used by this extension.

## Phase 2 Preconditioning Evidence

Phase 2 keeps original and preconditioned variants separate. Coordinate
preconditioning can reduce QSVT-compatible approximation difficulty for
selected alpha settings, but it changes the regularization geometry and can
degrade residual/RMSE. Transformed-penalty preconditioning preserves the
original \(x\)-space Ridge penalty while using the column-equilibrated matrix
for approximation diagnostics. The alpha-selection rows are diagnostic only
and are not field-calibrated operational rules.

## Block-Encoding Normalization

The weighted Jacobian is

```text
H_tilde = R^{-1/2} H
```

The extension normalizes it as

```text
A = H_tilde / beta,  beta >= ||H_tilde||_2
```

so that `||A||_2 <= 1`. The dense validation prototype constructs the Julia
operator

```text
U_A = [[A, sqrt(I - A A*)],
       [sqrt(I - A* A), -A*]]
```

whose top-left block is the normalized matrix. This is a small dense prototype
for validation, not a scalable oracle decomposition.

## QSVT Target Compared With Ridge

The end-to-end state demo solves the weighted system

```text
H_tilde Delta x ~= r_tilde
```

using Ridge/Tikhonov and the exact QSVT-target spectral simulator with the same
alpha. Because both use `sigma / (sigma^2 + alpha)`, the expected result is
numerical equivalence. The report records relative error, cosine similarity,
state-direction fidelity, residual norms, and weighted residual norms.

The bounded QSVT target is tracked separately through a scaling factor `C`:

```text
P_{alpha,bounded}(sigma) = (1 / C) * sigma / (sigma^2 + alpha)
```

Exact spectral equivalence tolerances are not mixed with polynomial or phase
approximation tolerances.

## Resource Estimates

The resource report includes matrix dimensions, nonzero density, rank,
condition number, beta, alpha, epsilon, degree estimate, query-count estimate,
logical and ancilla qubit estimates, depth proxy, controlled block-encoding
call proxy, state-preparation placeholder, and readout placeholder.

The degree estimate is a Chebyshev proxy over the observed singular-value
interval. It is not a full QSVT phase-synthesis, oracle-construction, or
fault-tolerant compilation result.

These resource estimates support feasibility discussion only. They do not
demonstrate quantum speedup or full IEEE-scale hardware execution.

## Readout Limitations

The readout analysis distinguishes:

- full vector reconstruction;
- selected bus or state-component readout;
- observable estimation.

For state estimation, full vector reconstruction can be a major limitation.
More realistic QSVT-oriented readout targets may include:

- `||Delta x||_2`;
- `||H_tilde Delta x - r_tilde||_2`;
- `|Delta theta_i|`;
- `|Delta V_i|`;
- weak-area correction magnitudes when metadata supports them.

The resource/readout script also writes `shot_readout_summary.csv` with a
simple bounded standard-error proxy for selected observables:

- selected state component;
- update-vector norm;
- residual-norm proxy.

These rows make readout scaling explicit for selected observables. They are not
backend sampling results and do not demonstrate full-vector reconstruction.

## Alpha Sensitivity

The alpha sensitivity report uses the grid:

```text
1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1
```

For each alpha, it reports condition number, maximum filter gain, bounded
scaling `C`, maximum bounded filter value, estimated QSVT degree, estimated
query count, Ridge RMSE/residual when available, and exact QSVT-target error
against Ridge.

The same script writes `selected_alpha_polynomial_validation.csv` for selected
alpha values `1e-4`, `1e-2`, and `1`. It fits a bounded odd polynomial to the
QSVT-compatible regularized target on the observed normalized singular-value
domain and reports bounded scaling `C`, polynomial degree, max pointwise target
error, query count, and pass/fail. This is backend-free polynomial validation,
not QSVT phase synthesis or hardware execution.

The standalone selected-alpha report writes the same style of evidence under
`outputs/qsvt_selected_alpha_phase_validation/` with a stricter default
tolerance of `1e-3` and an expanded odd-degree grid. It reports pass/fail per
alpha. A failed row is still useful evidence because it shows the degree grid
was insufficient for the requested tolerance; it is not hidden or reinterpreted
as phase synthesis.

## QSP/QSVT Approximation Strengthening

The approximation-strengthening layer adds:

- `outputs/qsvt_approximation_degree_sweep/`: degree-error-query rows for
  selected alpha values and polynomial degrees.
- `outputs/qsvt_adaptive_degree_selection/`: smallest configured degree meeting
  each tolerance, or explicit failure status.
- `outputs/qsvt_polynomial_method_comparison/`: comparison of the existing
  odd reduced-domain fallback, odd Chebyshev least-squares, odd Chebyshev
  minimax, and positive-interval Chebyshev interpolation diagnostics.
- `outputs/qsvt_optional_phase_synthesis_validation/`: dependency-safe optional
  phase synthesis; skipped and failed rows are reported explicitly.
- `outputs/qsvt_approximation_tradeoff/`: consolidated tolerance frontier and
  query-count report.
- `outputs/qsvt_multicase_approximation_diagnostics/`: resource-safe
  approximation diagnostics for multiple PYPOWER IEEE cases where feasible.

The degree-error-query trade-off indicates that tighter pointwise-error
tolerance may require substantially higher polynomial degree and query count,
especially for smaller alpha values or sharper spectral filters.

Polynomial fallback rows are bounded polynomial approximation diagnostics, not
full QSP/QSVT phase synthesis. Full phase synthesis should only be claimed when
the optional phase-synthesis report performs synthesis and the validation status
supports that claim.

See `docs/QSVT_APPROXIMATION_VALIDATION.md` for the approximation-specific
claim boundaries, methods, and limitations.

## Phase-Response Convention Diagnostics

`outputs/qsvt_phase_response_convention_diagnostics/` diagnoses the scalar
phase-response convention used for optional PennyLane QSVT validation. It
checks phase order, sign, endpoint offsets, signal convention, response
component, and monomial coefficient basis.

The diagnostic first validates known sanity polynomials:

- `x`;
- `0.5 x`;
- `x^3`;
- `0.5 x + 0.25 x^3`.

The canonical convention is PennyLane-style `RX(2 arccos x)` with `PCPhase`
projectors and `real(U[0,0])` response. Passing these sanity polynomials fixes
the scalar convention mismatch. It does not by itself prove that the bounded
Ridge/Tikhonov target passes a strict phase-response tolerance.

The Phase 1 stable target path adds:

- `outputs/qsvt_phase_backend_audit/`: records whether available tools accept
  monomial coefficients, Chebyshev coefficients, or function values and whether
  they can return phase angles.
- `outputs/qsvt_stable_phase_candidates/`: builds bounded Ridge/Tikhonov
  polynomial candidates in Chebyshev basis, checks approximation error,
  boundedness, odd parity, basis conversion, coefficient dynamic range, and
  post-conversion boundedness.
- `outputs/qsvt_phase_sanity_regression/`: validates `x`, `0.5 x`, `x^3`, and
  `0.5 x + 0.25 x^3` under the scalar convention.
- `outputs/qsvt_stable_target_phase_validation/`: attempts phase synthesis only
  for candidate rows marked safe and compares the scalar phase response with
  the bounded Ridge/Tikhonov target.

No unstable polynomial is forced into phase synthesis. The strict `1e-3`
target-level tolerance is not relaxed; failed and skipped rows remain visible.

Phase 1B extends this with `outputs/qsvt_phase_external_backend_audit/`,
`outputs/qsvt_external_backend_sanity_regression/`, and
`outputs/qsvt_external_backend_phase_validation/`. pyqsp is treated as a
Chebyshev-basis scalar QSP backend. PennyLane remains monomial-only, so unsafe
monomial candidates are still skipped rather than synthesized.

The latest Phase 1B result passes for `bounded_ridge_tikhonov_pyqsp` using
pyqsp symmetric-QSP phases. The accepted candidate uses Chebyshev-basis input,
degree 201, 202 phases, full-domain max error `4.668e-4`, and
actual-singular-value max error `8.673e-5`. The pyqsp sanity regression passed
before this row was interpreted. This is scalar phase-response validation only,
not hardware execution, not block-encoded matrix execution, and not quantum
speedup evidence.

## Non-Brute-Force Refinement

The follow-up refinement adds diagnostic scripts under
`src/robust_qsvt_se/qsvt/nonbruteforce_refinement.py` and `scripts/`:

- `scripts/diagnose_qsvt_phase_target_failure.py`
- `scripts/run_qsvt_stable_phase_validation_attempt.py`
- `scripts/diagnose_qsvt_ieee300_spectral_difficulty.py`
- `scripts/run_qsvt_spectrum_aware_diagnostics.py`
- `scripts/run_qsvt_ieee118_targeted_refinement.py`
- `scripts/build_qsvt_nonbruteforce_refinement_summary.py`

The generated summary is
`outputs/qsvt_nonbruteforce_refinement_summary/nonbruteforce_refinement_summary.md`.
It reports a partial pass: IEEE118 passes the strict full-interval
approximation diagnostic at degree 1501, IEEE300 remains failed at degree
1001, while the bounded Ridge/Tikhonov target now has a separate pyqsp scalar
full-domain phase-response pass. The older unresolved phase rows remain
historical PennyLane/monomial-path diagnostics.

No brute-force degree escalation is used. The IEEE118 refinement is limited to
the approved degree budget, and failed or numerically unstable rows remain
visible in the outputs. The strict `1e-3` tolerance is not relaxed to create a
passing result.

The new spectrum-aware rows are diagnostic only. Column equilibration can
reduce a diagnostic condition number and polynomial error for the scaled matrix,
but this does not change the main estimator results and does not prove quantum
speedup, quantum advantage, or QSVT superiority over Ridge/Tikhonov.

For the bounded Ridge/Tikhonov target, the report keeps phase-response status
separate from polynomial approximation status. A failed phase-response row means
the configured degree or stable phase-synthesis path did not meet the target
tolerance for that backend/path. Historical failed rows should be described as
backend-specific diagnostics superseded by the pyqsp Chebyshev-basis scalar
full-domain pass, not as hardware failure or hidden success.

See `docs/QSVT_PHASE_RESPONSE_CONVENTIONS.md`.

## Failure-Fix Follow-Up

The remaining-failure follow-up adds a second, stricter diagnostic layer in
`src/robust_qsvt_se/qsvt/failure_fix.py`:

- `scripts/fix_qsvt_phase_validation_stable_basis.py`
- `scripts/run_qsvt_preconditioned_ieee300_estimator.py`
- `scripts/diagnose_qsvt_ieee300_residual_weighted_error.py`
- `scripts/build_qsvt_failure_fix_summary.py`

The stable phase path evaluates candidate bounded Ridge/Tikhonov polynomials in
their native Chebyshev basis, checks boundedness and parity, then audits
monomial conversion before optional phase synthesis. Phase validation is claimed
only if approximation error, converted boundedness, sanity-polynomial status,
and scalar phase-response error all meet the strict `1e-3` boundary.

The formal preconditioned IEEE300 path creates explicitly labeled
column-equilibrated estimator variants. Coordinate-penalty Ridge and
transformed-penalty Ridge are separate rows, and neither overwrites the original
unpreconditioned Ridge/QSVT-target claims.

## Phase 2 Preconditioned Alpha Diagnostics

Phase 2 adds `outputs/qsvt_phase2_preconditioned_alpha_sweeps/`,
`outputs/qsvt_phase2_alpha_selection/`, and `outputs/qsvt_phase2_summary/`.
The new diagnostics evaluate IEEE118 and IEEE300 for:

- original Ridge;
- coordinate-preconditioned Ridge;
- transformed-penalty preconditioned Ridge;
- original QSVT diagnostic;
- preconditioned QSVT diagnostic.

Coordinate-preconditioned Ridge is a separate estimator and may degrade
residual/RMSE. It is not automatically a replacement for original Ridge.
Transformed-penalty preconditioning preserves the original x-space penalty and
is reported as a consistency-preserving formulation. Alpha selection is a
diagnostic score, not a field-calibrated rule. QSVT diagnostic rows report
approximation/resource metrics and do not imply QSVT superiority over
Ridge/Tikhonov under the same alpha.

The residual-weighted spectral diagnostic computes how pointwise filter error
aligns with `u_i^T r_tilde` singular-direction energy. This is useful for
solution-relevance analysis, but it is not full-interval validation.

See:

- `docs/QSVT_STABLE_PHASE_SYNTHESIS.md`
- `docs/QSVT_PRECONDITIONED_IEEE300_VARIANT.md`
- `docs/QSVT_RESIDUAL_WEIGHTED_SPECTRAL_ERROR.md`

## Adaptive Multicase Degree Search

`outputs/qsvt_adaptive_multicase_degree_search/` searches the configured odd
degree grid for IEEE/PYPOWER cases using the bounded target and
`odd_chebyshev_minimax_lp` method. The default alpha is `1e-2`, the strict
tolerance is `1e-3`, and the query-count proxy is `2 * degree + 1`.

The report records the first configured passing degree when one is found. If no
candidate degree passes, the row remains a failure row with the best tested
degree and explicit caveat. This makes larger-case approximation pressure
auditable without implying speedup or hardware execution.

See `docs/QSVT_MULTICASE_DEGREE_SEARCH.md`.

## Shot-Level Readout Model

The standalone shot-readout report writes
`outputs/qsvt_shot_readout/shot_readout_summary.csv` and
`observable_estimates.csv`. It models selected observables only:

- selected state-component amplitude/probability proxy;
- update-vector norm proxy;
- residual-norm proxy.

For probability proxies, the report uses the Bernoulli standard error
`sqrt(p(1-p)/N)` and a conservative additive-error shot estimate
`ceil(0.25 / epsilon_obs^2)`. This is a sampling-cost model for selected
observables. It is not hardware execution and does not solve full-vector
reconstruction.

## Hardware-Aware Proxy

`outputs/qsvt_hardware_aware/` reports dependency-free hardware-aware proxy
costs: logical and ancilla qubits, total qubits, QSVT degree, query count,
controlled block-encoding calls, one-qubit and two-qubit gate proxies, depth,
routing overhead, optional dependency availability, and a shot-budget
placeholder.

The default path records whether optional packages such as Qiskit, Qiskit Aer,
and PennyLane are importable, but it does not require or use them. The report is
not full IEEE-scale hardware execution and does not demonstrate quantum
advantage.

## Block-Encoding Scalability

The dense Julia block encoding is a validation prototype. The scalability
report under `outputs/qsvt_block_encoding_scalability/` records matrix size,
nonzeros, density, dense-encoding dimension, index-qubit proxy, beta, kappa
when feasible, and the caveat that dense block encodings are not scalable
sparse-access oracles. See `docs/QSVT_BLOCK_ENCODING_SCALABILITY.md`.

## Multi-Case Resource Diagnostics

The multi-case report under `outputs/qsvt_multicase_resource_diagnostics/`
attempts resource-only AC weighted-system construction for PYPOWER IEEE cases
and catches failures per case. It writes a summary table and `failure_log.csv`.
It does not trigger nonlinear IEEE300 experiments or hardware-native QSVT
execution.

## Artifact Audit

The audit report under `outputs/qsvt_engineering_extension_audit/` checks
expected files, selected columns, finite metrics, conservative thresholds,
manifests, claim-matrix coverage, and documentation wording. Forbidden phrases
inside explicit avoid-wording, do-not-claim, limitation, caveat, or boundary
contexts are classified as safe context.

## Preconditioning Diagnostics

The optional diagnostic uses column equilibration:

```text
H_tilde M^{-1} y ~= r_tilde,  Delta x = M^{-1} y
```

It compares condition number and degree/query estimates before and after
preconditioning, then reports residual and solution difference against the
unpreconditioned Ridge result. This is diagnostic evidence for possible
resource reduction. It does not prove quantum speedup.

## Paper-Ready Finalization

The final paper-support layer writes:

- `outputs/qsvt_preconditioned_variant_sweeps/`
- `outputs/qsvt_preconditioning_resource_comparison/`
- `outputs/paper_ready_qsvt_tables/`
- `outputs/final_qsvt_artifact_freeze/`
- `outputs/final_qsvt_claim_safety_audit/`

The sweep outputs evaluate IEEE118 and IEEE300 over controlled alpha, noise,
missing-row, and bad-data grids where construction succeeds. Original Ridge,
coordinate-preconditioned Ridge, transformed-penalty preconditioned Ridge,
unpreconditioned QSVT diagnostics, and preconditioned QSVT diagnostics are
separate rows. Coordinate-preconditioned Ridge is a new estimator variant. The
transformed-penalty row is a consistency-preserving check for the original
x-space penalty.

The paper-ready tables aggregate generated artifacts only. They do not invent
new measurements or convert diagnostic rows into full validation claims.

## Safe Claims

- QSVT-compatible implementation pathway.
- Regularized spectral filtering.
- Controlled IEEE/PYPOWER benchmark systems.
- Generated measurement rows from benchmark network models.
- Resource-aware feasibility analysis.
- Dense block-encoding validation for small normalized matrices.
- Exact QSVT-target/Ridge equivalence under the same alpha and filter.
- Selected-alpha bounded polynomial approximation diagnostics with explicit
  pass/fail rows.
- Degree-error-query trade-off diagnostics for bounded QSVT-compatible
  polynomial approximation.
- Stable-basis phase diagnostics with explicit pass/fail/skip rows.
- Formal preconditioned/equilibrated estimator variants, labeled separately
  from the original estimator claims.
- Residual-weighted spectral diagnostics for solution-relevance analysis only.
- Preconditioned variant sweeps and paper-ready artifact aggregation with
  explicit claim boundaries.
- Hardware-aware and readout reports as proxy feasibility analysis only.

## Claims To Avoid

- Quantum speedup.
- Quantum advantage.
- Full IEEE-scale hardware execution.
- Validation on PMU/SCADA field data.
- QSVT numerically beating Ridge/Tikhonov when the same alpha and filter are
  used.
- Deployment-ready quantum state estimation.
- Full validation inferred from residual-weighted or restricted-interval
  diagnostics.
- Original IEEE300 full-interval validation inferred from a preconditioned
  variant.
- Preconditioned coordinate Ridge replacing original Ridge when residual/RMSE
  degrade.
- Phase-level bounded Ridge/Tikhonov validation inferred from sanity
  polynomials alone.

## Limitations And Future Work

- Dense block encoding is a validation prototype, not a scalable oracle.
- Resource estimates are proxy estimates and omit state preparation, full
  readout sampling, error correction, and hardware-native compilation.
- Standalone shot-level readout is a selected-observable probability-proxy
  model, not full-vector tomography.
- Hardware-aware rows are dependency-free proxy estimates unless a future task
  explicitly adds backend execution.
- Dense block encoding is not a sparse-access or data-loading oracle.
- Multi-case diagnostics catch construction failures and do not imply every
  case is hardware executable.
- Optional phase synthesis depends on optional dependencies and may be skipped
  or fail validation; polynomial fallback results should not be described as
  full phase synthesis.
- The exact QSVT-target state demo is a classical spectral simulation.
- Large IEEE systems remain resource/readout-analysis targets unless a future
  task adds hardware-native block encoding and phase synthesis at scale.
- Preconditioning diagnostics are simple column scaling only.
- Stable phase synthesis remains unresolved unless a generated bounded
  Ridge/Tikhonov target row passes all declared criteria.
- Preconditioned IEEE300 rows are new variants and do not replace original
  unpreconditioned estimator claims.
- Residual-weighted spectral errors do not replace full-interval approximation
  validation.
- Paper-ready tables and artifact freezes aggregate evidence; they are not new
  experiments.
