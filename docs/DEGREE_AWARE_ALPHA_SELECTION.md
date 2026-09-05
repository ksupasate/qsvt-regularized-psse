# Degree-Aware Alpha Selection

Module: `robust_qsvt_se.paper.degree_aware_alpha`.

The classical Ridge/Tikhonov filter is `P_alpha(sigma) = sigma/(sigma^2+alpha)`.
The bounded QSVT-compatible target is `f_{alpha,bounded}(s) = (1/C) s/(s^2 +
alpha/beta^2)` on `s in [0,1]`, with `beta` the block-encoding normalization and
`C = max(1, max_{[0,1]} sigma/(sigma^2+alpha))`. This study quantifies the
trade-off between classical accuracy and the QSVT polynomial degree required to
approximate the bounded target to tolerance.

**Key conservative interpretation:** *the best classical regularization
parameter is not automatically the best QSVT-implementable choice under a degree
budget.* No speedup is claimed; the QSVT target uses the same alpha as Ridge.

## Implemented behavior

- For each case, alpha in `{1e-6 ... 1}`, tolerance in `{1e-2, 1e-3, 1e-4}`, and
  degree budget in `{25, 51, 101, 201}`, the workload reports: classical RMSE,
  residual norm, condition number, spectrum-point action degree, bounded-target
  approximation error, whether the target is met, whether the degree budget is
  met, the bounded scaling constant `C`, and the selected alpha under each rule.
- The required degree uses the repository's bounded-target convention
  (`fit_bounded_ridge_polynomial`, `bounded_ridge_normalization_C`,
  `bounded_ridge_target`, `qsvt_odd_degree`) with error measured at the matrix's
  actual normalized singular values.

Selection rules compared: `best_classical_rmse`, `default_alpha`,
`spectrum_based_alpha` (nearest grid alpha to `sigma_min^2`),
`degree_aware_under_dmax`, and `degree_aware_under_tolerance`.

## Observed behavior (well-posed linearized systems)

- Classical RMSE is nearly flat across alpha, marginally best at the largest alpha.
- Under the bounded convention the required degree **grows with alpha** while the
  bounded constant `C` **shrinks with alpha** — they trade off in opposite
  directions. Small alpha is degree-cheap but C-expensive (smaller success
  probability); large alpha is C-cheap but degree-expensive, often exceeding the
  budget.
- Consequently, for larger cases (e.g. ieee30/ieee57) the best-classical alpha can
  exceed the degree budget while a degree-aware alpha stays feasible at
  essentially the same RMSE.

## Degree distinction

- `spectrum_point_degree` measures action error only at the actual normalized
  singular values. It is empirical matrix-action evidence.
- `uniform_grid_degree` additionally checks a dense uniform domain, boundedness,
  and odd parity.
- `phase_synthesis_available` is false unless phases were actually synthesized
  for that row. The revised rows do not perform per-row synthesis.
- `(2d+1)` is retained only as `degree_derived_query_count`; it is not labeled a
  realizable QSVT query count unless boundedness, parity, uniform admissibility,
  and phase synthesis all pass.

## Proxy-level diagnostics

- Spectrum-point degree and `C` are empirical resource proxies.
- An independent classical alpha sweep
  (`outputs/full_alpha_sensitivity_classical/`) is reused as a cross-reference;
  the degree columns are computed here from the same weighted Jacobian.

## Excluded components

- Per-row phase synthesis and gate-level validation for the alpha grid.
- Any run on quantum hardware.

## Limitations and future work

- The conclusion depends on the bounded-target normalization convention; it is
  documented and matched to the repository's existing degree-alpha precision
  sweep. Future work could synthesize phase factors for the degree-aware-selected
configurations.

The observed trend is an implementation tradeoff for the tested bounded-target
convention and degree grid, not a universal QSVT scaling theorem.

## Required table columns (`degree_aware_alpha_grid.csv`)

`case, scenario, alpha, tolerance, degree_budget, rmse, residual_norm,
condition_number, degree_required, approx_error, target_met, phase_available,
degree_budget_met, bounded_scale_C, selection_rule, selected, notes`.
