# Experiment Model

## Common weighted problem

The classical experiment runner solves a weighted rectangular system

\[
\widetilde H x \approx \widetilde r,
\qquad
\widetilde H=R^{-1/2}H,
\qquad
\widetilde r=R^{-1/2}r.
\]

The stored `WeightedSystem` contains the weighted matrix, weighted right-hand
side, reference state or update, and row metadata. It does not require a dense
covariance matrix because the configured covariance is diagonal.

The regularized update is

\[
x_\alpha
=\arg\min_x \|\widetilde Hx-\widetilde r\|_2^2
+\alpha\|x\|_2^2.
\]

For \(\widetilde H=U\Sigma V^\mathsf{T}\),

\[
x_\alpha
=V\,\operatorname{diag}\!\left(
\frac{\sigma_i}{\sigma_i^2+\alpha}
\right)U^\mathsf{T}\widetilde r.
\]

This is both the Ridge/Tikhonov solution and the exact classical target called
`qsvt_regularized` when the same `alpha` is used. Any later difference must be
attributed to polynomial approximation, block/support restriction, circuit
implementation, quantization, postselection, or finite-shot readout—not to a
different exact classical estimator.

## Experiment families

| `system.mode` or config schema | Model constructed | Reference used for RMSE | Role |
|---|---|---|---|
| `synthetic_linearized` | Controlled SVD-generated weighted matrix | Generated state vector | Conditioning isolation |
| `dc_power_flow_linearized` | DC branch-flow, injection, and angle rows | Generated bus-angle state | Fast power-system fixture |
| `ac_power_flow_linearized` | One Jacobian at a perturbed operating point | True update from that point | Linearized AC comparison |
| `ac_iterative_state_estimation` | Repeated linearized updates on built-in fixture | Generated AC state | Iterative smoke/regression |
| `nonlinear_ac_state_estimation` | Raw generated AC measurements and refreshed Jacobian | Generated AC state | Nonlinear benchmark study |
| `demo` | Bounded scalar target, polynomial, and phase response | Exact regularized scalar target | QSVT phase feasibility |
| `scaling` | Deterministically selected small weighted-Jacobian blocks | Matched classical polynomial action | Circuit correctness/scaling boundary |
| `resource` | Full weighted IEEE Jacobian dimensions/spectra | Not an estimator RMSE experiment | Resource proxy |

The first five schemas are handled by
`robust_qsvt_se.experiments.runner`. The last three have specialized QSVT
runners. `scripts/run_experiment.py` dispatches all four configuration shapes
and redirects researcher outputs away from frozen evidence.

## Estimators

| Name | Numerical operation | Interpretation |
|---|---|---|
| `pseudoinverse` | SVD pseudoinverse with configured cutoff | Unregularized least-squares baseline |
| `normal_equation_wls` | Solve \(H^\mathsf{T}H x=H^\mathsf{T}r\) | Conditioning-sensitive diagnostic baseline |
| `ridge` | Exact SVD Ridge/Tikhonov filter | Regularized classical reference |
| `truncated_svd` | Invert singular values above `tau` | Spectral cutoff baseline |
| `qsvt_regularized` | Exact classical evaluation of the Ridge filter target | Target simulation, not a quantum execution |
| `qsvt_unregularized_inverse` | Cutoff inverse proxy | Unstable diagnostic ablation |
| `hhl_style_inverse_proxy` | Condition/precision-sensitive classical proxy | Not an HHL circuit |
| `huber_irls` | Iteratively reweighted least squares | Robust classical baseline |
| `lav` | Least-absolute-value linear program using SciPy/HiGHS | Robust classical baseline |

Estimator failures are written as structured result rows with a reason. They
must remain in trial counts and failure-rate calculations.

## Configuration resolution

Standard YAML configurations are deep-merged with defaults in
`robust_qsvt_se.utils.config`, then validated before a run. A resolved config is
written to the output directory. Important fields are:

- `seed`: base NumPy generator seed;
- `system`: case source, mode, dimensions, measurement inclusion, and
  linearization/iteration settings;
- `scenario`: noise, missing-row, weak-area, and bad-data settings;
- `sweeps`: parameter paths, values, and explicit seed lists;
- `estimators`: estimator names and numerical hyperparameters;
- `qsvt_resource`: classical polynomial/resource diagnostics;
- `output`: run identifier, plot switch, and overwrite policy.

Specialized QSVT configs use one of `demo`, `scaling`, or `resource` as their
top-level section and write their own resolved config.

## Experiment outputs

A single standard run writes `metrics.csv`; a sweep writes trial-level
`aggregate_metrics.csv` and grouped `summary_metrics.csv`. Nonlinear runs also
write iteration traces. All paths registered for publication are listed in
`outputs/reproducibility_audit/experiment_manifest.json`.

Runtime fields measure local execution and are expected to vary. Scientific
comparisons use the configured metrics and tolerances, not equality of wall
clock values.

## QSVT evidence ladder

The repository deliberately separates:

1. exact classical regularized target evaluation;
2. bounded polynomial approximation;
3. scalar phase-response validation;
4. small selected-block circuit action;
5. finite-shot or selected-output readout studies;
6. resource proxies for larger matrices.

A success at one level does not imply success at a later level. In particular,
small-circuit agreement with a polynomial does not establish full-system state
accuracy, practical oracle construction, favorable readout cost, or speedup.
