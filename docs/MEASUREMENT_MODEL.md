# Measurement Model

## Data boundary

IEEE/MATPOWER-compatible cases loaded through PYPOWER contain buses, branches,
generators, topology, electrical parameters, and an operating point. They do
not contain real PMU/SCADA streams, a utility sensor deployment, field sensor
covariance, or recorded field bad-data events.

Repository code generates all measurement systems. “IEEE14,” for example,
identifies the network model from which rows are constructed; it does not mean
that IEEE14 field measurements exist.

## Measurement rows

| Family | Generated row types |
|---|---|
| Synthetic | Abstract `synthetic_weighted_row` rows from a controlled SVD construction |
| DC-linearized | `branch_flow`, `bus_injection`, and configured `angle` rows |
| AC-linearized/nonlinear | `voltage_magnitude`, `p_injection`, `q_injection`, `p_branch_flow`, and `q_branch_flow`, subject to config switches |
| QSVT matrix paths | Rows of a generated weighted AC Jacobian, sometimes restricted to a deterministic selected block |

The built-in IEEE14-style fixture is bundled project data. Paper-scale network
models use PYPOWER package fixtures for IEEE14, IEEE30, IEEE57, IEEE118, and
IEEE300. Neither source is a field measurement archive.

## Nonlinear AC generation

The nonlinear workflow first generates raw measurements:

\[
z=h(x_{\mathrm{true}})+e+b,
\]

where \(h\) is the configured AC measurement function, \(e\) is controlled
zero-mean Gaussian noise, and \(b\) is a controlled sparse signed bad-data
vector when enabled. Missing rows are selected once and the same row indices are
used when the Jacobian is refreshed at each iteration.

At iteration \(k\), the runner builds

\[
r_k=z-h(x_k), \qquad
H_k=\left.\frac{\partial h}{\partial x}\right|_{x_k},
\]

weights the rows, and solves for an update. A convergence flag, iteration count,
and failure reason are recorded separately.

## Single-step generation

Synthetic, DC-linearized, and AC-linearized single-step experiments perturb an
already weighted residual:

\[
\widetilde r_{\mathrm{perturbed}}
=\widetilde r_{\mathrm{clean}}+\widetilde e+\widetilde b.
\]

This is not equivalent to perturbing a raw nonlinear measurement vector and
must not be described as such.

For AC-linearized experiments, the measurement equations are evaluated at a
generated reference state and a perturbed linearization point. RMSE compares the
estimated update with the generated true update between those points.

## Covariance and weighting

Each row has a configured standard deviation \(\sigma_i>0\). The covariance is
diagonal and implicit:

\[
R_{ii}=\sigma_i^2.
\]

Weighting is implemented row by row:

\[
\widetilde H_{i,:}=H_{i,:}/\sigma_i,
\qquad
\widetilde r_i=r_i/\sigma_i.
\]

Weak-area experiments multiply selected row standard deviations by a configured
factor. This is a controlled stress mechanism, not a field-estimated covariance
model.

## Noise, missing measurements, and bad data

- Noise uses a seeded NumPy `Generator` and a zero-mean Gaussian model.
- Missing measurements are random row drops at the configured ratio.
- Bad data are sparse signed additive perturbations applied after missing-row
  selection. Nonlinear bad data scale with the row standard deviation.
- Targeting may be random or restricted to configured weak-area eligibility.
- These mechanisms are experimental controls and are not calibrated to a named
  utility or measurement campaign.

## Dimensions and metadata

The manifest's `row_count` is the configured count before random missing-row
removal unless the entry explicitly describes a QSVT grid or selected block.
Actual retained rows, state dimension, case source, measurement types, standard
deviations, seed, and scenario parameters are stored in resolved configs and
output metadata.

## Condition number and RMSE

The primary condition number is computed from the weighted matrix:

\[
\kappa(\widetilde H)=
\sigma_{\max}(\widetilde H)/\sigma_{\min}(\widetilde H).
\]

Gain-matrix conditioning from normal-equation WLS is a separate diagnostic.
RMSE is computed against the stored generated reference vector:

\[
\operatorname{RMSE}=\sqrt{n^{-1}\sum_i(\hat x_i-x_i)^2}.
\]

## Limitations

- Inclusion switches generate placement; they do not reproduce utility sensor
  placement.
- Diagonal covariance omits cross-sensor correlation.
- Random row removal is not a communications or observability-contingency model.
- Sparse additive bad data do not model every cyber/physical failure process.
- Generated truth supports controlled error calculation but not field accuracy.

The more detailed implementation audit remains in
`docs/EXPERIMENT_MEASUREMENT_MODEL.md`.
