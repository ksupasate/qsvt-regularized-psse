# Selected Signed-Observable Readout

Modules: `robust_qsvt_se.qsvt.selected_observables`,
`robust_qsvt_se.qsvt.readout_diagnostics`. Workload:
`paper.selected_observable_workload`.

These diagnostics estimate **selected linear functionals** `y_l = l^T dx_alpha`
of the matched-alpha Ridge/Tikhonov update
`dx_alpha = (H~^T H~ + alpha I)^-1 H~^T r~`. Each readout estimates **one**
selected functional; this is **not full-vector readout** and recovers neither the
full signed update vector nor its sign for the energy-style observable.

## Implemented behavior

Observable builders, keyed to AC state metadata when available:

- voltage-magnitude correction at a bus, `e_i^T dV`;
- voltage-angle correction at a bus, `e_i^T dtheta`;
- branch angle-difference correction, `(e_i - e_j)^T dtheta`;
- area / selected-bus aggregate functional, `l^T dx`;
- energy-style observable `||Pi_A dx||^2` (for comparison).

All main-workload supports are predetermined before solving: the first valid bus,
the first valid branch, or fixed first-four angle/voltage sets. The former
solution-dependent ranking policy has been removed.

Exact matched values are computed directly from `dx_alpha`. Two unbiased
Monte-Carlo readout models are swept over shot budgets `{100, 1000, 10000,
100000}` with deterministic seeds:

- **sign-aware** (`hadamard_test_estimate`, reused from the existing signed-readout
  diagnostic): probability `p = (1+mu)/2`, estimate `||l|| ||dx|| (2 p_hat - 1)`;
- **basis-sampling** (`basis_sampling_energy_estimate`): estimate
  `||dx||^2 (successes/shots)` of the in-subspace probability.

Mean/max/std absolute error and relative error are reported per shot budget; error
falls on the order of `1/sqrt(shots)`.

## Modeled assumptions

- A measurement protocol is modeled; no circuit is synthesized.
- When bus/state metadata is unavailable, the builders fall back to
  coordinate-/block-level observables and label them as such.
- Physical recovery tracks `||r~||`, block-encoding normalization `beta`, bounded
  target scaling `C/beta`, and output-state norm. Postselection normalization is
  labeled `not_available` for the full-matrix synthetic diagnostic rather than
  inferred.

## Proxy-level diagnostics

- The shot sweep is a Monte-Carlo proxy for the readout cost of one functional.
- The IEEE-57 first-branch angle-difference functional has relative error
  `0.05589030178617602` at 100000 shots, above the 5% target; the miss is retained.

## Excluded components

- Full signed-vector recovery (one functional per coordinate) is out of scope.
- Sign recovery for the energy-style (squared-amplitude) observable.
- Any run on quantum hardware.

## Readout-model map

| Observable class | Readout model | Sign-aware | Basis-sampling | Full-vector |
| --- | --- | --- | --- | --- |
| voltage magnitude / angle / branch diff / area | sign-aware inner-product proxy | yes | no | no |
| energy-style `||Pi_A dx||^2` | basis-sampling squared-amplitude | no | yes | no |

## Limitations and future work

- `dx_alpha` is the matched-alpha Ridge/Tikhonov update, so no QSVT-over-Ridge
superiority is implied. The added 4x4 demonstration connects a phase-synthesized
  circuit transform to one predetermined signed functional, but its shot noise is
  still synthetic and is not a compiled readout circuit.

## Required table columns

`selected_observables.csv`: `case, matrix_source, observable_id, observable_type,
physical_meaning, selection_policy, selected_before_solving, depends_on_solution,
support_size, units_or_normalization, exact_value, residual_norm,
block_encoding_beta, bounded_scale_C, qsvt_target_scaling_C_over_beta,
postselection_normalization_status, output_state_norm, reported_value_domain,
physical_recovery_status, readout_model,
basis_sampling_accessible, sign_aware_required, full_vector_required, status,
notes`.

`readout_shot_sweep.csv`: `case, observable_id, shots, trials, mean_abs_error,
max_abs_error, std_abs_error, relative_error, target_relative_error, target_met,
target_status, readout_model, status`.
