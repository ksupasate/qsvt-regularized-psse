# QSVT Implementation Scope

This project separates QSVT-related evidence levels so that public descriptions
do not overclaim what was implemented.

## QSVT Implementation Levels

| Level | Description | Status |
| --- | --- | --- |
| Level 1 | Classical spectral simulation of the regularized filter on large IEEE benchmarks | Complete |
| Level 2 | QSP/QSVT phase synthesis and scalar phase-response validation | Complete |
| Level 3 | Dense-unitary matrix QSVT correctness circuit for research matrices | Complete |
| Level 4 | Explicit small-matrix block-encoding QSVT prototype with phase rotations | Complete for IEEE14 2x2 and 4x4 weighted-Jacobian submatrices |
| Level 5 | Full IEEE-scale QSVT resource and feasibility estimates | Complete |
| Level 6 | Full IEEE-scale hardware-native QSVT execution | Not complete and not claimed |

## Inverse-style diagnostic baselines

The executable estimator set includes two inverse-style diagnostic baselines:

- `qsvt_unregularized_inverse` applies the inverse target `1 / sigma` with a
  numerical cutoff. It is intentionally labeled as an unstable ablation and is
  not the proposed regularized QSVT filter.
- `hhl_style_inverse_proxy` is a classical diagnostic proxy for HHL-style
  condition-number sensitivity, target precision scaling, and output-readout
  caveats. It does not execute an HHL circuit and must not be cited as
  hardware-level HHL evidence.

## Large IEEE Benchmarks

The IEEE14/30/57/118/300 benchmark runs use classical SVD spectral filters and
resource proxy estimates. They do not execute QSVT circuits and do not imply
quantum speedup.

## Full Phase Synthesis

`configs/qsvt_phase_synthesis_full.yaml` fits a bounded odd polynomial for:

```text
P_alpha(sigma) = sigma / (sigma^2 + alpha)
```

The configured full demo uses `alpha = 0.01`, normalized domain `[0.2, 1.0]`,
degree `35`, and PennyLane `poly_to_angles(..., "QSVT", angle_solver="iterative")`.
It writes polynomial, phase, and phase-response validation artifacts to
`outputs/qsvt_phase_synthesis_full/`. Phase angles are cached under
`outputs/qsvt_phase_cache/` using a hash of coefficients, solver settings, and
fit metadata so repeated research-matrix runs do not recompute identical phase
lists.

The paper-validation config is:

```bash
.venv/bin/python -m robust_qsvt_se.qsvt.run_phase_demo \
  --config configs/qsvt_phase_validation_paper.yaml
```

It writes `phase_validation_report.json`, `qsp_validation_grid.csv`,
`phase_implemented_error.csv`, and `phase_validation_plot.png` under
`outputs/qsvt_phase_validation_paper/`. The report explicitly checks target
normalization, odd parity, boundedness, non-dummy finite phase angles, and
phase-response error against the configured thresholds.

## Research-Matrix Circuit Demos

The matrix demos extract weighted Jacobians from the PYPOWER AC benchmark
pipeline. The extraction layer can return the full weighted Jacobian or a
deterministic submatrix selected by high weighted column norms and row norms.
All outputs record the selected rows, selected columns, measurement labels,
state labels, singular values, condition number, normalization factor, and
whether the run used `full_matrix`, `submatrix`, or `resource_estimate_only`
scope.

PennyLane:

```bash
.venv/bin/python -m robust_qsvt_se.qsvt.pennylane_matrix_qsvt \
  --config configs/qsvt_pennylane_matrix_ieee14_full.yaml
.venv/bin/python -m robust_qsvt_se.qsvt.pennylane_matrix_qsvt \
  --config configs/qsvt_pennylane_matrix_ieee14_8x8.yaml
.venv/bin/python -m robust_qsvt_se.qsvt.pennylane_matrix_qsvt \
  --config configs/qsvt_pennylane_matrix_ieee30_8x8.yaml
```

Qiskit:

```bash
.venv/bin/python -m robust_qsvt_se.qsvt.qiskit_matrix_qsvt \
  --config configs/qsvt_qiskit_matrix_ieee14_full.yaml
.venv/bin/python -m robust_qsvt_se.qsvt.qiskit_matrix_qsvt \
  --config configs/qsvt_qiskit_matrix_ieee14_8x8.yaml
.venv/bin/python -m robust_qsvt_se.qsvt.qiskit_matrix_qsvt \
  --config configs/qsvt_qiskit_matrix_ieee30_4x4.yaml
```

The Qiskit path imports the dense QSVT unitary as a `QuantumCircuit` unitary
gate for correctness and transpiles that dense unitary only when the configured
qubit limit permits it. This is an auditable circuit-cost artifact for a dense
unitary correctness path, but it is not a scalable hardware-native
block-encoding oracle construction.

Explicit block-encoding hardware prototype:

```bash
.venv/bin/python -m robust_qsvt_se.qsvt.run_hardware_qsvt_demo \
  --config configs/qsvt_hardware_ieee14_2x2.yaml
.venv/bin/python -m robust_qsvt_se.qsvt.run_hardware_qsvt_demo \
  --config configs/qsvt_hardware_ieee14_4x4.yaml
```

This path extracts deterministic IEEE14 weighted-Jacobian submatrices, builds a
canonical square block encoding whose top-left block is the normalized matrix,
and applies a structured QSVT sequence of block-encoding calls, adjoint calls,
and explicit projector phase rotations. The canonical block-encoding primitive
is represented as a small dense unitary gate, but the whole QSVT transformation
is not imported as one opaque dense unitary. The outputs include block-encoding
validation, top-left block error, unitarity error, transpiled `rz/sx/x/cx` gate
counts, and comparison to the classical SVD spectral transform.

Circuit scaling:

```bash
.venv/bin/python -m robust_qsvt_se.qsvt.circuit_scaling \
  --config configs/qsvt_circuit_scaling.yaml
```

The scaling run evaluates deterministic IEEE14 and IEEE30 weighted-Jacobian
submatrices at 2x2, 4x4, and 8x8. It records 16x16 rows as infeasible under the
default laptop-safe policy rather than silently skipping them. These artifacts
support the claim that full IEEE118/300 hardware-level QSVT circuit simulation
is not currently claimed.

Unified full-matrix feasibility:

```bash
.venv/bin/python -m robust_qsvt_se.qsvt.run_full_matrix_qsvt \
  --config configs/qsvt_full_matrix_ieee14.yaml
.venv/bin/python -m robust_qsvt_se.qsvt.run_full_matrix_qsvt \
  --config configs/qsvt_full_matrix_ieee300.yaml
```

IEEE14 and IEEE30 full matrix paths are attempted within the configured
statevector limits. IEEE57 is attempted for PennyLane when feasible. IEEE118 and
IEEE300 produce full-matrix resource estimates and deterministic submatrix
QSVT demonstrations by default; the output metadata states this explicitly.

## Safe Claim Boundary

Safe wording:

> We provide small research-derived QSVT circuit demonstrations for normalized
> IEEE weighted-Jacobian matrices, full-matrix demonstrations where feasible,
> and resource estimates plus deterministic submatrix demonstrations for larger
> IEEE systems.

Unsafe wording:

> We execute full IEEE300 QSVT state estimation on quantum hardware or
> demonstrate quantum speedup.

## Audit Note

The clean-room audit preserves the QSVT evidence boundary:

- `qsvt_regularized` is the same numerical target filter as ridge/Tikhonov in
  the large classical simulator.
- `outputs/qsvt_phase_validation_paper/` is the final scalar phase-validation
  evidence.
- `outputs/qsvt_hardware_ieee14_2x2/` and
  `outputs/qsvt_hardware_ieee14_4x4/` are explicit small block-encoding
  prototypes.
- `outputs/qsvt_resource_full_ieee/` is a proxy resource and feasibility
  estimate, not hardware execution.

Non-final QSVT phase and matrix explorations are archived under
`outputs/historical/`.
