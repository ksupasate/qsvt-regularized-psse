# QSVT Phase 2 Preconditioned Alpha Diagnostics

## Purpose

Phase 2 evaluates original and preconditioned estimator variants separately for
IEEE118 and IEEE300. It adds alpha-sensitivity and alpha-selection diagnostics
without changing the original Ridge/Tikhonov solver, pseudoinverse solver,
measurement generation, or QSVT-target filter definition.

The variants are:

- `original_ridge`
- `coordinate_preconditioned_ridge`
- `transformed_penalty_preconditioned_ridge`
- `original_qsvt_diagnostic`
- `preconditioned_qsvt_diagnostic`

Coordinate-preconditioned Ridge uses a standard penalty in the equilibrated
coordinate and is a separate estimator. It may degrade residual or RMSE and is
not automatically a replacement for original Ridge. Transformed-penalty
preconditioned Ridge preserves the original x-space penalty and is reported as
a consistency-preserving formulation.

## Scripts And Outputs

```bash
.venv/bin/python scripts/run_qsvt_phase2_preconditioned_alpha_sweeps.py
.venv/bin/python scripts/build_qsvt_phase2_alpha_selection_report.py
.venv/bin/python scripts/build_qsvt_phase2_summary.py
.venv/bin/python scripts/build_qsvt_phase2_complete_summary.py
.venv/bin/python scripts/build_qsvt_phase2_figures.py
.venv/bin/python scripts/run_qsvt_phase2_optional_ieee57.py
```

The scripts write:

- `outputs/qsvt_phase2_preconditioned_alpha_sweeps/phase2_sweep_results.csv`
- `outputs/qsvt_phase2_preconditioned_alpha_sweeps/phase2_sweep_results.json`
- `outputs/qsvt_phase2_preconditioned_alpha_sweeps/phase2_sweep_summary.csv`
- `outputs/qsvt_phase2_preconditioned_alpha_sweeps/phase2_failure_log.csv`
- `outputs/qsvt_phase2_preconditioned_alpha_sweeps/manifest.json`
- `outputs/qsvt_phase2_alpha_selection/alpha_selection_summary.csv`
- `outputs/qsvt_phase2_alpha_selection/alpha_selection_summary.json`
- `outputs/qsvt_phase2_alpha_selection/alpha_selection_trace.csv`
- `outputs/qsvt_phase2_alpha_selection/alpha_selection_report.md`
- `outputs/qsvt_phase2_alpha_selection/alpha_selection_metric_definitions.md`
- `outputs/qsvt_phase2_alpha_selection/manifest.json`
- `outputs/qsvt_phase2_summary/phase2_summary.md`
- `outputs/qsvt_phase2_summary/phase2_summary.csv`
- `outputs/qsvt_phase2_summary/phase2_summary.json`
- `outputs/qsvt_phase2_summary/manifest.json`
- `outputs/qsvt_phase2_complete_summary/phase2_complete_summary.csv`
- `outputs/qsvt_phase2_figures/phase2_figure_captions.md`
- `outputs/qsvt_phase2_optional_ieee57/ieee57_phase2_status.json`

## Alpha Selection

The alpha-selection report includes residual-minimizing alpha,
RMSE-minimizing alpha when RMSE is available, QSVT-error-minimizing alpha,
query/degree-minimizing alpha, a legacy QSVT-resource-friendly alpha, and a
diagnostic joint score with default weights:

```text
w_r = 0.5
w_e = 0.3
w_q = 0.2
```

Alpha selection is diagnostic and controlled-benchmark-specific. It is not a
field-calibrated operational rule and does not replace estimator validation.

## Phase 1 Relationship

Phase 1 passed scalar full-domain phase-response validation using pyqsp for the
bounded Ridge/Tikhonov target at alpha `1e-2`. PennyLane's monomial path remains
limited by coefficient instability for the target; pyqsp accepted
Chebyshev-basis input after sanity regression passed.

## Claim Boundary

Phase 2 reports controlled IEEE/PYPOWER benchmark diagnostics. It does not
demonstrate hardware execution, block-encoded matrix execution, quantum
speedup, PMU/SCADA field-data validation, or QSVT superiority over
Ridge/Tikhonov.
