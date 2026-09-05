# QSVT Selected-Observable Workload

This workload adds a concrete QSVT *implementation-pathway* layer on top of the
existing controlled IEEE/PYPOWER benchmark. It does **not** change any estimator.
Ridge/Tikhonov remains the matched classical reference, and the QSVT-compatible
target implements the same regularized spectral filter `sigma/(sigma^2+alpha)` at
the same alpha. All artifacts are feasibility and boundary evidence: no speedup,
no QSVT-over-Ridge numerical superiority, no validation on real PMU/SCADA field
measurements, no full-vector readout, and no run on quantum hardware.

## Components

| Task | Module | Output |
| --- | --- | --- |
| A — audit & integration plan | `paper/selected_observable_audit.py` | `implementation_audit.md`, `repo_integration_plan.json` |
| B — sparse access | `qsvt/sparse_access.py`, `paper/sparse_access_workload.py` | `sparse_access_summary.csv`, `sparse_access_validation.csv`, `sparse_access_report.md` |
| C — selected readout | `qsvt/selected_observables.py`, `qsvt/readout_diagnostics.py`, `paper/selected_observable_workload.py` | `selected_observables.csv`, `readout_shot_sweep.csv`, `readout_map.csv`, `readout_summary.md` |
| D — degree-aware alpha | `paper/degree_aware_alpha.py` | `degree_aware_alpha_grid.csv`, `degree_aware_alpha_summary.csv`, `degree_aware_alpha_report.md` |
| E — cost accounting | `paper/selected_observable_cost.py` | `selected_observable_cost.csv`, `selected_observable_cost_summary.md` |
| F — consolidation | `paper/selected_observable_consolidation.py` | `claim_boundary_update.md`, `paper_ready_tables.md`, `manifest.json` |

## How to run

```bash
# Full workload (A-F), writes to outputs/qsvt_selected_observable_workload/
python scripts/run_selected_observable_workload.py

# Individual stages
python scripts/run_sparse_access_workload.py
python scripts/run_selected_observable_readout.py
python scripts/run_degree_aware_alpha_selection.py
python scripts/run_selected_observable_cost_accounting.py
```

All outputs land under `outputs/qsvt_selected_observable_workload/`. Existing
outputs elsewhere are never overwritten.

## Implemented behavior

- Weighted PSSE Jacobians are built by the existing `build_engineering_system`
  path (PYPOWER AC-linearized weighted update). The matched update is
  `dx_alpha = (H~^T H~ + alpha I)^-1 H~^T r~` via `ridge_svd_solution`.
- A validated classical sparse-access emulator with exact CSR index/value lookup.
- Physically meaningful selected observables (voltage magnitude/angle, branch
  angle difference, area aggregate, energy-style block) with exact matched values.
- Unbiased Monte-Carlo readout sweeps (sign-aware and basis-sampling).
- Degree-aware alpha selection using the repository's bounded QSVT-target
  convention; multiple selection rules are compared.
- Per-observable cost accounting with labeled cost terms.

## Modeled assumptions

- Block-encoding sparse access is a classical emulator, not a reversible circuit.
- Residual-state preparation is assumed (a loader is not synthesized).
- Amplitude amplification, if applied, is a modeled `O(1/sqrt(p))` overhead.

## Proxy-level diagnostics

- Postselection success probability is a proxy (ingested or amplitude-ratio).
- Readout shot budgets are Monte-Carlo proxies for the measurement cost.
- The classical sparse Ridge baseline is a conjugate-gradient flop proxy.

## Excluded components

- Full signed-vector recovery (one readout per state component).
- Phase-factor synthesis per row (only a documented degree-range hint is given).
- Any execution on quantum hardware.

## Limitations and future work

- Degree, C, and success probability are spectrum-level resource estimates, not a
  scalable compiled QSVT circuit.
- The trade-off study uses well-posed linearized systems where classical RMSE is
  nearly flat across alpha; ill-conditioned regimes are where the degree budget
  binds. Future work could synthesize phase factors for the degree-aware-selected
  configurations and validate gate-level output for the selected observables.

See `SPARSE_ACCESS_MODEL.md`, `SELECTED_SIGNED_READOUT.md`, and
`DEGREE_AWARE_ALPHA_SELECTION.md` for component detail.
