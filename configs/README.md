# Configuration Index

Canonical configurations remain at their established paths to preserve tests,
commands, and evidence provenance. They are indexed here instead of being moved
into new case folders.

## Smoke configurations

| Family | Config |
|---|---|
| Synthetic conditioning | `ieee14_spectral_smoke.yaml` |
| DC-linearized | `ieee14_dc_smoke.yaml` |
| AC-linearized | `ieee14_ac_smoke.yaml` |
| Iterative AC | `ieee14_ac_iterative_smoke.yaml` |
| Missing/bad-data stress | `ieee14_bad_data_smoke.yaml` |
| Robust estimators | `ieee14_robust_bad_data_smoke.yaml` |

Run all five lightweight smoke paths with:

```bash
python scripts/run_smoke_test.py
```

## Classical experiment families

- Synthetic sweeps: `ieee14_synthetic_sweeps.yaml`
- DC sweeps: `ieee14_dc_sweeps.yaml`
- AC-linearized sweeps: `ieee14_ac_sweeps.yaml`
- Iterative AC sweeps: `ieee14_ac_iterative_sweeps.yaml`
- Bad-data sweeps: `ieee14_bad_data_sweeps.yaml`
- Robust bad-data sweeps: `ieee14_robust_bad_data_sweeps.yaml`
- Missing-baseline diagnostics: `diagnostic_missing_baselines.yaml` and
  `real_ieee*_missing_baselines.yaml`

## PYPOWER benchmark cases

| Case | AC-linearized | Nonlinear AC |
|---|---|---|
| IEEE14 | `real_ieee14.yaml` | `nonlinear_ac_ieee14_seed10.yaml` |
| IEEE30 | `real_ieee30.yaml` | `nonlinear_ac_ieee30_seed10.yaml` |
| IEEE57 | `real_ieee57.yaml` | `nonlinear_ac_ieee57_seed10.yaml` |
| IEEE118 | `real_ieee118.yaml` | `nonlinear_ac_ieee118_seed10.yaml` |
| IEEE300 | `real_ieee300.yaml` | `nonlinear_ac_ieee300_seed10.yaml` |

The `real_` prefix is historical. These files load published benchmark network
cases from PYPOWER but generate all measurement data; they are not field-data
configs.

## QSVT configurations

- Scalar response and phase synthesis: `qsvt_phase_*.yaml`
- PennyLane matrix paths: `qsvt_pennylane_*.yaml`
- Qiskit matrix paths: `qsvt_qiskit_*.yaml`
- Small circuit scaling: `qsvt_circuit_scaling.yaml`
- Full IEEE resource proxies: `qsvt_resource_full_ieee.yaml`
- Hardware-oriented small demos: `qsvt_hardware_*.yaml`
- Boundary/resource campaigns: `qsvt_phase2_boundary.yaml` and
  `qsvt_phase3_resource_reproducibility.yaml`

QSVT configs do not all use the standard experiment schema. The public
`scripts/run_experiment.py` supports the standard, `demo`, `scaling`, and
`resource` schemas. Specialized matrix/campaign configs retain their documented
module entry points.

## Focused campaign subdirectories

- `cross_case_larger_block_validation/`
- `reviewer_blocking_tqe_evidence/` (historical directory name)
- `tqe_physical_alignment/`
- `tqe_reviewer_blocking/` (historical directory name)

These JSON campaigns contain their own training/held-out splits, seed formulas,
bootstrap settings, tie rules, and failure-retention policies. Do not replace
them with global defaults.

## Machine-readable map

The registered public subset, including model type, measurement types,
dimensions, estimators, metrics, and output path, is registered in:

```text
outputs/reproducibility_audit/experiment_manifest.json
```
