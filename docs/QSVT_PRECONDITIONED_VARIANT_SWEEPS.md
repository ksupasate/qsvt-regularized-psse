# QSVT Preconditioned Variant Sweeps

## Purpose

This document describes the controlled preconditioned-variant sweeps for
IEEE118 and IEEE300. The sweep extends the formal column-equilibrated IEEE300
diagnostic into a paper-support evidence package while keeping original and
preconditioned estimator claims separate.

## Script And Outputs

Run:

```bash
.venv/bin/python scripts/run_qsvt_preconditioned_variant_sweeps.py
```

Outputs:

- `outputs/qsvt_preconditioned_variant_sweeps/preconditioned_variant_sweep_results.csv`
- `outputs/qsvt_preconditioned_variant_sweeps/preconditioned_variant_sweep_results.json`
- `outputs/qsvt_preconditioned_variant_sweeps/preconditioned_variant_sweep_summary.csv`
- `outputs/qsvt_preconditioned_variant_sweeps/preconditioned_variant_sweep_summary.json`
- `outputs/qsvt_preconditioned_variant_sweeps/preconditioned_variant_failure_log.csv`
- `outputs/qsvt_preconditioned_variant_sweeps/preconditioned_variant_manifest.json`
- `outputs/qsvt_preconditioned_variant_sweeps/preconditioned_variant_sweep_report.md`

## Variants

- `unpreconditioned_ridge`
- `preconditioned_coordinate_ridge`
- `preconditioned_transformed_penalty_ridge`
- `unpreconditioned_qsvt_diagnostic`
- `preconditioned_qsvt_diagnostic`

Coordinate-preconditioned Ridge uses a standard penalty in the equilibrated
coordinate `y`; it is a separate estimator. Transformed-penalty Ridge preserves
the original x-space penalty and is a consistency check. QSVT diagnostic rows
report approximation/resource evidence only.

## Claim Boundary

Safe wording:

```text
Preconditioned estimator variants were evaluated across controlled
IEEE/PYPOWER sweeps and are reported separately from original estimators.
```

Avoid wording:

```text
Preconditioned coordinate Ridge replaces original Ridge, or preconditioned
IEEE300 diagnostics prove quantum speedup.
```
