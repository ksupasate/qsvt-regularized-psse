# QSVT Preconditioned IEEE300 Variant

## Purpose

This note documents the formal column-equilibrated IEEE300 estimator variant.
The variant is motivated by diagnostics showing that IEEE300 approximation
difficulty is strongly influenced by spectral spread. It is a new, labeled
estimator family and does not replace original unpreconditioned Ridge or
QSVT-target results.

## Script And Outputs

Run:

```bash
.venv/bin/python scripts/run_qsvt_preconditioned_ieee300_estimator.py
```

The script writes:

- `outputs/qsvt_preconditioned_ieee300_estimator/preconditioned_ieee300_estimator_summary.csv`
- `outputs/qsvt_preconditioned_ieee300_estimator/preconditioned_ieee300_estimator_summary.json`
- `outputs/qsvt_preconditioned_ieee300_estimator/preconditioned_ieee300_solution_metrics.csv`
- `outputs/qsvt_preconditioned_ieee300_estimator/preconditioned_ieee300_spectral_metrics.csv`
- `outputs/qsvt_preconditioned_ieee300_estimator/preconditioned_ieee300_qsvt_approximation.csv`
- `outputs/qsvt_preconditioned_ieee300_estimator/preconditioned_ieee300_report.md`
- `outputs/qsvt_preconditioned_ieee300_estimator/manifest.json`

## Variants

Let `A = H_tilde` and `b = r_tilde`. Column equilibration builds
`A_p = A M^{-1}` using diagonal column scaling.

The report keeps these rows separate:

- unpreconditioned Ridge/Tikhonov;
- coordinate-penalty Ridge in `y`, with `x = M^{-1} y`;
- transformed-penalty Ridge with the original `x`-space penalty;
- unpreconditioned QSVT-target spectral diagnostic;
- preconditioned QSVT-target spectral diagnostic.

Coordinate-penalty Ridge is a distinct estimator variant. Transformed-penalty
Ridge is a numerical consistency check for the original x-space penalty.

## Metrics

Rows report condition number, rank, RMSE when reference data is available,
solution relative error against unpreconditioned Ridge, residual norm, weighted
residual norm, full-interval approximation error, actual-singular-value
approximation error, degree, query count, runtime, status, and caveats.

## Claim Boundary

Safe wording:

```text
Column-equilibrated IEEE300 variants quantify whether preconditioning reduces
condition number and QSVT-compatible approximation difficulty for a separately
labeled estimator variant.
```

Avoid wording:

```text
Preconditioned IEEE300 rows prove quantum speedup, replace the original
unpreconditioned estimator result, or show QSVT outperforms Ridge/Tikhonov
under the same alpha.
```

Preconditioned approximation results are full-interval only for the
preconditioned matrix and configured variant. They do not make the original
unpreconditioned IEEE300 result pass.
