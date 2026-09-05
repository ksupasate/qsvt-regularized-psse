# QSVT Residual-Weighted Spectral Error

## Purpose

This diagnostic asks whether large IEEE300 pointwise approximation error aligns
with singular directions that have substantial residual energy. It is a
solution-relevance diagnostic and does not replace full-interval validation.

## Script And Outputs

Run:

```bash
.venv/bin/python scripts/diagnose_qsvt_ieee300_residual_weighted_error.py
```

The script writes:

- `outputs/qsvt_ieee300_residual_weighted_error/residual_weighted_error_summary.csv`
- `outputs/qsvt_ieee300_residual_weighted_error/residual_weighted_error_summary.json`
- `outputs/qsvt_ieee300_residual_weighted_error/singular_direction_contributions.csv`
- `outputs/qsvt_ieee300_residual_weighted_error/top_error_directions.csv`
- `outputs/qsvt_ieee300_residual_weighted_error/residual_weighted_error_report.md`
- `outputs/qsvt_ieee300_residual_weighted_error/manifest.json`

## Computed Quantities

For `A = U Sigma V^T` and `c_i = u_i^T b`, the report records:

```text
|P_alpha(sigma_i) c_i|
|p(sigma_i) - P_alpha(sigma_i)| |c_i|
```

The first quantity is a target directional contribution. The second is the
approximation-error contribution weighted by residual projection magnitude.

## Interpretation

Residual-weighted diagnostics indicate whether high pointwise approximation
error occurs in singular directions that materially affect the current right
hand side. They are useful for diagnosing practical solution relevance, but
they remain distinct from:

- full-interval approximation validation;
- actual-singular-value maximum-error diagnostics;
- restricted-interval diagnostics;
- phase-response validation.

## Claim Boundary

Safe wording:

```text
Residual-weighted diagnostics indicate whether high pointwise approximation
error aligns with high-energy residual directions.
```

Avoid wording:

```text
Residual-weighted diagnostics prove full QSVT validation or make IEEE300 pass
full-interval 1e-3 validation.
```

The diagnostic is not quantum speedup, quantum advantage, hardware execution,
field-data validation, or evidence that QSVT outperforms Ridge/Tikhonov under
the same `alpha`.
