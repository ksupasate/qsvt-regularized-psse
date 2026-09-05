# QSVT Adaptive Multicase Degree Search

## Purpose

This note documents the adaptive multicase degree search for bounded
QSVT-compatible polynomial approximations of the Ridge/Tikhonov spectral
filter. The goal is to quantify degree and query-count pressure across
controlled IEEE/PYPOWER benchmark matrices.

The report is a polynomial approximation and resource-proxy diagnostic. It does
not demonstrate quantum speedup, quantum advantage, full hardware execution,
field-data validation, or QSVT superiority over Ridge/Tikhonov under the same
alpha.

## Script And Outputs

Run:

```bash
python scripts/run_qsvt_adaptive_multicase_degree_search.py
```

The script writes:

- `outputs/qsvt_adaptive_multicase_degree_search/adaptive_multicase_summary.csv`
- `outputs/qsvt_adaptive_multicase_degree_search/adaptive_multicase_summary.json`
- `outputs/qsvt_adaptive_multicase_degree_search/adaptive_multicase_search_trace.csv`
- `outputs/qsvt_adaptive_multicase_degree_search/adaptive_multicase_failure_log.csv`
- `outputs/qsvt_adaptive_multicase_degree_search/manifest.json`

## Default Search

Default settings are:

```text
cases = ieee14, ieee30, ieee57, ieee118, ieee300
alpha = 1e-2
target_tolerance = 1e-3
method = odd_chebyshev_minimax_lp
degrees = 101, 151, 201, 301, 401, 601, 801, 1001
query_count = 2 * degree + 1
```

The search records the first configured degree that passes the strict
`1e-3` maximum pointwise-error tolerance. If no degree passes within the
configured schedule, the row is reported as failed and the failure log records
the best tested degree and reason.

## Matrix Construction

Each case uses resource-only AC weighted-system construction through existing
PYPOWER paths. The script catches failures per case and writes a failure log.
It does not trigger nonlinear IEEE300 experiments and does not execute
hardware-native QSVT circuits.

## Interpretation

The search can show that larger benchmark cases require higher degree and
query count than IEEE14 under the same alpha, method, grid, and tolerance. A
failed row means the configured degree schedule did not meet the target
tolerance; it is not a failed experiment hidden as success.

The non-brute-force refinement adds two follow-ups:

- IEEE118 targeted refinement tests only degrees `1201` and `1501` by default,
  with optional `2001` allowed by the script but not used unless configured.
  The current run records a degree-1201 numerical LP failure and a degree-1501
  strict `1e-3` pass.
- IEEE300 spectral difficulty diagnostics keep degree 1001 fixed and analyze
  spectrum, quantiles, histograms, error location, full-interval error,
  actual-singular-value error, and central interval diagnostics. The current
  IEEE300 run remains failed for full-interval `1e-3`; restricted-interval
  rows are diagnostic only.

Safe wording:

```text
Adaptive multicase diagnostics quantify configured polynomial degree and
query-count requirements for bounded QSVT-compatible filtering on controlled
IEEE/PYPOWER matrices.
```

Avoid wording:

```text
Adaptive multicase diagnostics demonstrate quantum advantage or full
IEEE-scale quantum hardware execution.
```

## Limitations

- The search uses a configured degree grid, not a proof of globally minimal
  QSP/QSVT degree.
- The query-count proxy omits oracle construction, data loading, state
  preparation, error correction, compilation, routing, and readout.
- High-degree linear programs can become numerically difficult; such failures
  are reported explicitly.
- Results depend on the generated benchmark matrix, alpha, grid, method, and
  normalization.
