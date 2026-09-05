# Phase 8: Bridge-Discrepancy Characterization

Classical audit of the selected-submatrix surrogate boundary across IEEE cases, block sizes, and selection rules at the executed normalized regularization lambda = 0.069018 (alpha = lambda * beta_B^2 per block). The three original bridge rows are reproduced verbatim as `provenance_bridge_row` entries.

## Rule-level medians (relative first-coordinate discrepancy)

| Case | Rule | blocks | median | min | max |
| --- | --- | --- | --- | --- | --- |
| ieee118 | column_leverage | 5 | 5.422 | 1.000 | 206.658 |
| ieee118 | largest_row_col_norms | 5 | 0.051 | 0.006 | 0.237 |
| ieee118 | seeded_random | 4 | 1.000 | 0.857 | 1.778 |
| ieee14 | column_leverage | 3 | 0.433 | 0.341 | 2.249 |
| ieee14 | largest_row_col_norms | 3 | 0.104 | 0.003 | 0.120 |
| ieee14 | seeded_random | 3 | 0.495 | 0.224 | 0.896 |
| ieee30 | column_leverage | 4 | 1.116 | 0.077 | 1.969 |
| ieee30 | largest_row_col_norms | 4 | 0.547 | 0.101 | 1.135 |
| ieee30 | seeded_random | 4 | 1.087 | 0.713 | 2.889 |
| ieee57 | column_leverage | 4 | 1.000 | 0.464 | 1.000 |
| ieee57 | largest_row_col_norms | 5 | 0.082 | 0.029 | 1.000 |
| ieee57 | seeded_random | 5 | 1.000 | 0.037 | 1.359 |
| __all_cases__ | column_leverage | 16 | 1.000 | 0.077 | 206.658 |
| __all_cases__ | largest_row_col_norms | 17 | 0.101 | 0.003 | 1.135 |
| __all_cases__ | seeded_random | 16 | 1.000 | 0.037 | 2.889 |

## Spearman correlation of the relative discrepancy with deleted-coupling diagnostics (computed rows, all rules)

| Diagnostic | Spearman rho |
| --- | --- |
| out_of_block_coupling_fraction | 0.662 |
| selected_rows_out_of_block_energy_fraction | 0.655 |
| selected_cols_out_of_block_energy_fraction | 0.644 |
| functional_column_leakage | 0.699 |
| block_frobenius_fraction | -0.613 |
| residual_energy_fraction | -0.423 |
| kappa_effective | -0.088 |

## Mechanism reading

The selected-submatrix surrogate discards out-of-block couplings. The discrepancy increases when the selected functional depends strongly on variables or equations outside the retained block, so the bridge result should be interpreted as a boundary on the selected-submatrix surrogate, not as evidence of full-system selected-output accuracy. The IEEE-30 16x16 provenance row (relative discrepancy 0.878) sits in the same regime as the corresponding characterized cell (0.878 at the anchor lambda) and its functional-column leakage 0.97 shows most of that column's energy lies outside the retained rows.

No speedup, no quantum execution, and no full-system selected-output claim is made; this is a boundary audit of the surrogate construction.
