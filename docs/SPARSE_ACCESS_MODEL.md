# Sparse-Access Model for Weighted PSSE Jacobians

Module: `robust_qsvt_se.qsvt.sparse_access` (wraps the existing
`sparse_access_oracle.SparseAccessOracle`). Workload: `paper.sparse_access_workload`.

The model exposes the standard sparse-access oracle interfaces used by QSVT
block-encoding constructions for a row-sparse weighted Jacobian `H~`:

```text
O_col : |i, k, z>  -> |i, k, z XOR c(i,k)>
O_val : |i, j, z>  -> |i, j, z XOR fix(H~_{ij})>
```

## Implemented behavior

- `build_sparse_access_model(matrix, ...)` converts the weighted Jacobian to a
  stable CSR/CSC representation and exposes:
  - `get_col(row, k)`, `get_val(row, col)`, `get_row_nonzeros(row)`;
  - `validate_against_dense_or_csr()` which checks **every** structural nonzero;
  - `num_rows`, `num_cols`, `nnz`, `row_nonzero_counts`, `max_row_nnz`,
    `mean_row_nnz`, `density`;
  - `index_qubits = ceil(log2(rows)) + ceil(log2(cols))`, `value_precision_bits`,
    `value_register_qubits`, `normalization_beta`, `query_assumptions()`.
- The exact-lookup validation reproduces every stored value (max error `0.0`),
  agrees with the CSR column structure, and rejects out-of-range indices.

## Modeled assumptions

- This is a **classical sparse-access emulator** and a **modeled sparse-access
  pathway**. It is **not a reversible quantum oracle circuit** and **not a
  quantum-hardware run**. `reversible_oracle_synthesized` is always `False`.
- Per-block-encoding query assumptions (`O_col`, `O_val`, diffusion = 1 each) are
  the modeled query model, not a compiled gate count.
- `value_register_qubits` equals `value_precision_bits` (a fixed-point register;
  sign folded into the two's-complement encoding).

## Proxy-level diagnostics

- Qubit and precision counts are resource estimates consistent with the existing
  oracle-model resource summary (`row_qubits = ceil(log2(rows))`,
  `max_row_nnz = max_row_sparsity`).

## Excluded components

- Reversible oracle synthesis, gate decomposition, and runs on quantum hardware.

The reversible register contract, invalid-index sentinel, signed fixed-point
encoding, row-sparsity bound `s_r`, normalization `beta`, live/work registers,
and block-encoding call boundary are specified in
`docs/SPARSE_ORACLE_BLOCK_ENCODING_SPEC.md`. That document is a specification,
not circuit-synthesis evidence.

## Limitations and future work

- The emulator validates correctness of access, not the cost of a compiled
  oracle. A future step could synthesize a small reversible oracle for one case
  and compare its gate count against the modeled query assumptions.

## Required table columns (`sparse_access_summary.csv`)

`case, matrix_source, shape_rows, shape_cols, nnz, density, max_row_nnz,
mean_row_nnz, index_qubits, value_precision_bits, value_register_qubits,
access_status, reversible_oracle_synthesized, notes`.
