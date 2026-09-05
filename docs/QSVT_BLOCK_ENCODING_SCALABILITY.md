# QSVT Block-Encoding Scalability

## Purpose

The dense block-encoding implementation in this repository is a validation
prototype. It checks normalization, contraction conditions, encoded-block
recovery, and unitarity on small weighted matrices or deterministic
submatrices. It is not a scalable sparse-access oracle for full IEEE-scale
power-system matrices.

## Dense Prototype

For a weighted Jacobian `H_tilde`, the prototype uses

```text
A = H_tilde / beta,  beta >= ||H_tilde||_2
```

and validates a dense Julia block encoding,

```text
U_A = [[A, sqrt(I - A A*)],
       [sqrt(I - A* A), -A*]]
```

The encoded matrix is the top-left rectangular block. The dense unitary has
dimension `m + n` for an `m x n` matrix, so the construction is useful for
small algebraic validation but not for scalable data access.

## Scalability Diagnostics

The scalability report records:

- matrix dimensions;
- number of nonzero entries;
- density and sparsity;
- dense block-encoding dimension;
- index-qubit proxy;
- normalization beta;
- condition number when feasible;
- explicit dense-prototype caveat.

These rows support engineering discussion about what would be required for a
future scalable implementation pathway. They do not implement sparse-access
oracles, state-preparation oracles, row/column lookup oracles, fault-tolerant
compilation, or full-vector readout.

## Future Scalable Path

A scalable QSVT pathway would need a data-access model such as sparse matrix
oracles, QRAM-style amplitude preparation, structured power-grid topology
oracles, or another explicit block-encoding construction. Such a path would
also need to account for measurement weights, data loading, row metadata,
state-preparation cost, error correction, and selected-observable readout.

## Safe Claim

Dense block encodings validate normalization and algebraic embedding on small
matrices. They are not scalable full-system block-encoding oracles.
