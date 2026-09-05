# Sparse-Oracle and Block-Encoding Interface Specification

## Scope and status

This document specifies a reversible interface for the generated weighted PSSE
Jacobian `H_tilde`. The repository implements and validates a classical CSR lookup
emulator. Reversible circuit synthesis, fault-tolerant compilation, and IEEE-scale
block-encoding circuits are **excluded**. The block-encoding pathway below is
**modeled**.

## Reversible interfaces

For row `i`, slot `k`, column `j`, and live target register `z`, the specified
unitaries are

```text
O_col: |i,k,z> -> |i,k,z XOR c(i,k)>
O_val: |i,j,z> -> |i,j,z XOR fix(H_tilde[i,j])>
```

The input row, slot, and column registers are preserved. XOR makes each mapping a
reversible extension even when multiple indices return the same value.

## Invalid-index behavior

- If `i` is outside the row range or `k >= s_r`, `O_col` XORs the all-ones column
  sentinel into `z`; valid column indices never use that sentinel.
- If `k` is below `s_r` but exceeds the stored nonzeros of row `i`, the same sentinel
  is returned.
- If `i` or `j` is outside the matrix range, `O_val` XORs the zero fixed-point word
  and sets a one-bit invalid flag in a work register.
- A structurally zero in-range entry returns the zero fixed-point word without the
  invalid flag.
- Any reversible implementation must uncompute the invalid flag and all work
  registers before completing a block-encoding use.

## Value representation

- Signed encoding: two's-complement fixed point.
- Default precision: 8 signed value bits, matching the emulator's reported value
  register. A concrete synthesis must expose the integer/fractional split and bound
  quantization error before changing status from modeled.
- Quantization rule: deterministic round-to-nearest with saturation at the
  representable endpoints.
- `fix(H_tilde[i,j])` denotes the resulting signed fixed-point word, not a floating-
  point register.

## Sparsity and normalization

- `s_r = max_i nnz(H_tilde[i,:])` is the validated row-sparsity bound reported by
  `sparse_access_summary.csv`.
- `beta >= ||H_tilde||_2` is the block-encoding normalization bound. The current
  selected-observable artifacts record `beta = sigma_max(H_tilde)` for analytic
  normalization; a scalable reversible construction using this value is modeled.
- The normalized matrix is `A = H_tilde / beta`.

## Registers

Live registers are `row`, `slot`, `column`, `value`, block-encoding ancillas, and the
prepared residual-state register. Work registers include the invalid flag, CSR
address arithmetic, fixed-point arithmetic scratch, and uncomputation ancillas.
The classical emulator validates the logical lookup values only; it does not
allocate or synthesize these quantum work registers.

## Block-encoding call contract

A sparse block-encoding use is specified to invoke a constant number of coherent
`O_col` and `O_val` calls plus their inverses, together with row-state preparation
and uncomputation. The exact constant, gate count, depth, work-qubit count, and
precision dependence are **not available** because the reversible construction is
not implemented. Cost ledgers therefore count one abstract block-encoding unitary
use and label access and preparation as **modeled**; they do not convert that use
to hardware gates.

## Evidence-status ledger

| Component | Status | Evidence or boundary |
| --- | --- | --- |
| CSR index/value lookup | available/validated | entrywise emulator checks against generated `H_tilde` |
| Row sparsity `s_r` | available/validated | maximum CSR row occupancy |
| Signed fixed-point format | specified/modeled | 8-bit two's-complement interface |
| Reversible `O_col`, `O_val` circuit | excluded | specification only; no synthesis |
| Residual-state preparation | modeled | normalized residual and norm recorded |
| Sparse block encoding | modeled | abstract interface and normalization only |
| Hardware gates/depth/probability | not available | no compiled reversible construction |

This specification is an auditable interface boundary, not evidence of reversible
oracle implementation, hardware execution, or quantum speedup.
