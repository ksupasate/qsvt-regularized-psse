# Integration Verdict

## Verdict A: Genuine single integrated sparse finite-shot circuit exists

The canonical workload is `ieee14_sparse_quantized_8x8_d31_selected_v1`.
For each of its three predetermined functionals, one measured eight-qubit
`QuantumCircuit` contains all of the following:

1. controlled preparation of the normalized residual input;
2. a six-qubit sparse wrapper with a three-slot index lookup, stored signed-value
   rotations, slot-controlled permutations, and inverse slot diffusion;
3. 16 controlled forward and 15 controlled inverse calls to that wrapper;
4. 32 projector-controlled QSVT phase operations for the fixed degree-31 phase
   sequence;
5. an aggregate postselection flag for the sparse encoded subspace;
6. a controlled selected-functional reference preparation and closing Hadamard;
7. measurement of both the postselection flag and readout sign; and
8. Aer finite-shot sampling, followed by the fixed physical recovery factor
   `(C/beta) ||r|| ||ell||`.

The decisive construction is
`build_integrated_sparse_selected_output_circuit` in
`src/robust_qsvt_se/qsvt/sparse_integrated_chain.py:577-700`. The experiment
builds this circuit at lines 1692-1705, compiles that same measured circuit at
lines 1743-1749, samples it through `sample_aer_counts` at lines 933-946, and
derives the executed resource row from the compiled primary-functional circuit
at lines 1766-1771. No classically computed QSVT output state is prepared.

The 186,191-gate, depth-180,380, 51,898-Toffoli, and 744-controlled-rotation
counts therefore belong to the same eight-qubit circuit and the same matrix
fingerprint as the finite-shot results. They do not belong to the separate
five-qubit dense readout circuit or the six-qubit sparse-wrapper-only
validation.

## Scope

This verdict is a circuit-composition statement for one simulator-scale,
quantized `8x8` workload. It is not evidence of hardware execution, scalable
IEEE-size sparse access, quantum speedup, quantum advantage, or practical
competitiveness.
