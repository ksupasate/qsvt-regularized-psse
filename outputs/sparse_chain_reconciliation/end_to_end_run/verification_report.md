# Sparse Integrated Chain Verification Report

- Configuration: `ieee14_sparse_quantized_8x8_d31_selected_v1`
- Execution: Executed 8x8 sparse-access selected-output chain on a classical simulator using the repository's enumerated slot/value wrapper. This is not an IEEE-scale sparse oracle, not a scalable state loader, not a hardware run, and not a speedup or practical-competitiveness claim.
- Relative sparse block reconstruction error: 1.009673e-12
- Sparse-versus-dense action error: 1.540259e-12
- QSVT-versus-exact polynomial SVT error: 1.115785e-08
- QSVT-versus-quantized-Ridge error: 1.627866e-04
- Sparse postselection probability: 0.6090421559
- Primary statevector selected output: 2.471211435826e-03
- Primary quantized Ridge output: 2.471715023872e-03
- Primary 1000000-shot mean over 10 seeds: 2.473512050019e-03

## Statistical interpretation

Each per-seed analytic standard error describes one finite-shot estimate. The across-seed standard deviation and the standard error of the seed mean are stored separately. The finite-shot counts are produced by actual Aer sampling of the measured circuits; exact statevector distributions are used only for validation and analytic variance references.

## Verification command status

Command results are appended after the repository verification commands run.
