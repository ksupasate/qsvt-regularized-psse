# Phase 8: Integrated 4x4 Finite-Shot Selected-Submatrix Chain

One shot-executed circuit composes dense residual initialization, the synthesized degree-31 QSVT sequence, measured ancilla postselection, and Hadamard-type signed-functional readout for the passing IEEE-14 4x4 selected-submatrix anchor. The classically computed postselected output state is used only as a validation reference; no gate in the sampled circuit prepares it (`output_state_used_for_preparation = false`).

- physical recovery: `y_hat = (C/beta)*||r_B||*||l|| * f_hat * Xbar_acc; f_hat is the measured acceptance frequency of the flag qubit, Xbar_acc the mean readout-ancilla sign among accepted shots; measured postselection probability p_hat = 2*f_hat - 1`
- statevector postselection probability (reference): 0.9904
- signal-unitary calls per attempt N_U = d = 31; projector phases N_phi = d+1 = 32; alternating length 2d+1 = 63; circuit qubits = 4

## Primary functional `state_correction_0` (first selected coordinate)

| Shots | measured p_succ | recovered mean | rel err vs Ridge | rel err vs statevector QSVT | isolated rel err vs Ridge | seeds |
| --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.9883 | -4.2611e-03 | 1.108e-01 | 1.108e-01 | 1.073e-01 | 30 |
| 10000 | 0.9900 | -4.2297e-03 | 3.242e-02 | 3.235e-02 | 3.277e-02 | 30 |
| 100000 | 0.9904 | -4.2490e-03 | 1.021e-02 | 1.016e-02 | 1.037e-02 | 30 |
| 1000000 | 0.9904 | -4.2529e-03 | 2.846e-03 | 2.653e-03 | 2.792e-03 | 30 |

The isolated column repeats the previous assumed-output-state-preparation Hadamard-overlap experiment for the same functional; the integrated chain replaces that assumption with the measured chain itself.

## Interpretation boundary

This is a 4x4 dense selected-submatrix demonstration on a statevector-based shot simulator with dense controlled-unitary gates. It does not imply scalable residual loading, IEEE-scale sparse block encoding, full selected-output PSSE execution, or quantum competitiveness. Larger blocks and IEEE-scale composition remain modeled.
