"""Phase 8: integrated 4x4 finite-shot selected-submatrix QSVT chain.

One shot-executed Qiskit circuit composes, for the passing IEEE-14 4x4
selected-submatrix anchor: (1) dense residual initialization (a unitary
``StatePreparation`` of the padded normalized residual -- the *input*, never
the output), (2) the synthesized degree-31 QSVT sequence with its exact
alternating projector-phase / signal-unitary structure, (3) measured ancilla
postselection (the block-encoding flag qubit is measured in every shot and
failures stay in the shot record), and (4) a Hadamard-type signed-overlap
readout of one predetermined functional via a reference branch that prepares
the normalized functional vector.

Interference estimator. With readout ancilla ``a`` in ``|+>``, branch ``a=0``
runs residual preparation followed by the QSVT sequence (encoded-block output
amplitudes ``x``, ``p_succ = ||x||^2``); branch ``a=1`` prepares the padded
unit functional ``l_hat`` (flag qubit untouched).  After the closing Hadamard,
measuring the flag qubit and ``a`` gives

    P(flag = 0) = (1 + p_succ)/2,
    E[z] = Re<l_hat|x>,  z = +/-1 on accepted shots (sign from ``a``), else 0.

The physical signed functional is recovered from *measured* statistics only:

    y_hat = (C/beta) ||r_B|| ||l|| * f_hat * Xbar_acc,

with ``f_hat`` the measured acceptance frequency, ``Xbar_acc`` the mean
ancilla sign among accepted shots (``f_hat * Xbar_acc`` is the per-shot mean
of ``z``), and measured postselection probability ``p_hat = 2 f_hat - 1``.
The classically computed postselected output state is used **only** as a
validation reference; no gate in the sampled circuit prepares it.

Scope / claim boundary: this remains a 4x4 dense selected-submatrix
demonstration on a statevector-based shot simulator.  Controlled operations
are realized as explicit dense controlled-unitary gates (the same small-scale
dense idiom as the canonical block encoding itself).  It does not imply
scalable residual loading, IEEE-scale sparse block encoding, full
selected-output PSSE execution, or quantum competitiveness.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.paper.tqe_revision_experiments_common import (
    assert_safe,
    write_experiment_manifest,
)
from robust_qsvt_se.paper.tqe_revision_readout_statistics import _build_headline
from robust_qsvt_se.qsvt.block_encoding import canonical_square_block_encoding
from robust_qsvt_se.qsvt.gate_level_qsvt import (
    projector_controlled_phase_matrix,
    qsvt_sequence_operation_counts,
)
from robust_qsvt_se.qsvt.gate_state_preparation import normalize_and_pad_for_gate_preparation
from robust_qsvt_se.qsvt.phase_synthesis import synthesize_pennylane_phases_cached
from robust_qsvt_se.utils.io import ensure_directory, write_json

PHASE8_READOUT_DIR = Path("outputs/phase8_integrated_readout")
DEMO_PHASE_CACHE = Path("outputs/selected_observable_qsvt_demo/phase_cache")
ISOLATED_SUMMARY_CSV = Path(
    "outputs/tqe_revision_experiments/readout_statistics/readout_shot_scaling_summary.csv"
)

INTEGRATED_READOUT_MODEL = (
    "integrated_interference_chain_residual_prep_qsvt_measured_postselection_signed_readout"
)
RECOVERY_CONVENTION = (
    "y_hat = (C/beta)*||r_B||*||l|| * f_hat * Xbar_acc; f_hat is the measured acceptance "
    "frequency of the flag qubit, Xbar_acc the mean readout-ancilla sign among accepted "
    "shots; measured postselection probability p_hat = 2*f_hat - 1"
)

FULL_SHOTS = (1_000, 10_000, 100_000, 1_000_000)
QUICK_SHOTS = (1_000, 10_000)

PER_SEED_COLUMNS = [
    "observable_label",
    "shots",
    "seed",
    "backend",
    "status",
    "total_attempts",
    "accepted_attempts",
    "effective_shots_after_postselection",
    "acceptance_frequency",
    "measured_postselection_probability",
    "measured_postselection_probability_standard_error",
    "direct_chain_measured_postselection_probability",
    "readout_sign_mean_accepted",
    "signed_overlap_estimate_z",
    "signed_overlap_standard_error",
    "physical_recovery_scale",
    "recovered_physical_functional",
    "recovered_physical_functional_standard_error",
    "exact_ridge_functional",
    "exact_qsvt_statevector_functional",
    "absolute_error_vs_ridge",
    "relative_error_vs_ridge",
    "absolute_error_vs_statevector_qsvt",
    "relative_error_vs_statevector_qsvt",
    "circuit_qubits",
    "signal_unitary_calls_per_attempt",
    "projector_phase_operations_per_attempt",
    "alternating_sequence_length",
    "output_state_used_for_preparation",
]

SUMMARY_COLUMNS = [
    "observable_label",
    "shots",
    "num_seeds",
    "mean_measured_postselection_probability",
    "std_measured_postselection_probability",
    "mean_postselection_probability_standard_error",
    "statevector_postselection_probability",
    "mean_signed_overlap_estimate_z",
    "mean_signed_overlap_standard_error",
    "mean_recovered_physical_functional",
    "std_recovered_physical_functional",
    "exact_ridge_functional",
    "exact_qsvt_statevector_functional",
    "mean_absolute_error_vs_ridge",
    "mean_relative_error_vs_ridge",
    "std_relative_error_vs_ridge",
    "mean_absolute_error_vs_statevector_qsvt",
    "mean_relative_error_vs_statevector_qsvt",
    "isolated_readout_mean_relative_error_vs_ridge",
    "mean_total_attempts",
    "mean_accepted_attempts",
    "mean_effective_shots_after_postselection",
    "circuit_qubits",
    "signal_unitary_calls_per_attempt",
    "projector_phase_operations_per_attempt",
    "alternating_sequence_length",
    "status",
]


@dataclass(slots=True)
class AnchorContext:
    """Rebuilt 4x4 anchor with the pieces needed to compose the integrated circuit."""

    result: Any
    matrix_source: str
    block_unitary: np.ndarray
    phases: np.ndarray
    padded_residual: np.ndarray
    residual_norm: float
    encoded_dimension: int
    alpha: float
    beta: float
    bound_c: float
    degree: int
    statevector_postselection_probability: float
    validation: dict[str, float] = field(default_factory=dict)


def _state_preparation_unitary(state: np.ndarray) -> np.ndarray:
    """Dense unitary of ``StatePreparation(state)`` (its first column is ``state``)."""

    from qiskit.circuit.library import StatePreparation  # type: ignore[import-not-found]
    from qiskit.quantum_info import Operator  # type: ignore[import-not-found]

    values = np.asarray(state, dtype=np.complex128)
    gate = StatePreparation(values)
    matrix = np.asarray(Operator(gate).data, dtype=np.complex128)
    prepared = matrix[:, 0]
    error = float(np.linalg.norm(prepared - values))
    if error > 1.0e-10:
        raise RuntimeError(f"state-preparation unitary column error {error:.3e}")
    return matrix


def _controlled_on_zero(op: np.ndarray) -> np.ndarray:
    """Controlled ``op`` applied when the (most-significant) control qubit is |0>."""

    dim = op.shape[0]
    out = np.eye(2 * dim, dtype=np.complex128)
    out[:dim, :dim] = op
    return out


def _controlled_on_one(op: np.ndarray) -> np.ndarray:
    dim = op.shape[0]
    out = np.eye(2 * dim, dtype=np.complex128)
    out[dim:, dim:] = op
    return out


def build_anchor_context(
    *,
    case: str = "ieee14",
    case_source: str = "pypower",
    system_seed: int = 123,
    degree: int = 31,
    alpha_mult: float = 4.0,
    phase_cache_dir: str | Path = DEMO_PHASE_CACHE,
) -> AnchorContext:
    """Rebuild the passing 4x4 anchor and re-derive its phases and block unitary.

    The phases are re-synthesized through the same cache used by the demo, so a
    cache hit returns bit-identical angles.  The rebuilt plain chain is
    validated against the demo's postselected statevector output before any
    sampling; the demo output state itself is never loaded by any circuit here.
    """

    result, matrix_source = _build_headline(case, case_source, system_seed, degree, alpha_mult)
    meta = result.pipeline_metadata
    if result.status_label != "pass":
        raise RuntimeError(f"anchor block status is '{result.status_label}', expected 'pass'")

    beta = float(meta["beta"])
    alpha = float(meta["alpha"])
    n = int(result.H_block.shape[0])
    A = np.asarray(result.H_block, dtype=np.float64).T / beta
    encoding = canonical_square_block_encoding(A, tolerance=1.0e-8)
    coefficients = np.asarray(meta["polynomial_coefficients_bounded"], dtype=np.float64)
    cached = synthesize_pennylane_phases_cached(
        coefficients,
        angle_solver=str(meta["angle_solver"]),
        cache_dir=phase_cache_dir,
        cache_metadata={
            "case": case,
            "block": f"{n}x{n}",
            "alpha": alpha,
            "degree": int(degree),
        },
    )
    phases = np.asarray(cached.phases, dtype=np.float64)
    if int(phases.size) != int(meta["phase_count"]):
        raise RuntimeError("re-synthesized phase count does not match the anchor metadata")

    preparation = normalize_and_pad_for_gate_preparation(
        np.asarray(result.r_block, dtype=np.float64),
        target_dimension=encoding.unitary.shape[0],
    )
    p_statevector = float(result.row_common["postselection_probability"])

    x_full = _chain_output_statevector(
        block_unitary=encoding.unitary,
        phases=phases,
        padded_residual=preparation.padded_state,
    )
    encoded = x_full[:n]
    reference = math.sqrt(p_statevector) * np.asarray(result.output_state, dtype=np.complex128)
    chain_state_error = float(np.linalg.norm(encoded - reference))
    chain_probability_error = abs(float(np.vdot(encoded, encoded).real) - p_statevector)
    if chain_state_error > 1.0e-8 or chain_probability_error > 1.0e-10:
        raise RuntimeError(
            "rebuilt plain chain disagrees with the demo pipeline: "
            f"state error {chain_state_error:.3e}, p error {chain_probability_error:.3e}"
        )

    return AnchorContext(
        result=result,
        matrix_source=matrix_source,
        block_unitary=np.asarray(encoding.unitary, dtype=np.complex128),
        phases=phases,
        padded_residual=np.asarray(preparation.padded_state, dtype=np.complex128),
        residual_norm=float(preparation.original_norm),
        encoded_dimension=n,
        alpha=alpha,
        beta=beta,
        bound_c=float(meta["bound_C"]),
        degree=int(degree),
        statevector_postselection_probability=p_statevector,
        validation={
            "plain_chain_encoded_state_error": chain_state_error,
            "plain_chain_probability_error": chain_probability_error,
            "phase_cache_hit": float(bool(cached.cache_hit)),
        },
    )


def _chain_output_statevector(
    *,
    block_unitary: np.ndarray,
    phases: np.ndarray,
    padded_residual: np.ndarray,
) -> np.ndarray:
    """Dense output amplitudes of residual preparation followed by the QSVT sequence."""

    state = np.asarray(padded_residual, dtype=np.complex128).copy()
    for operation, _label in _qsvt_operation_sequence(block_unitary, phases):
        state = operation @ state
    return state


def _qsvt_operation_sequence(
    block_unitary: np.ndarray, phases: np.ndarray
) -> list[tuple[np.ndarray, str]]:
    """The exact alternating QSVT operation list used by the anchor circuit.

    Mirrors ``build_structured_qsvt_operator_circuit`` (one leading projector
    phase, then alternating signal-unitary calls and projector phases) and
    fail-fasts if the signal-call count deviates from ``d``.
    """

    unitary = np.asarray(block_unitary, dtype=np.complex128)
    total_dimension = int(unitary.shape[0])
    encoded_dimension = total_dimension // 2
    phase_values = np.asarray(phases, dtype=np.float64)

    def pcphase(phi: float) -> np.ndarray:
        return projector_controlled_phase_matrix(
            phi, encoded_dimension=encoded_dimension, total_dimension=total_dimension
        )

    sequence: list[tuple[np.ndarray, str]] = [(pcphase(float(phase_values[0])), "PCPhase")]
    signal_calls = 0
    inverse = unitary.conj().T
    for index in range(1, phase_values.size - 1, 2):
        sequence.append((unitary, "U_A"))
        signal_calls += 1
        sequence.append((pcphase(float(phase_values[index])), "PCPhase"))
        sequence.append((inverse, "U_A_dagger"))
        signal_calls += 1
        sequence.append((pcphase(float(phase_values[index + 1])), "PCPhase"))
    if phase_values.size % 2 == 0:
        sequence.append((unitary, "U_A"))
        signal_calls += 1
        sequence.append((pcphase(float(phase_values[-1])), "PCPhase"))

    expected = qsvt_sequence_operation_counts(int(phase_values.size))
    if signal_calls != expected["signal_unitary_calls"]:
        raise RuntimeError("QSVT signal-unitary call accounting does not match the sequence")
    return sequence


def build_integrated_readout_circuit(
    *,
    block_unitary: np.ndarray,
    phases: np.ndarray,
    padded_residual: np.ndarray,
    functional_unit: np.ndarray,
    with_measurements: bool = True,
) -> tuple[Any, dict[str, int]]:
    """Build the one-circuit integrated chain for a normalized block functional.

    Qubits 0..1: system block coordinates; qubit 2: block-encoding flag;
    qubit 3: readout ancilla.  Classical bit 0 records the postselection flag,
    classical bit 1 the readout ancilla.  Every chain operation is a dense
    controlled-unitary on the readout ancilla; the reference branch prepares
    the functional vector on the system qubits only, so its flag stays |0>.
    The known postselected output state is never an input to this function.
    """

    from qiskit import QuantumCircuit  # type: ignore[import-not-found]

    unitary = np.asarray(block_unitary, dtype=np.complex128)
    total_dimension = int(unitary.shape[0])
    n_chain_qubits = int(np.log2(total_dimension))
    ell_hat = np.asarray(functional_unit, dtype=np.complex128)
    if ell_hat.size != total_dimension // 2:
        raise ValueError("functional must live on the encoded block dimension")
    if not np.isclose(float(np.linalg.norm(ell_hat)), 1.0, atol=1.0e-10):
        raise ValueError("functional must be normalized before circuit construction")

    residual_prep = _state_preparation_unitary(padded_residual)
    functional_prep = _state_preparation_unitary(ell_hat)

    n_qubits = n_chain_qubits + 1
    readout_ancilla = n_chain_qubits
    flag_qubit = n_chain_qubits - 1
    circuit = QuantumCircuit(n_qubits, 2 if with_measurements else 0, name="phase8_integrated")
    circuit.h(readout_ancilla)

    chain_qubits = list(range(n_chain_qubits))
    circuit.unitary(
        _controlled_on_zero(residual_prep),
        [*chain_qubits, readout_ancilla],
        label="c0_residual_prep",
    )
    signal_calls = 0
    phase_operations = 0
    for operation, op_label in _qsvt_operation_sequence(unitary, phases):
        circuit.unitary(
            _controlled_on_zero(operation),
            [*chain_qubits, readout_ancilla],
            label=f"c0_{op_label}",
        )
        if op_label.startswith("U_A"):
            signal_calls += 1
        else:
            phase_operations += 1
    system_qubits = list(range(n_chain_qubits - 1))
    circuit.unitary(
        _controlled_on_one(functional_prep),
        [*system_qubits, readout_ancilla],
        label="c1_functional_prep",
    )
    circuit.h(readout_ancilla)
    if with_measurements:
        circuit.measure(flag_qubit, 0)
        circuit.measure(readout_ancilla, 1)

    expected = qsvt_sequence_operation_counts(int(np.asarray(phases).size))
    if signal_calls != expected["signal_unitary_calls"]:
        raise RuntimeError("controlled QSVT signal-unitary call count does not match d")
    accounting = {
        "circuit_qubits": n_qubits,
        "signal_unitary_calls_per_attempt": signal_calls,
        "projector_phase_operations_per_attempt": phase_operations,
        "alternating_sequence_length": signal_calls + phase_operations,
    }
    return circuit, accounting


def build_direct_chain_circuit(
    *,
    block_unitary: np.ndarray,
    phases: np.ndarray,
    padded_residual: np.ndarray,
) -> Any:
    """Uncontrolled residual-prep + QSVT chain measuring only the postselection flag."""

    from qiskit import QuantumCircuit  # type: ignore[import-not-found]

    unitary = np.asarray(block_unitary, dtype=np.complex128)
    n_chain_qubits = int(np.log2(unitary.shape[0]))
    circuit = QuantumCircuit(n_chain_qubits, 1, name="phase8_direct_chain")
    circuit.unitary(
        _state_preparation_unitary(padded_residual),
        list(range(n_chain_qubits)),
        label="residual_prep",
    )
    for operation, op_label in _qsvt_operation_sequence(unitary, phases):
        circuit.unitary(operation, list(range(n_chain_qubits)), label=op_label)
    circuit.measure(n_chain_qubits - 1, 0)
    return circuit


def _measurement_free(circuit: Any) -> Any:
    return circuit.remove_final_measurements(inplace=False)


def _exact_clbit_distribution(circuit: Any, measured_qubits: list[int]) -> dict[str, float]:
    """Exact outcome distribution over the measured qubits (fallback / validation)."""

    from qiskit.quantum_info import Statevector  # type: ignore[import-not-found]

    state = Statevector(_measurement_free(circuit))
    probabilities = np.asarray(state.probabilities(measured_qubits), dtype=np.float64)
    n_bits = len(measured_qubits)
    distribution: dict[str, float] = {}
    for index, value in enumerate(probabilities):
        bits = format(index, f"0{n_bits}b")
        distribution[bits] = float(value)
    return distribution


def _sample_counts(
    circuit: Any, *, shots: int, seed: int, measured_qubits: list[int]
) -> tuple[dict[str, int], str]:
    """Sample the measured circuit; Aer shots preferred, exact-distribution fallback."""

    try:
        from qiskit_aer import AerSimulator  # type: ignore[import-not-found]

        simulator = AerSimulator()
        job = simulator.run(circuit, shots=int(shots), seed_simulator=int(seed))
        raw = job.result().get_counts()
        counts = {key.replace(" ", ""): int(value) for key, value in raw.items()}
        return counts, "aer_integrated_circuit_shot_sampling"
    except Exception:
        distribution = _exact_clbit_distribution(circuit, measured_qubits)
        keys = sorted(distribution)
        probabilities = np.asarray([distribution[key] for key in keys], dtype=np.float64)
        probabilities = probabilities / probabilities.sum()
        rng = np.random.default_rng(int(seed))
        draws = rng.multinomial(int(shots), probabilities)
        counts = {key: int(count) for key, count in zip(keys, draws, strict=True) if count > 0}
        return counts, "exact_distribution_multinomial_proxy"


def estimate_from_integrated_counts(
    counts: dict[str, int], *, physical_scale: float
) -> dict[str, float]:
    """Estimators from the joint (readout ancilla, flag) shot record.

    Keys are two-bit strings ``c1 c0`` (leftmost = readout ancilla, rightmost =
    flag bit; flag ``0`` accepts).  Recovery uses only measured statistics.
    """

    total = int(sum(counts.values()))
    if total <= 0:
        raise ValueError("empty shot record")
    accepted = sum(count for key, count in counts.items() if key[-1] == "0")
    plus = counts.get("00", 0)
    minus = counts.get("10", 0)
    if plus + minus != accepted:
        raise RuntimeError("accepted-shot bookkeeping mismatch in counts record")
    f_hat = accepted / total
    z_hat = (plus - minus) / total
    x_bar = (plus - minus) / accepted if accepted > 0 else float("nan")
    p_hat = 2.0 * f_hat - 1.0
    p_hat_se = 2.0 * math.sqrt(max(f_hat * (1.0 - f_hat), 0.0) / total)
    z_se = math.sqrt(max(f_hat - z_hat**2, 0.0) / total)
    recovered = physical_scale * z_hat
    return {
        "total_attempts": float(total),
        "accepted_attempts": float(accepted),
        "effective_shots_after_postselection": float(accepted),
        "acceptance_frequency": f_hat,
        "measured_postselection_probability": p_hat,
        "measured_postselection_probability_standard_error": p_hat_se,
        "readout_sign_mean_accepted": x_bar,
        "signed_overlap_estimate_z": z_hat,
        "signed_overlap_standard_error": z_se,
        "recovered_physical_functional": recovered,
        "recovered_physical_functional_standard_error": physical_scale * z_se,
    }


def _direct_chain_p_hat(counts: dict[str, int]) -> float:
    total = int(sum(counts.values()))
    return counts.get("0", 0) / total if total > 0 else float("nan")


def _classical_adjoint_reference(
    H: np.ndarray, r: np.ndarray, ell: np.ndarray, *, alpha: float, repeats: int = 30
) -> dict[str, float]:
    """Matched classical adjoint functional on the same block (host-specific timing)."""

    gram = H.T @ H + alpha * np.eye(H.shape[1])
    htr = H.T @ r

    def solve() -> float:
        w = np.linalg.solve(gram, ell)
        return float(w @ htr)

    for _ in range(3):
        solve()
    timings = []
    for _ in range(int(repeats)):
        start = time.perf_counter()
        value = solve()
        timings.append(time.perf_counter() - start)
    ordered = np.sort(np.asarray(timings, dtype=np.float64))
    return {
        "value": value,
        "median_runtime_seconds": float(np.median(ordered)),
        "iqr_low_seconds": float(np.percentile(ordered, 25)),
        "iqr_high_seconds": float(np.percentile(ordered, 75)),
        "num_timing_repeats": int(repeats),
    }


def _isolated_reference_frame() -> pd.DataFrame:
    if not ISOLATED_SUMMARY_CSV.is_file():
        return pd.DataFrame(columns=["observable_label", "shots", "mean_relative_error_vs_ridge"])
    frame = pd.read_csv(ISOLATED_SUMMARY_CSV)
    return frame[["observable_label", "shots", "mean_relative_error_vs_ridge"]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_phase8_integrated_readout(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    shots_grid = [int(value) for value in resolved["shots_grid"]]
    seeds = list(range(int(resolved["num_seeds"])))

    context = build_anchor_context(
        case=str(resolved["case"]),
        case_source=str(resolved["case_source"]),
        system_seed=int(resolved["seed"]),
        degree=int(resolved["degree"]),
        alpha_mult=float(resolved["alpha_mult"]),
        phase_cache_dir=resolved["phase_cache_dir"],
    )
    result = context.result
    signed = [obs for obs in result.observables if obs.sign_aware_required]
    counts_meta = qsvt_sequence_operation_counts(int(context.phases.size))

    direct_circuit = build_direct_chain_circuit(
        block_unitary=context.block_unitary,
        phases=context.phases,
        padded_residual=context.padded_residual,
    )

    isolated = _isolated_reference_frame()
    p_sv = context.statevector_postselection_probability
    ridge_update = np.asarray(result.ridge_update, dtype=np.float64)
    x_encoded = math.sqrt(p_sv) * np.asarray(result.output_state, dtype=np.complex128)

    per_seed_rows: list[dict[str, Any]] = []
    circuit_records: dict[str, dict[str, Any]] = {}
    counts_examples: dict[str, Any] = {}
    warnings: list[str] = []

    for observable in signed:
        ell = np.asarray(observable.vector, dtype=np.float64)
        ell_norm = float(np.linalg.norm(ell))
        ell_hat = ell / ell_norm
        physical_scale = context.bound_c / context.beta * context.residual_norm * ell_norm
        ridge_value = float(ell @ ridge_update)
        z_exact = float(np.real(np.vdot(ell_hat, x_encoded)))
        statevector_value = physical_scale * z_exact

        circuit, accounting = build_integrated_readout_circuit(
            block_unitary=context.block_unitary,
            phases=context.phases,
            padded_residual=context.padded_residual,
            functional_unit=ell_hat,
        )
        flag_qubit = accounting["circuit_qubits"] - 2
        readout_qubit = accounting["circuit_qubits"] - 1
        exact_distribution = _exact_clbit_distribution(circuit, [flag_qubit, readout_qubit])
        exact_accept = sum(prob for key, prob in exact_distribution.items() if key[-1] == "0")
        exact_z = exact_distribution.get("00", 0.0) - exact_distribution.get("10", 0.0)
        accept_error = abs(exact_accept - (1.0 + p_sv) / 2.0)
        z_error = abs(exact_z - z_exact)
        if accept_error > 1.0e-9 or z_error > 1.0e-9:
            raise RuntimeError(
                "integrated circuit disagrees with the demo pipeline: "
                f"acceptance error {accept_error:.3e}, overlap error {z_error:.3e}"
            )
        label_counts: dict[str, int] = {}
        for instruction in circuit.data:
            key = str(instruction.operation.label or instruction.operation.name)
            label_counts[key] = label_counts.get(key, 0) + 1
        circuit_records[observable.observable_id] = {
            **accounting,
            "exact_acceptance_probability": exact_accept,
            "exact_signed_overlap_z": exact_z,
            "acceptance_probability_validation_error": accept_error,
            "signed_overlap_validation_error": z_error,
            "gate_counts": {str(k): int(v) for k, v in circuit.count_ops().items()},
            "gate_label_counts": label_counts,
            "circuit_depth": int(circuit.depth()),
        }

        for shots in shots_grid:
            for seed in seeds:
                counts, backend = _sample_counts(
                    circuit, shots=shots, seed=seed, measured_qubits=[flag_qubit, readout_qubit]
                )
                direct_counts, _ = _sample_counts(
                    direct_circuit, shots=shots, seed=seed, measured_qubits=[2]
                )
                estimate = estimate_from_integrated_counts(counts, physical_scale=physical_scale)
                recovered = estimate["recovered_physical_functional"]
                abs_ridge = abs(recovered - ridge_value)
                abs_sv = abs(recovered - statevector_value)
                status = (
                    "executed_integrated_chain"
                    if backend.startswith("aer")
                    else "executed_integrated_chain_exact_distribution_proxy"
                )
                if (
                    backend == "exact_distribution_multinomial_proxy"
                    and "aer_unavailable" not in warnings
                ):
                    warnings.append("aer_unavailable")
                per_seed_rows.append(
                    {
                        "observable_label": observable.observable_id,
                        "shots": int(shots),
                        "seed": int(seed),
                        "backend": backend,
                        "status": status,
                        **{
                            key: estimate[key]
                            for key in (
                                "total_attempts",
                                "accepted_attempts",
                                "effective_shots_after_postselection",
                                "acceptance_frequency",
                                "measured_postselection_probability",
                                "measured_postselection_probability_standard_error",
                                "readout_sign_mean_accepted",
                                "signed_overlap_estimate_z",
                                "signed_overlap_standard_error",
                                "recovered_physical_functional",
                                "recovered_physical_functional_standard_error",
                            )
                        },
                        "direct_chain_measured_postselection_probability": _direct_chain_p_hat(
                            direct_counts
                        ),
                        "physical_recovery_scale": physical_scale,
                        "exact_ridge_functional": ridge_value,
                        "exact_qsvt_statevector_functional": statevector_value,
                        "absolute_error_vs_ridge": abs_ridge,
                        "relative_error_vs_ridge": abs_ridge / abs(ridge_value)
                        if abs(ridge_value) > 1.0e-15
                        else float("nan"),
                        "absolute_error_vs_statevector_qsvt": abs_sv,
                        "relative_error_vs_statevector_qsvt": abs_sv / abs(statevector_value)
                        if abs(statevector_value) > 1.0e-15
                        else float("nan"),
                        "circuit_qubits": accounting["circuit_qubits"],
                        "signal_unitary_calls_per_attempt": accounting[
                            "signal_unitary_calls_per_attempt"
                        ],
                        "projector_phase_operations_per_attempt": accounting[
                            "projector_phase_operations_per_attempt"
                        ],
                        "alternating_sequence_length": accounting["alternating_sequence_length"],
                        "output_state_used_for_preparation": False,
                    }
                )
                if seed == 0 and shots == max(shots_grid):
                    counts_examples[observable.observable_id] = {
                        "shots": int(shots),
                        "seed": int(seed),
                        "backend": backend,
                        "integrated_counts_c1c0": counts,
                        "direct_chain_counts_c0": direct_counts,
                        "bit_convention": (
                            "rightmost bit = postselection flag (0 accepts); leftmost bit = "
                            "readout ancilla after the closing Hadamard (0 -> +1, 1 -> -1)"
                        ),
                    }

    per_seed = pd.DataFrame(per_seed_rows, columns=PER_SEED_COLUMNS)
    summary = _summarize(per_seed, isolated, p_sv)

    adjoint = _classical_adjoint_reference(
        np.asarray(result.H_block, dtype=np.float64),
        np.asarray(result.r_block, dtype=np.float64),
        np.asarray(signed[0].vector, dtype=np.float64),
        alpha=context.alpha,
    )
    references = _reference_values(context, signed, ridge_update, x_encoded, adjoint, isolated)
    circuit_metadata = _circuit_metadata(context, counts_meta, circuit_records, resolved)

    artifacts = _write_outputs(
        output_dir=output_dir,
        per_seed=per_seed,
        summary=summary,
        references=references,
        circuit_metadata=circuit_metadata,
        counts_examples=counts_examples,
        context=context,
    )
    manifest = write_experiment_manifest(
        output_dir=output_dir,
        experiment_id="phase8_integrated_readout",
        script_name="scripts/run_phase8_integrated_readout.py",
        command=str(resolved["command"]),
        description=(
            "Integrated 4x4 finite-shot selected-submatrix chain: dense residual "
            "initialization, the synthesized degree-31 QSVT sequence, measured ancilla "
            "postselection, and Hadamard-type signed-functional readout composed in one "
            "shot-executed circuit. The classically computed postselected output state is "
            "used only as a validation reference and is never prepared by the sampled "
            "circuit."
        ),
        artifacts=artifacts,
        inputs_used=[
            f"build_engineering_system:{resolved['case']}:weighted_jacobian",
            str(ISOLATED_SUMMARY_CSV),
        ],
        random_seeds={"shot_seeds": seeds, "demo_system_seed": int(resolved["seed"])},
        warnings=warnings,
        failures=[],
        interpretation_boundary=(
            "One integrated 4x4 finite-shot selected-submatrix chain on a statevector-based "
            "shot simulator with dense controlled-unitary gates. It does not imply scalable "
            "residual loading, IEEE-scale sparse block encoding, full selected-output PSSE "
            "execution, or quantum competitiveness; larger blocks and IEEE-scale composition "
            "remain modeled."
        ),
        extra={
            "integrated_readout_model": INTEGRATED_READOUT_MODEL,
            "recovery_convention": RECOVERY_CONVENTION,
            "output_state_used_for_preparation": False,
            "signed_observables": [obs.observable_id for obs in signed],
            "shots_grid": shots_grid,
            "num_shot_seeds": len(seeds),
        },
    )
    artifacts["manifest"] = manifest
    return {
        "output_dir": output_dir,
        "per_seed": per_seed,
        "summary": summary,
        "references": references,
        "circuit_metadata": circuit_metadata,
        "artifacts": artifacts,
    }


def _summarize(
    per_seed: pd.DataFrame, isolated: pd.DataFrame, p_statevector: float
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (label, shots), block in per_seed.groupby(["observable_label", "shots"], sort=True):
        rel = block["relative_error_vs_ridge"].to_numpy(dtype=np.float64)
        rel = rel[np.isfinite(rel)]
        match = isolated[
            (isolated["observable_label"] == label) & (isolated["shots"] == int(shots))
        ]
        rows.append(
            {
                "observable_label": str(label),
                "shots": int(shots),
                "num_seeds": int(block["seed"].nunique()),
                "mean_measured_postselection_probability": float(
                    block["measured_postselection_probability"].mean()
                ),
                "std_measured_postselection_probability": float(
                    block["measured_postselection_probability"].std(ddof=0)
                ),
                "mean_postselection_probability_standard_error": float(
                    block["measured_postselection_probability_standard_error"].mean()
                ),
                "statevector_postselection_probability": float(p_statevector),
                "mean_signed_overlap_estimate_z": float(block["signed_overlap_estimate_z"].mean()),
                "mean_signed_overlap_standard_error": float(
                    block["signed_overlap_standard_error"].mean()
                ),
                "mean_recovered_physical_functional": float(
                    block["recovered_physical_functional"].mean()
                ),
                "std_recovered_physical_functional": float(
                    block["recovered_physical_functional"].std(ddof=0)
                ),
                "exact_ridge_functional": float(block["exact_ridge_functional"].iloc[0]),
                "exact_qsvt_statevector_functional": float(
                    block["exact_qsvt_statevector_functional"].iloc[0]
                ),
                "mean_absolute_error_vs_ridge": float(block["absolute_error_vs_ridge"].mean()),
                "mean_relative_error_vs_ridge": float(np.mean(rel)) if rel.size else float("nan"),
                "std_relative_error_vs_ridge": float(np.std(rel)) if rel.size else float("nan"),
                "mean_absolute_error_vs_statevector_qsvt": float(
                    block["absolute_error_vs_statevector_qsvt"].mean()
                ),
                "mean_relative_error_vs_statevector_qsvt": float(
                    block["relative_error_vs_statevector_qsvt"].mean()
                ),
                "isolated_readout_mean_relative_error_vs_ridge": float(
                    match["mean_relative_error_vs_ridge"].iloc[0]
                )
                if not match.empty
                else float("nan"),
                "mean_total_attempts": float(block["total_attempts"].mean()),
                "mean_accepted_attempts": float(block["accepted_attempts"].mean()),
                "mean_effective_shots_after_postselection": float(
                    block["effective_shots_after_postselection"].mean()
                ),
                "circuit_qubits": int(block["circuit_qubits"].iloc[0]),
                "signal_unitary_calls_per_attempt": int(
                    block["signal_unitary_calls_per_attempt"].iloc[0]
                ),
                "projector_phase_operations_per_attempt": int(
                    block["projector_phase_operations_per_attempt"].iloc[0]
                ),
                "alternating_sequence_length": int(block["alternating_sequence_length"].iloc[0]),
                "status": str(block["status"].iloc[0]),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _reference_values(
    context: AnchorContext,
    signed: list[Any],
    ridge_update: np.ndarray,
    x_encoded: np.ndarray,
    adjoint: dict[str, float],
    isolated: pd.DataFrame,
) -> dict[str, Any]:
    per_observable: dict[str, Any] = {}
    for observable in signed:
        ell = np.asarray(observable.vector, dtype=np.float64)
        ell_norm = float(np.linalg.norm(ell))
        scale = context.bound_c / context.beta * context.residual_norm * ell_norm
        z_exact = float(np.real(np.vdot(ell / ell_norm, x_encoded)))
        iso = isolated[isolated["observable_label"] == observable.observable_id]
        per_observable[observable.observable_id] = {
            "ridge_functional": float(ell @ ridge_update),
            "statevector_qsvt_functional": scale * z_exact,
            "exact_signed_overlap_z": z_exact,
            "physical_recovery_scale": scale,
            "isolated_readout_mean_relative_error_vs_ridge_by_shots": {
                str(int(row["shots"])): float(row["mean_relative_error_vs_ridge"])
                for _, row in iso.iterrows()
            },
        }
    return {
        "case": context.result.row_common["case"],
        "matrix_source": context.matrix_source,
        "block_shape": context.result.row_common["block_shape"],
        "alpha": context.alpha,
        "beta": context.beta,
        "bound_C": context.bound_c,
        "physical_recovery_factor_C_over_beta": context.bound_c / context.beta,
        "residual_norm": context.residual_norm,
        "degree": context.degree,
        "statevector_postselection_probability": (context.statevector_postselection_probability),
        "classical_adjoint_reference": {
            **adjoint,
            "timing_note": "Python wall-clock; diagnostic and environment-specific",
        },
        "observables": per_observable,
        "recovery_convention": RECOVERY_CONVENTION,
    }


def _circuit_metadata(
    context: AnchorContext,
    counts_meta: dict[str, int],
    circuit_records: dict[str, dict[str, Any]],
    resolved: dict[str, Any],
) -> dict[str, Any]:
    return {
        "integrated_readout_model": INTEGRATED_READOUT_MODEL,
        "output_state_used_for_preparation": False,
        "residual_state_preparation": (
            "unitary StatePreparation of the padded normalized residual (input state)"
        ),
        "controlled_operation_realization": (
            "explicit dense controlled-unitary gates (block-diagonal construction on the "
            "readout ancilla); same dense small-scale idiom as the canonical block encoding"
        ),
        "postselection_measurement": (
            "block-encoding flag qubit measured every shot; acceptance = flag 0; failures "
            "retained in the shot record"
        ),
        "signed_readout_measurement": (
            "readout ancilla Hadamard interference between the chain branch and the "
            "functional branch; sign from the ancilla bit on accepted shots"
        ),
        "signal_unitary_calls_per_attempt": counts_meta["signal_unitary_calls"],
        "projector_phase_operations_per_attempt": counts_meta["projector_phase_operations"],
        "alternating_sequence_length": counts_meta["alternating_sequence_length"],
        "degree": context.degree,
        "phase_count": int(context.phases.size),
        "anchor_validation": context.validation,
        "per_observable_circuits": circuit_records,
        "config": {
            key: resolved[key]
            for key in ("case", "case_source", "seed", "degree", "alpha_mult", "shots_grid")
        },
    }


def _write_outputs(
    *,
    output_dir: Path,
    per_seed: pd.DataFrame,
    summary: pd.DataFrame,
    references: dict[str, Any],
    circuit_metadata: dict[str, Any],
    counts_examples: dict[str, Any],
    context: AnchorContext,
) -> dict[str, Path]:
    per_seed_csv = output_dir / "integrated_readout_per_seed.csv"
    summary_csv = output_dir / "integrated_readout_summary.csv"
    references_json = output_dir / "integrated_readout_reference_values.json"
    metadata_json = output_dir / "integrated_readout_circuit_metadata.json"
    counts_json = output_dir / "integrated_readout_counts_summary.json"
    readme_md = output_dir / "README.md"

    per_seed.to_csv(per_seed_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    write_json(references_json, references)
    write_json(metadata_json, circuit_metadata)
    write_json(counts_json, counts_examples)
    readme_md.write_text(_readme(summary, references, context), encoding="utf-8")

    artifacts = {
        "integrated_readout_per_seed_csv": per_seed_csv,
        "integrated_readout_summary_csv": summary_csv,
        "integrated_readout_reference_values_json": references_json,
        "integrated_readout_circuit_metadata_json": metadata_json,
        "integrated_readout_counts_summary_json": counts_json,
        "readme_md": readme_md,
    }
    checksum_path = output_dir / "checksums.sha256"
    checksum_path.write_text(
        "".join(
            f"{_sha256(path)}  {path.name}\n"
            for _, path in sorted(artifacts.items(), key=lambda item: item[1].name)
        ),
        encoding="utf-8",
    )
    artifacts["checksums_sha256"] = checksum_path
    return artifacts


def _readme(summary: pd.DataFrame, references: dict[str, Any], context: AnchorContext) -> str:
    primary = summary[summary["observable_label"] == "state_correction_0"].sort_values("shots")
    lines = [
        "# Phase 8: Integrated 4x4 Finite-Shot Selected-Submatrix Chain",
        "",
        "One shot-executed circuit composes dense residual initialization, the synthesized "
        "degree-31 QSVT sequence, measured ancilla postselection, and Hadamard-type "
        "signed-functional readout for the passing IEEE-14 4x4 selected-submatrix anchor. "
        "The classically computed postselected output state is used only as a validation "
        "reference; no gate in the sampled circuit prepares it "
        "(`output_state_used_for_preparation = false`).",
        "",
        f"- physical recovery: `{RECOVERY_CONVENTION}`",
        f"- statevector postselection probability (reference): "
        f"{context.statevector_postselection_probability:.4f}",
        f"- signal-unitary calls per attempt N_U = d = {context.degree}; projector phases "
        f"N_phi = d+1 = {context.degree + 1}; alternating length 2d+1 = "
        f"{2 * context.degree + 1}; circuit qubits = 4",
        "",
        "## Primary functional `state_correction_0` (first selected coordinate)",
        "",
        "| Shots | measured p_succ | recovered mean | rel err vs Ridge | rel err vs "
        "statevector QSVT | isolated rel err vs Ridge | seeds |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in primary.iterrows():
        lines.append(
            f"| {int(row['shots'])} | {row['mean_measured_postselection_probability']:.4f} | "
            f"{row['mean_recovered_physical_functional']:.4e} | "
            f"{row['mean_relative_error_vs_ridge']:.3e} | "
            f"{row['mean_relative_error_vs_statevector_qsvt']:.3e} | "
            f"{row['isolated_readout_mean_relative_error_vs_ridge']:.3e} | "
            f"{int(row['num_seeds'])} |"
        )
    lines += [
        "",
        "The isolated column repeats the previous assumed-output-state-preparation "
        "Hadamard-overlap experiment for the same functional; the integrated chain replaces "
        "that assumption with the measured chain itself.",
        "",
        "## Interpretation boundary",
        "",
        "This is a 4x4 dense selected-submatrix demonstration on a statevector-based shot "
        "simulator with dense controlled-unitary gates. It does not imply scalable residual "
        "loading, IEEE-scale sparse block encoding, full selected-output PSSE execution, or "
        "quantum competitiveness. Larger blocks and IEEE-scale composition remain modeled.",
        "",
    ]
    text = "\n".join(lines)
    assert_safe(text)
    return text


MANUSCRIPT_TABLE_PATH = Path("manuscript/tables/phase8_integrated_readout.tex")


def _sci_tex(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "--"
    if value == 0.0:
        return "$0$"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return f"${mantissa:.{digits}f}\\times10^{{{exponent}}}$"


def build_manuscript_table(
    summary_csv: str | Path = PHASE8_READOUT_DIR / "integrated_readout_summary.csv",
    table_path: str | Path = MANUSCRIPT_TABLE_PATH,
) -> Path:
    """Write the compact manuscript table for the primary integrated functional."""

    summary = pd.read_csv(summary_csv)
    primary = summary[summary["observable_label"] == "state_correction_0"].sort_values("shots")
    if primary.empty:
        raise RuntimeError("summary CSV has no state_correction_0 rows")
    ridge = float(primary["exact_ridge_functional"].iloc[0])
    statevector = float(primary["exact_qsvt_statevector_functional"].iloc[0])
    p_sv = float(primary["statevector_postselection_probability"].iloc[0])
    seeds = int(primary["num_seeds"].iloc[0])
    lines = [
        "% Source: outputs/phase8_integrated_readout/integrated_readout_summary.csv",
        "% Regenerate: .venv/bin/python scripts/run_phase8_integrated_readout.py",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Integrated $4\times4$ finite-shot selected-submatrix chain: dense residual "
        r"initialization, the synthesized degree-31 QSVT sequence, measured ancilla "
        r"postselection, and Hadamard-type signed readout composed in one shot-executed "
        r"circuit (4 qubits; per attempt $N_U{=}31$ signal-unitary calls, $N_\phi{=}32$ "
        r"projector phases), for the primary functional (first selected coordinate), "
        f"mean over {seeds} shot seeds per cell. "
        r"Recovery uses the measured acceptance frequency and the recorded $C/\beta$; the "
        r"classically computed output state is never prepared by the sampled circuit. "
        r"References: matched Ridge "
        + _sci_tex(ridge, 4)
        + r", statevector QSVT "
        + _sci_tex(statevector, 4)
        + r", statevector $p_{\rm succ}="
        + f"{p_sv:.4f}"
        + r"$. "
        r"``Isolated'' repeats the previous assumed-output-state-preparation overlap "
        r"experiment for the same functional. This remains a selected-submatrix "
        r"demonstration; it does not imply scalable residual loading, IEEE-scale sparse "
        r"block encoding, full selected-output PSSE execution, or quantum competitiveness.}",
        r"\label{tab:phase8_integrated_readout}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{rcccccc}",
        r"\hline",
        r"Shots & meas.\ $p_{\rm succ}$ & accepted & recovered mean & rel.\ err vs Ridge & "
        r"rel.\ err vs SV & isolated \\",
        r"\hline",
    ]
    for _, row in primary.iterrows():
        lines.append(
            f"$10^{{{int(np.log10(row['shots']))}}}$ & "
            f"{float(row['mean_measured_postselection_probability']):.4f} & "
            f"{_sci_tex(float(row['mean_accepted_attempts']), 2)} & "
            f"{_sci_tex(float(row['mean_recovered_physical_functional']), 4)} & "
            f"{_sci_tex(float(row['mean_relative_error_vs_ridge']))} & "
            f"{_sci_tex(float(row['mean_relative_error_vs_statevector_qsvt']))} & "
            f"{_sci_tex(float(row['isolated_readout_mean_relative_error_vs_ridge']))} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}", ""]
    text = "\n".join(lines)
    assert_safe(text)
    destination = Path(table_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return destination


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": str(PHASE8_READOUT_DIR),
        "case": "ieee14",
        "case_source": "pypower",
        "seed": 123,
        "degree": 31,
        "alpha_mult": 4.0,
        "num_seeds": 30,
        "shots_grid": list(FULL_SHOTS),
        "phase_cache_dir": str(DEMO_PHASE_CACHE),
        "command": "run_phase8_integrated_readout",
    }
    if config:
        resolved.update({key: value for key, value in config.items() if value is not None})
    resolved["num_seeds"] = int(resolved["num_seeds"])
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 8: integrated 4x4 finite-shot QSVT chain")
    parser.add_argument("--output-dir", default=str(PHASE8_READOUT_DIR))
    parser.add_argument("--quick", action="store_true", help="fast smoke run (6 seeds)")
    parser.add_argument("--num-seeds", type=int, default=None)
    parser.add_argument("--table-path", default=str(MANUSCRIPT_TABLE_PATH))
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="rebuild the manuscript table from the existing summary CSV",
    )
    args = parser.parse_args(argv)

    if args.table_only:
        table = build_manuscript_table(
            Path(args.output_dir) / "integrated_readout_summary.csv", args.table_path
        )
        print(f"Manuscript table rebuilt: {table}")
        return

    if args.quick:
        num_seeds, shots = 6, list(QUICK_SHOTS)
    else:
        num_seeds, shots = 30, list(FULL_SHOTS)
    if args.num_seeds is not None:
        num_seeds = int(args.num_seeds)

    run = run_phase8_integrated_readout(
        {
            "output_dir": args.output_dir,
            "num_seeds": num_seeds,
            "shots_grid": shots,
            "command": "scripts/run_phase8_integrated_readout.py " + " ".join(argv or []),
        }
    )
    if not args.quick:
        table = build_manuscript_table(
            Path(args.output_dir) / "integrated_readout_summary.csv", args.table_path
        )
        print(f"Manuscript table: {table}")
    print(f"Integrated readout complete: {run['artifacts']['integrated_readout_summary_csv']}")
    print(run["summary"].to_string(index=False, max_colwidth=40))


if __name__ == "__main__":  # pragma: no cover
    main()
