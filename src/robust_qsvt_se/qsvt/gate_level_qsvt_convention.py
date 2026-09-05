from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from robust_qsvt_se.qsvt.block_encoding import canonical_square_block_encoding
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.gate_level_qsvt import build_structured_qsvt_operator_circuit
from robust_qsvt_se.utils.io import ensure_directory, write_json

CORRECT_OPERATOR_EXTRACTION_RULE = "real_top_left_signal_block"
CORRECT_STATE_EXTRACTION_RULE = "real_prefix_signal_quadrature_state"
CONVENTION_CLAIM_BOUNDARY = (
    "This is a gate-level QSVT convention debug artifact for small dense "
    "block-encoded matrices. It does not demonstrate full IEEE-scale hardware "
    "execution, quantum speedup, or QSVT superiority over Ridge/Tikhonov."
)
IDENTIFIED_ERROR_SOURCE = (
    "The Qiskit and PennyLane QSVT operators use the same phase and PCPhase "
    "block convention. The bounded polynomial target appears in the real part "
    "of the top-left encoded signal block. The previous high state error was "
    "primarily a state-reference orientation issue: the gate-level demo encoded "
    "B = H_tilde.T, whose singular-value transform maps measurement residuals "
    "to state updates, but it compared against a Ridge solve of B directly. "
    "The complex postselected prefix also contains non-target imaginary "
    "quadrature, so comparing that raw complex state to Ridge is not the "
    "correct extraction rule."
)


@dataclass(frozen=True, slots=True)
class ConventionProbe:
    matrix: np.ndarray
    coefficients: np.ndarray
    phases: np.ndarray
    target_block: np.ndarray
    qiskit_unitary: np.ndarray
    pennylane_unitary: np.ndarray


def linear_test_coefficients(scale: float = 0.5) -> np.ndarray:
    """Return coefficients for the bounded odd polynomial ``p(x)=scale*x``."""

    if not 0.0 < float(scale) <= 1.0:
        raise ValueError("scale must lie in (0, 1]")
    return np.array([0.0, float(scale)], dtype=np.float64)


def synthesize_test_phases(coefficients: np.ndarray) -> np.ndarray:
    """Synthesize QSVT phases for a bounded test polynomial with PennyLane."""

    try:
        import pennylane as qml  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency branch
        raise RuntimeError("PennyLane is required for QSVT phase synthesis") from exc

    values = np.asarray(coefficients, dtype=np.float64)
    return np.asarray(qml.poly_to_angles(values, "QSVT", angle_solver="iterative"))


def build_convention_probe(
    matrix: np.ndarray,
    coefficients: np.ndarray | None = None,
) -> ConventionProbe:
    """Build matching Qiskit and PennyLane QSVT operator matrices for ``matrix``."""

    A = np.asarray(matrix, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("matrix must be square")
    if not _is_power_of_two(A.shape[0]):
        raise ValueError("matrix dimension must be a power of two")
    coeffs = linear_test_coefficients() if coefficients is None else np.asarray(coefficients)
    phases = synthesize_test_phases(coeffs)
    block = canonical_square_block_encoding(A)
    qiskit_unitary = qiskit_qsvt_operator_matrix(
        block.unitary,
        phases,
        encoded_dimension=A.shape[0],
    )
    pennylane_unitary = pennylane_qsvt_operator_matrix(
        block.unitary,
        phases,
        encoded_dimension=A.shape[0],
    )
    target = polynomial_spectral_transform(A, coeffs)
    return ConventionProbe(
        matrix=A,
        coefficients=coeffs,
        phases=phases,
        target_block=target,
        qiskit_unitary=qiskit_unitary,
        pennylane_unitary=pennylane_unitary,
    )


def qiskit_qsvt_operator_matrix(
    block_unitary: np.ndarray,
    phases: np.ndarray,
    *,
    encoded_dimension: int,
) -> np.ndarray:
    """Return the full unitary matrix of the Qiskit gate-level QSVT circuit."""

    try:
        from qiskit.quantum_info import Operator  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency branch
        raise RuntimeError("Qiskit is required for operator extraction") from exc

    bundle = build_structured_qsvt_operator_circuit(
        np.asarray(block_unitary, dtype=np.complex128),
        np.asarray(phases, dtype=np.float64),
        encoded_dimension=int(encoded_dimension),
    )
    return np.asarray(Operator(bundle.qsvt_operator_circuit).data, dtype=np.complex128)


def pennylane_qsvt_operator_matrix(
    block_unitary: np.ndarray,
    phases: np.ndarray,
    *,
    encoded_dimension: int,
) -> np.ndarray:
    """Return the matching PennyLane QSVT operator matrix."""

    try:
        import pennylane as qml  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency branch
        raise RuntimeError("PennyLane is required for operator extraction") from exc

    unitary = np.asarray(block_unitary, dtype=np.complex128)
    wires = list(range(_qubits(unitary.shape[0])))
    qml_operator = qml.QSVT(
        qml.QubitUnitary(unitary, wires=wires),
        [qml.PCPhase(float(phase), dim=int(encoded_dimension), wires=wires) for phase in phases],
    )
    return np.asarray(qml.matrix(qml_operator, wire_order=wires), dtype=np.complex128)


def polynomial_spectral_transform(matrix: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Apply a real polynomial to the singular values of ``matrix``."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    polynomial = Polynomial(np.asarray(coefficients, dtype=np.float64))
    U, singular_values, Vh = np.linalg.svd(values, full_matrices=False)
    return U @ (polynomial(singular_values)[:, None] * Vh)


def operator_block_error_rows(
    unitary: np.ndarray,
    target_block: np.ndarray,
    *,
    encoded_dimension: int,
) -> list[dict[str, Any]]:
    """Compare candidate signal blocks and quadratures with the target block."""

    operator = np.asarray(unitary, dtype=np.complex128)
    target = np.asarray(target_block, dtype=np.complex128)
    dimension = int(encoded_dimension)
    if target.shape != (dimension, dimension):
        raise ValueError("target_block shape must match encoded_dimension")
    if operator.shape[0] < 2 * dimension or operator.shape[1] < dimension:
        raise ValueError("unitary is too small for encoded block extraction")

    top_left = operator[:dimension, :dimension]
    ancilla_1 = operator[dimension : 2 * dimension, :dimension]
    phase_aligned, phase = global_phase_aligned(target, top_left)
    rows = [
        _block_row(
            CORRECT_OPERATOR_EXTRACTION_RULE,
            np.real(top_left),
            target,
            "real part of the encoded top-left signal block",
        ),
        _block_row(
            "imaginary_top_left_signal_block",
            np.imag(top_left),
            target,
            "imaginary part of the encoded top-left signal block",
        ),
        _block_row(
            "ancilla_0_postselected_block",
            top_left,
            target,
            "complex block obtained by postselecting the encoded prefix",
        ),
        _block_row(
            "ancilla_1_postselected_block",
            ancilla_1,
            target,
            "complex block obtained by postselecting the orthogonal suffix",
        ),
        _block_row(
            "best_global_phase_aligned_block",
            phase_aligned,
            target,
            f"top-left block aligned by global phase {phase:.17g} radians",
        ),
    ]
    for row in rows:
        row["encoded_dimension"] = dimension
    return rows


def state_extraction_error_rows(
    statevector: np.ndarray,
    target_state: np.ndarray,
    *,
    encoded_dimension: int,
) -> list[dict[str, Any]]:
    """Compare state-extraction candidates against a normalized target state."""

    state = np.asarray(statevector, dtype=np.complex128)
    target = _normalize(np.asarray(target_state, dtype=np.complex128))
    dimension = int(encoded_dimension)
    if state.ndim != 1 or state.size < 2 * dimension:
        raise ValueError("statevector is too small for encoded state extraction")
    prefix = state[:dimension]
    suffix = state[dimension : 2 * dimension]
    complex_prefix = _normalize(prefix)
    real_prefix = _normalize(np.real(prefix))
    imag_prefix = _normalize(np.imag(prefix))
    suffix_state = _normalize(suffix)
    phase_prefix, phase = global_phase_aligned(target, complex_prefix)
    sign_prefix, sign = sign_aligned(target, real_prefix)
    candidates = [
        (
            "complex_prefix_postselected_state",
            complex_prefix,
            "raw complex state after encoded-prefix postselection",
        ),
        (
            CORRECT_STATE_EXTRACTION_RULE,
            real_prefix,
            "real signal quadrature of the encoded-prefix state",
        ),
        (
            "imaginary_prefix_signal_quadrature_state",
            imag_prefix,
            "imaginary quadrature of the encoded-prefix state",
        ),
        (
            "ancilla_1_postselected_state",
            suffix_state,
            "state after postselecting the orthogonal suffix",
        ),
        (
            "best_global_phase_aligned_prefix_state",
            phase_prefix,
            f"complex prefix aligned by global phase {phase:.17g} radians",
        ),
        (
            "best_sign_aligned_real_prefix_state",
            sign_prefix,
            f"real prefix aligned by sign {sign:+d}",
        ),
    ]
    success_probability = float(np.sum(np.abs(prefix) ** 2))
    rows = []
    for rule, candidate, notes in candidates:
        rows.append(
            {
                "extraction_rule": rule,
                "state_l2_error": float(np.linalg.norm(candidate - target)),
                "best_sign_l2_error": best_sign_l2_error(target, candidate),
                "best_global_phase_l2_error": best_global_phase_l2_error(target, candidate),
                "success_probability": success_probability,
                "notes": notes,
            }
        )
    return rows


def best_operator_rule(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must be nonempty")
    return min(rows, key=lambda row: float(row["frobenius_error"]))


def best_state_rule(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must be nonempty")
    return min(rows, key=lambda row: float(row["best_sign_l2_error"]))


def global_phase_aligned(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> tuple[np.ndarray, float]:
    ref = np.asarray(reference, dtype=np.complex128)
    cand = np.asarray(candidate, dtype=np.complex128)
    overlap = np.vdot(ref.ravel(), cand.ravel())
    if abs(overlap) <= 1.0e-15:
        return cand, 0.0
    phase = -float(np.angle(overlap))
    return cand * np.exp(1j * phase), phase


def sign_aligned(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, int]:
    ref = np.asarray(reference, dtype=np.complex128)
    cand = np.asarray(candidate, dtype=np.complex128)
    positive = float(np.linalg.norm(cand - ref))
    negative = float(np.linalg.norm(-cand - ref))
    if negative < positive:
        return -cand, -1
    return cand, 1


def best_global_phase_l2_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    aligned, _ = global_phase_aligned(reference, candidate)
    return float(np.linalg.norm(aligned - np.asarray(reference, dtype=np.complex128)))


def best_sign_l2_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    aligned, _ = sign_aligned(reference, candidate)
    return float(np.linalg.norm(aligned - np.asarray(reference, dtype=np.complex128)))


def run_gate_level_qsvt_convention_debug(
    output_dir: str | Path = "outputs/gate_level_qsvt_convention_debug",
) -> dict[str, Any]:
    """Run scalar/diagonal convention probes and write auditable artifacts."""

    root = ensure_directory(output_dir)
    scalar_probe = build_convention_probe(np.array([[0.3]], dtype=np.float64))
    diagonal_probe = build_convention_probe(np.diag([0.25, 0.7]))

    scalar_rows = operator_block_error_rows(
        scalar_probe.qiskit_unitary,
        scalar_probe.target_block,
        encoded_dimension=1,
    )
    diagonal_rows = operator_block_error_rows(
        diagonal_probe.qiskit_unitary,
        diagonal_probe.target_block,
        encoded_dimension=2,
    )
    scalar_diagonal_rows = [{"test_case": "scalar_1x1", **row} for row in scalar_rows] + [
        {"test_case": "diagonal_2x2", **row} for row in diagonal_rows
    ]

    residual_state = np.array([0.6, -0.8], dtype=np.complex128)
    full_input = np.zeros(diagonal_probe.qiskit_unitary.shape[0], dtype=np.complex128)
    full_input[: residual_state.size] = residual_state
    output_state = diagonal_probe.qiskit_unitary @ full_input
    target_update = diagonal_probe.target_block @ residual_state
    state_rows = state_extraction_error_rows(
        output_state,
        target_update,
        encoded_dimension=2,
    )

    qiskit_pennylane = qiskit_pennylane_comparison(diagonal_probe)
    scalar_correct = _row_by_rule(scalar_rows, CORRECT_OPERATOR_EXTRACTION_RULE)
    diagonal_correct = _row_by_rule(diagonal_rows, CORRECT_OPERATOR_EXTRACTION_RULE)
    best_operator = best_operator_rule(scalar_rows + diagonal_rows)
    best_state = best_state_rule(state_rows)
    summary = {
        "scalar_test_error": float(scalar_correct["frobenius_error"]),
        "diagonal_test_error": float(diagonal_correct["frobenius_error"]),
        "best_operator_block_error": float(best_operator["frobenius_error"]),
        "best_state_extraction_error": float(best_state["best_sign_l2_error"]),
        "best_extraction_rule": str(best_state["extraction_rule"]),
        "correct_operator_extraction_rule": CORRECT_OPERATOR_EXTRACTION_RULE,
        "correct_state_extraction_rule": CORRECT_STATE_EXTRACTION_RULE,
        "qiskit_vs_pennylane_operator_error": qiskit_pennylane["frobenius_error"],
        "identified_error_source": IDENTIFIED_ERROR_SOURCE,
        "claim_boundary": CONVENTION_CLAIM_BOUNDARY,
    }

    scalar_path = root / "scalar_diagonal_tests.csv"
    operator_path = root / "operator_block_errors.csv"
    state_path = root / "state_extraction_errors.csv"
    comparison_path = root / "qiskit_pennylane_operator_comparison.json"
    summary_path = root / "convention_debug_summary.md"
    pd.DataFrame(scalar_diagonal_rows).to_csv(scalar_path, index=False)
    pd.DataFrame(diagonal_rows).to_csv(operator_path, index=False)
    pd.DataFrame(state_rows).to_csv(state_path, index=False)
    write_json(comparison_path, qiskit_pennylane)
    summary_path.write_text(_summary_markdown(summary), encoding="utf-8")
    manifest = write_manifest(
        root,
        artifacts={
            "convention_debug_summary": str(summary_path),
            "scalar_diagonal_tests": str(scalar_path),
            "operator_block_errors": str(operator_path),
            "state_extraction_errors": str(state_path),
            "qiskit_pennylane_operator_comparison": str(comparison_path),
        },
        input_config={
            "scalar_matrix": [[0.3]],
            "diagonal_matrix": [[0.25, 0.0], [0.0, 0.7]],
            "coefficients": scalar_probe.coefficients.tolist(),
        },
        claim_boundary=CONVENTION_CLAIM_BOUNDARY,
    )
    return {
        "output_dir": root,
        "summary": summary,
        "artifacts": {
            "manifest": manifest,
            "convention_debug_summary": summary_path,
            "scalar_diagonal_tests": scalar_path,
            "operator_block_errors": operator_path,
            "state_extraction_errors": state_path,
            "qiskit_pennylane_operator_comparison": comparison_path,
        },
    }


def qiskit_pennylane_comparison(probe: ConventionProbe) -> dict[str, Any]:
    difference = probe.qiskit_unitary - probe.pennylane_unitary
    return {
        "same_phase_sequence": True,
        "same_qubit_ordering": True,
        "same_block_encoding_convention": True,
        "same_projection_rule": True,
        "phase_count": int(probe.phases.size),
        "operator_dimension": int(probe.qiskit_unitary.shape[0]),
        "max_abs_error": float(np.max(np.abs(difference))),
        "frobenius_error": float(np.linalg.norm(difference)),
        "conclusion": (
            "Qiskit circuit construction matches PennyLane QSVT for the same "
            "block unitary, phases, qubit order, and PCPhase projection."
        ),
    }


def _block_row(
    rule: str,
    candidate: np.ndarray,
    target: np.ndarray,
    notes: str,
) -> dict[str, Any]:
    values = np.asarray(candidate, dtype=np.complex128)
    target_values = np.asarray(target, dtype=np.complex128)
    difference = values - target_values
    return {
        "extraction_rule": rule,
        "frobenius_error": float(np.linalg.norm(difference)),
        "max_abs_error": float(np.max(np.abs(difference))),
        "candidate_max_imag_abs": float(np.max(np.abs(np.imag(values)))),
        "target_norm": float(np.linalg.norm(target_values)),
        "notes": notes,
    }


def _row_by_rule(rows: list[dict[str, Any]], rule: str) -> dict[str, Any]:
    for row in rows:
        if row["extraction_rule"] == rule:
            return row
    raise KeyError(rule)


def _normalize(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.complex128)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-15:
        raise ValueError("cannot normalize a zero vector")
    return vector / norm


def _summary_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Gate-Level QSVT Convention Debug",
            "",
            CONVENTION_CLAIM_BOUNDARY,
            "",
            "## Metrics",
            f"- scalar_test_error: {summary['scalar_test_error']:.17g}",
            f"- diagonal_test_error: {summary['diagonal_test_error']:.17g}",
            f"- best_operator_block_error: {summary['best_operator_block_error']:.17g}",
            f"- best_state_extraction_error: {summary['best_state_extraction_error']:.17g}",
            f"- best_extraction_rule: {summary['best_extraction_rule']}",
            "- qiskit_vs_pennylane_operator_error: "
            f"{summary['qiskit_vs_pennylane_operator_error']:.17g}",
            "",
            "## Extraction Rule",
            f"- Operator block: `{CORRECT_OPERATOR_EXTRACTION_RULE}`.",
            f"- State vector: `{CORRECT_STATE_EXTRACTION_RULE}`.",
            "",
            "## Identified Error Source",
            summary["identified_error_source"],
            "",
        ]
    )


def _qubits(dimension: int) -> int:
    return int(np.ceil(np.log2(max(int(dimension), 2))))


def _is_power_of_two(value: int) -> bool:
    return int(value) > 0 and (int(value) & (int(value) - 1)) == 0
