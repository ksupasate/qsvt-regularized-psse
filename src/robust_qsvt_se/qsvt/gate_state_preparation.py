from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class DenseStatePreparationResult:
    normalized_state: np.ndarray
    padded_state: np.ndarray
    original_norm: float
    input_dimension: int
    padded_dimension: int
    n_qubits: int
    padding_width: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalized_state_real": np.real(self.normalized_state).tolist(),
            "normalized_state_imag": np.imag(self.normalized_state).tolist(),
            "padded_state_real": np.real(self.padded_state).tolist(),
            "padded_state_imag": np.imag(self.padded_state).tolist(),
            "original_norm": self.original_norm,
            "input_dimension": self.input_dimension,
            "padded_dimension": self.padded_dimension,
            "n_qubits": self.n_qubits,
            "padding_width": self.padding_width,
            "state_preparation_method": "qiskit_initialize_dense_amplitude_loading",
            "limitation": (
                "Qiskit Initialize is used for small simulator evidence only; it is not "
                "an efficient scalable state-preparation proof."
            ),
        }


def normalize_and_pad_for_gate_preparation(
    vector: np.ndarray,
    *,
    target_dimension: int | None = None,
    eps: float = 1.0e-15,
) -> DenseStatePreparationResult:
    values = np.asarray(vector, dtype=np.complex128)
    if values.ndim != 1:
        raise ValueError("vector must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("vector entries must be finite")
    norm = float(np.linalg.norm(values))
    if norm <= float(eps):
        raise ValueError("cannot prepare a zero or near-zero state")
    normalized = values / norm
    padded_dimension = (
        _next_power_of_two(values.size) if target_dimension is None else int(target_dimension)
    )
    if padded_dimension < values.size:
        raise ValueError("target_dimension cannot be smaller than vector length")
    if not _is_power_of_two(padded_dimension):
        raise ValueError("target_dimension must be a power of two")
    padded = np.zeros(padded_dimension, dtype=np.complex128)
    padded[: values.size] = normalized
    return DenseStatePreparationResult(
        normalized_state=normalized,
        padded_state=padded,
        original_norm=norm,
        input_dimension=int(values.size),
        padded_dimension=int(padded_dimension),
        n_qubits=_qubits(padded_dimension),
        padding_width=int(padded_dimension - values.size),
    )


def build_initialize_circuit(state: np.ndarray, *, name: str = "dense_state_prep") -> Any:
    try:
        from qiskit import QuantumCircuit  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency branch
        raise RuntimeError("Qiskit is required for gate-level state preparation") from exc

    values = np.asarray(state, dtype=np.complex128)
    if values.ndim != 1 or not _is_power_of_two(values.size):
        raise ValueError("state must be a power-of-two one-dimensional vector")
    norm = float(np.linalg.norm(values))
    if not np.isclose(norm, 1.0, atol=1.0e-10):
        raise ValueError("state must be normalized before Initialize")
    circuit = QuantumCircuit(_qubits(values.size), name=name)
    circuit.initialize(values, list(range(circuit.num_qubits)))
    return circuit


def validate_initialize_circuit(circuit: Any, target_state: np.ndarray) -> dict[str, Any]:
    try:
        from qiskit.quantum_info import Statevector  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency branch
        raise RuntimeError("Qiskit is required for statevector validation") from exc

    target = np.asarray(target_state, dtype=np.complex128)
    state = np.asarray(Statevector.from_instruction(circuit).data, dtype=np.complex128)
    if state.shape != target.shape:
        raise ValueError("statevector and target_state have different dimensions")
    overlap = complex(np.vdot(target, state))
    error = float(np.linalg.norm(state - target))
    return {
        "state_preparation_l2_error": error,
        "state_preparation_fidelity": float(abs(overlap) ** 2),
        "prepared_state_norm": float(np.linalg.norm(state)),
    }


def _next_power_of_two(value: int) -> int:
    return 1 << (max(int(value), 1) - 1).bit_length()


def _qubits(dimension: int) -> int:
    return int(np.ceil(np.log2(max(int(dimension), 2))))


def _is_power_of_two(value: int) -> bool:
    return int(value) > 0 and (int(value) & (int(value) - 1)) == 0
