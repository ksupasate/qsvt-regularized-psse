from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def qsp_circuit_summary(phases: np.ndarray) -> dict[str, Any]:
    phase_values = np.asarray(phases, dtype=np.float64)
    if phase_values.ndim != 1 or phase_values.size == 0:
        raise ValueError("phases must be a non-empty 1D array")
    if not np.all(np.isfinite(phase_values)):
        raise ValueError("phases must be finite")

    try:
        from qiskit import QuantumCircuit  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        return {
            "qiskit_available": False,
            "qiskit_error": str(exc),
            "circuit_depth": None,
            "gate_counts": {},
            "gate_count_total": 0,
        }

    circuit = QuantumCircuit(1, name="qsp_phase_demo")
    for index, phase in enumerate(phase_values):
        circuit.rz(2.0 * float(phase), 0)
        if index < phase_values.size - 1:
            circuit.ry(np.pi / 2.0, 0)
    gate_counts: Mapping[str, int] = circuit.count_ops()
    return {
        "qiskit_available": True,
        "qiskit_error": None,
        "circuit_depth": int(circuit.depth()),
        "gate_counts": dict(gate_counts),
        "gate_count_total": int(sum(gate_counts.values())),
    }
