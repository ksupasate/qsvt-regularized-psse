"""Explicit binary-tree (Möttönen-style) amplitude loader for real vectors.

Phase 10 WP C: a *concrete* structured amplitude loader, distinct from Qiskit's
``initialize`` (Mode 1), so residual loading is an explicit circuit rather than
an unspecified assumption.

For a real vector of dimension ``N = 2^n`` the loader is a cascade of ``n``
uniformly controlled (multiplexed) RY rotations, one layer per qubit, with
``2^level`` rotation angles at layer ``level`` (``N - 1`` rotations total).  The
angle at a tree node with left/right child values ``L`` and ``R`` is
``theta = 2 * atan2(R, L)``.  Internal nodes carry subtree norms (nonnegative),
so their rotations split magnitude; leaf nodes carry the signed amplitudes, so
the signed ``atan2`` reproduces the exact real signs.  No auxiliary sign layer
is needed.

The angle computation is exact for signed real inputs (validated by classical
reconstruction at any size).  A Qiskit circuit built from these angles is
statevector-validated at the small feasible sizes; at larger sizes the rotation
and qubit counts are reported analytically and circuit compilation is flagged
expensive rather than silently claimed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class BinaryTreeLoaderPlan:
    """Angle plan for a real binary-tree amplitude loader (no circuit yet)."""

    n_qubits: int
    dimension: int
    input_dimension: int
    rotation_angles: tuple[tuple[int, int, float], ...]  # (level, control_state, theta)
    rotation_count: int
    nonzero_rotation_count: int
    reconstruction_error: float


def compute_binary_tree_plan(
    vector: np.ndarray, *, target_dimension: int | None = None, eps: float = 1.0e-15
) -> BinaryTreeLoaderPlan:
    """Compute the exact real binary-tree RY angle plan for ``vector``."""

    values = np.asarray(vector, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("vector must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("vector entries must be finite")
    norm = float(np.linalg.norm(values))
    if norm <= float(eps):
        raise ValueError("cannot load a zero or near-zero vector")
    input_dimension = int(values.size)
    dimension = (
        _next_power_of_two(input_dimension) if target_dimension is None else int(target_dimension)
    )
    if dimension < input_dimension:
        raise ValueError("target_dimension cannot be smaller than the vector length")
    if not _is_power_of_two(dimension):
        raise ValueError("target_dimension must be a power of two")
    n_qubits = round(math.log2(dimension))
    amplitudes = np.zeros(dimension, dtype=np.float64)
    amplitudes[:input_dimension] = values / norm

    # Build subtree-norm layers from leaves up to the root.
    layers = [amplitudes.copy()]
    current = amplitudes.copy()
    for _ in range(n_qubits):
        pairs = current.reshape(-1, 2)
        current = np.sqrt(pairs[:, 0] ** 2 + pairs[:, 1] ** 2)
        layers.append(current)
    layers = layers[::-1]  # layers[0] = root (size 1) ... layers[n] = leaves (size N)

    gates: list[tuple[int, int, float]] = []
    for level in range(n_qubits):
        parent = layers[level]
        child = layers[level + 1]
        for node in range(parent.size):
            left = float(child[2 * node])
            right = float(child[2 * node + 1])
            theta = 2.0 * math.atan2(right, left) if (abs(left) + abs(right)) > 0.0 else 0.0
            gates.append((level, node, theta))

    reconstruction = _reconstruct_from_gates(gates, n_qubits)
    error = float(np.max(np.abs(reconstruction - amplitudes))) if dimension else 0.0
    return BinaryTreeLoaderPlan(
        n_qubits=n_qubits,
        dimension=dimension,
        input_dimension=input_dimension,
        rotation_angles=tuple(gates),
        rotation_count=len(gates),
        nonzero_rotation_count=int(sum(1 for _, _, theta in gates if abs(theta) > 1.0e-12)),
        reconstruction_error=error,
    )


def build_binary_tree_circuit(plan: BinaryTreeLoaderPlan) -> Any:
    """Compile the multiplexed-RY loader circuit from an angle plan.

    The angle plan indexes basis states most-significant-bit first, while Qiskit
    statevectors are little-endian, so tree level ``l`` maps to physical qubit
    ``n - 1 - l`` (the MSB of the split is the highest-index qubit).  The control
    pattern (the tree node index) is passed as ``ctrl_state`` over the ascending
    control-qubit list ``[n - l, ..., n - 1]`` so that node bit ``i`` aligns with
    physical qubit ``n - l + i``.  Zero-angle rotations are skipped.
    """

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RYGate

    n = plan.n_qubits
    circuit = QuantumCircuit(n, name="binary_tree_real_loader")
    for level, control_state, theta in plan.rotation_angles:
        if abs(theta) <= 1.0e-15:
            continue
        target = n - 1 - level
        if level == 0:
            circuit.ry(theta, target)
            continue
        controls = list(range(n - level, n))
        gate = RYGate(theta).control(level, ctrl_state=control_state)
        circuit.append(gate, [*controls, target])
    return circuit


def validate_binary_tree_circuit(circuit: Any, plan: BinaryTreeLoaderPlan) -> dict[str, Any]:
    """Statevector-validate the compiled loader against the target amplitudes."""

    from qiskit.quantum_info import Statevector

    state = np.asarray(Statevector.from_instruction(circuit).data, dtype=np.complex128)
    target = _reconstruct_from_gates(plan.rotation_angles, plan.n_qubits).astype(np.complex128)
    error = float(np.linalg.norm(state - target))
    fidelity = float(abs(np.vdot(target, state)) ** 2)
    return {
        "state_preparation_l2_error": error,
        "state_preparation_fidelity": fidelity,
        "prepared_state_norm": float(np.linalg.norm(state)),
        "raw_gate_count": int(sum(circuit.count_ops().values())),
        "raw_depth": int(circuit.depth()),
    }


def _reconstruct_from_gates(
    gates: tuple[tuple[int, int, float], ...] | list[tuple[int, int, float]], n_qubits: int
) -> np.ndarray:
    dimension = 1 << n_qubits
    amplitudes = np.ones(dimension, dtype=np.float64)
    for level, node, theta in gates:
        cos = math.cos(theta / 2.0)
        sin = math.sin(theta / 2.0)
        shift = n_qubits - level
        bit_shift = n_qubits - level - 1
        for basis in range(dimension):
            if (basis >> shift) != node:
                continue
            amplitudes[basis] *= cos if ((basis >> bit_shift) & 1) == 0 else sin
    return amplitudes


def _next_power_of_two(value: int) -> int:
    return 1 << (max(int(value), 1) - 1).bit_length()


def _is_power_of_two(value: int) -> bool:
    return int(value) > 0 and (int(value) & (int(value) - 1)) == 0
