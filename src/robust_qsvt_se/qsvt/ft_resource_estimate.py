"""Modeled fault-tolerant logical resource estimate for the frozen QSVT workloads.

The frozen ledger records default-target transpiled counts (Toffolis, controlled value
rotations, gates, depth).  This module reduces the identity-verified rebuilt final circuits to
an explicit Clifford+T+Rz basis, classifies every ``rz`` angle exactly (Clifford / T-layer /
arbitrary), and applies a declared rotation-synthesis model to obtain a total logical T-count
and a serial Clifford+T depth estimate.

Declared conventions (all displayed in the generated table):

* Toffoli -> T: the exact ancilla-free 7T decomposition (the reduction Qiskit applies), with a
  4T measurement-assisted alternative reported as a modeled adjustment (-3T per frozen ledger
  Toffoli; Jones 2013).
* Rotation synthesis: ancilla-free Ross--Selinger z-rotation approximation with leading-order
  T-count ``T_rot(eps) = ceil(3*log2(1/eps))`` at a stated per-rotation accuracy
  ``eps_rot = 1e-10`` (so ``T_rot = 100``); each synthesized rotation is modeled as a serial
  Clifford+T sequence of length ``2*T_rot + 1``.
* Depth: the frozen circuits are measured to be >= 96% serial (depth/gates), so the
  Clifford+T depth estimate is the reduced-basis depth plus the serial expansion of every
  arbitrary rotation.

Everything here is MODELED-tier evidence: no error correction, no surface-code or factory
model, no hardware assumptions.
"""

from __future__ import annotations

import math
from typing import Any

CLIFFORD_T_BASIS = ("h", "s", "sdg", "x", "z", "cx", "t", "tdg", "rz")
EPS_ROTATION = 1.0e-10
ANGLE_TOLERANCE = 1.0e-10
T_PER_TOFFOLI_EXACT = 7
T_PER_TOFFOLI_JONES = 4


def ross_selinger_t_count(eps: float = EPS_ROTATION) -> int:
    """Leading-order Ross-Selinger T-count for one ancilla-free z-rotation at accuracy eps."""

    if not 0.0 < eps < 1.0:
        raise ValueError("eps must lie in (0, 1)")
    return math.ceil(3.0 * math.log2(1.0 / eps))


def rotation_sequence_length(eps: float = EPS_ROTATION) -> int:
    """Modeled serial Clifford+T gate-sequence length of one synthesized rotation."""

    return 2 * ross_selinger_t_count(eps) + 1


def classify_rz_angle(angle: float, tolerance: float = ANGLE_TOLERANCE) -> str:
    """Exact bucket for an rz angle: 'clifford' (k*pi/2), 't_layer' (odd k*pi/4), 'arbitrary'."""

    quarter = float(angle) / (math.pi / 4.0)
    nearest = round(quarter)
    if abs(quarter - nearest) <= tolerance:
        return "clifford" if nearest % 2 == 0 else "t_layer"
    return "arbitrary"


def reduce_to_clifford_t(circuit: Any) -> Any:
    """Transpile a final measured circuit to the declared Clifford+T+Rz basis."""

    from qiskit import transpile

    return transpile(
        circuit, basis_gates=list(CLIFFORD_T_BASIS), optimization_level=0
    )


def clifford_t_inventory(reduced: Any, tolerance: float = ANGLE_TOLERANCE) -> dict[str, Any]:
    """Exact gate inventory of a reduced circuit with per-angle rz classification."""

    counts: dict[str, int] = {}
    rz_clifford = rz_t_layer = rz_arbitrary = 0
    for instruction in reduced.data:
        name = str(instruction.operation.name)
        counts[name] = counts.get(name, 0) + 1
        if name == "rz":
            bucket = classify_rz_angle(float(instruction.operation.params[0]), tolerance)
            if bucket == "clifford":
                rz_clifford += 1
            elif bucket == "t_layer":
                rz_t_layer += 1
            else:
                rz_arbitrary += 1
    unexpected = sorted(
        name
        for name in counts
        if name not in {*CLIFFORD_T_BASIS, "measure", "barrier", "id", "global_phase"}
    )
    if unexpected:
        raise RuntimeError(f"non-Clifford+T basis gates remain after reduction: {unexpected}")
    total_gates = int(sum(v for k, v in counts.items() if k not in {"measure", "barrier"}))
    return {
        "reduced_gate_count": total_gates,
        "reduced_depth": int(reduced.depth()),
        "count_t": int(counts.get("t", 0)),
        "count_tdg": int(counts.get("tdg", 0)),
        "count_rz": int(counts.get("rz", 0)),
        "count_cx": int(counts.get("cx", 0)),
        "count_h": int(counts.get("h", 0)),
        "count_s": int(counts.get("s", 0)) + int(counts.get("sdg", 0)),
        "count_x": int(counts.get("x", 0)),
        "count_z": int(counts.get("z", 0)),
        "count_measure": int(counts.get("measure", 0)),
        "rz_clifford_angle": int(rz_clifford),
        "rz_t_layer_angle": int(rz_t_layer),
        "rz_arbitrary_angle": int(rz_arbitrary),
    }


def ft_estimate_record(
    workload_id: str,
    frozen_row: dict[str, str],
    inventory: dict[str, Any],
    *,
    eps_rotation: float = EPS_ROTATION,
) -> dict[str, Any]:
    """One modeled logical-resource record; arithmetic reproducible from the stored columns."""

    t_rot = ross_selinger_t_count(eps_rotation)
    seq_len = rotation_sequence_length(eps_rotation)
    toffoli_frozen = round(float(frozen_row["toffoli_count"]))
    direct_t = (
        inventory["count_t"] + inventory["count_tdg"] + inventory["rz_t_layer_angle"]
    )
    n_arbitrary = int(inventory["rz_arbitrary_angle"])
    synthesis_t = n_arbitrary * t_rot
    total_t_7 = direct_t + synthesis_t
    total_t_4 = total_t_7 - (T_PER_TOFFOLI_EXACT - T_PER_TOFFOLI_JONES) * toffoli_frozen
    depth_estimate = int(inventory["reduced_depth"]) + n_arbitrary * (seq_len - 1)
    frozen_gates = round(float(frozen_row["transpiled_gate_count"]))
    frozen_depth = round(float(frozen_row["transpiled_depth"]))
    return {
        "workload_id": workload_id,
        "evidence_tier": "modeled_logical_resource_estimate",
        "frozen_transpiled_gate_count": frozen_gates,
        "frozen_transpiled_depth": frozen_depth,
        "frozen_seriality_fraction": frozen_depth / frozen_gates,
        "frozen_toffoli_count": toffoli_frozen,
        "frozen_controlled_rotation_count": round(
            float(frozen_row["controlled_rotation_count"])
        ),
        "qubits": round(float(frozen_row["total_simultaneously_live_qubits"])),
        **inventory,
        "direct_t_count": int(direct_t),
        "expected_t_from_7t_toffoli": T_PER_TOFFOLI_EXACT * toffoli_frozen,
        "arbitrary_rotation_count": n_arbitrary,
        "eps_per_rotation": float(eps_rotation),
        "t_per_rotation_ross_selinger": int(t_rot),
        "rotation_sequence_length_model": int(seq_len),
        "rotation_synthesis_t_count": int(synthesis_t),
        "total_synthesis_error_bound": float(n_arbitrary * eps_rotation),
        "total_logical_t_count_7t_toffoli": int(total_t_7),
        "total_logical_t_count_4t_toffoli_jones": int(total_t_4),
        "serial_clifford_t_depth_estimate": int(depth_estimate),
    }


__all__ = [
    "ANGLE_TOLERANCE",
    "CLIFFORD_T_BASIS",
    "EPS_ROTATION",
    "T_PER_TOFFOLI_EXACT",
    "T_PER_TOFFOLI_JONES",
    "classify_rz_angle",
    "clifford_t_inventory",
    "ft_estimate_record",
    "reduce_to_clifford_t",
    "ross_selinger_t_count",
    "rotation_sequence_length",
]
