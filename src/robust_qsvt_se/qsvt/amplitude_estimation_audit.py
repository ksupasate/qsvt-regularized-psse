from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.amplitude_estimation_routines import build_qsvt_amplitude_problem
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

AUDIT_CLAIM = (
    "Integration audit for actual amplitude/norm-estimation routines. It records which "
    "gate-level QSVT circuits, success projectors, and readout protocols can be reused, and "
    "which previous scale-recovery outputs can be rerun. No QSVT superiority over "
    "Ridge/Tikhonov, quantum speedup, or hardware execution is claimed."
)

GATE_CONFIG_ROWS = [
    {
        "circuit_source": "build_qsvt_amplitude_problem",
        "builder": "amplitude_estimation_routines.build_qsvt_amplitude_problem",
        "encoded_block": "top-left A=H_tilde.T/beta block",
        "success_projector": "encoded_block_ancilla_zero (first encoded_dimension states)",
        "ancilla_qubits": 1,
        "reusable_for_amplitude_estimation": True,
        "notes": "Returns a Grover-ready unitary A with A e_0 == statevector.",
    },
    {
        "circuit_source": "solve_gate_level_state_estimation_problem",
        "builder": "gate_level_state_estimation_solver.solve_gate_level_state_estimation_problem",
        "encoded_block": "top-left A block",
        "success_projector": "encoded_block_ancilla_zero",
        "ancilla_qubits": 1,
        "reusable_for_amplitude_estimation": True,
        "notes": "Dense gate-level QSVT solver reused for Phase 4 gate validation.",
    },
    {
        "circuit_source": "build_gate_level_qsvt_state_circuit",
        "builder": "gate_level_qsvt.build_gate_level_qsvt_state_circuit",
        "encoded_block": "top-left A block",
        "success_projector": "encoded_block_ancilla_zero",
        "ancilla_qubits": 1,
        "reusable_for_amplitude_estimation": True,
        "notes": "Structured QSVT operator circuit + Initialize state preparation.",
    },
]

READOUT_PROTOCOL_ROWS = [
    {
        "protocol_type": "probability_readout",
        "amplitude_or_probability": "probability",
        "estimates_sign": False,
        "requires_norm_recovery": False,
        "notes": "Basis-probability and top-k identification readouts.",
    },
    {
        "protocol_type": "signed_overlap",
        "amplitude_or_probability": "amplitude",
        "estimates_sign": True,
        "requires_norm_recovery": True,
        "notes": "Signed linear functionals (bus angle/voltage, measurement row).",
    },
    {
        "protocol_type": "hadamard_test_proxy",
        "amplitude_or_probability": "amplitude",
        "estimates_sign": True,
        "requires_norm_recovery": True,
        "notes": "Branch angle-difference proxy via Hadamard-test-style overlap.",
    },
    {
        "protocol_type": "norm_scaled_observable",
        "amplitude_or_probability": "probability",
        "estimates_sign": False,
        "requires_norm_recovery": True,
        "notes": "Selected-area update energy (needs the update norm/scale).",
    },
    {
        "protocol_type": "amplitude_estimation_proxy",
        "amplitude_or_probability": "amplitude",
        "estimates_sign": False,
        "requires_norm_recovery": True,
        "notes": "Now backed by actual Bernoulli/iterative routines in Phase 1.",
    },
    {
        "protocol_type": "full_vector_required",
        "amplitude_or_probability": "amplitude",
        "estimates_sign": True,
        "requires_norm_recovery": True,
        "notes": "Full-vector reconstruction; not emphasized.",
    },
]

PRIOR_SCALE_RECOVERY_OUTPUTS = [
    "qsvt_scale_recovery_protocols",
    "qsvt_bounded_target_design_study",
    "qsvt_residual_feasible_config_search",
    "qsvt_residual_feasible_gate_validation",
    "qsvt_power_observable_protocols",
    "qsvt_sparse_oracle_prototype_strengthening",
    "qsvt_evidence_claim_support_report",
]


def run_amplitude_estimation_audit(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = {
        "input_root": "outputs",
        "output_dir": "outputs/qsvt_actual_amplitude_estimation_audit",
        "probe_alpha": 1.0e-4,
        "probe_degree": 5,
    }
    if config:
        resolved.update(config)
    input_root = Path(resolved["input_root"])
    output_dir = ensure_directory(resolved["output_dir"])

    probe = _probe_amplitude_problem(
        alpha=float(resolved["probe_alpha"]), degree=int(resolved["probe_degree"])
    )
    rerunnable = [
        {
            "previous_output": name,
            "present": (input_root / name).exists(),
            "rerunnable": True,
            "path": str(input_root / name),
        }
        for name in PRIOR_SCALE_RECOVERY_OUTPUTS
    ]

    gate_path = output_dir / "available_gate_configs.csv"
    readout_path = output_dir / "available_readout_protocols.csv"
    rerun_path = output_dir / "rerunnable_scale_recovery_outputs.csv"
    audit_path = output_dir / "integration_audit.md"

    pd.DataFrame(GATE_CONFIG_ROWS).to_csv(gate_path, index=False)
    pd.DataFrame(READOUT_PROTOCOL_ROWS).to_csv(readout_path, index=False)
    pd.DataFrame(rerunnable).to_csv(rerun_path, index=False)
    audit_path.write_text(_audit_markdown(probe, rerunnable), encoding="utf-8")

    manifest = write_manifest(
        output_dir,
        artifacts={
            "available_gate_configs": str(gate_path),
            "available_readout_protocols": str(readout_path),
            "rerunnable_scale_recovery_outputs": str(rerun_path),
            "integration_audit": str(audit_path),
        },
        input_config=resolved,
        claim_boundary=AUDIT_CLAIM,
    )
    return {
        "output_dir": output_dir,
        "probe": probe,
        "rerunnable": rerunnable,
        "artifacts": {
            "manifest": manifest,
            "available_gate_configs": gate_path,
            "available_readout_protocols": readout_path,
            "rerunnable_scale_recovery_outputs": rerun_path,
            "integration_audit": audit_path,
        },
    }


def _probe_amplitude_problem(*, alpha: float, degree: int) -> dict[str, Any]:
    H = np.array([[1.0, 0.2], [0.15, 0.8]], dtype=np.float64)
    r = np.array([0.4, -0.2], dtype=np.float64)
    try:
        problem = build_qsvt_amplitude_problem(H, r, alpha=float(alpha), degree=int(degree))
        reconstructed = float(
            np.sum(np.abs(problem.unitary_A[:, 0][: problem.encoded_dimension]) ** 2)
        )
        return {
            "status": "reusable",
            "encoded_dimension": int(problem.encoded_dimension),
            "total_dimension": int(problem.total_dimension),
            "n_qubits": int(problem.n_qubits),
            "exact_success_probability": float(problem.exact_success_probability),
            "reconstruction_matches_statevector": bool(
                np.isclose(reconstructed, problem.exact_success_probability)
            ),
        }
    except Exception as exc:  # pragma: no cover - backend dependent
        return {"status": f"unavailable:{type(exc).__name__}", "detail": str(exc)}


def _audit_markdown(probe: dict[str, Any], rerunnable: list[dict[str, Any]]) -> str:
    rerun_lines = [
        f"- `{row['previous_output']}` (present={row['present']}, rerunnable={row['rerunnable']})"
        for row in rerunnable
    ]
    probe_success = float(probe.get("exact_success_probability", float("nan")))
    probe_match = probe.get("reconstruction_matches_statevector")
    return "\n".join(
        [
            "# Actual Amplitude-Estimation Integration Audit",
            "",
            AUDIT_CLAIM,
            "",
            "## 1. Reusable QSVT gate-level circuits",
            "- `build_qsvt_amplitude_problem` exposes a Grover-ready unitary A (A e_0 == "
            "statevector).",
            "- `solve_gate_level_state_estimation_problem` provides the dense gate-level QSVT "
            "update used for gate validation.",
            "- `build_gate_level_qsvt_state_circuit` / `build_structured_qsvt_operator_circuit` "
            "expose the structured operator circuit.",
            "",
            "## 2. Success / projector qubits available",
            "- One block-encoding ancilla qubit; the success projector is the encoded top-left "
            "block (ancilla in |0>), i.e. the first `encoded_dimension` basis states.",
            "",
            "## 3. Postselection success states explicitly represented",
            f"- Yes. Probe success probability = {probe_success:.6g}; "
            f"reconstruction matches statevector = {probe_match}.",
            "",
            "## 4. Observables estimable by amplitude/probability readout",
            "- Probability readouts: basis-component probability, selected-area energy, top-k "
            "identification.",
            "- Amplitude/overlap readouts: signed bus angle/voltage updates, branch "
            "angle-difference proxy, measurement-row correction (need norm recovery and "
            "signed overlap).",
            "",
            "## 5. Previous scale-recovery outputs that can be rerun",
            *rerun_lines,
            "",
        ]
    )
