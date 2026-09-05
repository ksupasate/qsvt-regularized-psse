"""Small circuit-derived selected signed-functional demonstration.

The dense 4x4 IEEE-14 selected block is intentionally non-scalable. Its QSVT
transform and phase sequence are simulated, then one predetermined local state
functional is evaluated and passed through the existing synthetic shot-noise
readout model. The readout circuit itself is not synthesized.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.paper.signed_readout_diagnostic import hadamard_test_estimate
from robust_qsvt_se.qsvt.tqe_degree_alpha_precision_sweep import load_sweep_subproblem
from robust_qsvt_se.qsvt.tqe_end_to_end_qsvt_vs_ridge import ridge_update_svd
from robust_qsvt_se.qsvt.tqe_integrated_small_qsvt_circuit import (
    _resolve_config,
    qsvt_rescaled_update_from_transform,
    run_ieee_selected_block,
    run_sanity_check,
)
from robust_qsvt_se.utils.io import ensure_directory

DEMO_COLUMNS = [
    "case",
    "subproblem_size",
    "selection_policy",
    "selected_before_solving",
    "depends_on_solution",
    "observable_id",
    "physical_meaning",
    "alpha",
    "degree",
    "phase_synthesis_status",
    "qsvt_sequence_status",
    "block_encoding_beta",
    "bounded_scale_C",
    "manuscript_scaling_C_over_beta",
    "circuit_physical_recovery_factor",
    "residual_norm",
    "postselection_success_probability",
    "postselection_normalization",
    "output_state_norm",
    "normalized_state_observable",
    "ridge_physical_observable",
    "qsvt_circuit_physical_observable",
    "shot_estimate_physical_observable",
    "shots",
    "absolute_circuit_vs_ridge_error",
    "relative_circuit_vs_ridge_error",
    "absolute_shot_vs_ridge_error",
    "readout_implementation_status",
    "full_vector_recovery_included",
    "claim_boundary",
]


def run_integrated_small_selected_readout(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    options = dict(config or {})
    output_dir = ensure_directory(
        Path(options.pop("output_dir", "outputs/qsvt_selected_observable_workload"))
    )
    shots = int(options.pop("shots", 100_000))
    seed = int(options.pop("readout_seed", 20240629))
    resolved = _resolve_config(options)
    sanity = run_sanity_check(resolved)
    if sanity.row["qsvt_sequence_status"] != "sanity_passed":
        raise RuntimeError("QSVT phase-convention sanity check did not pass")
    evaluation = run_ieee_selected_block(resolved)
    if evaluation.transformed_block is None or evaluation.row["simulation_status"] != "completed":
        raise RuntimeError(str(evaluation.row["failure_or_skip_reason"]))

    subproblem = load_sweep_subproblem(
        dict(resolved["subproblem_spec"]), seed=int(resolved["seed"])
    )
    A = np.asarray(subproblem.H_tilde, dtype=np.float64)
    residual = np.asarray(subproblem.r_tilde, dtype=np.float64)
    transform = np.asarray(evaluation.transformed_block, dtype=np.float64)[
        : A.shape[0], : A.shape[1]
    ]
    ridge_update = ridge_update_svd(A, residual, alpha=float(resolved["alpha"]))
    qsvt_update = qsvt_rescaled_update_from_transform(
        transform, residual, C_alpha=float(evaluation.row["C_alpha"])
    )

    # Predetermined before solving: first local coordinate from the stored mapping.
    mapping = list(subproblem.metadata.get("state_index_mapping", []))
    first = mapping[0] if mapping else {"label": "local_state_0", "description": "local state 0"}
    functional = np.zeros(A.shape[1], dtype=np.float64)
    functional[0] = 1.0
    ridge_value = float(functional @ ridge_update)
    qsvt_value = float(functional @ qsvt_update)
    _true, shot_value, _mu = hadamard_test_estimate(
        functional,
        qsvt_update,
        shots=shots,
        rng=np.random.default_rng(seed),
    )
    output_norm = float(np.linalg.norm(qsvt_update))
    success = float(evaluation.row["success_probability_residual_state"])
    beta = float(evaluation.row["gamma"])
    bounded_c = float(evaluation.row["C_alpha"])
    row = {
        "case": evaluation.row["case_name"],
        "subproblem_size": int(evaluation.row["subproblem_size"]),
        "selection_policy": "first_local_state_from_saved_subproblem_mapping",
        "selected_before_solving": True,
        "depends_on_solution": False,
        "observable_id": str(first.get("label", "local_state_0")),
        "physical_meaning": str(first.get("description", "first local state correction")),
        "alpha": float(resolved["alpha"]),
        "degree": int(evaluation.row["degree"]),
        "phase_synthesis_status": evaluation.row["phase_synthesis_status"],
        "qsvt_sequence_status": evaluation.row["qsvt_sequence_status"],
        "block_encoding_beta": beta,
        "bounded_scale_C": bounded_c,
        "manuscript_scaling_C_over_beta": bounded_c / beta,
        "circuit_physical_recovery_factor": bounded_c,
        "residual_norm": float(np.linalg.norm(residual)),
        "postselection_success_probability": success,
        "postselection_normalization": float(np.sqrt(success)),
        "output_state_norm": output_norm,
        "normalized_state_observable": qsvt_value / output_norm,
        "ridge_physical_observable": ridge_value,
        "qsvt_circuit_physical_observable": qsvt_value,
        "shot_estimate_physical_observable": float(shot_value),
        "shots": shots,
        "absolute_circuit_vs_ridge_error": abs(qsvt_value - ridge_value),
        "relative_circuit_vs_ridge_error": abs(qsvt_value - ridge_value)
        / max(abs(ridge_value), 1.0e-15),
        "absolute_shot_vs_ridge_error": abs(float(shot_value) - ridge_value),
        "readout_implementation_status": (
            "synthetic_sign_aware_shot_model_applied_to_circuit_derived_update"
        ),
        "full_vector_recovery_included": False,
        "claim_boundary": (
            "4x4 dense simulator demonstration; signed readout is a synthetic shot-noise "
            "model, not a compiled readout circuit; not IEEE-scale hardware execution"
        ),
    }
    frame = pd.DataFrame([row], columns=DEMO_COLUMNS)
    output_path = output_dir / "integrated_small_qsvt_readout_demo.csv"
    frame.to_csv(output_path, index=False)
    return {"demo": frame, "artifact": output_path}


if __name__ == "__main__":  # pragma: no cover
    run_integrated_small_selected_readout()
