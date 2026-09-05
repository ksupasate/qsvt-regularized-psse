from __future__ import annotations

import numpy as np

from robust_qsvt_se.qsvt.power_observable_protocols import (
    AMPLITUDE_ESTIMATION_PROXY,
    FULL_VECTOR_REQUIRED,
    HADAMARD_TEST_PROXY,
    NORM_SCALED_OBSERVABLE,
    PROBABILITY_READOUT,
    SIGNED_OVERLAP,
    SUMMARY_COLUMNS,
    build_observable_protocols,
    evaluate_observable_protocol,
    run_qsvt_power_observable_protocols,
)

_SIGNED_PROTOCOLS = {SIGNED_OVERLAP, HADAMARD_TEST_PROXY}
_PROBABILITY_PROTOCOLS = {PROBABILITY_READOUT, NORM_SCALED_OBSERVABLE, AMPLITUDE_ESTIMATION_PROXY}


def _tiny_metadata() -> dict:
    return {
        "state_index_mapping": [
            {
                "local_index": 0,
                "source_index": 0,
                "label": "theta_4",
                "state_type": "angle",
                "bus_id": 4,
            },
            {
                "local_index": 1,
                "source_index": 1,
                "label": "theta_5",
                "state_type": "angle",
                "bus_id": 5,
            },
            {
                "local_index": 2,
                "source_index": 2,
                "label": "V_4",
                "state_type": "voltage",
                "bus_id": 4,
            },
            {
                "local_index": 3,
                "source_index": 3,
                "label": "V_5",
                "state_type": "voltage",
                "bus_id": 5,
            },
        ]
    }


def _tiny_problem() -> tuple[np.ndarray, np.ndarray]:
    H = np.array(
        [
            [1.0, 0.2, 0.1, 0.0],
            [0.2, 1.0, 0.0, 0.1],
            [0.1, 0.0, 1.0, 0.2],
            [0.0, 0.1, 0.2, 1.0],
        ],
        dtype=np.float64,
    )
    ridge = np.array([0.5, -0.3, 0.05, 0.02], dtype=np.float64)
    return H, ridge


def _by_name(protocols: list) -> dict[str, object]:
    return {protocol.observable_name: protocol for protocol in protocols}


def test_protocol_classification_flags() -> None:
    H, ridge = _tiny_problem()
    protocols = build_observable_protocols(
        H_tilde=H, ridge_update=ridge, metadata=_tiny_metadata(), topk=2
    )
    by_name = _by_name(protocols)

    # Signed linear functional: a bus-angle component update.
    angle = by_name["bus_4_angle_update"]
    assert angle.protocol_type in _SIGNED_PROTOCOLS
    assert angle.requires_signed_overlap is True
    assert angle.requires_norm_recovery is True
    assert angle.requires_full_vector_readout is False

    # Energy observable: norm-scaled, no signed overlap.
    energy = by_name["selected_subset_update_energy"]
    assert energy.protocol_type in _PROBABILITY_PROTOCOLS
    assert energy.requires_signed_overlap is False
    assert energy.requires_norm_recovery is True
    assert energy.requires_full_vector_readout is False

    # Branch angle-difference proxy is a signed difference functional.
    branch = by_name["branch_angle_difference_proxy"]
    assert branch.protocol_type in _SIGNED_PROTOCOLS
    assert branch.requires_signed_overlap is True

    # Measurement-row correction proxy: signed functional o = row of H.
    row = by_name["measurement_row_0_correction_proxy"]
    assert row.protocol_type in _SIGNED_PROTOCOLS
    assert row.requires_signed_overlap is True

    # Top-k identification: probability readout, no norm recovery or signed overlap.
    topk = by_name["topk_update_component_identification"]
    assert topk.protocol_type == PROBABILITY_READOUT
    assert topk.requires_norm_recovery is False
    assert topk.requires_signed_overlap is False
    assert topk.requires_full_vector_readout is False

    # Full-vector reconstruction pseudo-observable: should not be emphasized.
    full = by_name["full_state_vector_reconstruction"]
    assert full.protocol_type == FULL_VECTOR_REQUIRED
    assert full.requires_full_vector_readout is True


def test_topk_identifies_largest_components() -> None:
    H, ridge = _tiny_problem()
    protocols = build_observable_protocols(
        H_tilde=H, ridge_update=ridge, metadata=_tiny_metadata(), topk=2
    )
    topk = _by_name(protocols)["topk_update_component_identification"]
    # Largest |ridge| entries are indices 0 and 1.
    assert topk.state_indices == (0, 1)

    # Identical QSVT update reproduces ridge top-k => Jaccard 1.0.
    evaluation = evaluate_observable_protocol(topk, ridge_update=ridge, qsvt_update=ridge.copy())
    assert evaluation.ridge_value == 1.0
    assert evaluation.qsvt_value_scaled == 1.0


def test_signed_overlap_value_matches_dot_product() -> None:
    H, ridge = _tiny_problem()
    protocols = build_observable_protocols(
        H_tilde=H, ridge_update=ridge, metadata=_tiny_metadata(), topk=2
    )
    angle = _by_name(protocols)["bus_4_angle_update"]
    qsvt = ridge * 0.9
    evaluation = evaluate_observable_protocol(angle, ridge_update=ridge, qsvt_update=qsvt)
    # Component observable on index 0 -> the angle update entry itself.
    assert np.isclose(evaluation.ridge_value, ridge[0])
    assert np.isclose(evaluation.qsvt_value_scaled, qsvt[0])
    assert np.isclose(evaluation.absolute_error_scaled, abs(qsvt[0] - ridge[0]))


def test_run_writes_outputs_and_costs(tmp_path) -> None:
    output_dir = tmp_path / "protocols"
    run = run_qsvt_power_observable_protocols(
        {
            "case": "ieee14",
            "model": "ac_linearized",
            "submatrix_size": 4,
            "alpha": 1.0e-4,
            "degree": 9,
            "target_tolerances": [1.0e-1, 1.0e-2],
            "seed": 123,
            "output_dir": str(output_dir),
        }
    )
    summary_path = run["artifacts"]["observable_protocol_summary"]
    tradeoff_path = run["artifacts"]["observable_accuracy_cost_tradeoff"]
    interpretation_path = run["artifacts"]["observable_protocol_interpretation"]
    assert summary_path.exists()
    assert tradeoff_path.exists()
    assert interpretation_path.exists()

    import pandas as pd

    frame = pd.read_csv(summary_path)
    assert list(frame.columns) == SUMMARY_COLUMNS
    assert not frame.empty
    # O(1/eps^2) shots > O(1/eps) queries: at eps=1e-2 shots=2500 >> queries=100.
    tight = frame[np.isclose(frame["target_tolerance"], 1.0e-2)].iloc[0]
    assert int(tight["shot_cost_proxy"]) > int(tight["query_cost_proxy"])

    tradeoff = pd.read_csv(tradeoff_path)
    tight_tradeoff = tradeoff[np.isclose(tradeoff["target_tolerance"], 1.0e-2)]
    assert (tight_tradeoff["dominant_cost"] == "readout_shots").all()

    # A full-vector pseudo-observable must be present and flagged.
    full = frame[frame["observable_name"] == "full_state_vector_reconstruction"]
    assert bool(full["requires_full_vector_readout"].iloc[0]) is True
    assert full["protocol_type"].iloc[0] == FULL_VECTOR_REQUIRED
