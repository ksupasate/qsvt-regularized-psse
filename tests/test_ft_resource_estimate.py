"""FT logical resource estimate - conventions, arithmetic, and produced-row consistency."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from robust_qsvt_se.qsvt.ft_resource_estimate import (
    EPS_ROTATION,
    T_PER_TOFFOLI_EXACT,
    T_PER_TOFFOLI_JONES,
    classify_rz_angle,
    ft_estimate_record,
    ross_selinger_t_count,
    rotation_sequence_length,
)

ROWS_PATH = Path(__file__).resolve().parents[1] / (
    "outputs/ft_logical_resource_estimate/ft_logical_resource_rows.csv"
)


def test_ross_selinger_leading_order_formula():
    assert ross_selinger_t_count(1.0e-10) == 100
    assert ross_selinger_t_count(1.0e-2) == math.ceil(3.0 * math.log2(1.0e2))
    assert rotation_sequence_length(1.0e-10) == 201
    with pytest.raises(ValueError):
        ross_selinger_t_count(0.0)


def test_rz_angle_classification_is_exact():
    assert classify_rz_angle(0.0) == "clifford"
    assert classify_rz_angle(math.pi / 2) == "clifford"
    assert classify_rz_angle(-3.0 * math.pi) == "clifford"
    assert classify_rz_angle(math.pi / 4) == "t_layer"
    assert classify_rz_angle(-math.pi / 4) == "t_layer"
    assert classify_rz_angle(3.0 * math.pi / 4) == "t_layer"
    assert classify_rz_angle(0.1234) == "arbitrary"
    assert classify_rz_angle(math.pi / 4 + 1.0e-6) == "arbitrary"


def _synthetic_inventory() -> dict:
    return {
        "reduced_gate_count": 1000,
        "reduced_depth": 900,
        "count_t": 70,
        "count_tdg": 30,
        "count_rz": 60,
        "count_cx": 500,
        "count_h": 200,
        "count_s": 100,
        "count_x": 40,
        "count_z": 0,
        "count_measure": 2,
        "rz_clifford_angle": 10,
        "rz_t_layer_angle": 20,
        "rz_arbitrary_angle": 30,
    }


def test_ft_estimate_record_arithmetic_reproducible():
    frozen = {
        "toffoli_count": "10",
        "transpiled_gate_count": "1000",
        "transpiled_depth": "970",
        "controlled_rotation_count": "5",
        "total_simultaneously_live_qubits": "7",
    }
    record = ft_estimate_record("w", frozen, _synthetic_inventory())
    t_rot = ross_selinger_t_count(EPS_ROTATION)
    assert record["direct_t_count"] == 70 + 30 + 20
    assert record["rotation_synthesis_t_count"] == 30 * t_rot
    assert record["total_logical_t_count_7t_toffoli"] == 120 + 30 * t_rot
    assert record["total_logical_t_count_4t_toffoli_jones"] == (
        record["total_logical_t_count_7t_toffoli"]
        - (T_PER_TOFFOLI_EXACT - T_PER_TOFFOLI_JONES) * 10
    )
    assert record["serial_clifford_t_depth_estimate"] == 900 + 30 * (2 * t_rot + 1 - 1)
    assert record["expected_t_from_7t_toffoli"] == 70
    assert record["total_synthesis_error_bound"] == pytest.approx(30 * EPS_ROTATION)
    assert record["evidence_tier"] == "modeled_logical_resource_estimate"


@pytest.mark.skipif(not ROWS_PATH.is_file(), reason="estimate not yet generated")
def test_generated_rows_are_internally_consistent():
    rows = pd.read_csv(ROWS_PATH)
    assert len(rows) == 4
    t_rot = ross_selinger_t_count(EPS_ROTATION)
    for row in rows.itertuples():
        assert row.direct_t_count == row.count_t + row.count_tdg + row.rz_t_layer_angle
        assert row.rotation_synthesis_t_count == row.arbitrary_rotation_count * t_rot
        assert row.total_logical_t_count_7t_toffoli == (
            row.direct_t_count + row.rotation_synthesis_t_count
        )
        assert row.total_logical_t_count_4t_toffoli_jones == (
            row.total_logical_t_count_7t_toffoli - 3 * row.frozen_toffoli_count
        )
        # 7T-per-Toffoli content is a subset of the measured direct T gates.
        assert row.expected_t_from_7t_toffoli == 7 * row.frozen_toffoli_count
        assert row.direct_t_count >= row.expected_t_from_7t_toffoli
        assert row.count_rz == (
            row.rz_clifford_angle + row.rz_t_layer_angle + row.rz_arbitrary_angle
        )


@pytest.mark.skipif(not ROWS_PATH.is_file(), reason="estimate not yet generated")
def test_generated_rows_match_frozen_ledger_identities():
    rows = pd.read_csv(ROWS_PATH).set_index("workload_id")
    expected = {
        "ieee14_sparse_quantized_4x4_d31_scaling_anchor_v1": (60587, 13889, 7),
        "ieee14_sparse_quantized_8x8_d31_selected_v1": (186191, 51898, 8),
        "ieee30_sparse_quantized_8x8_d31_selected_v1": (186006, 51898, 8),
        "ieee14_sparse_quantized_16x16_d31_scaling_v1": (478753, 145773, 9),
    }
    for workload_id, (gates, toffolis, qubits) in expected.items():
        row = rows.loc[workload_id]
        assert int(row["frozen_transpiled_gate_count"]) == gates
        assert int(row["frozen_toffoli_count"]) == toffolis
        assert int(row["qubits"]) == qubits
