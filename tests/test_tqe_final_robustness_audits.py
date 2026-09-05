from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.tqe_final_robustness_audits import (
    ALPHA_COLUMNS,
    FINAL_SUMMARY_COLUMNS,
    NOISE_COLUMNS,
    PHASE_AUDIT_COLUMNS,
    SIGNED_READOUT_COLUMNS,
    TINY_ORACLE_COLUMNS,
    alpha_tradeoff_rows,
    build_measurement_config_for_row_set,
    build_tiny_index_oracle_circuit,
    integrated_qsvt_repeat_case,
    noise_sensitivity_integrated_qsvt,
    run_final_robustness_audits,
    signed_phase_aware_readout_diagnostic,
    tiny_reversible_sparse_oracle_lookup,
    verify_tiny_index_oracle_truth_table,
)
from robust_qsvt_se.qsvt.tqe_sparse_oracle_block_encoding_model import SparseJacobianOracle


def test_phase_synthesis_audit_schema_from_top_level_smoke(tmp_path: Path) -> None:
    run = _run_fast_smoke(tmp_path)
    frame = pd.read_csv(run["artifacts"]["phase_audit_csv"])

    assert set(PHASE_AUDIT_COLUMNS).issubset(frame.columns)


def test_signed_readout_labels_phase_access_requirement() -> None:
    frame = signed_phase_aware_readout_diagnostic(
        {
            "ridge_update": [1.0, -0.25, 0.5],
            "qsvt_update": [0.9, -0.2, 0.45],
            "metadata": {"source": "unit_test"},
        }
    )

    assert set(SIGNED_READOUT_COLUMNS).issubset(frame.columns)
    assert not frame["basis_sampling_accessible"].any()
    assert frame["phase_sign_access_required"].all()
    assert frame["sign_access_model"].str.contains("statevector phase-aware").all()


def test_noise_model_records_proxy_or_aer_status() -> None:
    frame = noise_sensitivity_integrated_qsvt(
        {
            "qsvt_update": [0.8, -0.4, 0.2],
            "ridge_update": [0.8, -0.4, 0.2],
            "metadata": {},
            "integrated_success_probability": 0.5,
            "integrated_raw_depth": 5,
            "integrated_cx_count": 3,
            "noise_models": [{"noise_model": "ideal", "p1": 0.0, "p2": 0.0, "readout_error": 0.0}],
            "noise_shots": [100],
            "seed": 1,
            "use_aer_if_available": False,
        }
    )

    assert set(NOISE_COLUMNS).issubset(frame.columns)
    assert len(frame) == 1
    assert frame.iloc[0]["simulation_status"] in {
        "completed_statevector_distribution_noise_proxy",
        "completed_aer_noise_model",
    }


def test_reactive_pq_row_set_construction() -> None:
    full_without_q = build_measurement_config_for_row_set("full_AC_without_Q_rows")
    branch_q_removed = build_measurement_config_for_row_set("full_AC_without_branch_Q_flow_rows")

    assert full_without_q["include_voltage_magnitudes"] is True
    assert full_without_q["include_p_injections"] is True
    assert full_without_q["include_p_branch_flows"] is True
    assert full_without_q["include_q_injections"] is False
    assert full_without_q["include_q_branch_flows"] is False
    assert branch_q_removed["include_q_injections"] is True
    assert branch_q_removed["include_q_branch_flows"] is False


def test_tiny_oracle_lookup_truth_table_on_small_matrix() -> None:
    oracle = SparseJacobianOracle.from_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
    circuit, layout = build_tiny_index_oracle_circuit(oracle)

    assert verify_tiny_index_oracle_truth_table(oracle, circuit, layout)


def test_tiny_oracle_output_schema() -> None:
    frame = tiny_reversible_sparse_oracle_lookup({"tiny_oracle_matrix": [[1.0, 0.0], [0.0, 2.0]]})

    assert set(TINY_ORACLE_COLUMNS).issubset(frame.columns)
    assert len(frame) == 1


def test_integrated_repeat_case_failure_is_recorded() -> None:
    frame = integrated_qsvt_repeat_case(
        {
            "force_repeat_case_failure": True,
            "repeat_case_spec": {"case_name": "ieee30", "subproblem_size": 4},
            "repeat_alpha": 1.0e-2,
            "repeat_epsilon": 1.0e-2,
            "repeat_degree": 11,
        }
    )

    assert len(frame) == 1
    assert frame.iloc[0]["simulation_status"] == "skipped_forced_failure"
    assert "forced repeat-case failure" in frame.iloc[0]["failure_or_skip_reason"]


def test_alpha_tradeoff_rows_include_rule_choices_and_degrees() -> None:
    H = np.array([[2.0, 0.0], [0.0, 0.5], [1.0, 1.0]], dtype=np.float64)
    r = np.array([1.0, -0.5, 0.25], dtype=np.float64)
    rows = alpha_tradeoff_rows(
        case_name="tiny",
        H=H,
        r=r,
        x_true=np.array([0.4, -0.2], dtype=np.float64),
        alpha_grid=[1.0e-2, 1.0e-3, 1.0e-4],
        epsilon_targets=[1.0e-2],
        degree_grid=[5, 10],
    )
    frame = pd.DataFrame(rows)

    assert set(ALPHA_COLUMNS).issubset(frame.columns)
    assert set(frame["rule_name"]) == {
        "oracle_best_alpha_using_x_true_diagnostic_only",
        "discrete_lcurve_corner_diagnostic",
    }
    assert frame["rule_status"].eq("completed").all()


def test_top_level_output_schema_and_manifest(tmp_path: Path) -> None:
    run = _run_fast_smoke(tmp_path)
    summary = pd.read_csv(run["artifacts"]["final_summary_table"])

    assert set(FINAL_SUMMARY_COLUMNS).issubset(summary.columns)
    assert len(summary) == 7
    assert run["artifacts"]["final_robustness_manifest"].is_file()
    assert run["artifacts"]["manifest"].is_file()


def _run_fast_smoke(tmp_path: Path) -> dict[str, object]:
    return run_final_robustness_audits(
        {
            "output_root": str(tmp_path),
            "phase_audit_settings": [],
            "ridge_update": [0.5, -0.25, 0.125],
            "qsvt_update": [0.49, -0.24, 0.13],
            "metadata": {"source": "unit_test"},
            "noise_models": [{"noise_model": "ideal", "p1": 0.0, "p2": 0.0, "readout_error": 0.0}],
            "noise_shots": [100],
            "use_aer_if_available": False,
            "pq_cases": [],
            "pq_row_sets": [],
            "tiny_oracle_matrix": [[1.0, 0.0], [0.0, 2.0]],
            "force_repeat_case_failure": True,
            "alpha_cases": [],
        }
    )
