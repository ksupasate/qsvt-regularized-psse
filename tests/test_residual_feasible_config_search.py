from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from robust_qsvt_se.qsvt import residual_feasible_config_search as rfcs
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import SelectedSubproblem


def test_feasibility_never_uses_gate_solver_source():
    """The feasibility decision must not reference any gate-level QSVT solver."""
    source = inspect.getsource(rfcs.decide_residual_feasibility)
    forbidden = (
        "solve_gate_level_state_estimation_problem",
        "gate_level",
        "GateLevel",
        "qsvt_update",
        "synthesized_degree",
        "phase_count",
        "shots",
    )
    for token in forbidden:
        assert token not in source, f"feasibility must not reference gate-level token {token!r}"

    signature = inspect.signature(rfcs.decide_residual_feasibility)
    assert set(signature.parameters) == {
        "scale_protocol",
        "residual_ratio_vs_no_update",
        "direction_error_vs_ridge",
    }


def test_module_does_not_import_gate_solver():
    """The whole module must avoid the gate-level solver entirely (poly/matrix-free)."""
    source = inspect.getsource(rfcs)
    # The gate-level circuit solver must never be imported or called in this phase.
    assert "solve_gate_level_state_estimation_problem" not in source
    assert "GateLevelSolverComputation" not in source
    # Only the safe value symbols may be pulled from the gate-level module.
    assert not hasattr(rfcs, "solve_gate_level_state_estimation_problem")


def test_best_scalar_diagnostic_never_feasible():
    """A best_scalar_diagnostic row can never be residual_feasible even with perfect metrics."""
    verdict = rfcs.decide_residual_feasibility(
        scale_protocol="best_scalar_diagnostic",
        residual_ratio_vs_no_update=0.0,
        direction_error_vs_ridge=0.0,
    )
    assert verdict.residual_feasible is False
    assert verdict.rejection_reason == "diagnostic_only_protocol_not_deployable"


def test_known_c_feasible_when_within_thresholds():
    verdict = rfcs.decide_residual_feasibility(
        scale_protocol="known_C",
        residual_ratio_vs_no_update=0.05,
        direction_error_vs_ridge=0.05,
    )
    assert verdict.residual_feasible is True
    assert verdict.rejection_reason == ""


def test_boundary_values_are_feasible():
    verdict = rfcs.decide_residual_feasibility(
        scale_protocol="success_amplitude_proxy",
        residual_ratio_vs_no_update=rfcs.RESIDUAL_RATIO_FEASIBLE_MAX,
        direction_error_vs_ridge=rfcs.DIRECTION_ERROR_FEASIBLE_MAX,
    )
    assert verdict.residual_feasible is True


def test_residual_ratio_above_threshold_rejected():
    verdict = rfcs.decide_residual_feasibility(
        scale_protocol="known_C",
        residual_ratio_vs_no_update=0.2,
        direction_error_vs_ridge=0.01,
    )
    assert verdict.residual_feasible is False
    assert "residual_ratio_vs_no_update" in verdict.rejection_reason


def test_direction_error_above_threshold_rejected():
    verdict = rfcs.decide_residual_feasibility(
        scale_protocol="amplitude_estimation_proxy",
        residual_ratio_vs_no_update=0.01,
        direction_error_vs_ridge=0.5,
    )
    assert verdict.residual_feasible is False
    assert "direction_error_vs_ridge" in verdict.rejection_reason


def test_non_finite_metrics_rejected():
    verdict = rfcs.decide_residual_feasibility(
        scale_protocol="known_C",
        residual_ratio_vs_no_update=float("nan"),
        direction_error_vs_ridge=0.01,
    )
    assert verdict.residual_feasible is False
    assert verdict.rejection_reason == "non_finite_residual_ratio"


def test_gate_validation_recommendation_only_for_feasible_manageable_stable():
    rows = [
        {
            "subproblem_id": "a",
            "target_design": "current_global",
            "scale_protocol": "known_C",
            "degree": 51,
            "condition_number": 10.0,
            "residual_ratio_vs_no_update": 0.01,
            "direction_error_vs_ridge": 0.01,
            "residual_feasible": True,
            "gate_validation_recommended": False,
        },
        {
            "subproblem_id": "b",
            "target_design": "current_global",
            "scale_protocol": "known_C",
            "degree": 35,
            "condition_number": 1.0e9,  # unstable -> excluded
            "residual_ratio_vs_no_update": 0.005,
            "direction_error_vs_ridge": 0.005,
            "residual_feasible": True,
            "gate_validation_recommended": False,
        },
        {
            "subproblem_id": "c",
            "target_design": "current_global",
            "scale_protocol": "known_C",
            "degree": 201,  # too large -> excluded
            "condition_number": 5.0,
            "residual_ratio_vs_no_update": 0.001,
            "direction_error_vs_ridge": 0.001,
            "residual_feasible": True,
            "gate_validation_recommended": False,
        },
        {
            "subproblem_id": "d",
            "target_design": "current_global",
            "scale_protocol": "known_C",
            "degree": 35,
            "condition_number": 5.0,
            "residual_ratio_vs_no_update": 0.5,  # not feasible
            "direction_error_vs_ridge": 0.5,
            "residual_feasible": False,
            "gate_validation_recommended": False,
        },
    ]
    marked = rfcs._mark_gate_validation_recommended(rows)
    recommended = {row["subproblem_id"] for row in marked if row["gate_validation_recommended"]}
    assert recommended == {"a"}


def test_gate_validation_recommendation_caps_at_three():
    rows = [
        {
            "subproblem_id": f"s{index}",
            "target_design": "current_global",
            "scale_protocol": "known_C",
            "degree": 51,
            "condition_number": 10.0,
            "residual_ratio_vs_no_update": 0.01 + 0.001 * index,
            "direction_error_vs_ridge": 0.01,
            "residual_feasible": True,
            "gate_validation_recommended": False,
        }
        for index in range(6)
    ]
    marked = rfcs._mark_gate_validation_recommended(rows)
    assert sum(1 for row in marked if row["gate_validation_recommended"]) == 3
    # Lowest residual ratios should win.
    recommended = sorted(
        row["subproblem_id"] for row in marked if row["gate_validation_recommended"]
    )
    assert recommended == ["s0", "s1", "s2"]


def _synthetic_subproblem(seed: int = 0) -> SelectedSubproblem:
    rng = np.random.default_rng(seed)
    base = np.diag([1.0, 0.6, 0.3, 0.12]).astype(np.float64)
    H = base + 0.02 * rng.standard_normal((4, 4))
    r = np.asarray([1.0, 0.5, 0.25, 0.1], dtype=np.float64)
    metadata = {
        "candidate_id": "synthetic_01",
        "selection_mode": "high_leverage",
        "selected_measurement_indices": [0, 1, 2, 3],
        "selected_state_indices": [0, 1, 2, 3],
    }
    return SelectedSubproblem(H_tilde=H, r_tilde=r, metadata=metadata)


def test_evaluate_config_point_produces_required_columns_and_protocols():
    subproblem = _synthetic_subproblem()
    rows = rfcs.evaluate_config_point(
        subproblem=subproblem,
        case="custom",
        model="weighted_linearized",
        alpha=1.0e-3,
        degree=35,
        target_designs=["current_global", "support_scaled"],
        scale_protocols=["known_C", "best_scalar_diagnostic"],
        grid_size=512,
    )
    assert len(rows) == 4
    for row in rows:
        assert set(rfcs.RESULT_COLUMNS).issubset(row.keys())
        assert row["selection_mode"] == "high_leverage"
        assert row["subproblem_id"] == "synthetic_01"
        assert int(row["qsvt_query_count"]) == 2 * 35 + 1
    # best_scalar_diagnostic rows must always be infeasible.
    for row in rows:
        if row["scale_protocol"] == "best_scalar_diagnostic":
            assert row["residual_feasible"] is False
            assert row["rejection_reason"] == "diagnostic_only_protocol_not_deployable"


def test_known_c_residual_independent_of_scaling_only_design():
    """Scaling-only designs share the known-C-rescaled residual (per the phase math fact)."""
    subproblem = _synthetic_subproblem()
    rows = rfcs.evaluate_config_point(
        subproblem=subproblem,
        case="custom",
        model="weighted_linearized",
        alpha=1.0e-3,
        degree=35,
        target_designs=["current_global", "support_scaled", "margin_1p05"],
        scale_protocols=["known_C"],
        grid_size=512,
    )
    residuals = [row["residual"] for row in rows]
    assert np.allclose(residuals, residuals[0], rtol=1e-9, atol=1e-12)


def test_write_outputs_creates_required_files(tmp_path):
    subproblem = _synthetic_subproblem()
    rows = rfcs.evaluate_config_point(
        subproblem=subproblem,
        case="custom",
        model="weighted_linearized",
        alpha=1.0e-3,
        degree=35,
        target_designs=["current_global"],
        scale_protocols=["known_C", "best_scalar_diagnostic"],
        grid_size=512,
    )
    artifacts = rfcs.write_config_search_outputs(tmp_path, {"seed": 0}, rows)
    for key in (
        "all_config_results",
        "residual_feasible_configs",
        "failed_or_rejected_configs",
        "config_search_interpretation",
        "manifest",
    ):
        assert artifacts[key].exists()
    all_frame = pd.read_csv(artifacts["all_config_results"])
    assert list(all_frame.columns) == rfcs.RESULT_COLUMNS
    # best_scalar_diagnostic must appear in the rejected file, never the feasible file.
    rejected = pd.read_csv(artifacts["failed_or_rejected_configs"])
    assert (rejected["scale_protocol"] == "best_scalar_diagnostic").any()


def test_select_policy_subproblems_uses_policy(monkeypatch):
    """Subproblem selection must come from the policy scoring, not QSVT residuals."""
    calls: dict[str, int] = {"generate": 0, "score": 0, "build": 0}
    fake_system = object()
    fake_subproblem = _synthetic_subproblem()

    def fake_build_system(*, case, model, case_source, seed):
        return fake_system, "fake_matrix_source"

    def fake_generate(**kwargs):
        calls["generate"] += 1
        assert kwargs["submatrix_size"] == 4
        return ["candidate"]

    def fake_score(**kwargs):
        calls["score"] += 1
        return [
            {
                "candidate_id": "high_leverage_01",
                "selected": True,
                "selection_source": "high_leverage",
            },
            {
                "candidate_id": "control",
                "selected": False,
                "selection_source": "worst_conditioned_control",
            },
        ]

    def fake_build(*, system, matrix_source, row):
        calls["build"] += 1
        assert row["selected"] is True
        return fake_subproblem

    monkeypatch.setattr(rfcs, "_build_system", fake_build_system)
    monkeypatch.setattr(rfcs, "generate_candidate_subproblems", fake_generate)
    monkeypatch.setattr(rfcs, "score_candidate_subproblems", fake_score)
    monkeypatch.setattr(rfcs, "build_selected_subproblem_from_policy_row", fake_build)

    selected = rfcs.select_policy_subproblems(
        case="ieee14",
        model="ac_linearized",
        case_source="pypower",
        submatrix_size=4,
        condition_threshold=1.0e8,
        policy_alpha=1.0e-4,
        max_selected=3,
        seed=123,
    )
    assert calls == {"generate": 1, "score": 1, "build": 1}
    assert len(selected) == 1


def test_direction_limited_assessment_reports_magnitude_limited():
    frame = pd.DataFrame(
        [
            {
                "scale_protocol": "known_C",
                "direction_error_vs_ridge": 0.05,
                "residual_feasible": False,
            },
            {
                "scale_protocol": "best_scalar_diagnostic",
                "direction_error_vs_ridge": 0.05,
                "residual_feasible": False,
            },
        ]
    )
    statement = rfcs._is_direction_limited(frame)
    assert "magnitude-limited" in statement


def test_direction_limited_assessment_reports_direction_limited():
    frame = pd.DataFrame(
        [
            {
                "scale_protocol": "known_C",
                "direction_error_vs_ridge": 0.9,
                "residual_feasible": False,
            },
        ]
    )
    statement = rfcs._is_direction_limited(frame)
    assert statement.startswith("yes")


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("current_global", "global_safe"),
        ("margin_1p05", "margin_1.05"),
        ("margin_1p10", "margin_1.1"),
        ("degree_adaptive", "degree_adaptive"),
    ],
)
def test_target_design_aliases(alias, expected):
    assert rfcs.TARGET_DESIGN_ALIASES[alias] == expected
