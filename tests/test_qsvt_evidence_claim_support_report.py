from __future__ import annotations

from pathlib import Path

from robust_qsvt_se.qsvt.evidence_claim_support_report import (
    CODESIGNED_LAYERS,
    CONSOLIDATION_LAYERS,
    CONTRIBUTION_STRENGTH_CATEGORIES,
    CROSS_CASE_LAYERS,
    FINAL_OPTIONAL_LAYERS,
    SOLVER_READINESS_CATEGORIES,
    build_evidence_claim_support_report,
    claim_support_rows,
    codesigned_target_gap_rows,
    contribution_strength_rows,
    cross_case_solver_prototype_rows,
    evidence_layer_rows,
    final_optional_evidence_rows,
    selected_solver_prototype_rows,
    solver_readiness_classification,
    tqe_readiness_classification,
)


def test_evidence_report_classifies_unsupported_claims(tmp_path: Path) -> None:
    rows = claim_support_rows(tmp_path)
    unsupported = [row for row in rows if row["claim_class"] == "unsupported_do_not_claim"]

    assert any("Quantum speedup" in row["claim"] for row in unsupported)
    assert any("QSVT beats Ridge" in row["claim"] for row in unsupported)


def test_evidence_report_writes_outputs(tmp_path: Path) -> None:
    input_root = tmp_path / "outputs"
    input_root.mkdir()
    (input_root / "gate_level_qsvt_ieee14_solver").mkdir()
    run = build_evidence_claim_support_report(
        {"input_root": str(input_root), "output_dir": str(tmp_path / "report")}
    )

    for name in [
        "manifest",
        "evidence_layer_table",
        "claim_support_table",
        "remaining_assumption_table",
        "bottleneck_ranking",
        "contribution_strength_table",
        "unresolved_gap_table",
        "amplitude_recovery_gap_table",
        "codesigned_target_gap_table",
        "selected_solver_prototype_table",
        "cross_case_solver_prototype_table",
        "final_optional_evidence_table",
        "full_solver_readiness_table",
        "paper_ready_summary",
        "tqe_readiness_summary",
        "limitation_ready_summary",
        "refinement_update_summary",
        "strongest_contribution_summary",
    ]:
        assert run["artifacts"][name].is_file()


def test_evidence_report_includes_refinement_layers(tmp_path: Path) -> None:
    rows = evidence_layer_rows(tmp_path)
    layers = {row["evidence_layer"] for row in rows}

    assert "alpha_degree_residual_refinement" in layers
    assert "degree_control_audit" in layers
    assert "polynomial_action_decomposition" in layers
    assert "best_config_gate_validation" in layers
    assert "hardware_aware_oracle_cost_model" in layers
    assert "criteria_based_subproblem_selection" in layers
    assert "refined_selected_subproblem_solver" in layers
    assert "observable_error_decomposition" in layers
    assert "accuracy_success_tradeoff" in layers


def test_tqe_readiness_summary_is_conservative(tmp_path: Path) -> None:
    for directory in [
        "qsvt_degree_control_audit",
        "qsvt_polynomial_action_decomposition",
        "qsvt_best_config_gate_validation",
        "hardware_aware_oracle_cost_model",
    ]:
        (tmp_path / directory).mkdir()
    evidence = evidence_layer_rows(tmp_path)
    bottlenecks = [
        {"bottleneck": "true QSVT degree control", "severity": "moderate"},
        {"bottleneck": "hardware-aware oracle cost", "severity": "high"},
    ]

    assert tqe_readiness_classification(evidence, bottlenecks) == (
        "draft_ready_with_major_limitations"
    )


def test_evidence_report_includes_strengthening_layers(tmp_path: Path) -> None:
    rows = evidence_layer_rows(tmp_path)
    layers = {row["evidence_layer"] for row in rows}

    for layer in [
        "scale_recovery_protocols",
        "bounded_target_design_study",
        "residual_feasible_config_search",
        "residual_feasible_gate_validation",
        "power_observable_protocols",
        "sparse_oracle_prototype_strengthening",
    ]:
        assert layer in layers


def test_contribution_strength_uses_allowed_categories_and_keeps_unsupported(
    tmp_path: Path,
) -> None:
    rows = contribution_strength_rows(tmp_path)
    categories = {row["contribution_strength"] for row in rows}

    assert categories.issubset(set(CONTRIBUTION_STRENGTH_CATEGORIES))
    # The two hard overclaims must always be classified unsupported regardless of artifacts.
    unsupported = {
        row["contribution"] for row in rows if row["contribution_strength"] == "unsupported"
    }
    assert any("superiority over Ridge" in item for item in unsupported)
    assert any("Quantum speedup" in item for item in unsupported)


def test_evidence_report_includes_amplitude_recovery_layers(tmp_path: Path) -> None:
    rows = evidence_layer_rows(tmp_path)
    layers = {row["evidence_layer"] for row in rows}

    for layer in [
        "actual_amplitude_estimation_routines",
        "norm_recovery_from_amplitude",
        "residual_feasible_with_amplitude_recovery",
        "gate_validation_with_amplitude_recovery",
        "observable_first_qsvt_solver",
    ]:
        assert layer in layers


def test_evidence_report_distinguishes_actual_amplitude_estimation(tmp_path: Path) -> None:
    input_root = tmp_path / "outputs"
    (input_root / "qsvt_amplitude_estimation_routines").mkdir(parents=True)
    claims = claim_support_rows(input_root)
    actual = [row for row in claims if "Actual small-circuit amplitude-estimation" in row["claim"]]
    assert actual
    assert actual[0]["claim_class"] == "supported_by_controlled_experiment"

    contributions = contribution_strength_rows(input_root)
    amplitude = [
        row for row in contributions if "amplitude-estimation routines" in row["contribution"]
    ]
    assert amplitude and amplitude[0]["contribution_strength"] == "strong_implemented_evidence"


def test_solver_readiness_is_conservative_when_layers_absent(tmp_path: Path) -> None:
    classification = solver_readiness_classification(evidence_layer_rows(tmp_path), tmp_path)
    assert classification in SOLVER_READINESS_CATEGORIES
    # No amplitude-recovery or co-designed artifacts present in a bare temp dir.
    assert classification == "not_ready"


def test_evidence_report_includes_codesigned_layers(tmp_path: Path) -> None:
    rows = evidence_layer_rows(tmp_path)
    layers = {row["evidence_layer"] for row in rows}

    for layer in CODESIGNED_LAYERS:
        assert layer in layers


def _make_codesigned_outputs(input_root: Path, *, feasible: bool, gate_feasible: bool) -> None:
    for layer_dir in [
        "qsvt_weighted_singular_support",
        "qsvt_codesigned_bounded_target_study",
        "qsvt_residual_feasible_codesigned_search",
        "qsvt_codesigned_gate_validation",
        "qsvt_observable_first_solver_v2",
    ]:
        (input_root / layer_dir).mkdir(parents=True)
    search = input_root / "qsvt_residual_feasible_codesigned_search"
    (search / "all_codesigned_configs.csv").write_text(
        "residual_ratio_vs_no_update,direction_error_vs_ridge\n0.0002,0.0009\n", encoding="utf-8"
    )
    feasible_body = (
        "alpha,degree,target_family\n0.0001,25,residual_aware\n"
        if feasible
        else "alpha,degree,target_family\n"
    )
    (search / "deployable_residual_feasible_configs.csv").write_text(
        feasible_body, encoding="utf-8"
    )
    (search / "diagnostic_feasible_configs.csv").write_text(
        "alpha,degree,target_family\n", encoding="utf-8"
    )
    gate = input_root / "qsvt_codesigned_gate_validation"
    (gate / "codesigned_gate_validation_results.csv").write_text(
        "residual_feasible_after_gate\n" + ("True\n" if gate_feasible else "False\n"),
        encoding="utf-8",
    )


def test_solver_readiness_codesigned_prototype_when_feasible(tmp_path: Path) -> None:
    _make_codesigned_outputs(tmp_path, feasible=True, gate_feasible=True)
    classification = solver_readiness_classification(evidence_layer_rows(tmp_path), tmp_path)
    assert classification == "solver_prototype_ready_selected_subproblems"

    gap = codesigned_target_gap_rows(tmp_path)
    closed = [row for row in gap if row["metric"] == "codesigned_gap_closed"]
    assert closed and closed[0]["value"] == "yes"


def test_solver_readiness_codesigned_negative_result_when_not_feasible(tmp_path: Path) -> None:
    _make_codesigned_outputs(tmp_path, feasible=False, gate_feasible=False)
    classification = solver_readiness_classification(evidence_layer_rows(tmp_path), tmp_path)
    assert classification == "engineering_framework_ready_with_negative_result"


def test_evidence_report_includes_consolidation_layers(tmp_path: Path) -> None:
    rows = evidence_layer_rows(tmp_path)
    layers = {row["evidence_layer"] for row in rows}

    for layer in CONSOLIDATION_LAYERS:
        assert layer in layers


def _make_consolidation_outputs(input_root: Path, *, robustness_modes: list[str]) -> None:
    for layer_dir in [
        "qsvt_degree_window_overshoot",
        "qsvt_degree_window_gate_validation",
        "qsvt_gate_observable_readout",
        "qsvt_codesigned_subproblem_robustness",
        "qsvt_robustness_gate_validation",
        "qsvt_selected_solver_prototype",
    ]:
        (input_root / layer_dir).mkdir(parents=True)
    (
        input_root / "qsvt_degree_window_gate_validation" / "degree_window_gate_results.csv"
    ).write_text("degree,residual_feasible_after_gate\n29,True\n", encoding="utf-8")
    body = "selection_mode,residual_feasible_after_gate\n"
    for mode in robustness_modes:
        body += f"{mode},True\n"
    (input_root / "qsvt_robustness_gate_validation" / "robustness_gate_results.csv").write_text(
        body, encoding="utf-8"
    )


def test_solver_readiness_consolidation_family_takes_precedence(tmp_path: Path) -> None:
    _make_consolidation_outputs(
        tmp_path, robustness_modes=["high_leverage", "metadata_mapped", "residual_supported"]
    )
    classification = solver_readiness_classification(evidence_layer_rows(tmp_path), tmp_path)
    assert classification == "solver_prototype_ready_selected_subproblem_family"
    assert classification in SOLVER_READINESS_CATEGORIES


def test_solver_readiness_consolidation_single_subproblem(tmp_path: Path) -> None:
    _make_consolidation_outputs(tmp_path, robustness_modes=[])
    classification = solver_readiness_classification(evidence_layer_rows(tmp_path), tmp_path)
    assert classification == "solver_prototype_ready_single_selected_subproblem"


def test_selected_solver_prototype_rows_report_classification(tmp_path: Path) -> None:
    _make_consolidation_outputs(tmp_path, robustness_modes=["high_leverage", "metadata_mapped"])
    rows = selected_solver_prototype_rows(tmp_path)
    classification = [row for row in rows if row["aspect"] == "final_prototype_classification"]
    assert classification
    assert classification[0]["value"] == "solver_prototype_ready_selected_subproblem_family"


def test_evidence_report_preserves_unsupported_claims(tmp_path: Path) -> None:
    # Even with the full consolidation stack present, the hard overclaims stay unsupported.
    _make_consolidation_outputs(
        tmp_path, robustness_modes=["high_leverage", "metadata_mapped", "residual_supported"]
    )
    claims = claim_support_rows(tmp_path)
    unsupported = [row for row in claims if row["claim_class"] == "unsupported_do_not_claim"]
    assert any("Quantum speedup" in row["claim"] for row in unsupported)
    assert any("QSVT beats Ridge" in row["claim"] for row in unsupported)

    contributions = contribution_strength_rows(tmp_path)
    unsupported_contrib = {
        row["contribution"]
        for row in contributions
        if row["contribution_strength"] == "unsupported"
    }
    assert any("superiority over Ridge" in item for item in unsupported_contrib)
    assert any("Quantum speedup" in item for item in unsupported_contrib)


def test_evidence_report_includes_cross_case_layers(tmp_path: Path) -> None:
    rows = evidence_layer_rows(tmp_path)
    layers = {row["evidence_layer"] for row in rows}

    for layer in CROSS_CASE_LAYERS:
        assert layer in layers


def _make_cross_case_outputs(input_root: Path, *, survivors: dict[str, bool]) -> None:
    for layer_dir in [
        "qsvt_cross_case_solver_audit",
        "qsvt_overshoot_mechanism_diagnostic",
        "qsvt_cross_case_codesigned_robustness",
        "qsvt_cross_case_gate_validation",
        "qsvt_cross_case_gate_observable_readout",
        "qsvt_cross_case_solver_prototype",
    ]:
        (input_root / layer_dir).mkdir(parents=True, exist_ok=True)
    rows = ["case,residual_feasible_after_gate"]
    for case, feasible in survivors.items():
        rows.append(f"{case},{'True' if feasible else 'False'}")
    (input_root / "qsvt_cross_case_gate_validation" / "cross_case_gate_results.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


def test_solver_readiness_cross_case_family_takes_precedence(tmp_path: Path) -> None:
    # Cross-case layers present and both IEEE30/57 survive -> cross-case selected family,
    # taking precedence over the consolidation pathway.
    _make_consolidation_outputs(
        tmp_path, robustness_modes=["high_leverage", "metadata_mapped", "residual_supported"]
    )
    _make_cross_case_outputs(tmp_path, survivors={"ieee14": True, "ieee30": True, "ieee57": True})
    classification = solver_readiness_classification(evidence_layer_rows(tmp_path), tmp_path)
    assert classification == "solver_prototype_ready_cross_case_selected_family"
    assert classification in SOLVER_READINESS_CATEGORIES


def test_solver_readiness_cross_case_partial(tmp_path: Path) -> None:
    _make_cross_case_outputs(tmp_path, survivors={"ieee14": True, "ieee30": True, "ieee57": False})
    classification = solver_readiness_classification(evidence_layer_rows(tmp_path), tmp_path)
    assert classification == "solver_prototype_ready_cross_case_partial"


def test_solver_readiness_cross_case_ieee14_only_when_no_transfer(tmp_path: Path) -> None:
    _make_cross_case_outputs(tmp_path, survivors={"ieee14": True, "ieee30": False, "ieee57": False})
    classification = solver_readiness_classification(evidence_layer_rows(tmp_path), tmp_path)
    assert classification == "solver_prototype_ready_ieee14_selected_family"


def test_cross_case_solver_prototype_rows_report_classification(tmp_path: Path) -> None:
    _make_cross_case_outputs(tmp_path, survivors={"ieee30": True, "ieee57": True})
    rows = cross_case_solver_prototype_rows(tmp_path)
    classification = [row for row in rows if row["aspect"] == "final_cross_case_classification"]
    assert classification
    assert classification[0]["value"] == "solver_prototype_ready_cross_case_selected_family"


def test_evidence_report_preserves_unsupported_claims_with_cross_case_stack(tmp_path: Path) -> None:
    # Even with the full cross-case stack present, the hard overclaims stay unsupported.
    _make_cross_case_outputs(tmp_path, survivors={"ieee14": True, "ieee30": True, "ieee57": True})
    claims = claim_support_rows(tmp_path)
    unsupported = [row for row in claims if row["claim_class"] == "unsupported_do_not_claim"]
    assert any("Quantum speedup" in row["claim"] for row in unsupported)
    assert any("QSVT beats Ridge" in row["claim"] for row in unsupported)


def test_evidence_report_includes_final_optional_layers(tmp_path: Path) -> None:
    rows = evidence_layer_rows(tmp_path)
    layers = {row["evidence_layer"] for row in rows}

    for layer in FINAL_OPTIONAL_LAYERS:
        assert layer in layers


def _make_final_optional_outputs(input_root: Path, *, ieee118_survives: bool) -> None:
    for layer_dir in [
        "qsvt_ieee118_extension_audit",
        "qsvt_direction_resolved_overshoot_decomposition",
        "qsvt_ieee118_selected_robustness",
        "qsvt_ieee118_gate_validation",
        "qsvt_ieee118_gate_observable_readout",
        "qsvt_final_optional_evidence_package",
    ]:
        (input_root / layer_dir).mkdir(parents=True, exist_ok=True)
    (
        input_root / "qsvt_ieee118_selected_robustness" / "ieee118_gate_validation_candidates.csv"
    ).write_text("subproblem_id,selection_mode\nhigh_leverage_00,high_leverage\n", encoding="utf-8")
    (input_root / "qsvt_ieee118_gate_validation" / "ieee118_gate_results.csv").write_text(
        "subproblem_id,gate_status,residual_feasible_after_gate\n"
        + (
            "high_leverage_00,completed,True\n"
            if ieee118_survives
            else "high_leverage_00,completed,False\n"
        ),
        encoding="utf-8",
    )


def test_solver_readiness_final_optional_plus_ieee118_takes_precedence(tmp_path: Path) -> None:
    # The final-optional pathway takes highest precedence over the cross-case pathway.
    _make_cross_case_outputs(tmp_path, survivors={"ieee14": True, "ieee30": True, "ieee57": True})
    _make_final_optional_outputs(tmp_path, ieee118_survives=True)
    classification = solver_readiness_classification(evidence_layer_rows(tmp_path), tmp_path)
    assert classification == "solver_prototype_ready_cross_case_plus_ieee118_selected"
    assert classification in SOLVER_READINESS_CATEGORIES


def test_solver_readiness_final_optional_boundary_when_no_survivor(tmp_path: Path) -> None:
    _make_cross_case_outputs(tmp_path, survivors={"ieee14": True, "ieee30": True, "ieee57": True})
    _make_final_optional_outputs(tmp_path, ieee118_survives=False)
    classification = solver_readiness_classification(evidence_layer_rows(tmp_path), tmp_path)
    assert classification == "solver_prototype_ready_cross_case_with_ieee118_boundary"


def test_final_optional_evidence_rows_report_classification(tmp_path: Path) -> None:
    _make_final_optional_outputs(tmp_path, ieee118_survives=True)
    rows = final_optional_evidence_rows(tmp_path)
    classification = [row for row in rows if row["aspect"] == "final_optional_classification"]
    assert classification
    assert classification[0]["value"] == "solver_prototype_ready_cross_case_plus_ieee118_selected"


def test_evidence_report_preserves_unsupported_claims_with_final_optional_stack(
    tmp_path: Path,
) -> None:
    # Even with the full final-optional stack present, the hard overclaims stay unsupported.
    _make_final_optional_outputs(tmp_path, ieee118_survives=True)
    claims = claim_support_rows(tmp_path)
    unsupported = [row for row in claims if row["claim_class"] == "unsupported_do_not_claim"]
    assert any("Quantum speedup" in row["claim"] for row in unsupported)
    assert any("QSVT beats Ridge" in row["claim"] for row in unsupported)

    contributions = contribution_strength_rows(tmp_path)
    unsupported_contrib = {
        row["contribution"]
        for row in contributions
        if row["contribution_strength"] == "unsupported"
    }
    assert any("superiority over Ridge" in item for item in unsupported_contrib)
    assert any("Quantum speedup" in item for item in unsupported_contrib)
