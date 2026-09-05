from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from robust_qsvt_se.qsvt.cross_case_solver_prototype_package import (
    cross_case_solver_prototype_classification,
)
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.final_optional_evidence_package import (
    final_optional_evidence_classification,
)
from robust_qsvt_se.qsvt.selected_solver_prototype_package import (
    selected_solver_prototype_classification,
)
from robust_qsvt_se.utils.io import ensure_directory

CLAIM_REPORT_BOUNDARY = (
    "This report separates implemented behavior, controlled evidence, resource "
    "models, assumptions, and unsupported claims. It does not claim quantum "
    "speedup, full IEEE-scale hardware execution, or QSVT superiority over "
    "Ridge/Tikhonov."
)


def build_evidence_claim_support_report(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "input_root": "outputs",
        "output_dir": "outputs/qsvt_evidence_claim_support_report",
    }
    resolved.update(config)
    input_root = Path(resolved["input_root"])
    output_dir = ensure_directory(resolved["output_dir"])
    evidence_rows = evidence_layer_rows(input_root)
    claim_rows = claim_support_rows(input_root)
    assumption_rows = remaining_assumption_rows(input_root)
    bottleneck_rows = bottleneck_rows_from_outputs(input_root)
    contribution_rows = contribution_strength_rows(input_root)
    gap_rows = unresolved_gap_rows(input_root, bottleneck_rows, assumption_rows)
    amplitude_gap_rows = amplitude_recovery_gap_rows(input_root)
    codesigned_gap_rows = codesigned_target_gap_rows(input_root)
    prototype_rows = selected_solver_prototype_rows(input_root)
    cross_case_prototype_rows = cross_case_solver_prototype_rows(input_root)
    final_optional_rows = final_optional_evidence_rows(input_root)
    solver_readiness_rows = full_solver_readiness_rows(input_root, evidence_rows)
    solver_readiness = solver_readiness_classification(evidence_rows, input_root)
    artifacts = _write_outputs(
        output_dir,
        resolved,
        evidence_rows=evidence_rows,
        claim_rows=claim_rows,
        assumption_rows=assumption_rows,
        bottleneck_rows=bottleneck_rows,
        contribution_rows=contribution_rows,
        gap_rows=gap_rows,
        amplitude_gap_rows=amplitude_gap_rows,
        codesigned_gap_rows=codesigned_gap_rows,
        prototype_rows=prototype_rows,
        cross_case_prototype_rows=cross_case_prototype_rows,
        final_optional_rows=final_optional_rows,
        solver_readiness_rows=solver_readiness_rows,
        solver_readiness=solver_readiness,
    )
    return {
        "output_dir": output_dir,
        "evidence_rows": evidence_rows,
        "claim_rows": claim_rows,
        "assumption_rows": assumption_rows,
        "bottleneck_rows": bottleneck_rows,
        "contribution_rows": contribution_rows,
        "gap_rows": gap_rows,
        "amplitude_gap_rows": amplitude_gap_rows,
        "codesigned_gap_rows": codesigned_gap_rows,
        "prototype_rows": prototype_rows,
        "cross_case_prototype_rows": cross_case_prototype_rows,
        "final_optional_rows": final_optional_rows,
        "solver_readiness_rows": solver_readiness_rows,
        "solver_readiness": solver_readiness,
        "artifacts": artifacts,
    }


def evidence_layer_rows(input_root: Path) -> list[dict[str, Any]]:
    layers = [
        (
            "regularized_spectral_filter_reference",
            "src/robust_qsvt_se/qsvt/filters.py",
            "implemented",
        ),
        ("phase_synthesis_validation", "qsvt_phase_cache", "controlled_or_cached"),
        (
            "dense_gate_level_convention",
            "gate_level_qsvt_convention_debug",
            "controlled_experiment",
        ),
        ("tiny_gate_level_solver", "gate_level_qsvt_tiny_solver", "controlled_experiment"),
        (
            "ieee14_selected_gate_level_solver",
            "gate_level_qsvt_ieee14_solver",
            "controlled_experiment",
        ),
        ("norm_residual_gap_audit", "qsvt_norm_residual_gap_audit", "controlled_experiment"),
        ("subproblem_sweep", "gate_level_qsvt_subproblem_sweep", "controlled_experiment"),
        ("power_observable_readout", "qsvt_power_observable_readout", "controlled_experiment"),
        (
            "dense_explicit_limit_study",
            "dense_explicit_qsvt_limit_study_v2",
            "resource_limit_study",
        ),
        ("sparse_oracle_assumption_ledger", "sparse_oracle_assumption_ledger", "assumption_ledger"),
        ("success_amplification_cost", "qsvt_success_amplification_cost", "cost_proxy"),
        (
            "alpha_degree_residual_refinement",
            "qsvt_alpha_degree_residual_refinement",
            "controlled_experiment",
        ),
        ("degree_control_audit", "qsvt_degree_control_audit", "controlled_experiment"),
        (
            "polynomial_action_decomposition",
            "qsvt_polynomial_action_decomposition",
            "controlled_experiment",
        ),
        (
            "best_config_gate_validation",
            "qsvt_best_config_gate_validation",
            "controlled_experiment",
        ),
        (
            "hardware_aware_oracle_cost_model",
            "hardware_aware_oracle_cost_model",
            "resource_model",
        ),
        (
            "criteria_based_subproblem_selection",
            "qsvt_subproblem_selection_policy",
            "controlled_experiment",
        ),
        (
            "refined_selected_subproblem_solver",
            "qsvt_refined_selected_subproblem_solver",
            "controlled_experiment",
        ),
        (
            "observable_error_decomposition",
            "qsvt_observable_error_decomposition",
            "controlled_experiment",
        ),
        (
            "accuracy_success_tradeoff",
            "qsvt_accuracy_success_tradeoff",
            "cost_proxy",
        ),
        ("matrix_free_qsvt_target_proxy", "qsvt_matrix_free_ieee_experiments", "target_proxy"),
        ("ieee_scale_resource_estimate", "qsvt_resource_full_ieee", "resource_model"),
        ("scale_recovery_protocols", "qsvt_scale_recovery_protocols", "controlled_experiment"),
        (
            "bounded_target_design_study",
            "qsvt_bounded_target_design_study",
            "controlled_experiment",
        ),
        (
            "residual_feasible_config_search",
            "qsvt_residual_feasible_config_search",
            "controlled_experiment",
        ),
        (
            "residual_feasible_gate_validation",
            "qsvt_residual_feasible_gate_validation",
            "controlled_experiment",
        ),
        (
            "power_observable_protocols",
            "qsvt_power_observable_protocols",
            "controlled_experiment",
        ),
        (
            "sparse_oracle_prototype_strengthening",
            "qsvt_sparse_oracle_prototype_strengthening",
            "resource_model",
        ),
        (
            "actual_amplitude_estimation_routines",
            "qsvt_amplitude_estimation_routines",
            "controlled_experiment",
        ),
        (
            "norm_recovery_from_amplitude",
            "qsvt_norm_recovery_from_amplitude",
            "controlled_experiment",
        ),
        (
            "residual_feasible_with_amplitude_recovery",
            "qsvt_residual_feasible_with_amplitude_recovery",
            "controlled_experiment",
        ),
        (
            "gate_validation_with_amplitude_recovery",
            "qsvt_gate_validation_with_amplitude_recovery",
            "controlled_experiment",
        ),
        (
            "observable_first_qsvt_solver",
            "qsvt_observable_first_solver",
            "controlled_experiment",
        ),
        (
            "weighted_singular_support",
            "qsvt_weighted_singular_support",
            "controlled_experiment",
        ),
        (
            "codesigned_bounded_target_study",
            "qsvt_codesigned_bounded_target_study",
            "controlled_experiment",
        ),
        (
            "residual_feasible_codesigned_search",
            "qsvt_residual_feasible_codesigned_search",
            "controlled_experiment",
        ),
        (
            "codesigned_gate_validation",
            "qsvt_codesigned_gate_validation",
            "controlled_experiment",
        ),
        (
            "observable_first_solver_v2",
            "qsvt_observable_first_solver_v2",
            "controlled_experiment",
        ),
        (
            "degree_window_overshoot",
            "qsvt_degree_window_overshoot",
            "controlled_experiment",
        ),
        (
            "degree_window_gate_validation",
            "qsvt_degree_window_gate_validation",
            "controlled_experiment",
        ),
        (
            "gate_observable_readout",
            "qsvt_gate_observable_readout",
            "controlled_experiment",
        ),
        (
            "codesigned_subproblem_robustness",
            "qsvt_codesigned_subproblem_robustness",
            "controlled_experiment",
        ),
        (
            "robustness_gate_validation",
            "qsvt_robustness_gate_validation",
            "controlled_experiment",
        ),
        (
            "selected_solver_prototype_package",
            "qsvt_selected_solver_prototype",
            "controlled_experiment",
        ),
        (
            "cross_case_solver_audit",
            "qsvt_cross_case_solver_audit",
            "controlled_experiment",
        ),
        (
            "overshoot_mechanism_diagnostic",
            "qsvt_overshoot_mechanism_diagnostic",
            "controlled_experiment",
        ),
        (
            "cross_case_codesigned_robustness",
            "qsvt_cross_case_codesigned_robustness",
            "controlled_experiment",
        ),
        (
            "cross_case_gate_validation",
            "qsvt_cross_case_gate_validation",
            "controlled_experiment",
        ),
        (
            "cross_case_gate_observable_readout",
            "qsvt_cross_case_gate_observable_readout",
            "controlled_experiment",
        ),
        (
            "cross_case_solver_prototype_package",
            "qsvt_cross_case_solver_prototype",
            "controlled_experiment",
        ),
        (
            "ieee118_extension_audit",
            "qsvt_ieee118_extension_audit",
            "controlled_experiment",
        ),
        (
            "direction_resolved_overshoot_decomposition",
            "qsvt_direction_resolved_overshoot_decomposition",
            "controlled_experiment",
        ),
        (
            "ieee118_selected_robustness",
            "qsvt_ieee118_selected_robustness",
            "controlled_experiment",
        ),
        (
            "ieee118_gate_validation",
            "qsvt_ieee118_gate_validation",
            "controlled_experiment",
        ),
        (
            "ieee118_gate_observable_readout",
            "qsvt_ieee118_gate_observable_readout",
            "controlled_experiment",
        ),
        (
            "final_optional_evidence_package",
            "qsvt_final_optional_evidence_package",
            "controlled_experiment",
        ),
    ]
    rows = []
    for layer, relative, evidence_type in layers:
        path = Path(relative) if str(relative).startswith("src/") else input_root / relative
        rows.append(
            {
                "evidence_layer": layer,
                "evidence_type": evidence_type,
                "artifact_path": str(path),
                "artifact_present": path.exists(),
                "claim_support_class": _layer_claim_class(layer, path.exists()),
            }
        )
    return rows


def claim_support_rows(input_root: Path) -> list[dict[str, Any]]:
    return [
        _claim(
            "QSVT signal-block convention is validated on scalar and diagonal tests",
            "supported_by_controlled_experiment",
            input_root / "gate_level_qsvt_convention_debug",
            "Use real top-left signal block and real prefix state quadrature.",
        ),
        _claim(
            "Selected IEEE14-derived 4x4 QSVT update reduces residual versus no update",
            "supported_by_controlled_experiment",
            input_root / "gate_level_qsvt_ieee14_solver",
            "Do not claim it matches Ridge residual accuracy.",
        ),
        _claim(
            "Residual gap is decomposed into norm, direction, polynomial, and conditioning factors",
            "supported_by_controlled_experiment",
            input_root / "qsvt_norm_residual_gap_audit",
            "Use audit-specific numbers.",
        ),
        _claim(
            "Alpha and degree sensitivity of the residual gap is directly tested",
            "supported_by_controlled_experiment",
            input_root / "qsvt_alpha_degree_residual_refinement",
            "Report whether higher requested degree changes synthesized degree.",
        ),
        _claim(
            "True QSVT degree control is audited with requested, constructed, "
            "synthesized, and effective degrees reported separately",
            "supported_by_controlled_experiment",
            input_root / "qsvt_degree_control_audit",
            "Do not claim high-degree convergence unless true higher degrees are synthesized.",
        ),
        _claim(
            "Polynomial-action convergence is separated from gate-level execution",
            "supported_by_controlled_experiment",
            input_root / "qsvt_polynomial_action_decomposition",
            "Use Ridge, exact target, bounded target, and polynomial-action rows separately.",
        ),
        _claim(
            "Best gate-level configurations are selected from polynomial-action residual evidence",
            "supported_by_controlled_experiment",
            input_root / "qsvt_best_config_gate_validation",
            "Selected 4x4 dense simulator validation only.",
        ),
        _claim(
            "Selected subproblems are chosen by criteria rather than QSVT residuals",
            "supported_by_controlled_experiment",
            input_root / "qsvt_subproblem_selection_policy",
            "Use numerical and metadata selection criteria before QSVT evaluation.",
        ),
        _claim(
            "Observable readout errors are decomposed into scaling, state, shot, "
            "and metadata sources",
            "supported_by_controlled_experiment",
            input_root / "qsvt_observable_error_decomposition",
            "Do not claim full readout bottleneck is solved.",
        ),
        _claim(
            "Accuracy improvements must be balanced against success-probability cost proxies",
            "supported_by_resource_model",
            input_root / "qsvt_accuracy_success_tradeoff",
            "Amplitude amplification remains a cost proxy unless implemented.",
        ),
        _claim(
            "Actual small-circuit amplitude-estimation routines are implemented "
            "(Bernoulli sampling and iterative Grover-power MLAE), distinct from the earlier "
            "success-probability cost proxy",
            "supported_by_controlled_experiment",
            input_root / "qsvt_amplitude_estimation_routines",
            "These are simulator-backed small-circuit routines; the earlier "
            "qsvt_success_amplification_cost rows remain cost proxies.",
        ),
        _claim(
            "Amplitude-based norm recovery is tested for the QSVT update scale on a selected "
            "IEEE14-derived subproblem",
            "supported_by_controlled_experiment",
            input_root / "qsvt_norm_recovery_from_amplitude",
            "Direction is from the polynomial action; only the scale is recovered from amplitude.",
        ),
        _claim(
            "Observable-first QSVT estimation avoids full-vector readout for selected "
            "power-system observables",
            "supported_by_controlled_experiment",
            input_root / "qsvt_observable_first_solver",
            "Practical observables avoid full-vector tomography; readout is not solved.",
        ),
        _claim(
            "Singular-support weighting identifies the high-impact singular directions for the "
            "Ridge update and the residual",
            "supported_by_controlled_experiment",
            input_root / "qsvt_weighted_singular_support",
            "Weights rank directions by residual projection, update magnitude, and residual "
            "sensitivity; used to focus the co-designed bounded target.",
        ),
        _claim(
            "Co-designed QSVT-safe bounded targets at low-to-moderate degree recover a "
            "Ridge-aligned, residual-feasible update on a selected IEEE14-derived subproblem",
            "supported_by_controlled_experiment",
            input_root / "qsvt_residual_feasible_codesigned_search",
            "Feasibility uses only QSVT-safe, deployable targets meeting the residual, "
            "direction, and success-amplitude thresholds; readout of the direction is assumed.",
        ),
        _claim(
            "The co-designed residual-feasible update survives dense gate-level validation with "
            "amplitude-based scale recovery",
            "supported_by_controlled_experiment",
            input_root / "qsvt_codesigned_gate_validation",
            "Phases are synthesized from the co-designed bounded coefficients; the gate output "
            "matches the polynomial action and the recovered residual stays feasible. Selected "
            "dense simulator evidence only.",
        ),
        _claim(
            "The co-designed targets are Ridge-aligned and residual-feasible over a concrete "
            "low-to-moderate odd-degree window, with overshoot beginning above it",
            "supported_by_controlled_experiment",
            input_root / "qsvt_degree_window_overshoot",
            "Singular-support-weighted and residual-aware targets stay feasible across the "
            "window; success-amplitude-aware and degree-scheduled clipped targets do not, and "
            "degree above the window overshoots off-support and loses the Ridge direction.",
        ),
        _claim(
            "The co-designed degree window is gate-validated over multiple degrees on the "
            "selected subproblem",
            "supported_by_controlled_experiment",
            input_root / "qsvt_degree_window_gate_validation",
            "Representative degree-window configurations are synthesized and dense-simulated; "
            "residual feasibility survives at several degrees, not a single point.",
        ),
        _claim(
            "Selected power-system observables are gate-confirmed from the QSVT output without "
            "full-vector readout",
            "supported_by_controlled_experiment",
            input_root / "qsvt_gate_observable_readout",
            "Top-k identification stays exact and signed/energy functionals match Ridge closely "
            "from the gate output; full-vector reconstruction stays not-recommended.",
        ),
        _claim(
            "The co-designed solver transfers to a small family of criteria-selected "
            "IEEE14-derived subproblems",
            "supported_by_controlled_experiment",
            input_root / "qsvt_robustness_gate_validation",
            "High-leverage, metadata-mapped, and residual-supported well-conditioned blocks are "
            "gate-validated; rank-deficient and worst-conditioned blocks are not.",
        ),
        _claim(
            "The overshoot onset above the safe degree window is a high-weight singular-support "
            "fit error that breaks the Ridge direction, not a boundedness violation or a dominant "
            "off-support peak",
            "supported_by_controlled_experiment",
            input_root / "qsvt_overshoot_mechanism_diagnostic",
            "The bounded polynomial stays within the QSVT bound at onset and the off-support "
            "margin is below the threshold; because C cancels in the known-C update, the corrupted "
            "on-support polynomial values are what break the direction.",
        ),
        _claim(
            "The co-designed selected-subproblem solver transfers across IEEE14/IEEE30/IEEE57 "
            "criteria-selected 4x4 blocks under dense gate validation",
            "supported_by_controlled_experiment",
            input_root / "qsvt_cross_case_gate_validation",
            "Cross-case candidates are reconstructed per case and gate-validated; this is "
            "selected-subproblem evidence on three IEEE cases, not full IEEE-scale solving.",
        ),
        _claim(
            "Selected power-system observables are gate-confirmed across IEEE14/30/57 without "
            "full-vector readout",
            "supported_by_controlled_experiment",
            input_root / "qsvt_cross_case_gate_observable_readout",
            "Top-k identification stays exact and signed/energy functionals match Ridge closely "
            "from the gate output in every tested case; full-vector reconstruction is excluded.",
        ),
        _claim(
            "The selected-subproblem solver prototype extends to IEEE118 selected 4x4 blocks "
            "under dense gate validation",
            "supported_by_controlled_experiment",
            input_root / "qsvt_ieee118_gate_validation",
            "Criteria-selected IEEE118 4x4 candidates are reconstructed and gate-validated; this "
            "is a larger benchmark selected-subproblem test, not a full IEEE118-scale QSVT solver.",
        ),
        _claim(
            "The degree-47 overshoot onset is a leading-direction amplitude distortion at the "
            "near-boundary high-weight singular directions, resolved per singular direction",
            "supported_by_controlled_experiment",
            input_root / "qsvt_direction_resolved_overshoot_decomposition",
            "The high-degree polynomial overshoots the Ridge filter value at the leading "
            "(sigma ~ sigma_max) directions while small-sigma directions stay fit; no sign flip "
            "at degree 47, and since C cancels in the known-C update this breaks the direction.",
        ),
        _claim(
            "Full IEEE-scale QSVT is an oracle-model pathway",
            "supported_by_assumption_only",
            input_root / "sparse_oracle_assumption_ledger",
            "Do not describe as hardware execution.",
        ),
        _claim(
            "Hardware-aware sparse-oracle cost estimates cover IEEE14/30/57/118/300",
            "supported_by_resource_model",
            input_root / "hardware_aware_oracle_cost_model",
            "This is an oracle-model estimate, not hardware-native oracle synthesis.",
        ),
        _claim(
            "Dense explicit block encoding is a small correctness/resource-limit route",
            "supported_by_resource_model",
            input_root / "dense_explicit_qsvt_limit_study_v2",
            "Do not extrapolate dense unitary synthesis to IEEE scale.",
        ),
        _claim(
            "QSVT beats Ridge/Tikhonov",
            "unsupported_do_not_claim",
            input_root / "gate_level_qsvt_ieee14_solver",
            "Ridge/Tikhonov remains the reference target.",
        ),
        _claim(
            "Quantum speedup or advantage",
            "unsupported_do_not_claim",
            input_root,
            "No complexity or hardware execution evidence supports this.",
        ),
    ]


def remaining_assumption_rows(input_root: Path) -> list[dict[str, Any]]:
    ledger_path = input_root / "sparse_oracle_assumption_ledger" / "oracle_assumption_ledger.csv"
    if ledger_path.is_file():
        frame = pd.read_csv(ledger_path)
        return frame[
            [
                "assumption_name",
                "current_status",
                "missing_piece",
                "risk_level",
                "next_action",
            ]
        ].to_dict("records")
    return [
        {
            "assumption_name": "sparse oracle pathway",
            "current_status": "ledger_missing",
            "missing_piece": "Run sparse-oracle assumption ledger script",
            "risk_level": "high",
            "next_action": "Generate assumption ledger",
        }
    ]


def bottleneck_rows_from_outputs(input_root: Path) -> list[dict[str, Any]]:
    rows = [
        {
            "bottleneck": "norm recovery",
            "severity": "high",
            "evidence": str(input_root / "qsvt_norm_residual_gap_audit"),
            "recommendation": "Implement norm-estimation or amplitude-estimation routine.",
        },
        {
            "bottleneck": "readout",
            "severity": "moderate",
            "evidence": str(input_root / "qsvt_power_observable_readout"),
            "recommendation": "Keep claims to partial observables.",
        },
        {
            "bottleneck": "sparse oracle synthesis",
            "severity": "high",
            "evidence": str(input_root / "sparse_oracle_assumption_ledger"),
            "recommendation": "Separate oracle-model estimates from implemented circuits.",
        },
        {
            "bottleneck": "success probability",
            "severity": _success_severity(input_root),
            "evidence": str(input_root / "qsvt_success_amplification_cost"),
            "recommendation": "Report postselection and amplification query proxies.",
        },
        {
            "bottleneck": "polynomial/filter degree",
            "severity": _refinement_severity(input_root),
            "evidence": str(input_root / "qsvt_alpha_degree_residual_refinement"),
            "recommendation": "Separate requested degree from synthesized dense phase degree.",
        },
        {
            "bottleneck": "true QSVT degree control",
            "severity": _degree_control_severity(input_root),
            "evidence": str(input_root / "qsvt_degree_control_audit"),
            "recommendation": (
                "Require true synthesized high-degree rows before convergence claims."
            ),
        },
        {
            "bottleneck": "polynomial versus gate-level error",
            "severity": _polynomial_decomposition_severity(input_root),
            "evidence": str(input_root / "qsvt_polynomial_action_decomposition"),
            "recommendation": "Use polynomial-action rows to select gate-level validations.",
        },
        {
            "bottleneck": "hardware-aware oracle cost",
            "severity": _hardware_cost_severity(input_root),
            "evidence": str(input_root / "hardware_aware_oracle_cost_model"),
            "recommendation": "Report full IEEE-scale work as oracle-model estimates only.",
        },
        {
            "bottleneck": "observable accuracy",
            "severity": _observable_severity(input_root),
            "evidence": str(input_root / "qsvt_observable_error_decomposition"),
            "recommendation": "Report exact scaling error and shot-error components separately.",
        },
    ]
    return rows


CONTRIBUTION_STRENGTH_CATEGORIES = (
    "strong_implemented_evidence",
    "moderate_selected_subproblem_evidence",
    "resource_model_evidence",
    "assumption_only",
    "unsupported",
)


def contribution_strength_rows(input_root: Path) -> list[dict[str, Any]]:
    """Classify each contribution into one of CONTRIBUTION_STRENGTH_CATEGORIES.

    An evidence-backed contribution keeps its intended strength only when its artifact
    is present; otherwise it degrades to ``unsupported``. The two hard overclaims are
    always ``unsupported``.
    """

    specs = [
        (
            "Regularized bounded spectral target and Ridge/Tikhonov reference filter",
            "strong_implemented_evidence",
            Path("src/robust_qsvt_se/qsvt/filters.py"),
            "Implemented and unit-tested; Ridge/Tikhonov is the only reference target.",
        ),
        (
            "Scale-recovery protocol comparison with deployability classification",
            "strong_implemented_evidence",
            input_root / "qsvt_scale_recovery_protocols",
            "known_C is the only oracle-deployable protocol; best-scalar/ridge-norm are "
            "classical diagnostics and the norm/amplitude proxies are simulator-metadata proxies.",
        ),
        (
            "Bounded-target design comparison (boundedness versus residual)",
            "strong_implemented_evidence",
            input_root / "qsvt_bounded_target_design_study",
            "The bounded-filter constant C cancels in the known-C-rescaled update; it controls "
            "boundedness margin and success probability, not the residual.",
        ),
        (
            "Readout-aware power-observable protocols",
            "strong_implemented_evidence",
            input_root / "qsvt_power_observable_protocols",
            "Selected observables avoid full-vector readout; readout shots dominate query cost.",
        ),
        (
            "Residual-feasible configuration search (criteria-based subproblem selection)",
            "moderate_selected_subproblem_evidence",
            input_root / "qsvt_residual_feasible_config_search",
            "Subproblems are selected by numerical/metadata criteria, never gate-level QSVT "
            "performance.",
        ),
        (
            "Dense gate-level validation of selected residual-feasible configurations",
            "moderate_selected_subproblem_evidence",
            input_root / "qsvt_residual_feasible_gate_validation",
            "Selected dense 4x4 simulator validation only; not IEEE-scale hardware execution.",
        ),
        (
            "Polynomial-action decomposition diagnostics",
            "moderate_selected_subproblem_evidence",
            input_root / "qsvt_polynomial_action_decomposition",
            "Separates Ridge, exact target, bounded target, and polynomial-action residuals.",
        ),
        (
            "Actual small-circuit amplitude-estimation routines (Bernoulli and iterative QAE)",
            "strong_implemented_evidence",
            input_root / "qsvt_amplitude_estimation_routines",
            "Implemented and unit-tested simulator-backed routines that estimate the QSVT "
            "postselection success amplitude; distinct from the earlier cost proxy.",
        ),
        (
            "Amplitude-based update-scale (norm) recovery",
            "moderate_selected_subproblem_evidence",
            input_root / "qsvt_norm_recovery_from_amplitude",
            "Recovers only the update scale on a selected IEEE14-derived subproblem; the "
            "direction comes from the polynomial action.",
        ),
        (
            "Residual-feasibility test with amplitude-based norm recovery",
            "moderate_selected_subproblem_evidence",
            input_root / "qsvt_residual_feasible_with_amplitude_recovery",
            "Deployable feasibility uses only shot-based and iterative-QAE recovery; exact and "
            "best-scalar rows are diagnostics.",
        ),
        (
            "Gate-level validation with amplitude recovery",
            "moderate_selected_subproblem_evidence",
            input_root / "qsvt_gate_validation_with_amplitude_recovery",
            "Selected dense simulator validation with the implemented amplitude routine; not "
            "IEEE-scale hardware execution.",
        ),
        (
            "Observable-first QSVT solver prototype",
            "moderate_selected_subproblem_evidence",
            input_root / "qsvt_observable_first_solver",
            "Estimates selected power-system observables without full-vector recovery; readout "
            "remains a bottleneck.",
        ),
        (
            "Weighted singular-support diagnostic for target co-design",
            "strong_implemented_evidence",
            input_root / "qsvt_weighted_singular_support",
            "Implemented and unit-tested; ranks the singular directions that drive the Ridge "
            "update and the residual.",
        ),
        (
            "Co-designed QSVT-safe bounded-target family with residual-feasible deployable "
            "configurations",
            "strong_implemented_evidence",
            input_root / "qsvt_codesigned_bounded_target_study",
            "Singular-support-weighted bounded targets at low-to-moderate degree achieve a "
            "Ridge-aligned, QSVT-safe, residual-feasible update under deployable scale recovery "
            "on the selected subproblem.",
        ),
        (
            "Gate-level validation of the co-designed residual-feasible target",
            "moderate_selected_subproblem_evidence",
            input_root / "qsvt_codesigned_gate_validation",
            "Phase synthesis from co-designed coefficients plus dense simulator validation; not "
            "IEEE-scale hardware execution.",
        ),
        (
            "Observable-first QSVT solver (v2) with co-designed targets",
            "moderate_selected_subproblem_evidence",
            input_root / "qsvt_observable_first_solver_v2",
            "Co-designed targets sharpen selected observable estimates; top-k identification and "
            "selected functionals avoid full-vector readout.",
        ),
        (
            "Degree-window analysis mapping the safe co-designed degree band and overshoot onset",
            "strong_implemented_evidence",
            input_root / "qsvt_degree_window_overshoot",
            "Implemented and unit-tested; the singular-support-weighted and residual-aware "
            "targets stay feasible over a concrete odd-degree window and overshoot above it.",
        ),
        (
            "Degree-window dense gate-level validation",
            "moderate_selected_subproblem_evidence",
            input_root / "qsvt_degree_window_gate_validation",
            "Selected dense simulator validation of representative degree-window configurations; "
            "not IEEE-scale hardware execution.",
        ),
        (
            "Gate-level observable-first readout with filled gate values",
            "moderate_selected_subproblem_evidence",
            input_root / "qsvt_gate_observable_readout",
            "Selected power-system observables are read from the gate output; full-vector "
            "reconstruction stays not-recommended.",
        ),
        (
            "Robustness across criteria-selected IEEE14-derived subproblems",
            "moderate_selected_subproblem_evidence",
            input_root / "qsvt_codesigned_subproblem_robustness",
            "Subproblems chosen by numerical/metadata criteria; the solver transfers to a small "
            "family of well-conditioned, residual-supported blocks, not to rank-deficient ones.",
        ),
        (
            "Robustness gate-level validation of additional selected subproblems",
            "moderate_selected_subproblem_evidence",
            input_root / "qsvt_robustness_gate_validation",
            "Selected dense simulator validation of non-high-leverage candidates; not IEEE-scale "
            "hardware execution.",
        ),
        (
            "Consolidated selected-subproblem solver-prototype evidence package",
            "moderate_selected_subproblem_evidence",
            input_root / "qsvt_selected_solver_prototype",
            "Consolidates the degree window, gate validation, observable readout, and robustness "
            "into a single conservative selected-subproblem prototype classification.",
        ),
        (
            "Overshoot-mechanism diagnostic explaining the safe degree boundary",
            "strong_implemented_evidence",
            input_root / "qsvt_overshoot_mechanism_diagnostic",
            "Implemented and unit-tested; attributes the overshoot onset to a high-weight "
            "singular-support fit error rather than a boundedness violation or off-support peak.",
        ),
        (
            "Cross-case selected-subproblem robustness across IEEE14/30/57",
            "moderate_selected_subproblem_evidence",
            input_root / "qsvt_cross_case_codesigned_robustness",
            "Criteria-selected 4x4 blocks on three IEEE cases; the solver transfers to "
            "well-conditioned residual-supported blocks, not rank-deficient or random ones.",
        ),
        (
            "Cross-case dense gate-level validation of IEEE30/57 candidates",
            "moderate_selected_subproblem_evidence",
            input_root / "qsvt_cross_case_gate_validation",
            "Selected dense-simulator validation of cross-case candidates; not IEEE-scale "
            "hardware execution.",
        ),
        (
            "Cross-case gate-level observable readout",
            "moderate_selected_subproblem_evidence",
            input_root / "qsvt_cross_case_gate_observable_readout",
            "Selected observables are read from the gate output across cases; full-vector "
            "reconstruction is excluded.",
        ),
        (
            "Cross-case selected-subproblem solver-prototype evidence package",
            "moderate_selected_subproblem_evidence",
            input_root / "qsvt_cross_case_solver_prototype",
            "Consolidates the overshoot mechanism and cross-case robustness, gate validation, and "
            "observable readout into a single conservative cross-case classification.",
        ),
        (
            "IEEE118 selected-subproblem extension (robustness, gate validation, observables)",
            "moderate_selected_subproblem_evidence",
            input_root / "qsvt_ieee118_gate_validation",
            "Criteria-selected IEEE118 4x4 blocks transfer at the matrix level and survive dense "
            "gate validation; a larger benchmark selected-subproblem test, not IEEE118-scale "
            "solving.",
        ),
        (
            "Direction-resolved overshoot decomposition of the degree-47 boundary",
            "strong_implemented_evidence",
            input_root / "qsvt_direction_resolved_overshoot_decomposition",
            "Implemented and unit-tested; attributes the degree-47 onset to a leading-direction "
            "amplitude distortion at the near-boundary high-weight singular directions.",
        ),
        (
            "Final optional evidence package (IEEE118 extension + direction-resolved overshoot)",
            "moderate_selected_subproblem_evidence",
            input_root / "qsvt_final_optional_evidence_package",
            "Separates positive IEEE118 evidence, the degree-47 boundary mechanism, "
            "assumption-only claims, and unsupported claims into one conservative classification.",
        ),
        (
            "Hardware-aware sparse-oracle cost model",
            "resource_model_evidence",
            input_root / "hardware_aware_oracle_cost_model",
            "Oracle-model resource estimate, not hardware-native oracle synthesis.",
        ),
        (
            "Sparse-oracle prototype with precision sweep and dense comparison",
            "resource_model_evidence",
            input_root / "qsvt_sparse_oracle_prototype_strengthening",
            "Toy prototype plus cost model; full hardware-native sparse-oracle synthesis is "
            "still missing.",
        ),
        (
            "Full IEEE-scale QSVT sparse-oracle pathway",
            "assumption_only",
            input_root / "sparse_oracle_assumption_ledger",
            "Oracle-model assumptions only; not implemented circuits.",
        ),
        (
            "QSVT numerical superiority over Ridge/Tikhonov",
            "unsupported",
            input_root / "qsvt_residual_feasible_gate_validation",
            "Ridge/Tikhonov remains the reference; no superiority is demonstrated.",
        ),
        (
            "Quantum speedup or advantage",
            "unsupported",
            input_root,
            "No complexity or hardware-execution evidence supports this.",
        ),
    ]
    rows = []
    for contribution, intended, path, note in specs:
        present = path.exists()
        strength = intended if (intended == "unsupported" or present) else "unsupported"
        rows.append(
            {
                "contribution": contribution,
                "contribution_strength": strength,
                "evidence_path": str(path),
                "artifact_present": present,
                "note": note,
            }
        )
    return rows


def unresolved_gap_rows(
    input_root: Path,
    bottleneck_rows: list[dict[str, Any]],
    assumption_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in bottleneck_rows:
        if str(row["severity"]) in {"moderate", "high", "severe", "unknown"}:
            rows.append(
                {
                    "gap": str(row["bottleneck"]),
                    "severity": str(row["severity"]),
                    "evidence": str(row["evidence"]),
                    "recommendation": str(row["recommendation"]),
                }
            )
    for row in assumption_rows:
        if str(row.get("risk_level", "")).lower() == "high":
            rows.append(
                {
                    "gap": str(row.get("assumption_name", "sparse oracle assumption")),
                    "severity": "high",
                    "evidence": str(input_root / "sparse_oracle_assumption_ledger"),
                    "recommendation": str(row.get("next_action", "Resolve assumption")),
                }
            )
    feasible_path = (
        input_root / "qsvt_residual_feasible_config_search" / "residual_feasible_configs.csv"
    )
    if feasible_path.is_file():
        try:
            frame = pd.read_csv(feasible_path)
            no_feasible = frame.empty
        except Exception:
            no_feasible = True
        if no_feasible:
            rows.append(
                {
                    "gap": "no residual-feasible configuration (magnitude-limited scale recovery)",
                    "severity": "high",
                    "evidence": str(feasible_path),
                    "recommendation": (
                        "No deployable scale-recovery protocol recovers a Ridge-feasible "
                        "update magnitude; keep this as an engineering negative result."
                    ),
                }
            )
    return rows


SOLVER_READINESS_CATEGORIES = (
    "not_ready",
    "draft_ready_with_major_limitations",
    "engineering_framework_ready_with_negative_result",
    "solver_prototype_ready_selected_subproblems",
    "solver_prototype_ready_single_selected_subproblem",
    "solver_prototype_ready_selected_subproblem_family",
    "solver_prototype_ready_ieee14_selected_family",
    "solver_prototype_ready_cross_case_partial",
    "solver_prototype_ready_cross_case_selected_family",
    "solver_prototype_ready_cross_case_plus_ieee118_selected",
    "solver_prototype_ready_cross_case_with_ieee118_boundary",
    "submission_ready_after_revision",
    "submission_ready",
)

AMPLITUDE_RECOVERY_LAYERS = (
    "actual_amplitude_estimation_routines",
    "norm_recovery_from_amplitude",
    "residual_feasible_with_amplitude_recovery",
    "gate_validation_with_amplitude_recovery",
    "observable_first_qsvt_solver",
)

CODESIGNED_LAYERS = (
    "weighted_singular_support",
    "codesigned_bounded_target_study",
    "residual_feasible_codesigned_search",
    "codesigned_gate_validation",
    "observable_first_solver_v2",
)

CONSOLIDATION_LAYERS = (
    "degree_window_overshoot",
    "degree_window_gate_validation",
    "gate_observable_readout",
    "codesigned_subproblem_robustness",
    "robustness_gate_validation",
    "selected_solver_prototype_package",
)

CROSS_CASE_LAYERS = (
    "cross_case_solver_audit",
    "overshoot_mechanism_diagnostic",
    "cross_case_codesigned_robustness",
    "cross_case_gate_validation",
    "cross_case_gate_observable_readout",
    "cross_case_solver_prototype_package",
)

FINAL_OPTIONAL_LAYERS = (
    "ieee118_extension_audit",
    "direction_resolved_overshoot_decomposition",
    "ieee118_selected_robustness",
    "ieee118_gate_validation",
    "ieee118_gate_observable_readout",
    "final_optional_evidence_package",
)

_DEPLOYABLE_NORM_METHODS = (
    "bernoulli_success_amplitude",
    "iterative_qae",
    "hybrid_known_C_amplitude",
)


def _csv_has_data_rows(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return not pd.read_csv(path).empty
    except Exception:
        return False


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def amplitude_recovery_gap_rows(input_root: Path) -> list[dict[str, Any]]:
    """Summarize whether actual amplitude/norm recovery closes the residual-feasibility gap."""

    base = input_root / "qsvt_residual_feasible_with_amplitude_recovery"
    all_path = base / "all_amplitude_recovery_configs.csv"
    if not all_path.is_file():
        return [
            {
                "metric": "amplitude_recovery_search",
                "value": "absent",
                "detail": "Run the residual-feasible search with amplitude recovery.",
            }
        ]
    try:
        frame = pd.read_csv(all_path)
    except Exception:
        frame = pd.DataFrame()
    deployable = (
        frame[frame["norm_recovery_method"].isin(_DEPLOYABLE_NORM_METHODS)]
        if "norm_recovery_method" in frame.columns
        else pd.DataFrame()
    )
    best_deployable_ratio = (
        float(pd.to_numeric(deployable["residual_ratio_vs_no_update"], errors="coerce").min())
        if not deployable.empty
        else float("nan")
    )
    best_direction = (
        float(pd.to_numeric(frame["direction_error_vs_ridge"], errors="coerce").min())
        if "direction_error_vs_ridge" in frame.columns and not frame.empty
        else float("nan")
    )
    deployable_feasible = _csv_has_data_rows(base / "residual_feasible_configs.csv")
    diagnostic_feasible = _csv_has_data_rows(base / "diagnostic_feasible_configs.csv")
    if deployable_feasible:
        bottleneck = "none (a deployable amplitude-recovered configuration is residual-feasible)"
    elif best_direction <= 0.1:
        bottleneck = (
            "scale recovery is not the binding magnitude constraint once direction agrees; the "
            "residual gap is set by the polynomial target/direction, not amplitude sampling"
        )
    else:
        bottleneck = "direction error vs Ridge (bounded QSVT output points away from Ridge)"
    return [
        {
            "metric": "deployable_residual_feasible",
            "value": "yes" if deployable_feasible else "no",
            "detail": f"best deployable residual_ratio_vs_no_update = {best_deployable_ratio:.6g}",
        },
        {
            "metric": "diagnostic_feasible_only",
            "value": "yes" if diagnostic_feasible and not deployable_feasible else "no",
            "detail": "feasibility reached only by non-deployable diagnostics",
        },
        {
            "metric": "best_direction_error_vs_ridge",
            "value": f"{best_direction:.6g}",
            "detail": "minimum bounded QSVT direction error across the amplitude-recovery search",
        },
        {
            "metric": "amplitude_recovery_gap_closed",
            "value": "yes" if deployable_feasible else "no",
            "detail": bottleneck,
        },
    ]


def _codesigned_gate_feasible(input_root: Path) -> bool:
    gate_path = (
        input_root / "qsvt_codesigned_gate_validation" / "codesigned_gate_validation_results.csv"
    )
    if not gate_path.is_file():
        return False
    try:
        gate_frame = pd.read_csv(gate_path)
    except Exception:
        return False
    return bool(
        gate_frame.get("residual_feasible_after_gate", pd.Series(dtype=bool))
        .astype(str)
        .str.lower()
        .eq("true")
        .any()
    )


def _amplitude_gate_feasible(input_root: Path) -> bool:
    gate_path = (
        input_root
        / "qsvt_gate_validation_with_amplitude_recovery"
        / "gate_amplitude_recovery_results.csv"
    )
    if not gate_path.is_file():
        return False
    try:
        gate_frame = pd.read_csv(gate_path)
    except Exception:
        return False
    return bool(
        gate_frame.get("residual_feasible_after_gate", pd.Series(dtype=bool))
        .astype(str)
        .str.lower()
        .eq("true")
        .any()
    )


def solver_readiness_classification(evidence_rows: list[dict[str, Any]], input_root: Path) -> str:
    """Conservative solver-readiness classification.

    The final-optional pathway (IEEE118 selected-block extension plus the direction-resolved
    overshoot decomposition) takes highest precedence when all of its layers are present: IEEE118
    blocks surviving gate validation upgrade the cross-case family to a cross-case-plus-IEEE118
    prototype, IEEE118 candidates that do not survive mark an IEEE118 boundary, and an unrun
    IEEE118 keeps the cross-case classification. With those layers absent, the cross-case pathway
    (overshoot mechanism plus IEEE14/30/57 cross-case robustness, gate validation, and observable
    readout) takes precedence: it reports whether the selected-subproblem prototype is IEEE14-only,
    cross-case partial, or a cross-case selected family by how many of IEEE30/IEEE57 survive gate
    validation. With the cross-case layers absent, the consolidation pathway takes precedence
    and distinguishes a single-selected-subproblem prototype from a selected-subproblem-family
    prototype. With both absent, the co-designed-target pathway takes precedence, and otherwise
    the classification falls back to the amplitude-recovery pathway.
    """

    present = {
        str(row["evidence_layer"])
        for row in evidence_rows
        if bool(row.get("artifact_present", False))
    }
    if set(FINAL_OPTIONAL_LAYERS).issubset(present):
        return final_optional_evidence_classification(input_root)
    if set(CROSS_CASE_LAYERS).issubset(present):
        return cross_case_solver_prototype_classification(input_root)
    if set(CONSOLIDATION_LAYERS).issubset(present):
        return selected_solver_prototype_classification(input_root)
    if set(CODESIGNED_LAYERS).issubset(present):
        deployable_feasible = _csv_has_data_rows(
            input_root
            / "qsvt_residual_feasible_codesigned_search"
            / "deployable_residual_feasible_configs.csv"
        )
        if not deployable_feasible:
            return "engineering_framework_ready_with_negative_result"
        if _codesigned_gate_feasible(input_root):
            return "solver_prototype_ready_selected_subproblems"
        return "draft_ready_with_major_limitations"

    if not set(AMPLITUDE_RECOVERY_LAYERS).issubset(present):
        return "not_ready"
    search_base = input_root / "qsvt_residual_feasible_with_amplitude_recovery"
    deployable_feasible = _csv_has_data_rows(search_base / "residual_feasible_configs.csv")
    if deployable_feasible and _amplitude_gate_feasible(input_root):
        return "solver_prototype_ready_selected_subproblems"
    return "draft_ready_with_major_limitations"


def codesigned_target_gap_rows(input_root: Path) -> list[dict[str, Any]]:
    """Summarize whether co-designed targets close the residual-feasibility gap."""

    base = input_root / "qsvt_residual_feasible_codesigned_search"
    all_path = base / "all_codesigned_configs.csv"
    if not all_path.is_file():
        return [
            {
                "metric": "codesigned_target_search",
                "value": "absent",
                "detail": "Run the residual-feasible co-designed search.",
            }
        ]
    try:
        frame = pd.read_csv(all_path)
    except Exception:
        frame = pd.DataFrame()
    deployable_feasible = _csv_has_data_rows(base / "deployable_residual_feasible_configs.csv")
    diagnostic_feasible = _csv_has_data_rows(base / "diagnostic_feasible_configs.csv")
    gate_feasible = _codesigned_gate_feasible(input_root)
    best_ratio = (
        float(pd.to_numeric(frame["residual_ratio_vs_no_update"], errors="coerce").min())
        if "residual_ratio_vs_no_update" in frame.columns and not frame.empty
        else float("nan")
    )
    best_direction = (
        float(pd.to_numeric(frame["direction_error_vs_ridge"], errors="coerce").min())
        if "direction_error_vs_ridge" in frame.columns and not frame.empty
        else float("nan")
    )
    if deployable_feasible and gate_feasible:
        bottleneck = (
            "closed: a co-designed QSVT-safe deployable target is residual-feasible and survives "
            "gate validation; readout of the output direction remains the outstanding bottleneck"
        )
    elif deployable_feasible:
        bottleneck = (
            "partially closed: a deployable co-designed target is residual-feasible at the matrix "
            "level but has not been confirmed by gate validation"
        )
    else:
        bottleneck = (
            "not closed: no co-designed QSVT-safe deployable target met the residual, direction, "
            "and success-amplitude thresholds"
        )
    return [
        {
            "metric": "deployable_residual_feasible_codesigned",
            "value": "yes" if deployable_feasible else "no",
            "detail": f"best residual_ratio_vs_no_update = {best_ratio:.6g}",
        },
        {
            "metric": "diagnostic_feasible_codesigned",
            "value": "yes" if diagnostic_feasible else "no",
            "detail": "feasibility reached by the non-deployable best-scalar lower bound",
        },
        {
            "metric": "codesigned_gate_feasible",
            "value": "yes" if gate_feasible else "no",
            "detail": "co-designed target residual-feasible after dense gate-level validation",
        },
        {
            "metric": "best_direction_error_vs_ridge_codesigned",
            "value": f"{best_direction:.6g}",
            "detail": "minimum bounded QSVT direction error across the co-designed search",
        },
        {
            "metric": "codesigned_gap_closed",
            "value": "yes" if (deployable_feasible and gate_feasible) else "no",
            "detail": bottleneck,
        },
    ]


def selected_solver_prototype_rows(input_root: Path) -> list[dict[str, Any]]:
    """Summarize the consolidated selected-subproblem solver prototype.

    Reads the degree-window, gate-validation, observable-readout, and robustness outputs to
    report the safe degree window, the residual ratio, gate survival, gate-confirmed
    observables, robustness breadth, and the final prototype classification.
    """

    classification = selected_solver_prototype_classification(input_root)

    degree_window = _read_csv(
        input_root / "qsvt_degree_window_overshoot" / "feasible_degree_window.csv"
    )
    feasible_degrees = (
        sorted({int(v) for v in pd.to_numeric(degree_window["degree"], errors="coerce").dropna()})
        if not degree_window.empty
        else []
    )
    best_ratio = (
        float(pd.to_numeric(degree_window["residual_ratio_vs_no_update"], errors="coerce").min())
        if not degree_window.empty
        else float("nan")
    )
    degree_gate = _read_csv(
        input_root / "qsvt_degree_window_gate_validation" / "degree_window_gate_results.csv"
    )
    gate_feasible = (
        bool(
            degree_gate.get("residual_feasible_after_gate", pd.Series(dtype=bool))
            .astype(str)
            .str.lower()
            .eq("true")
            .any()
        )
        if not degree_gate.empty
        else False
    )
    robustness_gate = _read_csv(
        input_root / "qsvt_robustness_gate_validation" / "robustness_gate_results.csv"
    )
    surviving_modes = []
    if not robustness_gate.empty and "residual_feasible_after_gate" in robustness_gate.columns:
        mask = robustness_gate["residual_feasible_after_gate"].astype(str).str.lower().eq("true")
        key = "selection_mode" if "selection_mode" in robustness_gate.columns else "subproblem_id"
        surviving_modes = sorted(
            set(robustness_gate[mask].get(key, pd.Series(dtype=str)).astype(str))
        )
    observables = _read_csv(
        input_root / "qsvt_gate_observable_readout" / "gate_observable_values.csv"
    )
    confirmed_observables = (
        sorted(set(observables["observable_name"].astype(str))) if not observables.empty else []
    )
    return [
        {
            "aspect": "final_prototype_classification",
            "value": classification,
            "detail": "consolidation pathway: degree window + gate + observable readout + "
            "robustness",
        },
        {
            "aspect": "safe_degree_window",
            "value": (
                f"{feasible_degrees[0]}..{feasible_degrees[-1]}" if feasible_degrees else "none"
            ),
            "detail": "odd degrees that stay QSVT-safe, Ridge-aligned, and residual-feasible",
        },
        {
            "aspect": "best_residual_ratio_vs_no_update",
            "value": f"{best_ratio:.6g}",
            "detail": "best deployable amplitude/known-C residual across the feasible degree "
            "window",
        },
        {
            "aspect": "degree_window_gate_feasible",
            "value": "yes" if gate_feasible else "no",
            "detail": "co-designed degree-window configurations survive dense gate validation",
        },
        {
            "aspect": "robustness_gate_surviving_modes",
            "value": ", ".join(surviving_modes) or "none",
            "detail": "criteria-selected subproblem modes residual-feasible after gate validation",
        },
        {
            "aspect": "gate_confirmed_observables",
            "value": ", ".join(confirmed_observables) or "none",
            "detail": "observables read out from the gate output and matched against Ridge",
        },
    ]


def cross_case_solver_prototype_rows(input_root: Path) -> list[dict[str, Any]]:
    """Summarize the cross-case selected-subproblem solver prototype.

    Reads the overshoot-mechanism, cross-case robustness, cross-case gate-validation, and
    cross-case observable-readout outputs to report which IEEE cases transfer, the safe degree
    boundary and overshoot mechanism, cross-case observable transfer, and the final cross-case
    classification.
    """

    classification = cross_case_solver_prototype_classification(input_root)

    by_case = _read_csv(
        input_root / "qsvt_cross_case_codesigned_robustness" / "cross_case_by_case_summary.csv"
    )
    cases_tested = sorted(set(by_case["case"].astype(str))) if not by_case.empty else []

    gate = _read_csv(input_root / "qsvt_cross_case_gate_validation" / "cross_case_gate_results.csv")
    surviving_cases: list[str] = []
    if not gate.empty and "residual_feasible_after_gate" in gate.columns:
        mask = gate["residual_feasible_after_gate"].astype(str).str.lower().eq("true")
        surviving_cases = sorted(set(gate[mask].get("case", pd.Series(dtype=str)).astype(str)))

    overshoot = _read_csv(
        input_root / "qsvt_overshoot_mechanism_diagnostic" / "overshoot_mechanism_summary.csv"
    )
    safe_boundary = "none"
    overshoot_mechanism = "not_mapped"
    if not overshoot.empty and "overshoot_failure_mode" in overshoot.columns:
        frame = overshoot.copy()
        frame["degree"] = pd.to_numeric(frame["degree"], errors="coerce")
        overshoot_modes = {
            "off_support_peak_breaks_direction",
            "singular_support_error_breaks_direction",
            "boundedness_violation",
        }
        feasible_modes = {"no_overshoot_residual_feasible", "off_support_peak_but_still_feasible"}
        overshoot_degrees = {
            int(v)
            for v in frame[frame["overshoot_failure_mode"].isin(overshoot_modes)]["degree"].dropna()
        }
        clean = {
            int(v)
            for v in frame[frame["overshoot_failure_mode"].isin(feasible_modes)]["degree"].dropna()
        } - overshoot_degrees
        onset = min(overshoot_degrees) if overshoot_degrees else None
        safest = max(clean) if clean else None
        if safest is not None:
            safe_boundary = f"safe up to degree {safest}; overshoot onset {onset}"
        failing = frame[frame["overshoot_failure_mode"].isin(overshoot_modes)]
        if not failing.empty:
            overshoot_mechanism = str(failing["overshoot_failure_mode"].value_counts().idxmax())

    observables = _read_csv(
        input_root
        / "qsvt_cross_case_gate_observable_readout"
        / "cross_case_gate_observable_values.csv"
    )
    observable_cases = sorted(set(observables["case"].astype(str))) if not observables.empty else []
    confirmed_observables = (
        sorted(set(observables["observable_name"].astype(str))) if not observables.empty else []
    )
    return [
        {
            "aspect": "final_cross_case_classification",
            "value": classification,
            "detail": "cross-case pathway: overshoot mechanism + IEEE14/30/57 robustness, gate "
            "validation, and observable readout",
        },
        {
            "aspect": "cases_tested",
            "value": ", ".join(cases_tested) or "none",
            "detail": "IEEE benchmark cases evaluated for selected-subproblem robustness",
        },
        {
            "aspect": "gate_surviving_cases",
            "value": ", ".join(surviving_cases) or "none",
            "detail": "cases with a residual-feasible result after dense gate validation",
        },
        {
            "aspect": "safe_degree_boundary",
            "value": safe_boundary,
            "detail": "largest safe degree and the overshoot onset from the mechanism diagnostic",
        },
        {
            "aspect": "overshoot_mechanism",
            "value": overshoot_mechanism,
            "detail": "dominant overshoot failure mode at and above the boundary degree",
        },
        {
            "aspect": "cross_case_observables",
            "value": f"{', '.join(confirmed_observables) or 'none'} (cases: "
            f"{', '.join(observable_cases) or 'none'})",
            "detail": "observables read from the gate output and matched against Ridge across "
            "cases",
        },
    ]


def final_optional_evidence_rows(input_root: Path) -> list[dict[str, Any]]:
    """Summarize the final optional evidence package (IEEE118 + direction-resolved overshoot).

    Reports the final-optional classification, whether IEEE118 selected blocks survive gate
    validation, the IEEE118 robustness breadth, the degree-47 overshoot mechanism, the recommended
    degree cutoff, and which IEEE118 observables were gate-confirmed.
    """

    classification = final_optional_evidence_classification(input_root)

    robustness = _read_csv(
        input_root / "qsvt_ieee118_selected_robustness" / "ieee118_by_subproblem_summary.csv"
    )
    feasible_modes: list[str] = []
    if not robustness.empty:
        non_control = robustness[robustness["is_control"].astype(str).str.lower() != "true"]
        feasible = non_control[
            non_control["any_residual_feasible"].astype(str).str.lower() == "true"
        ]
        feasible_modes = sorted(set(feasible["selection_mode"].astype(str)))

    gate = _read_csv(input_root / "qsvt_ieee118_gate_validation" / "ieee118_gate_results.csv")
    survivors: list[str] = []
    if not gate.empty and "residual_feasible_after_gate" in gate.columns:
        mask = gate["residual_feasible_after_gate"].astype(str).str.lower().eq("true")
        survivors = sorted(set(gate[mask].get("subproblem_id", pd.Series(dtype=str)).astype(str)))

    summary = _read_csv(
        input_root
        / "qsvt_direction_resolved_overshoot_decomposition"
        / "direction_resolved_error_summary.csv"
    )
    overshoot_mechanism = "not_mapped"
    reference_cutoff = "none"
    if not summary.empty and "failure_mechanism" in summary.columns:
        frame = summary.copy()
        frame["degree"] = pd.to_numeric(frame["degree"], errors="coerce")
        degree_47 = frame[frame["degree"] == 47]
        failing_47 = degree_47[degree_47["failure_mechanism"].astype(str) != "no_failure"]
        if not failing_47.empty:
            overshoot_mechanism = str(failing_47["failure_mechanism"].value_counts().idxmax())
        ieee14 = frame[frame["case"].astype(str) == "ieee14"]
        if not ieee14.empty:
            ieee14_failing = {
                int(v)
                for v in ieee14[ieee14["failure_mechanism"].astype(str) != "no_failure"][
                    "degree"
                ].dropna()
            }
            clean = {int(v) for v in ieee14["degree"].dropna()} - ieee14_failing
            if clean:
                reference_cutoff = f"degree {max(clean)} (IEEE14 reference; onset 47)"

    observables = _read_csv(
        input_root / "qsvt_ieee118_gate_observable_readout" / "ieee118_gate_observable_values.csv"
    )
    confirmed_observables = (
        sorted(set(observables["observable_name"].astype(str))) if not observables.empty else []
    )
    return [
        {
            "aspect": "final_optional_classification",
            "value": classification,
            "detail": "final-optional pathway: IEEE118 selected-block extension + "
            "direction-resolved overshoot decomposition",
        },
        {
            "aspect": "ieee118_extends_under_gate",
            "value": "yes" if survivors else "no",
            "detail": f"IEEE118 subproblems surviving gate validation: "
            f"{', '.join(survivors) or 'none'}",
        },
        {
            "aspect": "ieee118_feasible_modes",
            "value": ", ".join(feasible_modes) or "none",
            "detail": "criteria-selected IEEE118 modes residual-feasible at the matrix level",
        },
        {
            "aspect": "degree_47_overshoot_mechanism",
            "value": overshoot_mechanism,
            "detail": "dominant singular-direction mechanism of the degree-47 overshoot onset",
        },
        {
            "aspect": "recommended_degree_cutoff",
            "value": reference_cutoff,
            "detail": "manuscript-ready safe degree cutoff from the direction-resolved analysis",
        },
        {
            "aspect": "ieee118_gate_confirmed_observables",
            "value": ", ".join(confirmed_observables) or "none",
            "detail": "observables read from the IEEE118 gate output and matched against Ridge",
        },
    ]


def full_solver_readiness_rows(
    input_root: Path, evidence_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    present = {
        str(row["evidence_layer"]): bool(row.get("artifact_present", False))
        for row in evidence_rows
    }
    classification = solver_readiness_classification(evidence_rows, input_root)
    rows = [
        {
            "component": layer,
            "artifact_present": present.get(layer, False),
            "role": "amplitude-recovery pathway layer",
        }
        for layer in AMPLITUDE_RECOVERY_LAYERS
    ]
    rows.extend(
        {
            "component": layer,
            "artifact_present": present.get(layer, False),
            "role": "co-designed-target pathway layer",
        }
        for layer in CODESIGNED_LAYERS
    )
    rows.extend(
        {
            "component": layer,
            "artifact_present": present.get(layer, False),
            "role": "consolidation pathway layer",
        }
        for layer in CONSOLIDATION_LAYERS
    )
    rows.extend(
        {
            "component": layer,
            "artifact_present": present.get(layer, False),
            "role": "cross-case pathway layer",
        }
        for layer in CROSS_CASE_LAYERS
    )
    rows.extend(
        {
            "component": layer,
            "artifact_present": present.get(layer, False),
            "role": "final-optional (IEEE118 + direction-resolved) pathway layer",
        }
        for layer in FINAL_OPTIONAL_LAYERS
    )
    rows.append(
        {
            "component": "solver_readiness_classification",
            "artifact_present": classification,
            "role": "conservative overall classification",
        }
    )
    return rows


def _strongest_contribution_summary(
    contribution_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
) -> str:
    def names(strength: str) -> list[str]:
        return [
            str(row["contribution"])
            for row in contribution_rows
            if row["contribution_strength"] == strength and bool(row["artifact_present"])
        ]

    strong = names("strong_implemented_evidence")
    moderate = names("moderate_selected_subproblem_evidence")
    resource = names("resource_model_evidence")
    unsupported = [
        str(row["claim"]) for row in claim_rows if row["claim_class"] == "unsupported_do_not_claim"
    ]
    strongest = (
        strong[0] if strong else (moderate[0] if moderate else "regularized filter reference")
    )
    return "\n".join(
        [
            "# Strongest-Contribution Summary",
            "",
            CLAIM_REPORT_BOUNDARY,
            "",
            "## 1. What contribution is now strongest",
            f"- {strongest}",
            "- Strong implemented-evidence contributions:",
            *[f"  - {item}" for item in strong],
            "",
            "## 2. What contribution remains weak",
            "- Selected-subproblem-only evidence (not IEEE-scale):",
            *[f"  - {item}" for item in moderate],
            "- Resource/oracle-model-only evidence:",
            *[f"  - {item}" for item in resource],
            "",
            "## 3. Negative result to keep as an engineering finding",
            "- On the criteria-selected IEEE14-derived subproblems no deployable or proxy "
            "scale-recovery protocol produces a Ridge-feasible update: the bounded QSVT "
            "direction agrees with Ridge at low degree but no implementable magnitude "
            "recovery closes the residual, and high degree degrades the direction. The "
            "bounded-target scaling constant C does not change this (it only sets boundedness "
            "and success probability). This magnitude/direction-limited result is a legitimate "
            "engineering finding, not a defect.",
            "",
            "## 4. What should not be claimed",
            *[f"- {item}" for item in unsupported],
            "- Full IEEE-scale gate-level solving, solved readout, or hardware execution.",
            "",
            "## 5. What must be done before TQE submission",
            *[f"- {row['gap']}: {row['severity']}" for row in gap_rows],
            "",
        ]
    )


def _write_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    *,
    evidence_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
    assumption_rows: list[dict[str, Any]],
    bottleneck_rows: list[dict[str, Any]],
    contribution_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    amplitude_gap_rows: list[dict[str, Any]],
    codesigned_gap_rows: list[dict[str, Any]],
    prototype_rows: list[dict[str, Any]],
    cross_case_prototype_rows: list[dict[str, Any]],
    final_optional_rows: list[dict[str, Any]],
    solver_readiness_rows: list[dict[str, Any]],
    solver_readiness: str,
) -> dict[str, Path]:
    evidence_path = output_dir / "evidence_layer_table.csv"
    claim_path = output_dir / "claim_support_table.csv"
    assumption_path = output_dir / "remaining_assumption_table.csv"
    bottleneck_path = output_dir / "bottleneck_ranking.csv"
    contribution_path = output_dir / "contribution_strength_table.csv"
    gap_path = output_dir / "unresolved_gap_table.csv"
    amplitude_gap_path = output_dir / "amplitude_recovery_gap_table.csv"
    codesigned_gap_path = output_dir / "codesigned_target_gap_table.csv"
    prototype_path = output_dir / "selected_solver_prototype_table.csv"
    cross_case_prototype_path = output_dir / "cross_case_solver_prototype_table.csv"
    final_optional_path = output_dir / "final_optional_evidence_table.csv"
    solver_readiness_path = output_dir / "full_solver_readiness_table.csv"
    summary_path = output_dir / "paper_ready_summary.md"
    tqe_path = output_dir / "tqe_readiness_summary.md"
    limitation_path = output_dir / "limitation_ready_summary.md"
    refinement_path = output_dir / "refinement_update_summary.md"
    strongest_path = output_dir / "strongest_contribution_summary.md"
    pd.DataFrame(evidence_rows).to_csv(evidence_path, index=False)
    pd.DataFrame(claim_rows).to_csv(claim_path, index=False)
    pd.DataFrame(assumption_rows).to_csv(assumption_path, index=False)
    pd.DataFrame(bottleneck_rows).to_csv(bottleneck_path, index=False)
    pd.DataFrame(contribution_rows).to_csv(contribution_path, index=False)
    pd.DataFrame(gap_rows).to_csv(gap_path, index=False)
    pd.DataFrame(amplitude_gap_rows).to_csv(amplitude_gap_path, index=False)
    pd.DataFrame(codesigned_gap_rows).to_csv(codesigned_gap_path, index=False)
    pd.DataFrame(prototype_rows).to_csv(prototype_path, index=False)
    pd.DataFrame(cross_case_prototype_rows).to_csv(cross_case_prototype_path, index=False)
    pd.DataFrame(final_optional_rows).to_csv(final_optional_path, index=False)
    pd.DataFrame(solver_readiness_rows).to_csv(solver_readiness_path, index=False)
    summary_path.write_text(
        _paper_ready_summary(evidence_rows, claim_rows, bottleneck_rows),
        encoding="utf-8",
    )
    strongest_path.write_text(
        _strongest_contribution_summary(contribution_rows, gap_rows, claim_rows),
        encoding="utf-8",
    )
    tqe_path.write_text(
        _tqe_readiness_summary(
            evidence_rows,
            claim_rows,
            assumption_rows,
            bottleneck_rows,
            solver_readiness=solver_readiness,
            amplitude_gap_rows=amplitude_gap_rows,
        ),
        encoding="utf-8",
    )
    limitation_path.write_text(
        _limitation_ready_summary(evidence_rows, claim_rows, assumption_rows, bottleneck_rows),
        encoding="utf-8",
    )
    refinement_path.write_text(
        _refinement_update_summary(evidence_rows, claim_rows, bottleneck_rows),
        encoding="utf-8",
    )
    manifest = write_manifest(
        output_dir,
        artifacts={
            "evidence_layer_table": str(evidence_path),
            "claim_support_table": str(claim_path),
            "remaining_assumption_table": str(assumption_path),
            "bottleneck_ranking": str(bottleneck_path),
            "contribution_strength_table": str(contribution_path),
            "unresolved_gap_table": str(gap_path),
            "amplitude_recovery_gap_table": str(amplitude_gap_path),
            "codesigned_target_gap_table": str(codesigned_gap_path),
            "selected_solver_prototype_table": str(prototype_path),
            "cross_case_solver_prototype_table": str(cross_case_prototype_path),
            "final_optional_evidence_table": str(final_optional_path),
            "full_solver_readiness_table": str(solver_readiness_path),
            "paper_ready_summary": str(summary_path),
            "tqe_readiness_summary": str(tqe_path),
            "limitation_ready_summary": str(limitation_path),
            "refinement_update_summary": str(refinement_path),
            "strongest_contribution_summary": str(strongest_path),
        },
        input_config=resolved,
        claim_boundary=CLAIM_REPORT_BOUNDARY,
    )
    return {
        "manifest": manifest,
        "evidence_layer_table": evidence_path,
        "claim_support_table": claim_path,
        "remaining_assumption_table": assumption_path,
        "bottleneck_ranking": bottleneck_path,
        "contribution_strength_table": contribution_path,
        "unresolved_gap_table": gap_path,
        "amplitude_recovery_gap_table": amplitude_gap_path,
        "codesigned_target_gap_table": codesigned_gap_path,
        "selected_solver_prototype_table": prototype_path,
        "cross_case_solver_prototype_table": cross_case_prototype_path,
        "final_optional_evidence_table": final_optional_path,
        "full_solver_readiness_table": solver_readiness_path,
        "paper_ready_summary": summary_path,
        "tqe_readiness_summary": tqe_path,
        "limitation_ready_summary": limitation_path,
        "refinement_update_summary": refinement_path,
        "strongest_contribution_summary": strongest_path,
    }


def _paper_ready_summary(
    evidence_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
    bottleneck_rows: list[dict[str, Any]],
) -> str:
    classification = tqe_readiness_classification(evidence_rows, bottleneck_rows)
    confirmed = [
        row["claim"]
        for row in claim_rows
        if row["claim_class"]
        in {"supported_by_implementation", "supported_by_controlled_experiment"}
    ]
    unsupported = [
        row["claim"] for row in claim_rows if row["claim_class"] == "unsupported_do_not_claim"
    ]
    present_count = sum(bool(row["artifact_present"]) for row in evidence_rows)
    return "\n".join(
        [
            "# QSVT Evidence and Claim-Support Report",
            "",
            CLAIM_REPORT_BOUNDARY,
            "",
            f"## TQE Readiness Classification: {classification}",
            "",
            "## What Is Now Confirmed",
            *[f"- {item}" for item in confirmed],
            "",
            "## What Is Implemented",
            "- Regularized spectral filters and Ridge/Tikhonov reference calculations.",
            "- Dense scalar, diagonal, tiny, and selected-subproblem gate-level QSVT simulations.",
            "- Degree-control, polynomial-action, norm/residual, readout, and success-cost audits.",
            "",
            "## What Is Demonstrated On Toy Systems",
            "- Scalar/diagonal convention checks, tiny gate-level solver, and toy "
            "sparse-oracle gates.",
            "",
            "## What Is Demonstrated On IEEE-Derived Subproblems",
            "- Selected IEEE14-derived 4x4 dense QSVT update diagnostics and "
            "polynomial-action rows.",
            "",
            "## What Is Only Resource Or Oracle-Model Evidence",
            "- IEEE-scale sparse-oracle pathway and hardware-aware oracle cost estimates.",
            "",
            "## What Assumptions Are Required For Full IEEE-Scale QSVT",
            "- Sparse row/column lookup, value loading, fixed-point rotations, residual state "
            "preparation, amplitude amplification, norm recovery, and observable readout.",
            "",
            "## What Is Partially Supported",
            "- Dense explicit block encoding supports small-circuit evidence and "
            "resource-limit diagnostics.",
            "- Power-observable readout supports selected quantities, not full-vector tomography.",
            "- True-degree refinement and selected-subproblem runs improve the diagnostic basis "
            "but do not make QSVT numerically superior to Ridge.",
            "",
            "## What Claims Are Unsupported",
            *[f"- {item}" for item in unsupported],
            "",
            "## What Must Be Fixed Before TQE Submission",
            *[f"- {row['bottleneck']}: {row['severity']}" for row in bottleneck_rows],
            "",
            "## Recommended Manuscript Contribution Statement",
            "- We present a readout-aware dense-to-oracle QSVT engineering pathway for "
            "regularized IEEE-derived state-estimation updates, with degree-controlled "
            "polynomial-action diagnostics and hardware-aware sparse-oracle resource estimates.",
            "",
            "## Recommended Limitation Paragraph",
            "- The study provides dense simulator evidence on selected small subproblems and "
            "oracle-model resource estimates for larger IEEE cases. It does not demonstrate "
            "full IEEE-scale hardware execution, quantum speedup, QSVT superiority over "
            "Ridge/Tikhonov, full nonlinear AC solving, solved readout, or real PMU/SCADA "
            "validation.",
            "",
            "## Recommended Reviewer-Risk List",
            "- True high-degree synthesis may remain backend-limited for some alpha/degree pairs.",
            "- Oracle costs depend on sparse-access and state-preparation assumptions.",
            "- Dense 4x4 gate-level evidence does not establish IEEE-scale circuit feasibility.",
            "- Readout, norm recovery, and success probability remain major cost bottlenecks.",
            "",
            f"Evidence layers present: {present_count} / {len(evidence_rows)}.",
            "",
        ]
    )


def _refinement_update_summary(
    evidence_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
    bottleneck_rows: list[dict[str, Any]],
) -> str:
    present = {
        str(row["evidence_layer"])
        for row in evidence_rows
        if bool(row.get("artifact_present", False))
    }
    strengthened = [
        row["claim"]
        for row in claim_rows
        if row["claim_class"] == "supported_by_controlled_experiment"
        and bool(row.get("artifact_present", False))
    ]
    partial = [
        "Residual accuracy remains a selected-subproblem, dense gate-level diagnostic.",
        "Observable readout is decomposed, not solved.",
        "Success probability remains a cost proxy in resource-model rows.",
    ]
    failed_or_limited = [
        "Full IEEE-scale gate-level solving remains unsupported.",
        "QSVT numerical superiority over Ridge/Tikhonov remains unsupported.",
    ]
    bottlenecks = [f"- {row['bottleneck']}: {row['severity']}" for row in bottleneck_rows]
    return "\n".join(
        [
            "# Refinement Update Summary",
            "",
            CLAIM_REPORT_BOUNDARY,
            "",
            "## New Evidence Layers Present",
            *[
                f"- {layer}"
                for layer in [
                    "alpha_degree_residual_refinement",
                    "degree_control_audit",
                    "polynomial_action_decomposition",
                    "best_config_gate_validation",
                    "hardware_aware_oracle_cost_model",
                    "criteria_based_subproblem_selection",
                    "refined_selected_subproblem_solver",
                    "observable_error_decomposition",
                    "accuracy_success_tradeoff",
                ]
                if layer in present
            ],
            "",
            "## What Became Stronger",
            *[f"- {claim}" for claim in strengthened],
            "",
            "## What Remains Partial",
            *[f"- {item}" for item in partial],
            "",
            "## What Failed Or Should Be Reframed As A Limitation",
            *[f"- {item}" for item in failed_or_limited],
            "",
            "## Remaining Bottlenecks",
            *bottlenecks,
            "",
        ]
    )


def _tqe_readiness_summary(
    evidence_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
    assumption_rows: list[dict[str, Any]],
    bottleneck_rows: list[dict[str, Any]],
    *,
    solver_readiness: str = "draft_ready_with_major_limitations",
    amplitude_gap_rows: list[dict[str, Any]] | None = None,
) -> str:
    classification = tqe_readiness_classification(evidence_rows, bottleneck_rows)
    gap_lines = [
        f"- {row['metric']}: {row['value']} ({row['detail']})" for row in (amplitude_gap_rows or [])
    ]
    present = {
        str(row["evidence_layer"])
        for row in evidence_rows
        if bool(row.get("artifact_present", False))
    }
    unsupported = [
        row["claim"] for row in claim_rows if row["claim_class"] == "unsupported_do_not_claim"
    ]
    high_assumptions = [
        row["assumption_name"]
        for row in assumption_rows
        if str(row.get("risk_level", "")).lower() == "high"
    ]
    return "\n".join(
        [
            "# TQE Readiness Summary",
            "",
            CLAIM_REPORT_BOUNDARY,
            "",
            f"Overall classification: **{classification}**",
            "",
            f"Solver-readiness classification (amplitude-recovery pathway): **{solver_readiness}**",
            "",
            "## Amplitude/Norm Recovery Gap",
            *(gap_lines or ["- Amplitude-recovery search not yet run."]),
            "",
            "## What Is Implemented",
            "- Regularized spectral filter reference and classical Ridge/Tikhonov baseline.",
            "- Actual small-circuit amplitude-estimation routines (Bernoulli sampling and "
            "iterative Grover-power MLAE) when the amplitude-estimation artifact is present.",
            "- Dense scalar/diagonal convention validation and selected small gate-level solvers.",
            "- True-degree audit, polynomial-action decomposition, and hardware-aware cost tables "
            "when the corresponding artifacts are present.",
            "",
            "## What Is Demonstrated On Toy Systems",
            "- Tiny weighted state-estimation solver and toy sparse lookup/value-rotation "
            "circuits.",
            "",
            "## What Is Demonstrated On IEEE-Derived Subproblems",
            "- Selected IEEE14-derived 4x4 dense simulator updates and residual diagnostics.",
            "",
            "## What Is Only Resource Or Oracle-Model Evidence",
            "- Full IEEE-scale sparse-oracle estimates, amplification proxies, and "
            "readout-shot costs.",
            "",
            "## What Assumptions Are Required For Full IEEE-Scale QSVT",
            *[f"- {item}" for item in high_assumptions[:12]],
            "",
            "## What Claims Are Unsupported",
            *[f"- {item}" for item in unsupported],
            "",
            "## What Must Be Fixed Before TQE Submission",
            *[
                f"- {row['bottleneck']}: {row['severity']}"
                for row in bottleneck_rows
                if row["severity"] in {"high", "severe", "unknown"}
            ],
            "",
            "## Recommended Manuscript Contribution Statement",
            "- A conservative contribution is a readout-aware dense-to-oracle QSVT engineering "
            "pathway for regularized IEEE-derived state-estimation updates, supported by "
            "degree-controlled polynomial-action diagnostics and hardware-aware sparse-oracle "
            "cost modeling.",
            "",
            "## Recommended Limitation Paragraph",
            "- Full IEEE-scale QSVT remains an oracle-model pathway. The present evidence does "
            "not include hardware execution, quantum speedup, QSVT numerical superiority over "
            "Ridge/Tikhonov, a full nonlinear AC QSVT solver, solved readout, or real PMU/SCADA "
            "validation.",
            "",
            "## Recommended Reviewer-Risk List",
            "- Degree synthesis can be backend- or timeout-limited at high degree.",
            "- Polynomial-action quality and gate-level extraction quality must be reported "
            "separately.",
            "- Sparse-access, value-loading, and residual-state preparation assumptions dominate "
            "full-scale claims.",
            "- Observable readout and norm/success recovery remain bottlenecks.",
            "",
            f"New evidence layers present: {sorted(present)}",
            "",
        ]
    )


def _limitation_ready_summary(
    evidence_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
    assumption_rows: list[dict[str, Any]],
    bottleneck_rows: list[dict[str, Any]],
) -> str:
    _ = evidence_rows, claim_rows, assumption_rows
    return "\n".join(
        [
            "# Limitation-Ready Summary",
            "",
            "Recommended limitation paragraph:",
            "",
            "The present work provides dense explicit correctness evidence for scalar, diagonal, "
            "toy, and selected IEEE14-derived subproblem cases, plus polynomial-action and "
            "resource-model diagnostics for an oracle-model full IEEE-scale pathway. It does "
            "not demonstrate full IEEE-scale hardware execution, quantum speedup, QSVT "
            "numerical superiority over Ridge/Tikhonov, a full nonlinear AC QSVT solver, solved "
            "readout/norm recovery, or real PMU/SCADA validation. The full-scale pathway depends "
            "on sparse row/column lookup, value loading, fixed-point rotation synthesis, residual "
            "state preparation, amplitude amplification, and observable-readout assumptions.",
            "",
            "Must-fix items before stronger claims:",
            *[f"- {row['bottleneck']}: {row['severity']}" for row in bottleneck_rows],
            "",
        ]
    )


def tqe_readiness_classification(
    evidence_rows: list[dict[str, Any]],
    bottleneck_rows: list[dict[str, Any]],
) -> str:
    present = {
        str(row["evidence_layer"])
        for row in evidence_rows
        if bool(row.get("artifact_present", False))
    }
    required_new = {
        "degree_control_audit",
        "polynomial_action_decomposition",
        "hardware_aware_oracle_cost_model",
    }
    severe_or_unknown = {
        str(row["severity"])
        for row in bottleneck_rows
        if str(row["severity"]) in {"severe", "unknown"}
    }
    if not required_new.issubset(present):
        return "not_ready"
    if "best_config_gate_validation" not in present or severe_or_unknown:
        return "draft_ready_with_major_limitations"
    return "draft_ready_with_major_limitations"


def _claim(
    claim: str,
    claim_class: str,
    artifact_path: Path,
    note: str,
) -> dict[str, Any]:
    return {
        "claim": claim,
        "claim_class": claim_class,
        "artifact_path": str(artifact_path),
        "artifact_present": artifact_path.exists(),
        "note": note,
    }


def _layer_claim_class(layer: str, present: bool) -> str:
    if not present:
        return "unsupported_do_not_claim"
    if layer in {"regularized_spectral_filter_reference"}:
        return "supported_by_implementation"
    if layer in {
        "dense_gate_level_convention",
        "tiny_gate_level_solver",
        "ieee14_selected_gate_level_solver",
        "norm_residual_gap_audit",
        "subproblem_sweep",
        "power_observable_readout",
        "alpha_degree_residual_refinement",
        "degree_control_audit",
        "polynomial_action_decomposition",
        "best_config_gate_validation",
        "criteria_based_subproblem_selection",
        "refined_selected_subproblem_solver",
        "observable_error_decomposition",
        "scale_recovery_protocols",
        "bounded_target_design_study",
        "residual_feasible_config_search",
        "residual_feasible_gate_validation",
        "power_observable_protocols",
        "actual_amplitude_estimation_routines",
        "norm_recovery_from_amplitude",
        "residual_feasible_with_amplitude_recovery",
        "gate_validation_with_amplitude_recovery",
        "observable_first_qsvt_solver",
        "weighted_singular_support",
        "codesigned_bounded_target_study",
        "residual_feasible_codesigned_search",
        "codesigned_gate_validation",
        "observable_first_solver_v2",
        "degree_window_overshoot",
        "degree_window_gate_validation",
        "gate_observable_readout",
        "codesigned_subproblem_robustness",
        "robustness_gate_validation",
        "selected_solver_prototype_package",
        "cross_case_solver_audit",
        "overshoot_mechanism_diagnostic",
        "cross_case_codesigned_robustness",
        "cross_case_gate_validation",
        "cross_case_gate_observable_readout",
        "cross_case_solver_prototype_package",
        "ieee118_extension_audit",
        "direction_resolved_overshoot_decomposition",
        "ieee118_selected_robustness",
        "ieee118_gate_validation",
        "ieee118_gate_observable_readout",
        "final_optional_evidence_package",
    }:
        return "supported_by_controlled_experiment"
    if layer in {
        "dense_explicit_limit_study",
        "ieee_scale_resource_estimate",
        "success_amplification_cost",
        "accuracy_success_tradeoff",
        "hardware_aware_oracle_cost_model",
        "sparse_oracle_prototype_strengthening",
    }:
        return "supported_by_resource_model"
    if layer == "sparse_oracle_assumption_ledger":
        return "supported_by_assumption_only"
    return "supported_by_resource_model"


def _success_severity(input_root: Path) -> str:
    path = input_root / "qsvt_success_amplification_cost" / "success_probability_sweep.csv"
    if not path.is_file():
        return "unknown"
    frame = pd.read_csv(path)
    if "bottleneck_severity" not in frame.columns or frame.empty:
        return "unknown"
    order = {"low": 0, "moderate": 1, "high": 2, "severe": 3}
    return max(
        (str(value) for value in frame["bottleneck_severity"]),
        key=lambda value: order.get(value, -1),
    )


def _refinement_severity(input_root: Path) -> str:
    path = (
        input_root / "qsvt_alpha_degree_residual_refinement" / "alpha_degree_refinement_summary.csv"
    )
    if not path.is_file():
        return "unknown"
    frame = pd.read_csv(path)
    completed = frame[frame.get("run_status", "") == "completed"]
    if completed.empty:
        return "high"
    best = float(completed["residual_ratio_best_scalar_vs_no_update"].min())
    if best <= 0.1:
        return "moderate"
    if best <= 0.5:
        return "high"
    return "severe"


def _degree_control_severity(input_root: Path) -> str:
    path = input_root / "qsvt_degree_control_audit" / "degree_control_audit.csv"
    if not path.is_file():
        return "unknown"
    frame = pd.read_csv(path)
    completed = frame[frame.get("backend_status", "").isin(["synthesized", "cache_hit"])]
    if completed.empty:
        return "severe"
    synthesized = set(int(value) for value in completed["synthesized_phase_degree"].dropna())
    requested = set(int(value) for value in frame["requested_degree"].dropna())
    if any(value > 35 for value in synthesized):
        return "moderate"
    if any(value > 35 for value in requested):
        return "high"
    return "moderate"


def _polynomial_decomposition_severity(input_root: Path) -> str:
    path = input_root / "qsvt_polynomial_action_decomposition" / "polynomial_action_residuals.csv"
    if not path.is_file():
        return "unknown"
    frame = pd.read_csv(path)
    completed = frame[frame.get("run_status", "") == "completed"]
    if completed.empty:
        return "high"
    best_poly = float(completed["best_scalar_polynomial_residual"].min())
    best_ridge = float(completed["ridge_residual"].min())
    if best_poly <= max(100.0 * best_ridge, 1.0e-6):
        return "moderate"
    return "high"


def _hardware_cost_severity(input_root: Path) -> str:
    path = input_root / "hardware_aware_oracle_cost_model" / "qsvt_total_cost_estimate.csv"
    if not path.is_file():
        return "unknown"
    frame = pd.read_csv(path)
    if frame.empty:
        return "unknown"
    if any(str(value).startswith("failed") for value in frame["resource_status"]):
        return "high"
    if "high" in set(str(value) for value in frame["assumption_risk_level"]):
        return "high"
    return "moderate"


def _observable_severity(input_root: Path) -> str:
    path = input_root / "qsvt_observable_error_decomposition" / "observable_error_decomposition.csv"
    if not path.is_file():
        return "unknown"
    frame = pd.read_csv(path)
    if frame.empty:
        return "unknown"
    max_scaled = float(frame["best_scalar_absolute_error"].max())
    if max_scaled <= 1.0e-3:
        return "low"
    if max_scaled <= 1.0e-1:
        return "moderate"
    return "high"
