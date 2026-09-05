"""Phase 6: QSVT resource and phase analysis paper-ready summary.

Consolidates existing QSVT target/phase, polynomial-action, gate-validation,
amplitude/norm, observable-readout, and resource-model evidence into
manuscript-ready tables and figure-ready data. Selected-subproblem solver
evidence is kept separate from full-case resource-model estimates; unsupported
claims are preserved as unsupported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from robust_qsvt_se.paper._common import read_csv, rows_to_table
from robust_qsvt_se.qsvt.direction_resolved_overshoot_decomposition import NO_FAILURE_MECHANISMS
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

QSVT_CLAIM_BOUNDARY = (
    "QSVT-compatible implementation pathway and resource-aware feasibility analysis only. "
    "Selected (4x4) subproblem solver-prototype evidence is separated from full-case "
    "resource-model estimates. Ridge/Tikhonov is the reference filter; no quantum speedup, "
    "quantum advantage, QSVT superiority over Ridge/Tikhonov, full IEEE-scale hardware "
    "execution, or real PMU/SCADA validation is claimed."
)

TARGET_WINDOW_COLUMNS = [
    "case",
    "subproblem_id",
    "selection_mode",
    "target_family",
    "alpha",
    "safe_degree_min",
    "safe_degree_max",
    "overshoot_onset_degree",
    "failure_mechanism",
    "qsvt_safe",
    "source_artifact",
    "notes",
]

PHASE_ERROR_COLUMNS = [
    "target",
    "basis",
    "degree",
    "phase_count",
    "grid_points",
    "max_phase_abs_error",
    "max_phase_scaled_abs_error",
    "max_phase_vs_polynomial_abs_error",
    "status",
    "source_artifact",
    "claim_boundary",
    "notes",
]

GATE_COLUMNS = [
    "case",
    "subproblem_id",
    "selection_mode",
    "alpha",
    "degree",
    "target_family",
    "gate_status",
    "residual_ratio_vs_no_update",
    "state_error_gate_vs_polynomial",
    "state_error_gate_vs_ridge",
    "direction_error_gate_vs_ridge",
    "success_probability_exact",
    "success_probability_estimated",
    "circuit_depth",
    "two_qubit_gates",
    "source_artifact",
    "claim_boundary",
    "notes",
]

OBSERVABLE_COLUMNS = [
    "case",
    "subproblem_id",
    "selection_mode",
    "observable_name",
    "physical_meaning",
    "degree",
    "target_family",
    "ridge_value",
    "gate_value",
    "relative_error_gate_vs_ridge",
    "top_k_match_if_applicable",
    "requires_norm_recovery",
    "requires_signed_overlap",
    "requires_full_vector_readout",
    "readout_protocol",
    "source_artifact",
    "claim_boundary",
    "notes",
]

RESOURCE_COLUMNS = [
    "case",
    "matrix_shape",
    "subproblem_or_full",
    "resource_model_type",
    "alpha",
    "degree",
    "phase_count",
    "query_count",
    "qubit_estimate",
    "circuit_depth",
    "two_qubit_gates",
    "success_probability",
    "readout_cost_proxy",
    "source_artifact",
    "claim_boundary",
    "notes",
]

CLAIM_BOUNDARY_COLUMNS = [
    "claim",
    "evidence_level",
    "support_status",
    "allowed_wording",
    "disallowed_wording",
    "source_artifact",
    "notes",
]

MISSING_COLUMNS = [
    "missing_output",
    "needed_for",
    "importance",
    "reason_missing",
    "recommended_action",
]

FIG_TARGET_COLUMNS = [
    "case",
    "subproblem_id",
    "target_family",
    "degree",
    "singular_index",
    "sigma",
    "ridge_filter_value",
    "qsvt_polynomial_value",
    "signed_filter_error",
]

FIG_DEGREE_COLUMNS = [
    "case",
    "subproblem_id",
    "alpha",
    "target_family",
    "degree",
    "overshoot_margin",
    "direction_error_vs_ridge",
    "residual_ratio_vs_no_update",
    "qsvt_safe",
]

FIG_RESOURCE_COLUMNS = [
    "case",
    "matrix_columns",
    "estimated_total_qubits",
    "estimated_qsvt_query_count",
    "degree",
    "phase_count",
    "source_artifact",
]

_STABLE_FAMILIES = ("weighted_support_ls", "residual_aware")


def build_qsvt_resource_phase_summary(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "input_root": "outputs",
        "output_dir": "outputs/final_manuscript_package/phase6_qsvt_resource_phase",
    }
    resolved.update(config)
    input_root = Path(resolved["input_root"])
    output_dir = ensure_directory(resolved["output_dir"])

    target_rows = target_and_degree_window_rows(input_root)
    phase_rows = phase_error_rows(input_root)
    gate_rows = gate_validation_rows(input_root)
    observable_rows = observable_readout_rows(input_root)
    resource_rows = resource_summary_rows(input_root)
    claim_rows = qsvt_claim_boundary_rows()
    missing_rows = missing_qsvt_rows(
        target_rows, phase_rows, gate_rows, observable_rows, resource_rows
    )
    fig_target = figure_target_vs_polynomial(input_root)
    fig_degree = figure_degree_window(input_root)
    fig_resource = figure_resource_scaling(input_root)

    artifacts = _write_outputs(
        output_dir,
        resolved,
        target_rows=target_rows,
        phase_rows=phase_rows,
        gate_rows=gate_rows,
        observable_rows=observable_rows,
        resource_rows=resource_rows,
        claim_rows=claim_rows,
        missing_rows=missing_rows,
        fig_target=fig_target,
        fig_degree=fig_degree,
        fig_resource=fig_resource,
    )
    return {
        "output_dir": output_dir,
        "target_rows": target_rows,
        "phase_rows": phase_rows,
        "gate_rows": gate_rows,
        "observable_rows": observable_rows,
        "resource_rows": resource_rows,
        "claim_rows": claim_rows,
        "missing_rows": missing_rows,
        "artifacts": artifacts,
    }


def target_and_degree_window_rows(input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary = read_csv(
        input_root
        / "qsvt_direction_resolved_overshoot_decomposition"
        / "direction_resolved_error_summary.csv"
    )
    mode_map = _selection_mode_map(summary, input_root)
    mechanism_lookup = _mechanism_lookup(summary)

    # IEEE14 reference window from the residual-feasibility overshoot boundary (15 / 45 / 47).
    boundary = read_csv(input_root / "qsvt_degree_window_overshoot" / "overshoot_boundary.csv")
    boundary_cases: set[str] = set()
    if not boundary.empty:
        for _, r in boundary.iterrows():
            case = str(r.get("case", ""))
            boundary_cases.add(case)
            subproblem = str(r.get("subproblem_id", ""))
            family = str(r.get("target_family", ""))
            alpha = r.get("alpha", "")
            onset = _opt_int(r.get("overshoot_onset_degree"))
            safe_max = _opt_int(r.get("max_feasible_degree"))
            count = pd.to_numeric(r.get("feasible_degree_count"), errors="coerce")
            mechanism = mechanism_lookup.get((case, subproblem, str(alpha), family, onset), "")
            rows.append(
                {
                    "case": case,
                    "subproblem_id": subproblem,
                    "selection_mode": mode_map.get(subproblem, _mode_from_id(subproblem)),
                    "target_family": family,
                    "alpha": alpha,
                    "safe_degree_min": _opt_int(r.get("min_feasible_degree")),
                    "safe_degree_max": safe_max,
                    "overshoot_onset_degree": onset,
                    "failure_mechanism": mechanism
                    or ("leading_direction_amplitude_distortion" if onset != "" else "no_failure"),
                    "qsvt_safe": "yes" if (not pd.isna(count) and count > 0) else "no",
                    "source_artifact": "qsvt_degree_window_overshoot/overshoot_boundary.csv",
                    "notes": "residual-feasibility degree window (IEEE14 reference)",
                }
            )

    # Remaining cross-case (ieee30/57) from the direction-resolved no_failure semantics.
    if not summary.empty:
        summary = summary.copy()
        summary["degree"] = pd.to_numeric(summary["degree"], errors="coerce")
        summary["safe_flag"] = summary["failure_mechanism"].isin(NO_FAILURE_MECHANISMS)
        keys = ["case", "subproblem_id", "selection_mode", "target_family", "alpha"]
        for key, group in summary.groupby(keys, sort=True):
            case, subproblem, mode, family, alpha = key
            if case in boundary_cases:
                continue
            safe = group[group["safe_flag"]]["degree"].dropna()
            failing = group[~group["safe_flag"]].sort_values("degree")
            safe_max = safe.max() if not safe.empty else -1
            after = failing[failing["degree"] > safe_max]
            onset = int(after.iloc[0]["degree"]) if not after.empty else ""
            mechanism = str(after.iloc[0]["failure_mechanism"]) if not after.empty else "no_failure"
            rows.append(
                {
                    "case": case,
                    "subproblem_id": subproblem,
                    "selection_mode": mode,
                    "target_family": family,
                    "alpha": alpha,
                    "safe_degree_min": int(safe.min()) if not safe.empty else "",
                    "safe_degree_max": int(safe.max()) if not safe.empty else "",
                    "overshoot_onset_degree": onset,
                    "failure_mechanism": mechanism,
                    "qsvt_safe": "yes" if not safe.empty else "no",
                    "source_artifact": "qsvt_direction_resolved_overshoot_decomposition/"
                    "direction_resolved_error_summary.csv",
                    "notes": "cross-case direction-resolved no_failure window",
                }
            )

    # IEEE118 from the selected-robustness configuration sweep.
    configs = read_csv(input_root / "qsvt_ieee118_selected_robustness" / "ieee118_all_configs.csv")
    if not configs.empty:
        configs = configs.copy()
        configs["degree"] = pd.to_numeric(configs["degree"], errors="coerce")
        configs["feasible"] = configs["residual_feasible"].astype(str).str.lower().eq("true")
        keys = ["case", "subproblem_id", "selection_mode", "target_family", "alpha"]
        for key, group in configs.groupby(keys, sort=True):
            case, subproblem, mode, family, alpha = key
            feasible = group[group["feasible"]]["degree"].dropna()
            unsafe = group[~group["qsvt_safe"].astype(str).str.lower().eq("true")]
            unsafe = unsafe.sort_values("degree")
            safe_min = int(feasible.min()) if not feasible.empty else ""
            safe_max = int(feasible.max()) if not feasible.empty else ""
            onset = int(unsafe.iloc[0]["degree"]) if not unsafe.empty else ""
            rows.append(
                {
                    "case": case,
                    "subproblem_id": subproblem,
                    "selection_mode": mode,
                    "target_family": family,
                    "alpha": alpha,
                    "safe_degree_min": safe_min,
                    "safe_degree_max": safe_max,
                    "overshoot_onset_degree": onset,
                    "failure_mechanism": "overshoot" if onset != "" else "no_failure",
                    "qsvt_safe": "yes" if not feasible.empty else "no",
                    "source_artifact": "qsvt_ieee118_selected_robustness/ieee118_all_configs.csv",
                    "notes": "IEEE118 selected-block residual-feasible degree window",
                }
            )
    return rows


def _selection_mode_map(summary: pd.DataFrame, input_root: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    frames = [summary]
    frames.append(
        read_csv(input_root / "qsvt_ieee118_selected_robustness" / "ieee118_all_configs.csv")
    )
    for frame in frames:
        if frame.empty or "subproblem_id" not in frame.columns:
            continue
        if "selection_mode" not in frame.columns:
            continue
        for subproblem, mode in zip(
            frame["subproblem_id"].astype(str), frame["selection_mode"].astype(str), strict=False
        ):
            mapping.setdefault(subproblem, mode)
    return mapping


def _mechanism_lookup(summary: pd.DataFrame) -> dict[tuple[str, str, str, str, Any], str]:
    lookup: dict[tuple[str, str, str, str, Any], str] = {}
    if summary.empty:
        return lookup
    for _, r in summary.iterrows():
        key = (
            str(r.get("case", "")),
            str(r.get("subproblem_id", "")),
            str(r.get("alpha", "")),
            str(r.get("target_family", "")),
            _opt_int(r.get("degree")),
        )
        lookup[key] = str(r.get("failure_mechanism", ""))
    return lookup


def _mode_from_id(subproblem_id: str) -> str:
    for mode in (
        "high_leverage",
        "metadata_mapped",
        "residual_supported",
        "best_conditioned",
        "worst_conditioned_control",
        "random_seeded_pool",
    ):
        if mode in subproblem_id:
            return mode
    return ""


def _opt_int(value: Any) -> Any:
    numeric = pd.to_numeric(value, errors="coerce")
    return "" if pd.isna(numeric) else int(numeric)


def phase_error_rows(input_root: Path) -> list[dict[str, Any]]:
    base = input_root / "qsvt_phase_validation_paper"
    grid = read_csv(base / "phase_implemented_error.csv")
    if grid.empty:
        return []
    angles = read_csv(base / "phase_angles.csv")
    phase_count = len(angles) if not angles.empty else ""
    degree = phase_count - 1 if isinstance(phase_count, int) else ""
    max_abs = _safe_max(grid, "phase_abs_error")
    max_scaled = _safe_max(grid, "phase_scaled_abs_error")
    max_vs_poly = _safe_max(grid, "phase_vs_polynomial_abs_error")
    status = (
        "validated_within_tolerance"
        if isinstance(max_scaled, float) and max_scaled <= 1e-3
        else "review_tolerance"
    )
    return [
        {
            "target": "bounded Ridge/Tikhonov scalar target",
            "basis": "Chebyshev",
            "degree": degree,
            "phase_count": phase_count,
            "grid_points": len(grid),
            "max_phase_abs_error": max_abs,
            "max_phase_scaled_abs_error": max_scaled,
            "max_phase_vs_polynomial_abs_error": max_vs_poly,
            "status": status,
            "source_artifact": "qsvt_phase_validation_paper/phase_implemented_error.csv",
            "claim_boundary": QSVT_CLAIM_BOUNDARY,
            "notes": "phase-response synthesis reproduces the bounded target",
        }
    ]


def gate_validation_rows(input_root: Path) -> list[dict[str, Any]]:
    sources = [
        (
            "qsvt_cross_case_gate_validation/cross_case_gate_results.csv",
            input_root / "qsvt_cross_case_gate_validation" / "cross_case_gate_results.csv",
        ),
        (
            "qsvt_ieee118_gate_validation/ieee118_gate_results.csv",
            input_root / "qsvt_ieee118_gate_validation" / "ieee118_gate_results.csv",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for source, path in sources:
        frame = read_csv(path)
        for _, record in frame.iterrows():
            rows.append(
                {
                    "case": record.get("case", ""),
                    "subproblem_id": record.get("subproblem_id", ""),
                    "selection_mode": record.get("selection_mode", ""),
                    "alpha": record.get("alpha", ""),
                    "degree": record.get("degree", ""),
                    "target_family": record.get("target_family", ""),
                    "gate_status": record.get("gate_status", ""),
                    "residual_ratio_vs_no_update": record.get("residual_ratio_vs_no_update", ""),
                    "state_error_gate_vs_polynomial": record.get(
                        "state_error_gate_vs_polynomial", ""
                    ),
                    "state_error_gate_vs_ridge": record.get("state_error_gate_vs_ridge", ""),
                    "direction_error_gate_vs_ridge": record.get(
                        "direction_error_gate_vs_ridge", ""
                    ),
                    "success_probability_exact": record.get("success_probability_exact", ""),
                    "success_probability_estimated": record.get(
                        "success_probability_estimated", ""
                    ),
                    "circuit_depth": record.get("circuit_depth", ""),
                    "two_qubit_gates": record.get("two_qubit_gates", ""),
                    "source_artifact": source,
                    "claim_boundary": QSVT_CLAIM_BOUNDARY,
                    "notes": "selected 4x4 subproblem gate validation",
                }
            )
    return rows


def observable_readout_rows(input_root: Path) -> list[dict[str, Any]]:
    sources = [
        (
            "qsvt_cross_case_gate_observable_readout/cross_case_gate_observable_values.csv",
            input_root
            / "qsvt_cross_case_gate_observable_readout"
            / "cross_case_gate_observable_values.csv",
        ),
        (
            "qsvt_ieee118_gate_observable_readout/ieee118_gate_observable_values.csv",
            input_root
            / "qsvt_ieee118_gate_observable_readout"
            / "ieee118_gate_observable_values.csv",
        ),
        (
            "qsvt_gate_observable_readout/gate_observable_values.csv",
            input_root / "qsvt_gate_observable_readout" / "gate_observable_values.csv",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for source, path in sources:
        frame = read_csv(path)
        for _, record in frame.iterrows():
            rows.append(
                {
                    "case": record.get("case", "ieee14"),
                    "subproblem_id": record.get("subproblem_id", ""),
                    "selection_mode": record.get("selection_mode", ""),
                    "observable_name": record.get("observable_name", ""),
                    "physical_meaning": record.get("physical_meaning", ""),
                    "degree": record.get("degree", ""),
                    "target_family": record.get("target_family", ""),
                    "ridge_value": record.get("ridge_value", ""),
                    "gate_value": record.get("gate_value", ""),
                    "relative_error_gate_vs_ridge": record.get("relative_error_gate_vs_ridge", ""),
                    "top_k_match_if_applicable": record.get("top_k_match_if_applicable", ""),
                    "requires_norm_recovery": record.get("requires_norm_recovery", ""),
                    "requires_signed_overlap": record.get("requires_signed_overlap", ""),
                    "requires_full_vector_readout": record.get("requires_full_vector_readout", ""),
                    "readout_protocol": record.get("readout_protocol", ""),
                    "source_artifact": source,
                    "claim_boundary": QSVT_CLAIM_BOUNDARY,
                    "notes": "observable-first readout from gate output",
                }
            )
    return rows


def resource_summary_rows(input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    oracle = read_csv(
        input_root / "qsvt_oracle_model_resources" / "oracle_model_resource_summary.csv"
    )
    for _, r in oracle.iterrows():
        rows.append(
            _resource_row(
                r.get("case"),
                _shape(r, "matrix_rows", "matrix_cols"),
                "selected",
                "oracle_model",
                r.get("alpha"),
                r.get("degree"),
                r.get("phase_count"),
                r.get("qsvt_query_count"),
                r.get("total_logical_qubits_padded_convention"),
                "",
                "",
                r.get("success_probability_proxy"),
                r.get("readout_shots"),
                "qsvt_oracle_model_resources/oracle_model_resource_summary.csv",
            )
        )

    hardware = read_csv(
        input_root / "hardware_aware_oracle_cost_model" / "qsvt_total_cost_estimate.csv"
    )
    for _, r in hardware.iterrows():
        rows.append(
            _resource_row(
                r.get("case"),
                r.get("matrix_shape"),
                "full",
                "hardware_aware_oracle_cost",
                r.get("value_precision_bits"),
                r.get("qsvt_degree"),
                "",
                r.get("qsvt_query_count"),
                _sum_qubits(r),
                "",
                r.get("estimated_total_gate_cost_proxy"),
                r.get("success_probability_proxy"),
                r.get("observable_readout_shots"),
                "hardware_aware_oracle_cost_model/qsvt_total_cost_estimate.csv",
            )
        )

    full = read_csv(input_root / "qsvt_resource_full_ieee" / "qsvt_resource_estimates.csv")
    for _, r in full.iterrows():
        rows.append(
            _resource_row(
                r.get("case_name"),
                r.get("matrix_shape"),
                "full",
                "full_statevector_resource_estimate",
                r.get("alpha"),
                r.get("polynomial_degree"),
                r.get("phase_count"),
                r.get("estimated_qsvt_query_count"),
                r.get("estimated_total_qubits"),
                r.get("estimated_circuit_depth_proxy"),
                r.get("estimated_gate_count_proxy"),
                "",
                "",
                "qsvt_resource_full_ieee/qsvt_resource_estimates.csv",
            )
        )

    hw_summary = read_csv(
        input_root / "full_qsvt_ieee_hardware_resources" / "hardware_resource_summary.csv"
    )
    for _, r in hw_summary.iterrows():
        rows.append(
            _resource_row(
                r.get("case_name"),
                r.get("matrix_shape"),
                "full",
                "hardware_resource_summary",
                "",
                r.get("qsvt_degree"),
                r.get("phase_count"),
                r.get("query_count"),
                r.get("total_logical_qubits"),
                r.get("circuit_depth_proxy"),
                r.get("two_qubit_gate_proxy"),
                "",
                r.get("readout_shots"),
                "full_qsvt_ieee_hardware_resources/hardware_resource_summary.csv",
            )
        )
    return rows


def qsvt_claim_boundary_rows() -> list[dict[str, Any]]:
    return [
        _claim(
            "Bounded QSVT target reproduces the Ridge/Tikhonov filter under phase synthesis",
            "target_phase_validation",
            "supported",
            "QSVT-compatible implementation pathway; regularized spectral filter",
            "quantum advantage",
            "qsvt_phase_validation_paper",
        ),
        _claim(
            "Polynomial action reproduces the regularized filter on the singular support",
            "polynomial_action",
            "supported",
            "regularized spectral filter",
            "QSVT beats Ridge",
            "qsvt_direction_resolved_overshoot_decomposition",
        ),
        _claim(
            "Single-step selected 4x4 update is gate-validated across IEEE14/30/57/118 blocks",
            "gate_validation",
            "supported",
            "single-step QSVT state-estimation update solver prototype; gate-validated "
            "selected-subproblem evidence",
            "full IEEE-scale quantum state estimator",
            "qsvt_cross_case_gate_validation; qsvt_ieee118_gate_validation",
        ),
        _claim(
            "Amplitude/norm recovery yields a deployable update scale on selected blocks",
            "amplitude_norm_estimation",
            "supported_with_limitations",
            "QSVT-compatible implementation pathway",
            "deployment-ready quantum solver",
            "qsvt_amplitude_estimation_routines; qsvt_norm_recovery_from_amplitude",
        ),
        _claim(
            "Observable-first readout recovers selected observables without full-vector tomography",
            "observable_readout",
            "supported_with_limitations",
            "gate-validated selected-subproblem evidence",
            "full IEEE-scale quantum state estimator",
            "qsvt_cross_case_gate_observable_readout; qsvt_ieee118_gate_observable_readout",
        ),
        _claim(
            "Oracle/hardware resource models estimate qubits, queries, and readout cost",
            "resource_model",
            "supported_with_limitations",
            "QSVT-compatible implementation pathway",
            "quantum speedup; deployment-ready quantum solver",
            "qsvt_oracle_model_resources; hardware_aware_oracle_cost_model",
        ),
        _claim(
            "Full IEEE-scale QSVT sparse-oracle pathway with full output-direction readout",
            "assumption_only",
            "assumption_only",
            "QSVT-compatible implementation pathway",
            "full IEEE-scale quantum state estimator",
            "sparse_oracle_assumption_ledger",
        ),
        _claim(
            "QSVT numerical superiority over Ridge/Tikhonov",
            "unsupported",
            "unsupported_do_not_claim",
            "Ridge/Tikhonov reference",
            "QSVT beats Ridge",
            "none",
        ),
        _claim(
            "Quantum speedup or quantum advantage for state estimation",
            "unsupported",
            "unsupported_do_not_claim",
            "QSVT-compatible implementation pathway",
            "quantum speedup; quantum advantage",
            "none",
        ),
    ]


def missing_qsvt_rows(
    target_rows: list[dict[str, Any]],
    phase_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    observable_rows: list[dict[str, Any]],
    resource_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    checks = [
        (
            phase_rows,
            "QSVT phase-error validation table",
            "phase-response validation figure",
            "high",
            "qsvt_phase_validation_paper outputs absent",
            "rerun the phase-validation config",
        ),
        (
            gate_rows,
            "QSVT gate-validation table",
            "solver-prototype results",
            "critical",
            "gate-validation outputs absent",
            "rerun cross-case/IEEE118 gate validation",
        ),
        (
            observable_rows,
            "QSVT observable-readout table",
            "readout analysis",
            "high",
            "observable-readout outputs absent",
            "rerun gate observable readout",
        ),
        (
            resource_rows,
            "QSVT resource-model table",
            "resource analysis",
            "high",
            "resource-model outputs absent",
            "rerun oracle/hardware resource models",
        ),
        (
            target_rows,
            "QSVT degree-window table",
            "degree-window figure",
            "high",
            "degree-window outputs absent",
            "rerun direction-resolved / robustness sweeps",
        ),
    ]
    for present, name, needed, importance, reason, action in checks:
        if not present:
            rows.append(
                {
                    "missing_output": name,
                    "needed_for": needed,
                    "importance": importance,
                    "reason_missing": reason,
                    "recommended_action": action,
                }
            )
    # Standing assumption gaps regardless of artifact presence.
    rows.append(
        {
            "missing_output": "full IEEE-scale gate-level QSVT execution",
            "needed_for": "full-scale solver claim (out of scope)",
            "importance": "documented_limitation",
            "reason_missing": "only selected 4x4 subproblems are gate-executed; full case is "
            "resource-model only",
            "recommended_action": "remains future work; not claimed",
        }
    )
    rows.append(
        {
            "missing_output": "full output-direction (full-vector) readout",
            "needed_for": "complete state readout",
            "importance": "documented_limitation",
            "reason_missing": "observable-first readout only recovers selected observables",
            "recommended_action": "remains an assumption; not claimed",
        }
    )
    return rows


def figure_target_vs_polynomial(input_root: Path) -> list[dict[str, Any]]:
    frame = read_csv(
        input_root
        / "qsvt_direction_resolved_overshoot_decomposition"
        / "per_direction_error_components.csv"
    )
    if frame.empty:
        return []
    frame = frame.copy()
    frame["degree"] = pd.to_numeric(frame["degree"], errors="coerce")
    mask = (
        frame["case"].astype(str).eq("ieee14")
        & frame["selection_mode"].astype(str).eq("high_leverage")
        & frame["target_family"].astype(str).eq("weighted_support_ls")
        & frame["degree"].isin([45, 47])
    )
    selected = frame[mask]
    rows: list[dict[str, Any]] = []
    for _, r in selected.iterrows():
        rows.append(
            {
                "case": r.get("case", ""),
                "subproblem_id": r.get("subproblem_id", ""),
                "target_family": r.get("target_family", ""),
                "degree": r.get("degree", ""),
                "singular_index": r.get("singular_index", ""),
                "sigma": r.get("sigma", ""),
                "ridge_filter_value": r.get("ridge_filter_value", ""),
                "qsvt_polynomial_value": r.get("qsvt_polynomial_value", ""),
                "signed_filter_error": r.get("signed_filter_error", ""),
            }
        )
    return rows


def figure_degree_window(input_root: Path) -> list[dict[str, Any]]:
    frame = read_csv(input_root / "qsvt_degree_window_overshoot" / "degree_window_summary.csv")
    if frame.empty:
        return []
    frame = frame.copy()
    keep = frame["target_family"].astype(str).isin(_STABLE_FAMILIES)
    frame = frame[keep]
    rows: list[dict[str, Any]] = []
    for _, r in frame.iterrows():
        rows.append(
            {
                "case": r.get("case", ""),
                "subproblem_id": r.get("subproblem_id", ""),
                "alpha": r.get("alpha", ""),
                "target_family": r.get("target_family", ""),
                "degree": r.get("degree", ""),
                "overshoot_margin": r.get("overshoot_margin", ""),
                "direction_error_vs_ridge": r.get("direction_error_vs_ridge", ""),
                "residual_ratio_vs_no_update": r.get("residual_ratio_vs_no_update", ""),
                "qsvt_safe": r.get("qsvt_safe", ""),
            }
        )
    return rows


def figure_resource_scaling(input_root: Path) -> list[dict[str, Any]]:
    frame = read_csv(input_root / "qsvt_resource_full_ieee" / "qsvt_resource_estimates.csv")
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, r in frame.iterrows():
        rows.append(
            {
                "case": r.get("case_name", ""),
                "matrix_columns": r.get("matrix_columns", ""),
                "estimated_total_qubits": r.get("estimated_total_qubits", ""),
                "estimated_qsvt_query_count": r.get("estimated_qsvt_query_count", ""),
                "degree": r.get("polynomial_degree", ""),
                "phase_count": r.get("phase_count", ""),
                "source_artifact": "qsvt_resource_full_ieee/qsvt_resource_estimates.csv",
            }
        )
    return rows


def _resource_row(
    case: Any,
    shape: Any,
    scope: str,
    model: str,
    alpha: Any,
    degree: Any,
    phase_count: Any,
    query_count: Any,
    qubits: Any,
    depth: Any,
    two_qubit: Any,
    success: Any,
    readout: Any,
    source: str,
) -> dict[str, Any]:
    return {
        "case": _clean(case),
        "matrix_shape": _clean(shape),
        "subproblem_or_full": scope,
        "resource_model_type": model,
        "alpha": _clean(alpha),
        "degree": _clean(degree),
        "phase_count": _clean(phase_count),
        "query_count": _clean(query_count),
        "qubit_estimate": _clean(qubits),
        "circuit_depth": _clean(depth),
        "two_qubit_gates": _clean(two_qubit),
        "success_probability": _clean(success),
        "readout_cost_proxy": _clean(readout),
        "source_artifact": source,
        "claim_boundary": QSVT_CLAIM_BOUNDARY,
        "notes": "selected-subproblem" if scope == "selected" else "full-case resource-model only",
    }


def _claim(
    claim: str, level: str, status: str, allowed: str, disallowed: str, source: str
) -> dict[str, Any]:
    return {
        "claim": claim,
        "evidence_level": level,
        "support_status": status,
        "allowed_wording": allowed,
        "disallowed_wording": disallowed,
        "source_artifact": source,
        "notes": "",
    }


def _shape(record: pd.Series, rows_col: str, cols_col: str) -> str:
    rows = record.get(rows_col)
    cols = record.get(cols_col)
    if pd.isna(rows) or pd.isna(cols):
        return ""
    return f"{int(rows)}x{int(cols)}"


def _sum_qubits(record: pd.Series) -> Any:
    work = pd.to_numeric(record.get("work_qubits"), errors="coerce")
    ancilla = pd.to_numeric(record.get("ancilla_qubits"), errors="coerce")
    if pd.isna(work) and pd.isna(ancilla):
        return ""
    return int((0 if pd.isna(work) else work) + (0 if pd.isna(ancilla) else ancilla))


def _safe_max(frame: pd.DataFrame, column: str) -> Any:
    if column not in frame.columns:
        return ""
    value = pd.to_numeric(frame[column], errors="coerce").dropna()
    return round(float(value.max()), 8) if not value.empty else ""


def _clean(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return value


def _write_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    *,
    target_rows: list[dict[str, Any]],
    phase_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    observable_rows: list[dict[str, Any]],
    resource_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
    fig_target: list[dict[str, Any]],
    fig_degree: list[dict[str, Any]],
    fig_resource: list[dict[str, Any]],
) -> dict[str, Path]:
    paths = {
        "paper_table_qsvt_target_and_degree_window": rows_to_table(
            target_rows,
            output_dir / "paper_table_qsvt_target_and_degree_window.csv",
            TARGET_WINDOW_COLUMNS,
        ),
        "paper_table_qsvt_phase_error": rows_to_table(
            phase_rows, output_dir / "paper_table_qsvt_phase_error.csv", PHASE_ERROR_COLUMNS
        ),
        "paper_table_qsvt_gate_validation": rows_to_table(
            gate_rows, output_dir / "paper_table_qsvt_gate_validation.csv", GATE_COLUMNS
        ),
        "paper_table_qsvt_observable_readout": rows_to_table(
            observable_rows,
            output_dir / "paper_table_qsvt_observable_readout.csv",
            OBSERVABLE_COLUMNS,
        ),
        "paper_table_qsvt_resource_summary": rows_to_table(
            resource_rows, output_dir / "paper_table_qsvt_resource_summary.csv", RESOURCE_COLUMNS
        ),
        "paper_table_qsvt_claim_boundaries": rows_to_table(
            claim_rows,
            output_dir / "paper_table_qsvt_claim_boundaries.csv",
            CLAIM_BOUNDARY_COLUMNS,
        ),
        "missing_qsvt_outputs": rows_to_table(
            missing_rows, output_dir / "missing_qsvt_outputs.csv", MISSING_COLUMNS
        ),
        "figure_data_qsvt_target_vs_polynomial": rows_to_table(
            fig_target,
            output_dir / "figure_data_qsvt_target_vs_polynomial.csv",
            FIG_TARGET_COLUMNS,
        ),
        "figure_data_qsvt_degree_window": rows_to_table(
            fig_degree, output_dir / "figure_data_qsvt_degree_window.csv", FIG_DEGREE_COLUMNS
        ),
        "figure_data_qsvt_resource_scaling": rows_to_table(
            fig_resource,
            output_dir / "figure_data_qsvt_resource_scaling.csv",
            FIG_RESOURCE_COLUMNS,
        ),
    }
    status_path = output_dir / "qsvt_resource_phase_status.md"
    status_path.write_text(
        _status_markdown(target_rows, gate_rows, observable_rows, resource_rows, missing_rows),
        encoding="utf-8",
    )
    paths["qsvt_resource_phase_status"] = status_path

    manifest = write_manifest(
        output_dir,
        artifacts={key: str(value) for key, value in paths.items()},
        input_config=resolved,
        claim_boundary=QSVT_CLAIM_BOUNDARY,
    )
    paths["manifest"] = manifest
    return paths


def _status_markdown(
    target_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    observable_rows: list[dict[str, Any]],
    resource_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
) -> str:
    selected = [r for r in resource_rows if r["subproblem_or_full"] == "selected"]
    full = [r for r in resource_rows if r["subproblem_or_full"] == "full"]
    return "\n".join(
        [
            "# QSVT Resource and Phase Analysis Summary",
            "",
            QSVT_CLAIM_BOUNDARY,
            "",
            "## Evidence separation",
            f"- Degree-window rows: {len(target_rows)} (selected-subproblem feasibility).",
            f"- Gate-validation rows: {len(gate_rows)} (selected 4x4 blocks).",
            f"- Observable-readout rows: {len(observable_rows)}.",
            f"- Resource-model rows: {len(resource_rows)} "
            f"({len(selected)} selected-subproblem, {len(full)} full-case resource-model only).",
            f"- Missing / documented-limitation rows: {len(missing_rows)}.",
            "",
            "## Conclusion",
            "Bounded QSVT target f_{alpha,bounded}(sigma) = (1/C) sigma/(sigma^2+alpha) reproduces "
            "the Ridge/Tikhonov filter under phase synthesis. Selected 4x4 subproblem evidence is "
            "gate-validated; full-case rows are resource-model estimates only. No quantum speedup "
            "or QSVT-over-Ridge superiority is claimed.",
            "",
        ]
    )
