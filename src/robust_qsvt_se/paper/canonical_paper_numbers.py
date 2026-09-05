"""Phase 7 hardening: canonical paper-number extraction and checking.

Builds a single canonical source of the numerical facts the manuscript cites, each tied
to the artifact it came from, so the paper text cannot drift from the evidence package.
Every number is read from a real artifact; an absent optional artifact is recorded as
``unavailable`` (not silently zero), and the JSON and Markdown outputs are generated from
the same dictionary so they always agree.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from robust_qsvt_se.paper import PAPER_CLAIM_BOUNDARY
from robust_qsvt_se.paper._common import read_csv, rows_to_table
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

SOURCE_SCRIPT = "scripts/check_paper_numbers.py"

SOURCE_MAP_COLUMNS = ["number_name", "value", "source_artifact", "status", "notes"]
QSVT_SCOPE = "selected 4x4 single-step QSVT state-estimation update solver prototype"
_UNAVAILABLE = "unavailable"

_CLAIM_CATEGORIES = (
    ("claim_count_supported", "supported"),
    ("claim_count_supported_with_limitations", "supported_with_limitations"),
    ("claim_count_assumption_only", "assumption_only"),
    ("claim_count_unsupported_do_not_claim", "unsupported_do_not_claim"),
)


def build_canonical_paper_numbers(config: dict[str, Any]) -> dict[str, Any]:
    input_root = Path(config.get("input_root", "outputs/final_manuscript_package"))
    output_dir = Path(config.get("output_dir", input_root / "canonical_numbers"))
    ensure_directory(output_dir)

    numbers: dict[str, Any] = {}
    source_rows: list[dict[str, Any]] = []

    def record(name: str, value: Any, source: str, notes: str = "") -> None:
        status = "unavailable" if value == _UNAVAILABLE else "recorded"
        numbers[name] = value
        source_rows.append(
            {
                "number_name": name,
                "value": value,
                "source_artifact": source,
                "status": status,
                "notes": notes,
            }
        )

    _record_test_status(input_root, record)
    _record_figures_tables(input_root, record)
    _record_claim_counts(input_root, record)
    record(
        "qsvt_scope", QSVT_SCOPE, "claim boundary", "selected-subproblem prototype, not full-scale"
    )
    _record_classical_alpha(input_root, record)
    _record_ablation_cases(input_root, record)
    _record_optional_rows(input_root, record)
    _record_full_vector_readout(input_root, record)
    _record_test_quality(input_root, record)
    _record_implementation_verification(input_root, record)
    _record_pre_manuscript_usability(input_root, record)
    _record_full_repo_audit(input_root, record)
    _record_pre_manuscript_final_gate(input_root, record)

    consistency = _consistency_checks(numbers)
    return _write_outputs(
        output_dir=output_dir,
        numbers=numbers,
        source_rows=source_rows,
        consistency=consistency,
        input_config={"input_root": str(input_root), "output_dir": str(output_dir)},
    )


def _record_test_status(input_root: Path, record: Any) -> None:
    freeze = input_root / "evidence_freeze" / "EVIDENCE_FREEZE.md"
    passed: Any = _UNAVAILABLE
    warnings: Any = _UNAVAILABLE
    if freeze.is_file():
        text = freeze.read_text(encoding="utf-8")
        passed_match = re.search(r"(\d+)\s+passed", text)
        warning_match = re.search(r"(\d+)\s+(?:pre-existing\s+)?warning", text)
        if passed_match:
            passed = int(passed_match.group(1))
        warnings = int(warning_match.group(1)) if warning_match else 0
    rel = "evidence_freeze/EVIDENCE_FREEZE.md"
    record("tests_passed", passed, rel, "parsed from the recorded full-suite status")
    record("test_warning_count", warnings, rel, "pre-existing warnings recorded at freeze")


def _record_figures_tables(input_root: Path, record: Any) -> None:
    figures = read_csv(input_root / "final_figures" / "final_figure_index.csv")
    record(
        "final_figures_rendered",
        int((figures.get("status", "").astype(str) == "rendered").sum())
        if not figures.empty
        else 0,
        "final_figures/final_figure_index.csv",
    )
    main = read_csv(input_root / "final_tables" / "main_table_index.csv")
    record(
        "main_tables_selected", 0 if main.empty else len(main), "final_tables/main_table_index.csv"
    )
    appendix = read_csv(input_root / "final_tables" / "appendix_table_index.csv")
    record(
        "appendix_tables_selected",
        0 if appendix.empty else len(appendix),
        "final_tables/appendix_table_index.csv",
    )


def _record_claim_counts(input_root: Path, record: Any) -> None:
    matrix = read_csv(input_root / "claim_support_matrix_final.csv")
    rel = "claim_support_matrix_final.csv"
    if matrix.empty or "support_status" not in matrix.columns:
        for name, _ in _CLAIM_CATEGORIES:
            record(name, _UNAVAILABLE, rel, "claim matrix not present")
        record("claim_count_total", _UNAVAILABLE, rel, "claim matrix not present")
        return
    counts = matrix["support_status"].astype(str).value_counts().to_dict()
    for name, status in _CLAIM_CATEGORIES:
        record(name, int(counts.get(status, 0)), rel, f"support_status == {status}")
    record("claim_count_total", len(matrix), rel, "total claims in the matrix")


def _record_classical_alpha(input_root: Path, record: Any) -> None:
    rel = "phase3_full_classical_alpha_sweep/figure_data_rmse_vs_alpha_all_cases.csv"
    frame = read_csv(input_root / rel)
    if frame.empty or "case" not in frame.columns:
        record("classical_alpha_cases", _UNAVAILABLE, rel, "alpha sweep not present")
        record("classical_alpha_grid", _UNAVAILABLE, rel, "alpha sweep not present")
        return
    record("classical_alpha_cases", int(frame["case"].astype(str).nunique()), rel)
    grid = int(frame["alpha"].nunique()) if "alpha" in frame.columns else _UNAVAILABLE
    record("classical_alpha_grid", grid, rel, "distinct alpha values in the sweep")


def _record_ablation_cases(input_root: Path, record: Any) -> None:
    ablation = read_csv(
        input_root / "phase5_measurement_type_ablation" / "condition_by_measurement_subset.csv"
    )
    record(
        "measurement_ablation_cases",
        int(ablation["case"].astype(str).nunique()) if "case" in ablation.columns else _UNAVAILABLE,
        "phase5_measurement_type_ablation/condition_by_measurement_subset.csv",
    )
    reactive = read_csv(
        input_root / "reactive_power_conditioning" / "reactive_power_conditioning_table.csv"
    )
    record(
        "reactive_conditioning_cases",
        int(reactive["case"].astype(str).nunique()) if "case" in reactive.columns else _UNAVAILABLE,
        "reactive_power_conditioning/reactive_power_conditioning_table.csv",
    )


def _record_optional_rows(input_root: Path, record: Any) -> None:
    traj_dir = input_root / "nonlinear_trajectories"
    total = 0
    found = False
    for name in (
        "figure_data_nonlinear_residual_vs_iteration.csv",
        "figure_data_nonlinear_update_norm_vs_iteration.csv",
        "figure_data_nonlinear_rmse_vs_iteration.csv",
        "figure_data_nonlinear_kappa_vs_iteration.csv",
    ):
        frame = read_csv(traj_dir / name)
        if not frame.empty:
            found = True
            total += len(frame)
    record(
        "nonlinear_trajectory_rows",
        total if found else _UNAVAILABLE,
        "nonlinear_trajectories/figure_data_nonlinear_*_vs_iteration.csv",
    )

    init_summary = read_csv(
        input_root / "initializer_ablation" / "initializer_ablation_summary.csv"
    )
    if init_summary.empty or "n_trials" not in init_summary.columns:
        record(
            "initializer_ablation_rows",
            _UNAVAILABLE,
            "initializer_ablation/initializer_ablation_summary.csv",
            "optional ablation not run",
        )
    else:
        record(
            "initializer_ablation_rows",
            int(init_summary["n_trials"].fillna(0).astype(int).sum()),
            "initializer_ablation/initializer_ablation_summary.csv",
            "sum of computed trials across initializers",
        )

    ieee300 = read_csv(input_root / "ieee300_runtime_extension" / "ieee300_alpha_sweep_reduced.csv")
    record(
        "ieee300_extension_rows",
        len(ieee300) if not ieee300.empty else _UNAVAILABLE,
        "ieee300_runtime_extension/ieee300_alpha_sweep_reduced.csv",
        "optional reduced IEEE300 extension",
    )


_FULL_VECTOR_SCOPE = (
    "selected small QSVT subproblems only; not efficient full IEEE-scale full-vector readout"
)


def _record_full_vector_readout(input_root: Path, record: Any) -> None:
    base = "full_vector_readout"
    statevector = read_csv(input_root / base / "statevector_full_vector_summary.csv")
    sampling = read_csv(input_root / base / "sampling_error_vs_shots.csv")
    signed = read_csv(input_root / base / "signed_vector_reconstruction_summary.csv")
    sv_rel = f"{base}/statevector_full_vector_summary.csv"

    if statevector.empty:
        record("full_vector_readout_subproblems", _UNAVAILABLE, sv_rel, "readout not run")
        record("full_vector_readout_cases", _UNAVAILABLE, sv_rel, "readout not run")
        record("full_vector_statevector_successes", _UNAVAILABLE, sv_rel, "readout not run")
        record("full_vector_best_relative_error", _UNAVAILABLE, sv_rel, "readout not run")
    else:
        record("full_vector_readout_subproblems", len(statevector), sv_rel)
        record(
            "full_vector_readout_cases",
            int(statevector["case"].astype(str).nunique()),
            sv_rel,
        )
        record(
            "full_vector_statevector_successes",
            int((statevector["status"].astype(str) == "pass").sum()),
            sv_rel,
            "vector_relative_l2_error_vs_ridge < 1e-5",
        )
        record(
            "full_vector_best_relative_error",
            float(statevector["vector_relative_l2_error_vs_ridge"].min()),
            sv_rel,
            "exact QSVT target vs Ridge at matched alpha",
        )

    if sampling.empty or "shots" not in sampling.columns:
        record(
            "full_vector_sampling_shot_levels",
            _UNAVAILABLE,
            f"{base}/sampling_error_vs_shots.csv",
            "readout not run",
        )
    else:
        record(
            "full_vector_sampling_shot_levels",
            int(sampling["shots"].nunique()),
            f"{base}/sampling_error_vs_shots.csv",
        )

    if signed.empty or "fraction_reliable_coordinates" not in signed.columns:
        record(
            "full_vector_reliable_sign_fraction",
            _UNAVAILABLE,
            f"{base}/signed_vector_reconstruction_summary.csv",
            "readout not run",
        )
    else:
        record(
            "full_vector_reliable_sign_fraction",
            round(float(signed["fraction_reliable_coordinates"].mean()), 6),
            f"{base}/signed_vector_reconstruction_summary.csv",
        )

    record(
        "full_vector_readout_scope",
        _FULL_VECTOR_SCOPE,
        "claim boundary",
        "selected-subproblem readout prototype, not full-scale",
    )
    _record_readout_hardening(input_root, record)


def _record_readout_hardening(input_root: Path, record: Any) -> None:
    base = "full_vector_readout"
    gate = read_csv(input_root / base / "gate_level_full_vector_summary.csv")
    gate_rel = f"{base}/gate_level_full_vector_summary.csv"
    if gate.empty or "gate_available" not in gate.columns:
        record("full_vector_gate_level_subproblems", _UNAVAILABLE, gate_rel, "readout not run")
        record("full_vector_exact_target_subproblems", _UNAVAILABLE, gate_rel, "readout not run")
    else:
        available = gate["gate_available"].astype(str).str.lower().isin({"true", "1"})
        record("full_vector_gate_level_subproblems", int(available.sum()), gate_rel)
        record("full_vector_exact_target_subproblems", len(gate), gate_rel)

    missing = read_csv(input_root / base / "missing_full_vector_readout_cases.csv")
    record(
        "full_vector_missing_subproblems",
        len(missing) if not missing.empty else 0,
        f"{base}/missing_full_vector_readout_cases.csv",
        "missing_phase_data combinations excluded from the success denominator",
    )

    trials = read_csv(input_root / base / "readout_confidence_intervals.csv")
    if trials.empty or "trials" not in trials.columns:
        record("full_vector_trials_per_shot_level", _UNAVAILABLE, f"{base}/...", "readout not run")
        record("full_vector_sampling_ci_available", False, f"{base}/...", "readout not run")
    else:
        record(
            "full_vector_trials_per_shot_level",
            int(trials["trials"].max()),
            f"{base}/readout_confidence_intervals.csv",
        )
        record(
            "full_vector_sampling_ci_available",
            True,
            f"{base}/readout_confidence_intervals.csv",
        )

    sign_audit = read_csv(input_root / base / "global_sign_audit.csv")
    rel = f"{base}/global_sign_audit.csv"
    if sign_audit.empty or "uses_ridge_for_reconstruction" not in sign_audit.columns:
        record("full_vector_global_sign_audit_passed", _UNAVAILABLE, rel, "readout not run")
        record("full_vector_no_ridge_leakage", _UNAVAILABLE, rel, "readout not run")
    else:
        no_leak = bool(
            (sign_audit["uses_ridge_for_reconstruction"].astype(str).str.lower() == "no").all()
        )
        record("full_vector_global_sign_audit_passed", no_leak, rel)
        record("full_vector_no_ridge_leakage", no_leak, rel, "no Ridge sign used in reconstruction")

    norm = read_csv(input_root / base / "shot_based_norm_recovery.csv")
    record(
        "full_vector_shot_based_norm_available",
        bool(not norm.empty and (norm.get("status", "").astype(str) == "norm_sampled").any()),
        f"{base}/shot_based_norm_recovery.csv",
    )

    decomposition = read_csv(input_root / base / "readout_error_decomposition.csv")
    record(
        "full_vector_error_decomposition_rows",
        len(decomposition) if not decomposition.empty else _UNAVAILABLE,
        f"{base}/readout_error_decomposition.csv",
    )

    total_cost = read_csv(input_root / base / "full_vector_readout_total_cost.csv")
    record(
        "full_vector_total_cost_model_available",
        bool(not total_cost.empty),
        f"{base}/full_vector_readout_total_cost.csv",
    )

    family = read_csv(input_root / "claim_boundaries" / "readout_claim_family_table.csv")
    if family.empty:
        family = read_csv(input_root / base / "readout_claim_family_table.csv")
    fam_rel = "claim_boundaries/readout_claim_family_table.csv"
    if family.empty or "counts_as_independent_claim" not in family.columns:
        record("full_vector_readout_claim_split_clean", _UNAVAILABLE, fam_rel, "split not present")
        record("full_vector_readout_independent_claims", _UNAVAILABLE, fam_rel, "split not present")
    else:
        by_id = dict(
            zip(
                family["claim_id"].astype(str),
                family["counts_as_independent_claim"].astype(str).str.lower(),
                strict=False,
            )
        )
        independent = int(sum(1 for value in by_id.values() if value == "yes"))
        # Clean iff the legacy parent (C10) is not double-counted and the two children are.
        clean = (
            by_id.get("C10") == "no" and by_id.get("C10a") == "yes" and by_id.get("C10b") == "yes"
        )
        record("full_vector_readout_claim_split_clean", bool(clean), fam_rel)
        record(
            "full_vector_readout_independent_claims",
            independent,
            fam_rel,
            "C10a + C10b; legacy parent C10 not counted",
        )

    _record_readout_codesign(input_root, record)


def _record_readout_codesign(input_root: Path, record: Any) -> None:
    base = "full_vector_readout"

    def _finite_max(frame: Any, column: str) -> Any:
        if frame.empty or column not in frame.columns:
            return _UNAVAILABLE
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        return float(series.max()) if not series.empty else _UNAVAILABLE

    classification = read_csv(input_root / base / "readout_config_diagnostic_classification.csv")
    sweep = read_csv(input_root / base / "readout_alpha_degree_sweep.csv")
    best = read_csv(input_root / base / "readout_selected_best_configs.csv")
    matched = read_csv(input_root / base / "readout_matched_alpha_view.csv")
    validated = read_csv(input_root / base / "readout_per_subproblem_validated_view.csv")
    summary = read_csv(input_root / base / "readout_two_view_summary.csv")
    decomp_fig = read_csv(input_root / base / "figure_data_error_decomposition_high_error.csv")

    if classification.empty and sweep.empty and summary.empty:
        for name in (
            "readout_codesign_targeted_configs",
            "readout_codesign_sweep_rows",
            "readout_codesign_best_configs",
            "readout_two_view_summary_rows",
            "readout_matched_alpha_max_error",
            "readout_validated_view_max_error",
            "readout_high_error_improved_count",
            "readout_polynomial_limited_count",
        ):
            record(name, _UNAVAILABLE, f"{base}/readout_two_view_summary.csv", "co-design not run")
        record("readout_two_view_available", False, f"{base}/readout_matched_alpha_view.csv")
        record(
            "readout_error_decomposition_high_error_available",
            False,
            f"{base}/figure_data_error_decomposition_high_error.csv",
        )
        return

    record(
        "readout_codesign_targeted_configs",
        len(best) if not best.empty else 0,
        f"{base}/readout_selected_best_configs.csv",
    )
    record(
        "readout_codesign_sweep_rows",
        len(sweep) if not sweep.empty else 0,
        f"{base}/readout_alpha_degree_sweep.csv",
    )
    selected = (
        int((pd.to_numeric(best["best_degree"], errors="coerce") > 0).sum())
        if not best.empty and "best_degree" in best.columns
        else 0
    )
    record(
        "readout_codesign_best_configs",
        selected,
        f"{base}/readout_selected_best_configs.csv",
        "configs with a bounded, in-window, gate-computed best selection",
    )
    record(
        "readout_two_view_summary_rows",
        len(summary) if not summary.empty else 0,
        f"{base}/readout_two_view_summary.csv",
    )
    record(
        "readout_matched_alpha_max_error",
        _finite_max(matched, "gate_error_vs_ridge"),
        f"{base}/readout_matched_alpha_view.csv",
        "diagnostic common-reference max gate-vs-Ridge error",
    )
    record(
        "readout_validated_view_max_error",
        _finite_max(validated, "gate_error_vs_ridge"),
        f"{base}/readout_per_subproblem_validated_view.csv",
        "per-subproblem QSVT-validated max gate-vs-Ridge error",
    )
    if not summary.empty and "final_status" in summary.columns:
        status = summary["final_status"].astype(str)
        record(
            "readout_high_error_improved_count",
            int((status == "improved_by_codesign").sum()),
            f"{base}/readout_two_view_summary.csv",
        )
        record(
            "readout_polynomial_limited_count",
            int((status == "polynomial_limited_after_sweep").sum()),
            f"{base}/readout_two_view_summary.csv",
        )
    else:
        record(
            "readout_high_error_improved_count", _UNAVAILABLE, f"{base}/...", "co-design not run"
        )
        record("readout_polynomial_limited_count", _UNAVAILABLE, f"{base}/...", "co-design not run")

    record(
        "readout_two_view_available",
        bool(not matched.empty and not validated.empty and not summary.empty),
        f"{base}/readout_two_view_summary.csv",
    )
    record(
        "readout_error_decomposition_high_error_available",
        bool(not decomp_fig.empty),
        f"{base}/figure_data_error_decomposition_high_error.csv",
    )
    _record_phase_refinement(input_root, record)


def _record_phase_refinement(input_root: Path, record: Any) -> None:
    base = "phase_synthesis_refinement"
    names = (
        "phase_refinement_targeted_configs",
        "phase_refinement_degree_47_49_rows",
        "phase_refinement_solver_variants_tested",
        "phase_refinement_fit_modes_tested",
        "phase_refinement_three_view_rows",
        "phase_refinement_pass_count",
        "phase_refinement_polynomial_limited_count",
        "phase_refinement_phase_limited_count",
        "phase_refinement_best_error",
        "phase_refinement_improved_below_0p1_count",
    )
    baseline = read_csv(input_root / base / "baseline_current_readout_codesign.csv")
    diagnostic = read_csv(input_root / base / "degree_47_49_diagnostic_sweep.csv")
    variants = read_csv(input_root / base / "phase_solver_variant_comparison.csv")
    fits = read_csv(input_root / base / "polynomial_fit_mode_comparison.csv")
    three_view = read_csv(input_root / base / "three_view_readout_summary.csv")
    leakage = read_csv(input_root / base / "selection_leakage_audit.csv")
    cost = read_csv(input_root / base / "cost_accuracy_tradeoff.csv")
    src = f"{base}/three_view_readout_summary.csv"

    if three_view.empty and baseline.empty:
        for name in names:
            record(name, _UNAVAILABLE, src, "phase-synthesis refinement not run")
        record("phase_refinement_selection_leakage_passed", _UNAVAILABLE, src, "refinement not run")
        record("phase_refinement_cost_accuracy_available", False, src)
        return

    relaxed = (
        three_view[three_view["view"] == "per_subproblem_relaxed_alpha_qsvt_validated"]
        if not three_view.empty and "view" in three_view.columns
        else three_view
    )
    status = relaxed["status"].astype(str) if "status" in relaxed.columns else pd.Series(dtype=str)
    errors = (
        pd.to_numeric(relaxed["gate_error_vs_matching_ridge_alpha"], errors="coerce").dropna()
        if "gate_error_vs_matching_ridge_alpha" in relaxed.columns
        else pd.Series(dtype=float)
    )
    record(
        "phase_refinement_targeted_configs",
        len(baseline),
        f"{base}/baseline_current_readout_codesign.csv",
    )
    record(
        "phase_refinement_degree_47_49_rows",
        len(diagnostic),
        f"{base}/degree_47_49_diagnostic_sweep.csv",
    )
    record(
        "phase_refinement_solver_variants_tested",
        int(variants["phase_solver_variant"].nunique())
        if "phase_solver_variant" in variants.columns
        else 0,
        f"{base}/phase_solver_variant_comparison.csv",
    )
    record(
        "phase_refinement_fit_modes_tested",
        int(fits["fit_mode"].nunique()) if "fit_mode" in fits.columns else 0,
        f"{base}/polynomial_fit_mode_comparison.csv",
    )
    record("phase_refinement_three_view_rows", len(three_view), src)
    record("phase_refinement_pass_count", int((status == "pass").sum()), src)
    record(
        "phase_refinement_polynomial_limited_count",
        int((status == "qsvt_polynomial_limited").sum()),
        src,
    )
    record(
        "phase_refinement_phase_limited_count",
        int((diagnostic["phase_synthesis_status"].astype(str) == "phase_synthesis_degraded").sum())
        if not diagnostic.empty and "phase_synthesis_status" in diagnostic.columns
        else 0,
        f"{base}/degree_47_49_diagnostic_sweep.csv",
    )
    record(
        "phase_refinement_best_error",
        float(errors.min()) if not errors.empty else _UNAVAILABLE,
        src,
        "lowest relaxed-view gate-vs-Ridge error after refinement",
    )
    record(
        "phase_refinement_improved_below_0p1_count",
        int((errors < 0.1).sum()) if not errors.empty else 0,
        src,
    )
    if not leakage.empty and "uses_true_state" in leakage.columns:
        uses_true = leakage["uses_true_state"].astype(str).str.lower().isin({"true", "1"}).any()
        rmse = (
            leakage["uses_state_rmse"].astype(str).str.lower().isin({"true", "1"}).any()
            if "uses_state_rmse" in leakage.columns
            else False
        )
        record(
            "phase_refinement_selection_leakage_passed",
            bool(not uses_true and not rmse),
            f"{base}/selection_leakage_audit.csv",
            "no selection stage uses true state or true-state RMSE",
        )
    else:
        record("phase_refinement_selection_leakage_passed", _UNAVAILABLE, src, "audit missing")
    record(
        "phase_refinement_cost_accuracy_available",
        bool(not cost.empty),
        f"{base}/cost_accuracy_tradeoff.csv",
    )


def _record_test_quality(input_root: Path, record: Any) -> None:
    rel = "test_quality_appendix/test_quality_appendix_table.csv"
    table = read_csv(input_root / rel)
    mapping = {}
    if not table.empty and {"metric", "value"}.issubset(table.columns):
        mapping = dict(zip(table["metric"].astype(str), table["value"], strict=False))
    record(
        "test_quality_total_tests",
        _as_int(mapping.get("total_tests_inventoried", _UNAVAILABLE)),
        rel,
    )
    record(
        "test_quality_real_validation_tests",
        _as_int(mapping.get("real_validation", _UNAVAILABLE)),
        rel,
    )
    record(
        "test_quality_smoke_only_tests",
        _as_int(mapping.get("smoke_only", _UNAVAILABLE)),
        rel,
    )
    # Full engineering pytest suite vs the smoke-free scientific-validation suite. The two
    # totals differ because pytest expands parametrized functions into multiple items.
    record(
        "full_pytest_tests_passed",
        _as_int(mapping.get("full_pytest_tests_passed", _UNAVAILABLE)),
        rel,
        "pytest items (parametrized cases counted separately)",
    )
    record(
        "test_quality_inventory_total",
        _as_int(mapping.get("test_quality_inventory_total", _UNAVAILABLE)),
        rel,
        "distinct test functions counted by the AST audit",
    )
    record(
        "scientific_validation_suite_tests",
        _as_int(mapping.get("scientific_validation_suite_tests", _UNAVAILABLE)),
        rel,
        "real_validation + claim_boundary_guard + integration_regression + artifact_schema",
    )
    record(
        "scientific_validation_suite_smoke_only_tests",
        _as_int(mapping.get("scientific_validation_suite_smoke_only_tests", _UNAVAILABLE)),
        rel,
        "zero by construction",
    )
    record(
        "engineering_smoke_tests",
        _as_int(mapping.get("engineering_smoke_tests", _UNAVAILABLE)),
        rel,
        "kept for engineering coverage; not scientific validation",
    )
    record(
        "remaining_smoke_only_tests",
        _as_int(mapping.get("remaining_smoke_only_tests", _UNAVAILABLE)),
        rel,
        "smoke tests still smoke after conversion",
    )


def _record_implementation_verification(input_root: Path, record: Any) -> None:
    jac_base = "jacobian_validation"
    ac = read_csv(input_root / jac_base / "ac_jacobian_validation_summary.csv")
    dc = read_csv(input_root / jac_base / "dc_jacobian_validation_summary.csv")
    weighted = read_csv(input_root / jac_base / "weighted_jacobian_consistency_audit.csv")
    metadata = read_csv(input_root / "measurement_row_metadata_audit" / "row_metadata_audit.csv")
    subsets = read_csv(
        input_root / "measurement_row_metadata_audit" / "subset_row_composition_summary.csv"
    )
    masks = read_csv(
        input_root / "measurement_row_metadata_audit" / "row_mask_consistency_checks.csv"
    )
    stats_dir = input_root / "statistical_summary"
    main_dir = input_root / "main_tables"
    stat_manifest = read_csv(stats_dir / "statistical_aggregation_manifest.csv")
    main_manifest = read_csv(main_dir / "main_tables_manifest.csv")

    ac_status, ac_cases, ac_error = _jacobian_numbers(ac)
    dc_status, dc_cases, dc_error = _jacobian_numbers(dc)
    record(
        "jacobian_validation_ac_cases",
        ac_cases,
        f"{jac_base}/ac_jacobian_validation_summary.csv",
    )
    record(
        "jacobian_validation_dc_cases",
        dc_cases,
        f"{jac_base}/dc_jacobian_validation_summary.csv",
    )
    record(
        "jacobian_validation_ac_max_relative_error",
        ac_error,
        f"{jac_base}/ac_jacobian_validation_summary.csv",
    )
    record(
        "jacobian_validation_dc_max_relative_error",
        dc_error,
        f"{jac_base}/dc_jacobian_validation_summary.csv",
    )
    jac_status = _combined_status([ac_status, dc_status, _weighted_status(weighted)])
    record("jacobian_validation_status", jac_status, jac_base)

    metadata_cases = (
        int(metadata["case"].astype(str).nunique()) if "case" in metadata.columns else _UNAVAILABLE
    )
    subsets_checked = (
        int((subsets["status"].astype(str) != "warning_subset_not_implemented").sum())
        if "status" in subsets.columns
        else _UNAVAILABLE
    )
    mask_failures = (
        int(masks["status"].astype(str).str.startswith("fail").sum())
        if "status" in masks.columns
        else _UNAVAILABLE
    )
    metadata_status = _metadata_status(subsets, masks)
    record(
        "measurement_metadata_cases",
        metadata_cases,
        "measurement_row_metadata_audit/row_metadata_audit.csv",
    )
    record(
        "measurement_metadata_subsets_checked",
        subsets_checked,
        "measurement_row_metadata_audit/subset_row_composition_summary.csv",
    )
    record(
        "measurement_metadata_mask_failures",
        mask_failures,
        "measurement_row_metadata_audit/row_mask_consistency_checks.csv",
    )
    record("measurement_metadata_status", metadata_status, "measurement_row_metadata_audit")

    statistical_tables = _count_expected_files(
        stats_dir,
        (
            "estimator_seed_variability_summary.csv",
            "nonlinear_convergence_summary.csv",
            "measurement_ablation_statistical_summary.csv",
            "reactive_conditioning_statistical_summary.csv",
            "readout_sampling_statistical_summary.csv",
            "phase_refinement_statistical_summary.csv",
        ),
    )
    # T1-T8 have descriptive names, so count by prefix instead of constructing every name.
    main_tables = len(sorted(main_dir.glob("T[1-8]_*.csv"))) if main_dir.is_dir() else _UNAVAILABLE
    record("statistical_summary_tables", statistical_tables, "statistical_summary")
    record("main_paper_tables", main_tables if main_dir.is_dir() else _UNAVAILABLE, "main_tables")
    main_status = (
        "pass"
        if not main_manifest.empty
        and "status" in main_manifest.columns
        and not (main_manifest["status"].astype(str) != "pass").any()
        else (_UNAVAILABLE if main_manifest.empty else "warning")
    )
    record("main_table_manifest_status", main_status, "main_tables/main_tables_manifest.csv")

    complete = (
        jac_status == "pass"
        and metadata_status in {"pass", "warning"}
        and mask_failures == 0
        and not stat_manifest.empty
        and main_status == "pass"
    )
    record(
        "implementation_verification_complete",
        bool(complete),
        "jacobian_validation; measurement_row_metadata_audit; statistical_summary; main_tables",
    )


def _jacobian_numbers(frame: pd.DataFrame) -> tuple[str, Any, Any]:
    if frame.empty or "status" not in frame.columns:
        return _UNAVAILABLE, _UNAVAILABLE, _UNAVAILABLE
    statuses = frame["status"].astype(str)
    if statuses.str.contains("fail|build_failed").any():
        status = "fail"
    elif statuses[~statuses.isin(["pass", "not_implemented"])].empty:
        status = "pass" if (statuses == "pass").any() else "not_applicable"
    else:
        status = "warning"
    cases = (
        int(frame.loc[statuses == "pass", "case"].astype(str).nunique())
        if "case" in frame.columns
        else _UNAVAILABLE
    )
    if "frobenius_relative_error" not in frame.columns:
        return status, cases, _UNAVAILABLE
    values = pd.to_numeric(frame["frobenius_relative_error"], errors="coerce").dropna()
    return status, cases, (float(values.max()) if not values.empty else _UNAVAILABLE)


def _weighted_status(frame: pd.DataFrame) -> str:
    if frame.empty or "status" not in frame.columns:
        return _UNAVAILABLE
    statuses = frame["status"].astype(str)
    if statuses.isin(["build_failed", "invalid_variance"]).any():
        return "fail"
    return "pass" if (statuses == "consistent").all() else "warning"


def _metadata_status(subsets: pd.DataFrame, masks: pd.DataFrame) -> str:
    if subsets.empty and masks.empty:
        return _UNAVAILABLE
    statuses = []
    if "status" in subsets.columns:
        statuses.extend(subsets["status"].astype(str).tolist())
    if "status" in masks.columns:
        statuses.extend(masks["status"].astype(str).tolist())
    if any(status.startswith("fail") for status in statuses):
        return "fail"
    if any(status.startswith("warning") for status in statuses):
        return "warning"
    return "pass"


def _combined_status(statuses: list[Any]) -> Any:
    if all(status == _UNAVAILABLE for status in statuses):
        return _UNAVAILABLE
    if any(status == "fail" for status in statuses):
        return "fail"
    if any(status == "warning" for status in statuses):
        return "warning"
    if any(status == _UNAVAILABLE for status in statuses):
        return "warning"
    return "pass"


def _record_pre_manuscript_usability(input_root: Path, record: Any) -> None:
    rel = "pre_manuscript_usability_audit/pre_manuscript_usability_scorecard.csv"
    scorecard = read_csv(input_root / rel)
    if scorecard.empty or "category" not in scorecard.columns:
        for name in (
            "pre_manuscript_blocker_count",
            "pre_manuscript_warning_count",
            "pre_manuscript_usable_main_artifacts",
            "pre_manuscript_usable_appendix_artifacts",
            "pre_manuscript_diagnostic_only_artifacts",
            "pre_manuscript_future_work_artifacts",
            "pre_manuscript_overall_status",
        ):
            record(name, _UNAVAILABLE, rel, "pre-manuscript usability audit not present")
        return
    overall_rows = scorecard[scorecard["category"].astype(str) == "overall"]
    overall = overall_rows.iloc[0] if not overall_rows.empty else scorecard.iloc[-1]
    record(
        "pre_manuscript_blocker_count",
        _as_int(overall.get("blocker_count", _UNAVAILABLE)),
        rel,
    )
    record(
        "pre_manuscript_warning_count",
        _as_int(overall.get("warning_count", _UNAVAILABLE)),
        rel,
    )
    record(
        "pre_manuscript_usable_main_artifacts",
        _as_int(overall.get("usable_main_count", _UNAVAILABLE)),
        rel,
    )
    record(
        "pre_manuscript_usable_appendix_artifacts",
        _as_int(overall.get("usable_appendix_count", _UNAVAILABLE)),
        rel,
    )
    record(
        "pre_manuscript_diagnostic_only_artifacts",
        _as_int(overall.get("diagnostic_only_count", _UNAVAILABLE)),
        rel,
    )
    record(
        "pre_manuscript_future_work_artifacts",
        _as_int(overall.get("future_work_count", _UNAVAILABLE)),
        rel,
    )
    record(
        "pre_manuscript_overall_status",
        str(overall.get("status", _UNAVAILABLE)),
        rel,
    )


def _record_full_repo_audit(input_root: Path, record: Any) -> None:
    base = input_root / "full_repo_evidence_audit"
    score_rel = "full_repo_evidence_audit/full_repo_evidence_scorecard.csv"
    completion_rel = "full_repo_evidence_audit/experiment_completion_matrix.csv"
    tests_rel = "full_repo_evidence_audit/test_quality_full_repo_audit.csv"
    suspicious_rel = "full_repo_evidence_audit/suspicious_result_audit.csv"
    claims_rel = "full_repo_evidence_audit/claim_boundary_full_repo_audit.csv"

    scorecard = read_csv(input_root / score_rel)
    overall = _scorecard_overall(scorecard)
    if overall is None:
        for name in (
            "full_repo_audit_overall_status",
            "full_repo_audit_blocker_count",
            "full_repo_audit_needs_review_count",
            "full_repo_audit_warning_count",
        ):
            record(name, _UNAVAILABLE, score_rel, "full-repo audit not present")
    else:
        record("full_repo_audit_overall_status", str(overall.get("status", "")), score_rel)
        record(
            "full_repo_audit_blocker_count",
            _as_int(overall.get("blocker_count", _UNAVAILABLE)),
            score_rel,
        )
        record(
            "full_repo_audit_needs_review_count",
            _as_int(overall.get("needs_review_count", _UNAVAILABLE)),
            score_rel,
        )
        record(
            "full_repo_audit_warning_count",
            _as_int(overall.get("warning_count", _UNAVAILABLE)),
            score_rel,
        )

    completion = read_csv(input_root / completion_rel)
    _record_completion_count(
        record,
        "full_repo_audit_experiments_complete",
        completion,
        "complete",
        completion_rel,
    )
    _record_completion_count(
        record,
        "full_repo_audit_experiments_complete_with_limitations",
        completion,
        "complete_with_limitations",
        completion_rel,
    )
    _record_completion_count(
        record,
        "full_repo_audit_experiments_future_work",
        completion,
        "future_work",
        completion_rel,
    )

    tests = read_csv(base / "test_quality_full_repo_audit.csv")
    if tests.empty or not {"supports_scientific_claim", "is_smoke_only"}.issubset(tests.columns):
        record(
            "full_repo_audit_scientific_validation_smoke_only_count",
            _UNAVAILABLE,
            tests_rel,
            "test-quality audit not present",
        )
    else:
        sci = tests["supports_scientific_claim"].astype(str).str.lower().isin({"yes", "true"})
        smoke = tests["is_smoke_only"].astype(str).str.lower().isin({"yes", "true"})
        record(
            "full_repo_audit_scientific_validation_smoke_only_count",
            int((sci & smoke).sum()),
            tests_rel,
        )

    suspicious = read_csv(input_root / suspicious_rel)
    if suspicious.empty or "is_blocker" not in suspicious.columns:
        record(
            "full_repo_audit_suspicious_blockers",
            _UNAVAILABLE,
            suspicious_rel,
            "suspicious-result audit not present",
        )
    else:
        record(
            "full_repo_audit_suspicious_blockers",
            int(suspicious["is_blocker"].astype(str).str.lower().isin({"yes", "true"}).sum()),
            suspicious_rel,
        )

    claims = read_csv(input_root / claims_rel)
    if claims.empty or "risk_level" not in claims.columns:
        record(
            "full_repo_audit_claim_high_risk_count",
            _UNAVAILABLE,
            claims_rel,
            "claim-boundary full-repo audit not present",
        )
    else:
        high = claims["risk_level"].astype(str).isin({"high", "blocker"})
        record("full_repo_audit_claim_high_risk_count", int(high.sum()), claims_rel)


def _record_pre_manuscript_final_gate(input_root: Path, record: Any) -> None:
    base = input_root / "pre_manuscript_final_gate"
    decision_rel = "pre_manuscript_final_gate/manuscript_readiness_decision.csv"
    triage_rel = "pre_manuscript_final_gate/needs_review_triage.csv"
    freeze_rel = "pre_manuscript_final_gate/final_evidence_freeze_manifest.csv"

    decision = read_csv(input_root / decision_rel)
    if decision.empty or "decision_item" not in decision.columns:
        status = _UNAVAILABLE
        record(
            "pre_manuscript_final_gate_status",
            status,
            decision_rel,
            "pre-manuscript final gate not present",
        )
        record(
            "pre_manuscript_final_gate_ready_for_manuscript",
            _UNAVAILABLE,
            decision_rel,
            "pre-manuscript final gate not present",
        )
    else:
        row = decision[decision["decision_item"].astype(str) == "manuscript_readiness"]
        status = str(row.iloc[0].get("value", _UNAVAILABLE)) if not row.empty else _UNAVAILABLE
        record("pre_manuscript_final_gate_status", status, decision_rel)
        record(
            "pre_manuscript_final_gate_ready_for_manuscript",
            "yes" if status in {"ready_for_manuscript", "ready_with_minor_notes"} else "no",
            decision_rel,
            "yes means the gate permits manuscript writing with any recorded notes",
        )

    triage = read_csv(base / "needs_review_triage.csv")
    if triage.empty or "triage_label" not in triage.columns:
        for name in (
            "pre_manuscript_final_gate_needs_review_total",
            "pre_manuscript_final_gate_blocker_count",
            "pre_manuscript_final_gate_main_claim_note_count",
            "pre_manuscript_final_gate_appendix_note_count",
            "pre_manuscript_final_gate_diagnostic_only_count",
            "pre_manuscript_final_gate_future_work_count",
            "pre_manuscript_final_gate_ignore_safe_count",
        ):
            record(name, _UNAVAILABLE, triage_rel, "needs-review triage not present")
    else:
        labels = triage["triage_label"].astype(str)
        record("pre_manuscript_final_gate_needs_review_total", len(triage), triage_rel)
        record(
            "pre_manuscript_final_gate_blocker_count",
            int((labels == "main_claim_blocker").sum()),
            triage_rel,
        )
        record(
            "pre_manuscript_final_gate_main_claim_note_count",
            int((labels == "main_claim_note").sum()),
            triage_rel,
        )
        record(
            "pre_manuscript_final_gate_appendix_note_count",
            int((labels == "appendix_note").sum()),
            triage_rel,
        )
        record(
            "pre_manuscript_final_gate_diagnostic_only_count",
            int((labels == "diagnostic_only_ok").sum()),
            triage_rel,
        )
        record(
            "pre_manuscript_final_gate_future_work_count",
            int((labels == "future_work_note").sum()),
            triage_rel,
        )
        record(
            "pre_manuscript_final_gate_ignore_safe_count",
            int((labels == "ignore_safe").sum()),
            triage_rel,
        )

    freeze = read_csv(base / "final_evidence_freeze_manifest.csv")
    if freeze.empty or not {"baseline_item", "value"}.issubset(freeze.columns):
        record(
            "pre_manuscript_final_gate_scientific_validation_smoke_only_count",
            _UNAVAILABLE,
            freeze_rel,
            "final-gate freeze manifest not present",
        )
    else:
        item = freeze[
            freeze["baseline_item"].astype(str) == "scientific_validation_smoke_only_count"
        ]
        value = _UNAVAILABLE if item.empty else _as_int(item.iloc[0].get("value", _UNAVAILABLE))
        record(
            "pre_manuscript_final_gate_scientific_validation_smoke_only_count",
            value,
            freeze_rel,
        )


def _scorecard_overall(scorecard: pd.DataFrame) -> pd.Series[Any] | None:
    if scorecard.empty or "category" not in scorecard.columns:
        return None
    overall = scorecard[scorecard["category"].astype(str) == "overall"]
    if overall.empty:
        return None
    return overall.iloc[0]


def _record_completion_count(
    record: Any,
    name: str,
    completion: pd.DataFrame,
    status: str,
    source: str,
) -> None:
    if completion.empty or "completion_status" not in completion.columns:
        record(name, _UNAVAILABLE, source, "completion matrix not present")
        return
    record(name, int((completion["completion_status"].astype(str) == status).sum()), source)


def _count_expected_files(directory: Path, names: tuple[str, ...]) -> Any:
    if not directory.is_dir():
        return _UNAVAILABLE
    return sum(1 for name in names if (directory / name).is_file())


def _as_int(value: Any) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return _UNAVAILABLE


def _consistency_checks(numbers: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    category_values = [numbers.get(name) for name, _ in _CLAIM_CATEGORIES]
    total = numbers.get("claim_count_total")
    if all(isinstance(value, int) for value in category_values) and isinstance(total, int):
        checks["claim_counts_sum_to_total"] = sum(category_values) == total
    else:
        checks["claim_counts_sum_to_total"] = "unavailable"
    scope = str(numbers.get("qsvt_scope", ""))
    checks["qsvt_scope_is_selected_subproblem"] = (
        "selected" in scope.lower() and "full ieee-scale" not in scope.lower()
    )
    smoke_in_suite = numbers.get("scientific_validation_suite_smoke_only_tests")
    checks["scientific_validation_suite_is_smoke_free"] = (
        smoke_in_suite == 0 if isinstance(smoke_in_suite, int) else "unavailable"
    )
    return checks


def _markdown(numbers: dict[str, Any], consistency: dict[str, Any]) -> str:
    lines = [
        "# Canonical Paper Numbers",
        "",
        PAPER_CLAIM_BOUNDARY,
        "",
        "Single source of the numerical facts the manuscript cites. Each value is traced to "
        "an artifact in `number_source_map.csv`; an absent optional artifact is `unavailable` "
        "(never silently zero).",
        "",
        "## Numbers",
        *[f"- `{name}` = {value}" for name, value in numbers.items()],
        "",
        "## Consistency checks",
        *[f"- {name}: {value}" for name, value in consistency.items()],
        "",
    ]
    return "\n".join(lines)


def _write_outputs(
    *,
    output_dir: Path,
    numbers: dict[str, Any],
    source_rows: list[dict[str, Any]],
    consistency: dict[str, Any],
    input_config: dict[str, Any],
) -> dict[str, Any]:
    json_path = output_dir / "canonical_paper_numbers.json"
    json_path.write_text(
        json.dumps({"numbers": numbers, "consistency": consistency}, indent=2),
        encoding="utf-8",
    )
    md_path = output_dir / "canonical_paper_numbers.md"
    md_path.write_text(_markdown(numbers, consistency), encoding="utf-8")
    map_path = rows_to_table(source_rows, output_dir / "number_source_map.csv", SOURCE_MAP_COLUMNS)

    artifacts = {
        "canonical_paper_numbers_json": str(json_path),
        "canonical_paper_numbers_md": str(md_path),
        "number_source_map": str(map_path),
    }
    write_manifest(
        output_dir,
        artifacts=artifacts,
        input_config=input_config,
        claim_boundary=PAPER_CLAIM_BOUNDARY,
    )
    return {
        "output_dir": output_dir,
        "numbers": numbers,
        "source_rows": source_rows,
        "consistency": consistency,
        "artifacts": artifacts,
    }
