from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory


def build_engineering_extension_summary(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    rows = _claim_rows()
    frame = pd.DataFrame(rows)
    csv_path = output_dir / "claim_support_matrix.csv"
    summary_path = output_dir / "summary.md"
    final_summary_path = output_dir / "final_engineering_summary.md"
    frame.to_csv(csv_path, index=False)
    summary_path.write_text(_summary_markdown(rows), encoding="utf-8")
    final_summary_path.write_text(_final_summary_markdown(rows), encoding="utf-8")
    manifest_path = write_manifest(
        output_dir,
        artifacts={
            "claim_support_matrix_csv": str(csv_path),
            "summary_md": str(summary_path),
            "final_engineering_summary_md": str(final_summary_path),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": frame,
        "artifacts": {
            "claim_support_matrix_csv": csv_path,
            "summary_md": summary_path,
            "final_engineering_summary_md": final_summary_path,
            "manifest": manifest_path,
        },
    }


def _claim_rows() -> list[dict[str, str]]:
    return [
        {
            "claim": "Dense block-encoding prototype was validated on small normalized matrices.",
            "support_status": "supported_for_dense_prototype",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/block_encoding.py",
                    "src/robust_qsvt_se/qsvt/block_encoding_demo.py",
                ]
            ),
            "supporting_outputs": "outputs/qsvt_block_encoding/block_encoding_summary.csv",
            "strength": "small dense validation",
            "limitations": "Dense Julia block encoding only; no scalable oracle decomposition.",
            "recommended_wording": (
                "We validate dense block-encoding prototypes for normalized weighted "
                "Jacobian matrices and submatrices."
            ),
            "avoid_wording": "We implement scalable full-system block-encoding oracles.",
        },
        {
            "claim": (
                "Exact QSVT-target spectral filtering matches Ridge/Tikhonov under the same alpha."
            ),
            "support_status": "supported",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/state_demo.py",
                    "src/robust_qsvt_se/estimators/ridge.py",
                    "src/robust_qsvt_se/estimators/qsvt_spectral.py",
                ]
            ),
            "supporting_outputs": "outputs/qsvt_end_to_end_state_demo/state_demo_summary.csv",
            "strength": "exact numerical equivalence check",
            "limitations": (
                "This is expected equivalence of the same spectral filter, not superiority."
            ),
            "recommended_wording": (
                "The exact QSVT-target spectral simulator reproduces Ridge/Tikhonov "
                "for the same alpha."
            ),
            "avoid_wording": "QSVT numerically outperforms Ridge for the same alpha.",
        },
        {
            "claim": "Selected-alpha bounded polynomial/phase approximations were validated.",
            "support_status": "supported_with_pass_fail_rows",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/selected_alpha_validation.py",
                    "src/robust_qsvt_se/qsvt/polynomial.py",
                ]
            ),
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_selected_alpha_phase_validation/phase_validation_summary.csv",
                    "outputs/qsvt_selected_alpha_phase_validation/pointwise_errors.csv",
                ]
            ),
            "strength": "backend-free bounded polynomial approximation diagnostic",
            "limitations": (
                "Rows may fail the configured tolerance; this is polynomial fallback "
                "validation, not full QSVT phase synthesis or hardware execution."
            ),
            "recommended_wording": (
                "Selected alpha values were checked against a bounded odd-polynomial "
                "approximation target with explicit pass/fail tolerance."
            ),
            "avoid_wording": "Selected-alpha validation proves hardware-ready QSVT phases.",
        },
        {
            "claim": "Selected-alpha polynomial approximation diagnostics were implemented.",
            "support_status": "supported_with_diagnostics",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/selected_alpha_validation.py",
                    "src/robust_qsvt_se/qsvt/polynomial_approximation.py",
                ]
            ),
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_selected_alpha_phase_validation/phase_validation_summary.csv",
                    "outputs/qsvt_polynomial_method_comparison/method_comparison_summary.csv",
                ]
            ),
            "strength": "selected-alpha approximation diagnostics",
            "limitations": (
                "Polynomial diagnostics are not full QSP/QSVT phase synthesis unless "
                "the optional phase-synthesis report succeeds."
            ),
            "recommended_wording": (
                "Selected-alpha polynomial diagnostics quantify approximation error "
                "for bounded QSVT-compatible targets."
            ),
            "avoid_wording": "Selected-alpha polynomial diagnostics are hardware validation.",
        },
        {
            "claim": "Degree sweep quantifies approximation error versus resource cost.",
            "support_status": "supported",
            "supporting_files": "src/robust_qsvt_se/qsvt/approximation_degree_sweep.py",
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_approximation_degree_sweep/degree_sweep_summary.csv",
                    "outputs/qsvt_approximation_degree_sweep/degree_sweep_pointwise_errors.csv",
                ]
            ),
            "strength": "degree-error-query sweep",
            "limitations": (
                "Resource cost is represented by query-count proxy `2 * degree + 1`; "
                "oracle, loading, and fault-tolerant costs are not implemented."
            ),
            "recommended_wording": (
                "The degree sweep quantifies maximum pointwise error and query-count "
                "proxy as polynomial degree increases."
            ),
            "avoid_wording": "The degree sweep demonstrates quantum speedup.",
        },
        {
            "claim": "Adaptive degree selection identifies whether target tolerances can be met.",
            "support_status": "supported",
            "supporting_files": "src/robust_qsvt_se/qsvt/adaptive_degree_selection.py",
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_adaptive_degree_selection/adaptive_degree_summary.csv",
                    "outputs/qsvt_adaptive_degree_selection/adaptive_search_trace.csv",
                ]
            ),
            "strength": "configured adaptive search trace",
            "limitations": (
                "The selected degree is the smallest passing value in the configured "
                "candidate grid, not a proof of global minimax optimality."
            ),
            "recommended_wording": (
                "Adaptive degree selection reports the smallest configured degree "
                "meeting each target tolerance, or an explicit failure status."
            ),
            "avoid_wording": "Adaptive degree selection proves optimal QSP degree.",
        },
        {
            "claim": (
                "Preconditioning reduces QSVT-compatible approximation difficulty "
                "for selected alpha settings."
            ),
            "support_status": "supported_with_phase2_diagnostics",
            "supporting_files": "src/robust_qsvt_se/qsvt/phase2_completion.py",
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_phase2_complete_summary/phase2_complete_summary.csv",
                    "outputs/qsvt_phase2_figures/fig_phase2_ieee300_qsvt_error_vs_alpha.png",
                ]
            ),
            "strength": "controlled alpha/preconditioning diagnostics",
            "limitations": (
                "Alpha-dependent diagnostic; preconditioned IEEE300 rows do not "
                "make original IEEE300 pass."
            ),
            "recommended_wording": (
                "Preconditioning reduces QSVT-compatible approximation difficulty "
                "for selected alpha settings in the controlled benchmark."
            ),
            "avoid_wording": (
                "Preconditioning proves the original IEEE300 matrix passes the same diagnostic."
            ),
        },
        {
            "claim": (
                "Coordinate-preconditioned Ridge is a separate estimator and can "
                "degrade residual/RMSE."
            ),
            "support_status": "supported_with_phase2_diagnostics",
            "supporting_files": "src/robust_qsvt_se/qsvt/phase2_preconditioned_alpha.py",
            "supporting_outputs": (
                "outputs/qsvt_phase2_complete_summary/phase2_variant_comparison.csv"
            ),
            "strength": "variant-separated metrics",
            "limitations": "Coordinate penalty changes the regularization geometry.",
            "recommended_wording": (
                "Coordinate-preconditioned Ridge is evaluated as a separate estimator "
                "and can degrade residual/RMSE."
            ),
            "avoid_wording": "Coordinate-preconditioned Ridge replaces original Ridge.",
        },
        {
            "claim": (
                "Transformed-penalty preconditioning preserves the original x-space Ridge penalty."
            ),
            "support_status": "supported_by_equation_and_diagnostic",
            "supporting_files": "src/robust_qsvt_se/qsvt/phase2_completion.py",
            "supporting_outputs": (
                "outputs/qsvt_phase2_manuscript_text/transformed_penalty_explanation.md"
            ),
            "strength": "objective-level consistency explanation",
            "limitations": "Consistency-preserving diagnostic, not a new superiority claim.",
            "recommended_wording": (
                "The transformed-penalty formulation preserves the original x-space "
                "Ridge penalty while using the preconditioned matrix for diagnostics."
            ),
            "avoid_wording": "Coordinate and transformed-penalty formulations are equivalent.",
        },
        {
            "claim": "Alpha selection is diagnostic, not field-calibrated.",
            "support_status": "explicit_boundary",
            "supporting_files": "src/robust_qsvt_se/qsvt/phase2_preconditioned_alpha.py",
            "supporting_outputs": ("outputs/qsvt_phase2_alpha_selection/alpha_selection_report.md"),
            "strength": "explicit report caveat",
            "limitations": "No operational field-calibrated alpha rule is claimed.",
            "recommended_wording": (
                "Alpha selection is diagnostic and controlled-benchmark-specific."
            ),
            "avoid_wording": "The Phase 2 alpha score is a deployment-ready rule.",
        },
        {
            "claim": "Optional phase synthesis is performed only if dependencies are available.",
            "support_status": "supported_with_skip_path",
            "supporting_files": ("src/robust_qsvt_se/qsvt/optional_phase_synthesis_validation.py"),
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_optional_phase_synthesis_validation/phase_synthesis_summary.csv",
                    "outputs/qsvt_optional_phase_synthesis_validation/phase_angles.csv",
                ]
            ),
            "strength": "dependency-safe optional phase attempt",
            "limitations": (
                "A synthesized phase sequence may still fail response validation; "
                "skipped and failed rows are reported explicitly."
            ),
            "recommended_wording": (
                "Optional phase synthesis is attempted only when the configured "
                "dependency is available, and pass/fail status is reported."
            ),
            "avoid_wording": "Full phase synthesis is always available and passing.",
        },
        {
            "claim": (
                "Phase-response convention diagnostics validate the PennyLane scalar "
                "response convention."
            ),
            "support_status": "supported_for_scalar_diagnostic",
            "supporting_files": "src/robust_qsvt_se/qsvt/phase_response_conventions.py",
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_phase_response_convention_diagnostics/"
                    "convention_search_summary.csv",
                    "outputs/qsvt_phase_response_convention_diagnostics/best_convention_report.md",
                ]
            ),
            "strength": "scalar phase-response convention diagnostic",
            "limitations": (
                "This validates the scalar response convention and coefficient basis; "
                "it is not hardware execution or a scalable matrix-level circuit."
            ),
            "recommended_wording": (
                "Phase-response diagnostics identify a PennyLane `RX`/`PCPhase` "
                "scalar convention with `real(U[0,0])` response."
            ),
            "avoid_wording": (
                "The phase-response convention diagnostic executes full quantum hardware."
            ),
        },
        {
            "claim": (
                "Known sanity-polynomial QSP/QSVT responses are checked before "
                "Ridge-target validation."
            ),
            "support_status": "supported",
            "supporting_files": "src/robust_qsvt_se/qsvt/phase_response_conventions.py",
            "supporting_outputs": (
                "outputs/qsvt_phase_response_convention_diagnostics/sanity_polynomial_results.csv"
            ),
            "strength": "sanity-polynomial response tests",
            "limitations": (
                "Passing sanity polynomials does not imply the bounded Ridge target "
                "passes a stricter tolerance at the configured degree."
            ),
            "recommended_wording": (
                "Known polynomials such as `x`, `0.5x`, and `x^3` are used to "
                "verify the response convention before reporting Ridge-target status."
            ),
            "avoid_wording": "Sanity polynomials prove full Ridge-target phase validation.",
        },
        {
            "claim": (
                "Full phase-level Ridge/Tikhonov target validation remains unresolved "
                "when reported failed."
            ),
            "support_status": "explicit_boundary",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/optional_phase_synthesis_validation.py",
                    "docs/QSVT_PHASE_RESPONSE_CONVENTIONS.md",
                ]
            ),
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_optional_phase_synthesis_validation/phase_synthesis_summary.csv",
                    "outputs/qsvt_phase_response_convention_diagnostics/"
                    "convention_search_summary.csv",
                ]
            ),
            "strength": "pass/fail phase-level boundary",
            "limitations": (
                "High-degree stable phase synthesis in monomial basis remains a future "
                "work item when the optional phase row fails the target tolerance."
            ),
            "recommended_wording": (
                "The convention issue is diagnosed separately from the remaining "
                "phase-level approximation tolerance for the bounded Ridge target."
            ),
            "avoid_wording": "The bounded Ridge target has fully passing QSP/QSVT phases.",
        },
        {
            "claim": "Polynomial fallback is not full QSP/QSVT phase synthesis.",
            "support_status": "explicit_boundary",
            "supporting_files": _join(
                [
                    "docs/QSVT_APPROXIMATION_VALIDATION.md",
                    "docs/QSVT_ENGINEERING_EXTENSION.md",
                ]
            ),
            "supporting_outputs": (
                "outputs/qsvt_polynomial_method_comparison/method_comparison_summary.csv"
            ),
            "strength": "claim boundary",
            "limitations": (
                "Polynomial approximation evidence does not by itself provide phase "
                "angles or hardware execution."
            ),
            "recommended_wording": (
                "Polynomial fallback rows are bounded polynomial approximation "
                "diagnostics, not full QSP/QSVT phase synthesis."
            ),
            "avoid_wording": "Polynomial fallback is full QSP/QSVT phase synthesis.",
        },
        {
            "claim": "Passing/failing 1e-3 tolerance is reported explicitly.",
            "support_status": "supported",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/approximation_degree_sweep.py",
                    "src/robust_qsvt_se/qsvt/adaptive_degree_selection.py",
                    "src/robust_qsvt_se/qsvt/polynomial_method_comparison.py",
                ]
            ),
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_approximation_degree_sweep/degree_sweep_summary.csv",
                    "outputs/qsvt_adaptive_degree_selection/adaptive_degree_summary.csv",
                    "outputs/qsvt_polynomial_method_comparison/method_comparison_summary.csv",
                ]
            ),
            "strength": "explicit tolerance columns and statuses",
            "limitations": "Tolerance status depends on the configured grid and method.",
            "recommended_wording": (
                "The reports explicitly identify which degree and method pass or fail "
                "the strict 1e-3 maximum pointwise-error tolerance."
            ),
            "avoid_wording": "The 1e-3 tolerance passed when the report shows failure.",
        },
        {
            "claim": "Query count increases with polynomial degree.",
            "support_status": "supported_by_resource_proxy",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/approximation_degree_sweep.py",
                    "src/robust_qsvt_se/qsvt/approximation_tradeoff_report.py",
                ]
            ),
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_approximation_degree_sweep/degree_sweep_summary.csv",
                    "outputs/qsvt_approximation_tradeoff/query_vs_degree.csv",
                ]
            ),
            "strength": "query-count proxy",
            "limitations": (
                "Query count is a proxy `2 * degree + 1`; it excludes oracle "
                "decomposition and state-preparation cost."
            ),
            "recommended_wording": (
                "Higher polynomial degree increases the QSVT query-count proxy."
            ),
            "avoid_wording": "Query count is the only relevant hardware cost.",
        },
        {
            "claim": "Approximation diagnostics support feasibility discussion only.",
            "support_status": "supported_with_caveat",
            "supporting_files": _join(
                [
                    "docs/QSVT_APPROXIMATION_VALIDATION.md",
                    "src/robust_qsvt_se/qsvt/approximation_tradeoff_report.py",
                ]
            ),
            "supporting_outputs": ("outputs/qsvt_approximation_tradeoff/tradeoff_report.md"),
            "strength": "claim-safe approximation evidence",
            "limitations": (
                "Diagnostics omit complete fault-tolerant compilation, oracle "
                "synthesis, and full readout."
            ),
            "recommended_wording": (
                "Approximation diagnostics support resource-aware feasibility "
                "discussion for a QSVT-compatible pathway."
            ),
            "avoid_wording": ("Approximation diagnostics demonstrate quantum advantage."),
        },
        {
            "claim": "QSVT resource estimates support feasibility discussion only.",
            "support_status": "supported_with_caveat",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/resource_estimator.py",
                    "src/robust_qsvt_se/qsvt/readout_analysis.py",
                ]
            ),
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_resource_readout/resource_summary.csv",
                    "outputs/qsvt_resource_readout/resource_assumptions.md",
                ]
            ),
            "strength": "proxy resource analysis",
            "limitations": (
                "No oracle synthesis, state preparation implementation, fault-tolerant "
                "compilation, or full readout sampling model."
            ),
            "recommended_wording": "The estimates support resource-aware feasibility discussion.",
            "avoid_wording": "The resource estimates demonstrate quantum speedup.",
        },
        {
            "claim": (
                "Shot-level readout analysis quantifies sampling cost for selected observables."
            ),
            "support_status": "supported_as_analysis",
            "supporting_files": "src/robust_qsvt_se/qsvt/shot_readout_model.py",
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_shot_readout/shot_readout_summary.csv",
                    "outputs/qsvt_shot_readout/observable_estimates.csv",
                ]
            ),
            "strength": "selected-observable Bernoulli sampling proxy",
            "limitations": (
                "Shot rows are probability-proxy estimates, not backend sampling or "
                "full-vector tomography."
            ),
            "recommended_wording": (
                "Shot-level selected-observable proxies make readout sampling cost explicit."
            ),
            "avoid_wording": "The implementation solves full state-vector readout.",
        },
        {
            "claim": "Full-vector readout remains a limitation.",
            "support_status": "supported_as_limitation",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/readout_analysis.py",
                    "src/robust_qsvt_se/qsvt/shot_readout_model.py",
                ]
            ),
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_resource_readout/readout_summary.md",
                    "outputs/qsvt_shot_readout/readout_caveats.md",
                ]
            ),
            "strength": "readout caveat and selected-observable report",
            "limitations": (
                "The repository does not implement full-vector reconstruction or "
                "tomographic readout."
            ),
            "recommended_wording": (
                "Full-vector recovery remains a readout limitation; selected observables "
                "are a more realistic target for feasibility discussion."
            ),
            "avoid_wording": "The implementation efficiently reads out the full state vector.",
        },
        {
            "claim": "Hardware-aware analysis is simulation/proxy only.",
            "support_status": "supported_with_caveat",
            "supporting_files": "src/robust_qsvt_se/qsvt/hardware_aware_report.py",
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_hardware_aware/hardware_aware_summary.csv",
                    "outputs/qsvt_hardware_aware/hardware_assumptions.md",
                ]
            ),
            "strength": "dependency-free hardware-aware cost proxy",
            "limitations": (
                "No calibrated backend run, noise model, fault-tolerant compilation, "
                "or full oracle implementation is included."
            ),
            "recommended_wording": (
                "Hardware-aware proxy estimates provide qubit, gate, depth, and shot "
                "budget diagnostics."
            ),
            "avoid_wording": "The report executes full IEEE-scale QSVT on quantum hardware.",
        },
        {
            "claim": "Dense block encoding is not a scalable oracle.",
            "support_status": "explicit_limitation",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/block_encoding_scalability.py",
                    "docs/QSVT_BLOCK_ENCODING_SCALABILITY.md",
                ]
            ),
            "supporting_outputs": (
                "outputs/qsvt_block_encoding_scalability/scalability_summary.csv"
            ),
            "strength": "scalability caveat and matrix-size accounting",
            "limitations": (
                "The dense Julia construction is a validation prototype; sparse-access "
                "oracles and data-loading costs remain future work."
            ),
            "recommended_wording": (
                "Dense block encodings validate normalization and algebraic embedding "
                "on small matrices but are not scalable oracle constructions."
            ),
            "avoid_wording": "The dense block encoding is a scalable full-system oracle.",
        },
        {
            "claim": "Multi-case resource diagnostics extend beyond IEEE14 where feasible.",
            "support_status": "supported_with_failure_log",
            "supporting_files": "src/robust_qsvt_se/qsvt/multicase_resource_diagnostics.py",
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_multicase_resource_diagnostics/multicase_resource_summary.csv",
                    "outputs/qsvt_multicase_resource_diagnostics/failure_log.csv",
                ]
            ),
            "strength": "resource-only multi-case diagnostics",
            "limitations": (
                "Failures are logged per case; no nonlinear IEEE300 experiment or "
                "hardware QSVT execution is triggered."
            ),
            "recommended_wording": (
                "Resource-only diagnostics are attempted for multiple PYPOWER IEEE "
                "cases, with failures reported explicitly."
            ),
            "avoid_wording": (
                "The repository executes full quantum state estimation on all IEEE cases."
            ),
        },
        {
            "claim": (
                "Adaptive multicase degree search quantifies larger-case degree and "
                "query requirements."
            ),
            "support_status": "supported_with_failure_log",
            "supporting_files": ("src/robust_qsvt_se/qsvt/adaptive_multicase_degree_search.py"),
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_adaptive_multicase_degree_search/adaptive_multicase_summary.csv",
                    "outputs/qsvt_adaptive_multicase_degree_search/"
                    "adaptive_multicase_search_trace.csv",
                    "outputs/qsvt_adaptive_multicase_degree_search/"
                    "adaptive_multicase_failure_log.csv",
                ]
            ),
            "strength": "adaptive degree-query search trace",
            "limitations": (
                "The search is bounded to configured candidate degrees and reports "
                "failures explicitly; it is not a proof of optimal QSP degree."
            ),
            "recommended_wording": (
                "Adaptive multicase diagnostics report the first passing configured "
                "degree or an explicit failure status for each IEEE/PYPOWER case."
            ),
            "avoid_wording": ("Adaptive multicase diagnostics prove scalable quantum advantage."),
        },
        {
            "claim": (
                "Some larger IEEE cases require higher degree than IEEE14 under the same tolerance."
            ),
            "support_status": "supported_by_adaptive_search",
            "supporting_files": ("src/robust_qsvt_se/qsvt/adaptive_multicase_degree_search.py"),
            "supporting_outputs": (
                "outputs/qsvt_adaptive_multicase_degree_search/adaptive_multicase_summary.csv"
            ),
            "strength": "multi-case approximation scaling diagnostic",
            "limitations": (
                "The trend is tied to the configured matrix construction, alpha, "
                "method, grid, and candidate degree cap."
            ),
            "recommended_wording": (
                "Under the configured alpha and tolerance, larger benchmark cases "
                "can require higher polynomial degree and query count than IEEE14."
            ),
            "avoid_wording": "The larger-case degree trend proves quantum speedup.",
        },
        {
            "claim": "Phase-target failure was diagnosed with coefficient and basis evidence.",
            "support_status": "supported_with_diagnostics",
            "supporting_files": "src/robust_qsvt_se/qsvt/nonbruteforce_refinement.py",
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_phase_target_failure_diagnostics/"
                    "phase_target_failure_summary.csv",
                    "outputs/qsvt_phase_target_failure_diagnostics/coefficient_diagnostics.csv",
                    "outputs/qsvt_phase_target_failure_diagnostics/"
                    "basis_conversion_diagnostics.csv",
                ]
            ),
            "strength": "coefficient, parity, boundedness, and response-error diagnostics",
            "limitations": (
                "Diagnosis is scalar phase-response evidence and optional-dependency "
                "dependent when phase synthesis is attempted."
            ),
            "recommended_wording": (
                "The bounded Ridge/Tikhonov phase-target failure was diagnosed with "
                "coefficient, basis-conversion, boundedness, parity, and response-error "
                "evidence."
            ),
            "avoid_wording": "The bounded Ridge/Tikhonov target passed phase validation.",
        },
        {
            "claim": "Sanity phase-response validation passed before target-level claims.",
            "support_status": "supported_for_sanity_polynomials",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/phase_response_conventions.py",
                    "src/robust_qsvt_se/qsvt/nonbruteforce_refinement.py",
                ]
            ),
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_phase_response_convention_diagnostics/"
                    "sanity_polynomial_results.csv",
                    "outputs/qsvt_stable_phase_validation_attempt/"
                    "stable_phase_validation_summary.csv",
                ]
            ),
            "strength": "sanity-polynomial phase-response check",
            "limitations": (
                "Sanity-polynomial passes do not imply bounded Ridge/Tikhonov target "
                "phase validation."
            ),
            "recommended_wording": (
                "Scalar phase-response convention is validated on sanity polynomials, "
                "with bounded Ridge/Tikhonov target status reported separately."
            ),
            "avoid_wording": "Sanity-polynomial validation proves the Ridge target passes.",
        },
        {
            "claim": (
                "Bounded Ridge/Tikhonov target phase validation status is reported honestly."
            ),
            "support_status": "explicit_pass_fail_boundary",
            "supporting_files": "src/robust_qsvt_se/qsvt/nonbruteforce_refinement.py",
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_phase_target_failure_diagnostics/"
                    "phase_target_failure_summary.csv",
                    "outputs/qsvt_stable_phase_validation_attempt/"
                    "stable_phase_validation_summary.csv",
                ]
            ),
            "strength": "explicit phase-target status rows",
            "limitations": (
                "Rows may remain failed or skipped; passing is claimed only when the "
                "target phase response meets the declared tolerance."
            ),
            "recommended_wording": (
                "Bounded Ridge/Tikhonov phase validation is claimed only for rows where "
                "the target phase response itself passes the strict tolerance."
            ),
            "avoid_wording": "Phase validation passed when only sanity polynomials passed.",
        },
        {
            "claim": "Phase backend capabilities were audited.",
            "support_status": "supported_with_backend_inventory",
            "supporting_files": "src/robust_qsvt_se/qsvt/phase_backend_audit.py",
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_phase_backend_audit/phase_backend_audit_summary.csv",
                    "outputs/qsvt_phase_backend_audit/phase_backend_capabilities.md",
                ]
            ),
            "strength": "dependency and capability audit",
            "limitations": (
                "Backend availability does not imply target-level phase validation; "
                "PennyLane remains monomial-coefficient based in this environment."
            ),
            "recommended_wording": (
                "Available phase and polynomial backends were audited before target "
                "phase validation was attempted."
            ),
            "avoid_wording": "A Chebyshev-basis phase backend was available if the audit says no.",
        },
        {
            "claim": "Stable polynomial candidates were tested.",
            "support_status": "supported_with_candidate_gates",
            "supporting_files": "src/robust_qsvt_se/qsvt/stable_phase_candidates.py",
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_stable_phase_candidates/stable_phase_candidate_summary.csv",
                    "outputs/qsvt_stable_phase_candidates/candidate_report.md",
                ]
            ),
            "strength": "candidate approximation, boundedness, conversion, and parity gates",
            "limitations": (
                "A candidate can pass polynomial approximation while still failing "
                "basis-conversion or coefficient-stability gates."
            ),
            "recommended_wording": (
                "Stable polynomial candidates were tested with explicit safety gates "
                "before phase synthesis."
            ),
            "avoid_wording": "A stable polynomial candidate implies phase validation passed.",
        },
        {
            "claim": (
                "Bounded Ridge/Tikhonov target phase validation passed only if all gates passed."
            ),
            "support_status": "explicit_pass_fail_boundary",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/stable_phase_candidates.py",
                    "src/robust_qsvt_se/qsvt/stable_phase_validation.py",
                ]
            ),
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_stable_phase_candidates/stable_phase_candidate_summary.csv",
                    "outputs/qsvt_stable_target_phase_validation/"
                    "stable_target_phase_validation_summary.csv",
                ]
            ),
            "strength": "explicit all-gates pass rule",
            "limitations": (
                "Rows with failed candidate gates or failed phase response remain "
                "unresolved and are not reinterpreted as passing."
            ),
            "recommended_wording": (
                "Bounded Ridge/Tikhonov target phase validation is claimed only for "
                "rows passing all candidate and phase-response gates."
            ),
            "avoid_wording": "The bounded Ridge/Tikhonov target passed without a passing row.",
        },
        {
            "claim": "Sanity-polynomial phase response passed.",
            "support_status": "supported_for_scalar_sanity_rows",
            "supporting_files": "src/robust_qsvt_se/qsvt/phase_sanity_regression.py",
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_phase_sanity_regression/phase_sanity_regression_summary.csv",
                    "outputs/qsvt_phase_sanity_regression/phase_sanity_response_values.csv",
                ]
            ),
            "strength": "known-polynomial scalar phase-response regression",
            "limitations": (
                "Sanity-polynomial passes validate the response convention only; "
                "they do not validate the bounded Ridge/Tikhonov target."
            ),
            "recommended_wording": (
                "The scalar phase-response convention passed known-polynomial sanity "
                "checks before target-level validation was interpreted."
            ),
            "avoid_wording": "Sanity-polynomial passes prove the Ridge target passed.",
        },
        {
            "claim": "Chebyshev-to-monomial conversion instability was measured.",
            "support_status": "supported_with_coefficient_diagnostics",
            "supporting_files": "src/robust_qsvt_se/qsvt/stable_phase_candidates.py",
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_stable_phase_candidates/stable_phase_candidate_summary.csv",
                    "outputs/qsvt_stable_phase_candidates/candidate_coefficients_monomial.csv",
                ]
            ),
            "strength": "conversion error and dynamic-range diagnostics",
            "limitations": (
                "Measured instability motivates skip decisions; it is not a claim "
                "about all possible phase-synthesis algorithms."
            ),
            "recommended_wording": (
                "Chebyshev-to-monomial conversion error and coefficient dynamic range "
                "were measured before phase synthesis."
            ),
            "avoid_wording": "High-degree monomial conversion is always safe.",
        },
        {
            "claim": "No unstable polynomial was forced into phase synthesis.",
            "support_status": "explicit_safety_gate",
            "supporting_files": "src/robust_qsvt_se/qsvt/stable_phase_validation.py",
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_stable_phase_candidates/stable_phase_candidate_summary.csv",
                    "outputs/qsvt_stable_target_phase_validation/phase_angles.csv",
                ]
            ),
            "strength": "phase synthesis limited to safe_for_phase_synthesis rows",
            "limitations": (
                "Skipped rows remain unresolved; safety gating does not itself "
                "produce a passing phase response."
            ),
            "recommended_wording": (
                "Unsafe candidates were skipped before phase synthesis and recorded "
                "with failure reasons."
            ),
            "avoid_wording": "Unstable high-degree coefficients were synthesized anyway.",
        },
        {
            "claim": "No tolerance relaxation was used.",
            "support_status": "explicit_method_boundary",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/stable_phase_candidates.py",
                    "src/robust_qsvt_se/qsvt/stable_phase_validation.py",
                ]
            ),
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_stable_phase_candidates/stable_phase_candidate_summary.csv",
                    "outputs/qsvt_stable_target_phase_validation/"
                    "stable_target_phase_validation_summary.csv",
                ]
            ),
            "strength": "fixed 1e-3 target-level tolerance",
            "limitations": (
                "Rows may fail or remain unresolved when the strict tolerance is not met."
            ),
            "recommended_wording": (
                "The strict 1e-3 tolerance was preserved, with failed and skipped "
                "rows reported explicitly."
            ),
            "avoid_wording": "The tolerance was loosened to obtain a passing phase row.",
        },
        {
            "claim": "No quantum speedup or hardware execution is claimed.",
            "support_status": "explicit_claim_boundary",
            "supporting_files": _join(
                [
                    "docs/QSVT_ENGINEERING_EXTENSION.md",
                    "docs/QSVT_STABLE_PHASE_SYNTHESIS.md",
                    "docs/QSVT_PHASE_RESPONSE_CONVENTIONS.md",
                ]
            ),
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_engineering_extension/claim_support_matrix.csv",
                    "outputs/qsvt_stable_target_phase_validation/"
                    "stable_target_phase_validation_report.md",
                ]
            ),
            "strength": "documentation and claim-matrix boundary",
            "limitations": (
                "The repository provides scalar diagnostics and resource-aware "
                "feasibility evidence, not hardware execution."
            ),
            "recommended_wording": (
                "The results support stable phase-synthesis diagnostics and "
                "resource-aware feasibility discussion only."
            ),
            "avoid_wording": "The results demonstrate quantum speedup or hardware execution.",
        },
        {
            "claim": "External QSP/QSVT phase backends were audited.",
            "support_status": "supported_with_external_backend_inventory",
            "supporting_files": "src/robust_qsvt_se/qsvt/external_phase_backend_audit.py",
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_phase_external_backend_audit/external_backend_audit_summary.csv",
                    "outputs/qsvt_phase_external_backend_audit/external_backend_capabilities.md",
                ]
            ),
            "strength": "external backend install and API audit",
            "limitations": (
                "Backend availability is separate from sanity and target-level "
                "phase-response validation."
            ),
            "recommended_wording": (
                "External QSP/QSVT phase backends were audited with explicit install "
                "and API capability records."
            ),
            "avoid_wording": "Backend installation alone validates the bounded target.",
        },
        {
            "claim": (
                "pyqsp/QSPPACK/PennyLane/local optimization backend availability was tested."
            ),
            "support_status": "supported_with_backend_status_rows",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/external_phase_backend_audit.py",
                    "src/robust_qsvt_se/qsvt/phase_backend_adapters.py",
                ]
            ),
            "supporting_outputs": (
                "outputs/qsvt_phase_external_backend_audit/external_backend_audit_summary.csv"
            ),
            "strength": "per-backend availability and adapter status",
            "limitations": (
                "QSPPACK is recorded as not directly callable from Python unless a "
                "usable package is present."
            ),
            "recommended_wording": (
                "pyqsp, QSPPACK, PennyLane, Qiskit utility availability, and a local "
                "optimization backend were tested and reported separately."
            ),
            "avoid_wording": "All external backends were available and passing.",
        },
        {
            "claim": "Backend sanity regression was performed.",
            "support_status": "supported_for_backend_sanity_rows",
            "supporting_files": "src/robust_qsvt_se/qsvt/external_backend_sanity.py",
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_external_backend_sanity_regression/"
                    "external_backend_sanity_summary.csv",
                    "outputs/qsvt_external_backend_sanity_regression/"
                    "external_backend_sanity_response_values.csv",
                ]
            ),
            "strength": "four-polynomial backend sanity regression",
            "limitations": (
                "Passing backend sanity regression is required before target validation "
                "but does not itself validate the bounded target."
            ),
            "recommended_wording": (
                "Backend sanity regression was run on four known QSP/QSVT polynomial "
                "targets before target-level results were trusted."
            ),
            "avoid_wording": "Sanity regression alone validates the Ridge/Tikhonov target.",
        },
        {
            "claim": (
                "The bounded Ridge/Tikhonov target passed scalar full-domain "
                "phase-response validation using pyqsp symmetric-QSP phases."
            ),
            "support_status": "supported",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/external_backend_phase_validation.py",
                    "src/robust_qsvt_se/qsvt/phase_backend_adapters.py",
                ]
            ),
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_external_backend_phase_validation/"
                    "external_backend_phase_validation_summary.csv",
                    "outputs/qsvt_phase1_finalization/phase1_finalization_summary.csv",
                ]
            ),
            "strength": "scalar full-domain phase-response validation",
            "limitations": (
                "This is scalar phase-response validation only; it is not hardware "
                "execution or block-encoded matrix execution."
            ),
            "recommended_wording": (
                "The bounded Ridge/Tikhonov target passed scalar full-domain "
                "phase-response validation using pyqsp symmetric-QSP phases."
            ),
            "avoid_wording": ("The pyqsp scalar phase-response pass proves hardware execution."),
        },
        {
            "claim": "The full-domain max phase-response error was below 1e-3.",
            "support_status": "supported",
            "supporting_files": "src/robust_qsvt_se/qsvt/external_backend_phase_validation.py",
            "supporting_outputs": (
                "outputs/qsvt_external_backend_phase_validation/"
                "external_backend_phase_validation_summary.csv"
            ),
            "strength": "strict tolerance pass",
            "limitations": "The tolerance applies to scalar phase response on the declared grid.",
            "recommended_wording": (
                "The pyqsp row reports full-domain maximum phase-response error "
                "4.668e-4, below the strict 1e-3 tolerance."
            ),
            "avoid_wording": "The tolerance pass proves end-to-end quantum state estimation.",
        },
        {
            "claim": (
                "The validation used Chebyshev-basis input and avoided unsafe monomial conversion."
            ),
            "support_status": "supported",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/phase_backend_adapters.py",
                    "src/robust_qsvt_se/qsvt/external_phase_candidates.py",
                ]
            ),
            "supporting_outputs": (
                "outputs/qsvt_external_backend_phase_validation/"
                "external_backend_phase_validation_summary.csv"
            ),
            "strength": "basis-support and safety-gate evidence",
            "limitations": (
                "PennyLane monomial-path rows remain historical backend-specific "
                "failures or skipped rows."
            ),
            "recommended_wording": (
                "pyqsp accepted the Chebyshev-basis candidate directly, avoiding the "
                "unsafe high-degree monomial conversion path."
            ),
            "avoid_wording": "High-degree monomial conversion was safe for all backends.",
        },
        {
            "claim": "The result is scalar phase-response validation only.",
            "support_status": "explicit_boundary",
            "supporting_files": _join(
                [
                    "docs/QSVT_EXTERNAL_PHASE_BACKENDS.md",
                    "docs/QSVT_PHASE1_FINALIZATION.md",
                ]
            ),
            "supporting_outputs": "outputs/qsvt_phase1_finalization/phase1_finalization_summary.md",
            "strength": "claim boundary",
            "limitations": "No matrix block encoding or state-estimation circuit is executed.",
            "recommended_wording": (
                "The pyqsp pass is scalar full-domain phase-response validation only."
            ),
            "avoid_wording": "Scalar phase-response validation proves end-to-end QSVT execution.",
        },
        {
            "claim": "The result does not demonstrate hardware execution.",
            "support_status": "explicit_boundary",
            "supporting_files": "docs/QSVT_PHASE1_FINALIZATION.md; README.md",
            "supporting_outputs": "outputs/qsvt_phase1_finalization/phase1_finalization_summary.md",
            "strength": "claim boundary",
            "limitations": "No quantum device or hardware-native block encoding is used.",
            "recommended_wording": ("The Phase 1 pass does not constitute hardware execution."),
            "avoid_wording": "The Phase 1 pass is hardware validation.",
        },
        {
            "claim": "The result does not demonstrate quantum speedup.",
            "support_status": "explicit_boundary",
            "supporting_files": "docs/QSVT_PHASE1_FINALIZATION.md; README.md",
            "supporting_outputs": "outputs/qsvt_phase1_finalization/phase1_finalization_summary.md",
            "strength": "claim boundary",
            "limitations": "The result is a scalar validation artifact, not a runtime comparison.",
            "recommended_wording": (
                "The Phase 1 pass supports a QSVT-compatible implementation pathway "
                "without making a speedup claim."
            ),
            "avoid_wording": "The Phase 1 pass demonstrates quantum speedup.",
        },
        {
            "claim": "The result does not demonstrate QSVT superiority over Ridge/Tikhonov.",
            "support_status": "explicit_boundary",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/estimators/ridge.py",
                    "src/robust_qsvt_se/estimators/qsvt_spectral.py",
                    "docs/QSVT_PHASE1_FINALIZATION.md",
                ]
            ),
            "supporting_outputs": "outputs/qsvt_engineering_extension/claim_support_matrix.csv",
            "strength": "same-filter equivalence boundary",
            "limitations": (
                "The QSVT target filter is the Ridge/Tikhonov filter under the same alpha."
            ),
            "recommended_wording": (
                "QSVT-target equivalence to Ridge/Tikhonov under the same alpha is a "
                "correctness check, not a performance win."
            ),
            "avoid_wording": "QSVT outperforms Ridge/Tikhonov under the same alpha.",
        },
        {
            "claim": (
                "Target-level bounded Ridge/Tikhonov phase validation passed only if "
                "full-domain error <= 1e-3."
            ),
            "support_status": "explicit_full_domain_pass_boundary",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/external_phase_candidates.py",
                    "src/robust_qsvt_se/qsvt/external_backend_phase_validation.py",
                ]
            ),
            "supporting_outputs": (
                "outputs/qsvt_external_backend_phase_validation/"
                "external_backend_phase_validation_summary.csv"
            ),
            "strength": "full-domain phase-response pass/fail columns",
            "limitations": (
                "The pass is scalar phase-response validation over the declared "
                "normalized interval, not hardware execution."
            ),
            "recommended_wording": (
                "Target-level bounded Ridge/Tikhonov phase validation is reported "
                "only where full-domain phase-response error is at most 1e-3."
            ),
            "avoid_wording": "Actual-singular-value-only validation is a full-domain pass.",
        },
        {
            "claim": "Actual-singular-value-only validation is not full-domain validation.",
            "support_status": "explicit_domain_boundary",
            "supporting_files": "src/robust_qsvt_se/qsvt/external_backend_phase_validation.py",
            "supporting_outputs": (
                "outputs/qsvt_external_backend_phase_validation/"
                "external_backend_phase_validation_summary.csv"
            ),
            "strength": "separate full-domain and actual-spectrum columns",
            "limitations": (
                "Actual singular values are a useful diagnostic but do not replace "
                "the dense full-domain grid."
            ),
            "recommended_wording": (
                "Full-domain and actual-singular-value phase-response errors are "
                "reported separately."
            ),
            "avoid_wording": "Actual singular values alone establish full QSP/QSVT validation.",
        },
        {
            "claim": "No unsafe monomial candidate was forced into phase synthesis.",
            "support_status": "explicit_monomial_safety_gate",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/external_phase_candidates.py",
                    "src/robust_qsvt_se/qsvt/external_backend_phase_validation.py",
                ]
            ),
            "supporting_outputs": (
                "outputs/qsvt_external_backend_phase_validation/"
                "external_backend_phase_validation_summary.csv"
            ),
            "strength": "monomial candidates require conversion safety gates",
            "limitations": (
                "Chebyshev-capable backends may validate candidates that monomial-only "
                "backends must skip."
            ),
            "recommended_wording": (
                "Unsafe monomial candidates were skipped, while Chebyshev-capable "
                "backend paths were evaluated separately."
            ),
            "avoid_wording": "Unstable high-degree monomial coefficients were synthesized.",
        },
        {
            "claim": "IEEE300 spectral difficulty was analyzed without degree brute force.",
            "support_status": "supported_with_spectral_diagnostics",
            "supporting_files": "src/robust_qsvt_se/qsvt/nonbruteforce_refinement.py",
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_ieee300_spectral_difficulty/spectral_difficulty_summary.csv",
                    "outputs/qsvt_ieee300_spectral_difficulty/error_location_diagnostics.csv",
                    "outputs/qsvt_ieee300_spectral_difficulty/interval_restriction_diagnostics.csv",
                ]
            ),
            "strength": "spectrum, quantile, histogram, and error-location diagnostics",
            "limitations": (
                "Diagnostics explain approximation difficulty; they do not by "
                "themselves make a full validation row pass."
            ),
            "recommended_wording": (
                "IEEE300 difficulty is analyzed through spectral spread, singular-value "
                "density, error location, and interval diagnostics."
            ),
            "avoid_wording": "IEEE300 passed full QSVT validation after diagnostics.",
        },
        {
            "claim": (
                "Full-interval error and actual-singular-value error are reported separately."
            ),
            "support_status": "supported_with_separate_columns",
            "supporting_files": "src/robust_qsvt_se/qsvt/nonbruteforce_refinement.py",
            "supporting_outputs": (
                "outputs/qsvt_ieee300_spectral_difficulty/spectral_difficulty_summary.csv"
            ),
            "strength": "separate full-interval and actual-spectrum metrics",
            "limitations": (
                "Actual-singular-value error is diagnostic only and must not be "
                "reported as full-interval validation."
            ),
            "recommended_wording": (
                "Full-interval approximation error and actual-singular-value error are "
                "reported as distinct diagnostics."
            ),
            "avoid_wording": (
                "Actual-singular-value error is equivalent to full-interval QSVT validation."
            ),
        },
        {
            "claim": ("Restricted-interval diagnostics are not claimed as full validation."),
            "support_status": "explicit_boundary",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/nonbruteforce_refinement.py",
                    "docs/QSVT_IEEE300_SPECTRAL_DIAGNOSTIC.md",
                ]
            ),
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_ieee300_spectral_difficulty/interval_restriction_diagnostics.csv",
                    "outputs/qsvt_spectrum_aware_diagnostics/"
                    "preconditioning_interval_diagnostics.csv",
                ]
            ),
            "strength": "explicit restricted-interval caveats",
            "limitations": "Restricted intervals are spectrum-aware diagnostics only.",
            "recommended_wording": (
                "Restricted-interval results are diagnostic only and are not presented "
                "as full-interval QSVT validation."
            ),
            "avoid_wording": "Restricted intervals make IEEE300 pass full validation.",
        },
        {
            "claim": (
                "IEEE118 targeted refinement was attempted within a justified degree budget."
            ),
            "support_status": "supported_with_degree_budget",
            "supporting_files": "src/robust_qsvt_se/qsvt/nonbruteforce_refinement.py",
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_ieee118_targeted_refinement/ieee118_refinement_summary.csv",
                    "outputs/qsvt_ieee118_targeted_refinement/ieee118_refinement_trace.csv",
                ]
            ),
            "strength": "targeted degree list only",
            "limitations": (
                "Only approved degrees 1201, 1501, and optionally 2001 are permitted; "
                "the script stops on pass or exhaustion."
            ),
            "recommended_wording": (
                "IEEE118 was refined only over a small justified degree budget because "
                "it narrowly missed the 1e-3 tolerance at degree 1001."
            ),
            "avoid_wording": "IEEE118 was tuned through arbitrary high-degree search.",
        },
        {
            "claim": ("Spectrum-aware and preconditioning diagnostics are diagnostic only."),
            "support_status": "explicit_boundary",
            "supporting_files": "src/robust_qsvt_se/qsvt/nonbruteforce_refinement.py",
            "supporting_outputs": (
                "outputs/qsvt_spectrum_aware_diagnostics/spectrum_aware_summary.csv"
            ),
            "strength": "preconditioning and interval caveats",
            "limitations": (
                "These rows do not alter main estimator results or prove quantum speedup."
            ),
            "recommended_wording": (
                "Preconditioning and spectrum-aware diagnostics quantify spectral-spread "
                "and low-density-region effects only."
            ),
            "avoid_wording": "Preconditioning diagnostics prove quantum advantage.",
        },
        {
            "claim": "No brute-force degree escalation was used in the refinement.",
            "support_status": "explicit_method_boundary",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/nonbruteforce_refinement.py",
                    "scripts/run_qsvt_ieee118_targeted_refinement.py",
                ]
            ),
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_ieee118_targeted_refinement/ieee118_refinement_report.md",
                    "outputs/qsvt_nonbruteforce_refinement_summary/"
                    "nonbruteforce_refinement_summary.md",
                ]
            ),
            "strength": "bounded degree budget and diagnostic scripts",
            "limitations": (
                "The scripts still evaluate configured degrees; they do not prove optimality."
            ),
            "recommended_wording": (
                "The follow-up used targeted diagnostics and a bounded IEEE118 degree "
                "budget, not brute-force escalation."
            ),
            "avoid_wording": "The result was obtained by searching arbitrary high degrees.",
        },
        {
            "claim": "No tolerance relaxation was used to claim success.",
            "support_status": "explicit_boundary",
            "supporting_files": "src/robust_qsvt_se/qsvt/nonbruteforce_refinement.py",
            "supporting_outputs": (
                "outputs/qsvt_nonbruteforce_refinement_summary/nonbruteforce_refinement_summary.md"
            ),
            "strength": "strict 1e-3 pass/fail boundary",
            "limitations": "Failed rows remain failed and are documented.",
            "recommended_wording": (
                "The strict 1e-3 tolerance is preserved; failures are reported rather "
                "than converted to passes."
            ),
            "avoid_wording": "A relaxed tolerance establishes phase or IEEE300 success.",
        },
        {
            "claim": "Stable phase-synthesis diagnostics were implemented.",
            "support_status": "supported_with_pass_fail_rows",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/failure_fix.py",
                    "scripts/fix_qsvt_phase_validation_stable_basis.py",
                ]
            ),
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_phase_validation_stable_basis/"
                    "candidate_polynomial_diagnostics.csv",
                    "outputs/qsvt_phase_validation_stable_basis/stable_phase_validation_report.md",
                ]
            ),
            "strength": "stage-by-stage stable-basis phase diagnostics",
            "limitations": (
                "Rows can remain failed or skipped; target phase validation is claimed "
                "only when all declared criteria pass."
            ),
            "recommended_wording": (
                "Stable phase-synthesis diagnostics evaluate native approximation "
                "error, boundedness, coefficient conversion, and phase response."
            ),
            "avoid_wording": (
                "Stable diagnostics prove target phase validation without a passing row."
            ),
        },
        {
            "claim": ("High-precision or basis-stable coefficient conversion was evaluated."),
            "support_status": "supported_with_diagnostics",
            "supporting_files": "src/robust_qsvt_se/qsvt/failure_fix.py",
            "supporting_outputs": (
                "outputs/qsvt_phase_validation_stable_basis/coefficient_stability_diagnostics.csv"
            ),
            "strength": "float64 and numpy.longdouble conversion diagnostics",
            "limitations": (
                "No direct Chebyshev-basis phase backend is available in the current "
                "dependency set."
            ),
            "recommended_wording": (
                "Coefficient conversion stability was evaluated before attempting phase synthesis."
            ),
            "avoid_wording": "Unstable high-degree monomial coefficients were accepted.",
        },
        {
            "claim": (
                "Bounded Ridge/Tikhonov target phase validation passed only if all "
                "declared criteria were met."
            ),
            "support_status": "explicit_pass_fail_boundary",
            "supporting_files": "src/robust_qsvt_se/qsvt/failure_fix.py",
            "supporting_outputs": (
                "outputs/qsvt_phase_validation_stable_basis/"
                "phase_validation_stable_basis_summary.csv"
            ),
            "strength": "strict phase-validation gate",
            "limitations": "If no row passes, the bounded target remains unresolved.",
            "recommended_wording": (
                "A phase target is called passing only when approximation error, "
                "boundedness, and phase-response error all meet 1e-3."
            ),
            "avoid_wording": "Phase validation passed from sanity-polynomial tests alone.",
        },
        {
            "claim": "A formal preconditioned IEEE300 estimator variant was implemented.",
            "support_status": "supported_as_new_variant",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/failure_fix.py",
                    "scripts/run_qsvt_preconditioned_ieee300_estimator.py",
                ]
            ),
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_preconditioned_ieee300_estimator/"
                    "preconditioned_ieee300_estimator_summary.csv",
                    "outputs/qsvt_preconditioned_ieee300_estimator/"
                    "preconditioned_ieee300_solution_metrics.csv",
                ]
            ),
            "strength": "separate coordinate and transformed-penalty variants",
            "limitations": "Preconditioned rows do not overwrite original Ridge/QSVT results.",
            "recommended_wording": (
                "Column-equilibrated Ridge variants are reported as formal new estimator variants."
            ),
            "avoid_wording": "Preconditioned results replace the original estimator claims.",
        },
        {
            "claim": "Preconditioned results are separate from original estimator claims.",
            "support_status": "explicit_boundary",
            "supporting_files": "src/robust_qsvt_se/qsvt/failure_fix.py",
            "supporting_outputs": (
                "outputs/qsvt_preconditioned_ieee300_estimator/preconditioned_ieee300_report.md"
            ),
            "strength": "explicit variant labeling",
            "limitations": "Coordinate-penalty and transformed-penalty variants differ.",
            "recommended_wording": (
                "Preconditioned coordinate-penalty and transformed-penalty rows are "
                "labeled separately."
            ),
            "avoid_wording": "Coordinate and transformed-penalty variants are the same claim.",
        },
        {
            "claim": (
                "IEEE300 approximation difficulty is reduced under column equilibration "
                "if supported by metrics."
            ),
            "support_status": "supported_when_metrics_pass",
            "supporting_files": "src/robust_qsvt_se/qsvt/failure_fix.py",
            "supporting_outputs": (
                "outputs/qsvt_preconditioned_ieee300_estimator/"
                "preconditioned_ieee300_qsvt_approximation.csv"
            ),
            "strength": "before/after spectral approximation metrics",
            "limitations": (
                "Improved approximation difficulty is not a quantum speedup claim and "
                "does not imply the original unpreconditioned case passed."
            ),
            "recommended_wording": (
                "Column equilibration can reduce QSVT-compatible approximation "
                "difficulty for the formal preconditioned variant."
            ),
            "avoid_wording": "IEEE300 original unpreconditioned validation passed.",
        },
        {
            "claim": "Residual-weighted spectral diagnostics were implemented.",
            "support_status": "supported_as_diagnostic",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/failure_fix.py",
                    "scripts/diagnose_qsvt_ieee300_residual_weighted_error.py",
                ]
            ),
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_ieee300_residual_weighted_error/"
                    "residual_weighted_error_summary.csv",
                    "outputs/qsvt_ieee300_residual_weighted_error/"
                    "singular_direction_contributions.csv",
                ]
            ),
            "strength": "SVD residual-projection weighted error diagnostic",
            "limitations": "Residual-weighted evidence is not full-interval validation.",
            "recommended_wording": (
                "Residual-weighted diagnostics indicate whether pointwise error aligns "
                "with high-energy residual directions."
            ),
            "avoid_wording": "Residual-weighted diagnostics prove full QSVT validation.",
        },
        {
            "claim": "Residual-weighted diagnostics do not replace full-interval validation.",
            "support_status": "explicit_boundary",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/failure_fix.py",
                    "docs/QSVT_RESIDUAL_WEIGHTED_SPECTRAL_ERROR.md",
                ]
            ),
            "supporting_outputs": (
                "outputs/qsvt_ieee300_residual_weighted_error/residual_weighted_error_report.md"
            ),
            "strength": "explicit residual-weighted caveat",
            "limitations": "Full-interval rows remain the validation source of truth.",
            "recommended_wording": (
                "Residual-weighted diagnostics are reported separately from "
                "full-interval validation."
            ),
            "avoid_wording": "Residual-weighted results are full validation.",
        },
        {
            "claim": "No restricted interval was reported as full validation.",
            "support_status": "explicit_boundary",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/nonbruteforce_refinement.py",
                    "src/robust_qsvt_se/qsvt/failure_fix.py",
                ]
            ),
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_ieee300_spectral_difficulty/interval_restriction_diagnostics.csv",
                    "outputs/qsvt_failure_fix_summary/failure_fix_summary.md",
                ]
            ),
            "strength": "claim boundary across interval and failure-fix reports",
            "limitations": "Restricted intervals remain diagnostic only.",
            "recommended_wording": (
                "Restricted-interval diagnostics are not reported as full validation."
            ),
            "avoid_wording": "Restricted intervals establish full QSVT validation.",
        },
        {
            "claim": "Preconditioned variant sweeps were implemented for paper support.",
            "support_status": "supported_with_sweep_outputs",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/paper_finalization.py",
                    "scripts/run_qsvt_preconditioned_variant_sweeps.py",
                ]
            ),
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_preconditioned_variant_sweeps/"
                    "preconditioned_variant_sweep_results.csv",
                    "outputs/qsvt_preconditioned_variant_sweeps/"
                    "preconditioned_variant_sweep_summary.csv",
                ]
            ),
            "strength": "controlled alpha/noise/missing/bad-data variant sweeps",
            "limitations": (
                "Coordinate-preconditioned Ridge is a separate estimator and may "
                "degrade residual/RMSE in some scenarios."
            ),
            "recommended_wording": (
                "Preconditioned estimator variants were evaluated across controlled "
                "scenario sweeps and reported separately from original estimators."
            ),
            "avoid_wording": "Preconditioned coordinate Ridge replaces original Ridge.",
        },
        {
            "claim": (
                "Phase 2 preconditioned alpha sweeps were implemented for IEEE118 and IEEE300."
            ),
            "support_status": "supported_with_phase2_outputs",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/phase2_preconditioned_alpha.py",
                    "scripts/run_qsvt_phase2_preconditioned_alpha_sweeps.py",
                ]
            ),
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_phase2_preconditioned_alpha_sweeps/phase2_sweep_results.csv",
                    "outputs/qsvt_phase2_preconditioned_alpha_sweeps/phase2_sweep_summary.csv",
                ]
            ),
            "strength": ("controlled original/preconditioned variant alpha and perturbation sweep"),
            "limitations": (
                "Coordinate-preconditioned Ridge is a separate estimator and may "
                "degrade residual/RMSE; QSVT rows are diagnostics only."
            ),
            "recommended_wording": (
                "Phase 2 evaluates original and preconditioned variants separately "
                "for IEEE118 and IEEE300 across alpha and controlled stress settings."
            ),
            "avoid_wording": (
                "Coordinate-preconditioned Ridge replaces original Ridge, or "
                "preconditioned IEEE300 means original IEEE300 passed."
            ),
        },
        {
            "claim": "Phase 2 alpha-selection diagnostics were generated.",
            "support_status": "supported_as_diagnostic",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/phase2_preconditioned_alpha.py",
                    "scripts/build_qsvt_phase2_alpha_selection_report.py",
                ]
            ),
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_phase2_alpha_selection/alpha_selection_summary.csv",
                    "outputs/qsvt_phase2_alpha_selection/alpha_selection_trace.csv",
                    "outputs/qsvt_phase2_alpha_selection/alpha_selection_report.md",
                ]
            ),
            "strength": "residual, RMSE, QSVT-resource, and joint-score diagnostics",
            "limitations": "Alpha selection is diagnostic only and is not field calibrated.",
            "recommended_wording": (
                "Diagnostic alpha-selection criteria summarize residual, RMSE, "
                "QSVT approximation error, and query-count tradeoffs."
            ),
            "avoid_wording": "The alpha-selection score is a field-calibrated operating rule.",
        },
        {
            "claim": "QSVT preconditioning resource comparison was generated.",
            "support_status": "supported_with_proxy_resource_rows",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/paper_finalization.py",
                    "scripts/build_qsvt_preconditioning_resource_comparison.py",
                ]
            ),
            "supporting_outputs": (
                "outputs/qsvt_preconditioning_resource_comparison/"
                "preconditioning_resource_comparison.csv"
            ),
            "strength": "before/after approximation and query proxy comparison",
            "limitations": (
                "Proxy resources omit oracle construction, state preparation, "
                "fault tolerance, compilation, and full readout."
            ),
            "recommended_wording": (
                "The preconditioned matrix reduces approximation difficulty under "
                "the configured proxy model where the generated rows support it."
            ),
            "avoid_wording": "The preconditioned QSVT estimator achieves quantum speedup.",
        },
        {
            "claim": "Paper-ready QSVT evidence tables were generated from artifacts.",
            "support_status": "supported_with_aggregation_outputs",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/paper_finalization.py",
                    "scripts/build_paper_ready_qsvt_tables.py",
                ]
            ),
            "supporting_outputs": "outputs/paper_ready_qsvt_tables/",
            "strength": "table aggregation from generated CSV/JSON outputs",
            "limitations": "Tables aggregate existing evidence and do not invent measurements.",
            "recommended_wording": (
                "Paper-ready tables aggregate generated diagnostics with explicit "
                "claim boundaries and limitations."
            ),
            "avoid_wording": "Paper-ready tables create new validation evidence.",
        },
        {
            "claim": "Final artifact freeze and claim-safety audit were implemented.",
            "support_status": "supported_with_traceability_outputs",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/paper_finalization.py",
                    "scripts/freeze_qsvt_manuscript_artifacts.py",
                    "scripts/run_final_qsvt_claim_safety_audit.py",
                ]
            ),
            "supporting_outputs": _join(
                [
                    "outputs/final_qsvt_artifact_freeze/artifact_inventory.csv",
                    "outputs/final_qsvt_claim_safety_audit/claim_safety_audit.csv",
                ]
            ),
            "strength": "artifact inventory, checksums, and unsafe wording audit",
            "limitations": (
                "Automated wording classification can still require manual review "
                "for ambiguous prose."
            ),
            "recommended_wording": (
                "The final evidence package includes artifact inventory and "
                "claim-safety audit outputs."
            ),
            "avoid_wording": "The artifact freeze validates quantum hardware execution.",
        },
        {
            "claim": "Original and preconditioned estimator claims are separated.",
            "support_status": "explicit_boundary",
            "supporting_files": _join(
                [
                    "docs/QSVT_PRECONDITIONED_VARIANT_SWEEPS.md",
                    "docs/QSVT_PAPER_READY_TABLES.md",
                    "src/robust_qsvt_se/qsvt/paper_finalization.py",
                ]
            ),
            "supporting_outputs": _join(
                [
                    "outputs/qsvt_preconditioned_variant_sweeps/"
                    "preconditioned_variant_sweep_report.md",
                    "outputs/paper_ready_qsvt_tables/table_9_claim_boundary_matrix.csv",
                ]
            ),
            "strength": "separate labels and paper-ready claim matrix",
            "limitations": "Preconditioned rows are variant-specific evidence.",
            "recommended_wording": (
                "Original and preconditioned estimator results are reported as "
                "separate evidence tracks."
            ),
            "avoid_wording": ("A preconditioned IEEE300 result makes original IEEE300 pass."),
        },
        {
            "claim": "No QSVT-over-Ridge superiority is claimed.",
            "support_status": "explicit_boundary",
            "supporting_files": _join(
                [
                    "docs/QSVT_ENGINEERING_EXTENSION.md",
                    "src/robust_qsvt_se/qsvt/failure_fix.py",
                ]
            ),
            "supporting_outputs": ("outputs/qsvt_failure_fix_summary/failure_fix_summary.md"),
            "strength": "same-filter Ridge/QSVT boundary",
            "limitations": "Exact QSVT-target and Ridge use the same spectral filter.",
            "recommended_wording": (
                "QSVT-target diagnostics are resource/implementation-pathway evidence, "
                "not superiority over Ridge."
            ),
            "avoid_wording": "QSVT outperforms Ridge under the same alpha.",
        },
        {
            "claim": "No quantum speedup or hardware execution is claimed.",
            "support_status": "explicit_boundary",
            "supporting_files": _join(
                [
                    "docs/QSVT_ENGINEERING_EXTENSION.md",
                    "docs/QSVT_PRECONDITIONED_IEEE300_VARIANT.md",
                    "docs/QSVT_STABLE_PHASE_SYNTHESIS.md",
                ]
            ),
            "supporting_outputs": ("outputs/qsvt_failure_fix_summary/failure_fix_summary.md"),
            "strength": "global quantum-claim boundary",
            "limitations": "Reports are classical diagnostics and proxy resource analyses.",
            "recommended_wording": (
                "The results support resource-aware feasibility analysis only."
            ),
            "avoid_wording": (
                "The implementation demonstrates quantum speedup or hardware execution."
            ),
        },
        {
            "claim": "The extension does not demonstrate quantum speedup.",
            "support_status": "explicit_boundary",
            "supporting_files": "docs/QSVT_ENGINEERING_EXTENSION.md; README.md",
            "supporting_outputs": "outputs/qsvt_engineering_extension/summary.md",
            "strength": "claim boundary",
            "limitations": "The repository reports simulations and proxy estimates only.",
            "recommended_wording": (
                "The extension strengthens quantum-engineering feasibility evidence "
                "without claiming speedup."
            ),
            "avoid_wording": "The extension demonstrates quantum speedup.",
        },
        {
            "claim": "The extension does not demonstrate quantum advantage.",
            "support_status": "explicit_boundary",
            "supporting_files": _join(
                [
                    "docs/QSVT_ENGINEERING_EXTENSION.md",
                    "outputs/qsvt_hardware_aware/hardware_assumptions.md",
                ]
            ),
            "supporting_outputs": "outputs/qsvt_engineering_extension/final_engineering_summary.md",
            "strength": "claim boundary",
            "limitations": "The repository reports simulators, prototypes, and proxy estimates.",
            "recommended_wording": ("The extension does not make a quantum-advantage claim."),
            "avoid_wording": "The extension demonstrates quantum advantage.",
        },
        {
            "claim": "The extension does not execute full IEEE-scale QSVT on quantum hardware.",
            "support_status": "explicit_boundary",
            "supporting_files": _join(
                [
                    "docs/QSVT_ENGINEERING_EXTENSION.md",
                    "docs/qsvt_implementation_scope.md",
                ]
            ),
            "supporting_outputs": "outputs/qsvt_resource_readout/resource_assumptions.md",
            "strength": "scope limitation",
            "limitations": "Only dense prototypes and resource/readout reports are added.",
            "recommended_wording": (
                "Full IEEE-scale hardware-native QSVT execution remains future work."
            ),
            "avoid_wording": "The repository executes full IEEE-scale QSVT on quantum hardware.",
        },
        {
            "claim": "The extension does not use real PMU/SCADA field data.",
            "support_status": "explicit_boundary",
            "supporting_files": "docs/QSVT_ENGINEERING_EXTENSION.md; docs/dataset_strategy.md",
            "supporting_outputs": "outputs/qsvt_engineering_extension/summary.md",
            "strength": "dataset boundary",
            "limitations": "Uses generated measurement rows from benchmark network models.",
            "recommended_wording": (
                "The matrix sources are controlled IEEE/PYPOWER benchmark systems, not field data."
            ),
            "avoid_wording": "The extension is validated on PMU/SCADA field data.",
        },
        {
            "claim": (
                "QSVT does not numerically outperform Ridge/Tikhonov under the same alpha/filter."
            ),
            "support_status": "explicit_boundary",
            "supporting_files": _join(
                [
                    "src/robust_qsvt_se/qsvt/state_demo.py",
                    "docs/QSVT_ENGINEERING_EXTENSION.md",
                ]
            ),
            "supporting_outputs": "outputs/qsvt_end_to_end_state_demo/state_demo_summary.csv",
            "strength": "mathematical equivalence boundary",
            "limitations": (
                "The exact QSVT-target simulator and Ridge use the same spectral filter."
            ),
            "recommended_wording": (
                "Under identical alpha and filter, the exact QSVT-target spectral "
                "simulator reproduces Ridge/Tikhonov rather than outperforming it."
            ),
            "avoid_wording": "QSVT beats Ridge for the same alpha and spectral filter.",
        },
    ]


def _join(values: list[str]) -> str:
    return "; ".join(values)


def _summary_markdown(rows: list[dict[str, str]]) -> str:
    supported = sum(1 for row in rows if row["support_status"].startswith("supported"))
    return f"""# QSVT Engineering Extension Summary

This bundle documents claim support for the QSVT-compatible implementation pathway extension.

The extension strengthens the quantum-engineering feasibility evidence. It does
not demonstrate quantum speedup, full hardware execution, or numerical
superiority over Ridge/Tikhonov in the classical simulator.

- Claims tracked: {len(rows)}
- Supported or caveated-support claims: {supported}
- Explicit boundary claims: {len(rows) - supported}

Use the claim-support matrix for manuscript wording and limitations.
"""


def _final_summary_markdown(rows: list[dict[str, str]]) -> str:
    claim_lines = "\n".join(f"- {row['claim']} (`{row['support_status']}`)" for row in rows)
    approximation_status = _approximation_status_markdown()
    return f"""# Final QSVT Engineering Extension Summary

The strengthened extension adds standalone audit, selected-alpha polynomial
validation, selected-observable shot-readout modeling, hardware-aware proxy
costs, block-encoding scalability analysis, and multi-case resource diagnostics.

Tracked claims: {len(rows)}

## Claim Inventory

{claim_lines}

## Boundary

The extension supports a QSVT-compatible implementation pathway and
resource-aware feasibility analysis. It does not demonstrate quantum speedup,
quantum advantage, full IEEE-scale hardware execution, PMU/SCADA field-data
validation, or numerical superiority over Ridge/Tikhonov under the same alpha
and spectral filter.

Approximation reports explicitly separate exact spectral equivalence,
polynomial fallback diagnostics, and optional phase-synthesis attempts. Passing
or failing the strict 1e-3 maximum pointwise-error tolerance is reported in the
generated tables rather than inferred.

## Approximation Status

{approximation_status}
"""


def _approximation_status_markdown() -> str:
    selected_path = Path(
        "outputs/qsvt_selected_alpha_phase_validation/phase_validation_summary.csv"
    )
    degree_path = Path("outputs/qsvt_approximation_degree_sweep/degree_sweep_summary.csv")
    phase_path = Path(
        "outputs/qsvt_optional_phase_synthesis_validation/phase_synthesis_summary.csv"
    )
    convention_path = Path(
        "outputs/qsvt_phase_response_convention_diagnostics/sanity_polynomial_results.csv"
    )
    multicase_path = Path(
        "outputs/qsvt_adaptive_multicase_degree_search/adaptive_multicase_summary.csv"
    )
    external_phase_path = Path(
        "outputs/qsvt_external_backend_phase_validation/"
        "external_backend_phase_validation_summary.csv"
    )
    lines = []
    if selected_path.is_file():
        selected = pd.read_csv(selected_path)
        passed = int(selected["passed"].sum()) if "passed" in selected else 0
        lines.append(
            "- Selected-alpha polynomial fallback strict 1e-3 rows passed: "
            f"{passed}/{len(selected)}."
        )
    else:
        lines.append("- Selected-alpha polynomial fallback output has not been generated.")
    if degree_path.is_file():
        degree = pd.read_csv(degree_path)
        strict = degree[degree["passed_tol_1e_minus_3"] == True]  # noqa: E712
        if strict.empty:
            lines.append("- Degree sweep: no tested degree met the strict 1e-3 tolerance.")
        else:
            first = strict.sort_values(["alpha", "degree"]).groupby("alpha").first()
            pieces = [
                f"alpha {alpha:g}: degree {int(row['degree'])}, "
                f"queries {int(row['query_count_estimate'])}"
                for alpha, row in first.iterrows()
            ]
            lines.append(
                "- Degree sweep first strict 1e-3 passing rows: " + "; ".join(pieces) + "."
            )
    else:
        lines.append("- Degree-sweep output has not been generated.")
    if phase_path.is_file():
        phase = pd.read_csv(phase_path)
        statuses = ", ".join(sorted(set(phase["status"].astype(str))))
        lines.append(f"- Optional phase-synthesis statuses: {statuses}.")
    else:
        lines.append("- Optional phase-synthesis output has not been generated.")
    if convention_path.is_file():
        sanity = pd.read_csv(convention_path)
        passed = int((sanity["best_status"].astype(str) == "passed").sum())
        max_error = float(sanity["best_max_pointwise_error"].max())
        lines.append(
            "- Phase-response sanity polynomials passed: "
            f"{passed}/{len(sanity)}; maximum best error {max_error:.3g}."
        )
    else:
        lines.append("- Phase-response convention diagnostics have not been generated.")
    if multicase_path.is_file():
        multicase = pd.read_csv(multicase_path)
        passed = int((multicase["status"].astype(str) == "passed").sum())
        details = [
            f"{row.case_name}: degree {row.selected_degree}, status {row.status}"
            for row in multicase.itertuples()
        ]
        lines.append(
            "- Adaptive multicase degree-search passing rows: "
            f"{passed}/{len(multicase)} ({'; '.join(details)})."
        )
    else:
        lines.append("- Adaptive multicase degree-search output has not been generated.")
    if external_phase_path.is_file():
        external = pd.read_csv(external_phase_path)
        pyqsp = external[
            (external["backend_name"].astype(str) == "pyqsp_sym_qsp")
            & (
                external["candidate_name"].astype(str)
                == "coefficient_conditioned_chebyshev_degree_201_lambda_1e-04"
            )
        ]
        if pyqsp.empty:
            lines.append("- pyqsp target-level phase validation row is not present.")
        else:
            row = pyqsp.iloc[0]
            lines.append(
                "- pyqsp bounded Ridge/Tikhonov scalar full-domain phase-response "
                f"status: {row.status}; full-domain max error "
                f"{float(row.phase_response_max_error_full_domain):.3g}; "
                f"phase count {int(row.phase_count)}."
            )
    else:
        lines.append("- External backend phase-validation output has not been generated.")
    return "\n".join(lines)


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {"output_dir": "outputs/qsvt_engineering_extension"}
    if config:
        resolved.update(config)
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build QSVT engineering claim-support summary")
    parser.parse_args(argv)
    run = build_engineering_extension_summary()
    print(f"QSVT engineering extension summary complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
