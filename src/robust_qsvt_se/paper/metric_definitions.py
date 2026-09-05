"""Phase 4: metric definitions and interpretation notes.

Writes a manuscript-ready metrics document so the paper never conflates single-step
update RMSE, nonlinear full-state RMSE, residual fit, weighted residual, gate-vs-target
circuit error, observable error, and conditioning. This module defines vocabulary only;
it computes no new results. The interpretation notes are deliberately conservative:
residual fit is not state accuracy, gate-vs-polynomial error validates the circuit
pathway (not estimator superiority), and QSVT-target equals Ridge for matched alpha.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robust_qsvt_se.paper import PAPER_CLAIM_BOUNDARY
from robust_qsvt_se.paper._common import rows_to_table
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

SOURCE_SCRIPT = "scripts/build_metric_definitions.py"

METRIC_COLUMNS = [
    "metric_name",
    "formula",
    "applies_to_workflow",
    "measures",
    "does_not_measure",
    "used_in_sections",
    "claim_boundary",
    "notes",
]
USAGE_COLUMNS = ["manuscript_section", "primary_metrics", "notes"]

_METRICS: tuple[dict[str, str], ...] = (
    {
        "metric_name": "single_step_update_rmse",
        "formula": r"\sqrt{\tfrac{1}{n}\sum_i (\hat{x}_i - x^{\star}_i)^2} for one AC-linearized "
        r"weighted update",
        "applies_to_workflow": "single_step_ac_linearized",
        "measures": "accuracy of one weighted-update state estimate vs the true linearized state",
        "does_not_measure": "nonlinear full-state accuracy after iteration",
        "used_in_sections": "5; 8; 9",
        "claim_boundary": "single-step AC-linearized update; not a full nonlinear solve",
        "notes": "the headline RMSE for the single-step prototype",
    },
    {
        "metric_name": "nonlinear_full_state_rmse",
        "formula": r"\sqrt{\tfrac{1}{n}\sum_i (\hat{x}_i - x^{\star}_i)^2} after Gauss-Newton "
        r"convergence",
        "applies_to_workflow": "nonlinear_ac_iterative",
        "measures": "final state accuracy of the iterative nonlinear AC estimate",
        "does_not_measure": "single-step linearized update quality",
        "used_in_sections": "8",
        "claim_boundary": "classical nonlinear consistency evidence; not a QSVT loop",
        "notes": "distinct from single_step_update_rmse; do not conflate the two",
    },
    {
        "metric_name": "angle_rmse",
        "formula": r"\sqrt{\tfrac{1}{|A|}\sum_{i\in A} (\hat{\theta}_i - \theta^{\star}_i)^2}",
        "applies_to_workflow": "single_step_ac_linearized; nonlinear_ac_iterative",
        "measures": "bus voltage-angle error component",
        "does_not_measure": "voltage-magnitude error",
        "used_in_sections": "8",
        "claim_boundary": "controlled benchmark state split",
        "notes": "angle sub-block of the state RMSE",
    },
    {
        "metric_name": "voltage_magnitude_rmse",
        "formula": r"\sqrt{\tfrac{1}{|V|}\sum_{i\in V} (\hat{V}_i - V^{\star}_i)^2}",
        "applies_to_workflow": "single_step_ac_linearized; nonlinear_ac_iterative",
        "measures": "bus voltage-magnitude error component",
        "does_not_measure": "angle error",
        "used_in_sections": "8",
        "claim_boundary": "controlled benchmark state split",
        "notes": "voltage sub-block of the state RMSE",
    },
    {
        "metric_name": "residual_norm",
        "formula": r"\lVert \tilde r - \tilde H \hat{x} \rVert_2",
        "applies_to_workflow": "single_step_ac_linearized; nonlinear_ac_iterative",
        "measures": "measurement fit of the estimate",
        "does_not_measure": "state accuracy (a low residual can hide a poorly observed state)",
        "used_in_sections": "8",
        "claim_boundary": "fit diagnostic; not a state-accuracy claim",
        "notes": "measurement-fit residual, not state error",
    },
    {
        "metric_name": "weighted_residual_norm",
        "formula": r"\lVert W^{1/2}(z - h(\hat{x})) \rVert_2 (already-weighted rows)",
        "applies_to_workflow": "single_step_ac_linearized; nonlinear_ac_iterative",
        "measures": "noise-weighted measurement fit",
        "does_not_measure": "state accuracy",
        "used_in_sections": "8",
        "claim_boundary": "weighted fit diagnostic; not a state-accuracy claim",
        "notes": "convergence/fit metric for the iterative loop",
    },
    {
        "metric_name": "gate_vs_polynomial_state_error",
        "formula": r"\lVert x_{\mathrm{gate}} - x_{\mathrm{poly}} \rVert_2",
        "applies_to_workflow": "qsvt_selected_subproblem",
        "measures": "agreement between the gate-level circuit and the classical QSVT polynomial",
        "does_not_measure": "estimator quality vs Ridge; quantum advantage",
        "used_in_sections": "9",
        "claim_boundary": "circuit-implementation validation only",
        "notes": "validates the QSVT-compatible pathway, not superiority",
    },
    {
        "metric_name": "gate_vs_ridge_state_error",
        "formula": r"\lVert x_{\mathrm{gate}} - x_{\mathrm{ridge}} \rVert_2",
        "applies_to_workflow": "qsvt_selected_subproblem",
        "measures": "agreement between the gate output and the Ridge/Tikhonov reference",
        "does_not_measure": "any QSVT-over-Ridge improvement",
        "used_in_sections": "9",
        "claim_boundary": "agreement, not superiority; QSVT-target equals Ridge for matched alpha",
        "notes": "small by construction; the QSVT target IS the Ridge filter",
    },
    {
        "metric_name": "direction_error_vs_ridge",
        "formula": r"|P_{\mathrm{qsvt}}(\sigma) - P_{\alpha}(\sigma)| per singular direction",
        "applies_to_workflow": "qsvt_selected_subproblem",
        "measures": "per-direction filter mismatch driving the high-degree overshoot",
        "does_not_measure": "global estimator accuracy",
        "used_in_sections": "9",
        "claim_boundary": "explains the safe degree window; does not widen it",
        "notes": "leading-direction amplitude distortion diagnostic",
    },
    {
        "metric_name": "observable_relative_error",
        "formula": r"|o_{\mathrm{gate}} - o_{\mathrm{ref}}| / |o_{\mathrm{ref}}|",
        "applies_to_workflow": "qsvt_observable_readout",
        "measures": "relative error of a selected physical observable read from the gate output",
        "does_not_measure": "full update-vector accuracy",
        "used_in_sections": "10",
        "claim_boundary": "observable-first readout; not full-vector tomography",
        "notes": "selected observables avoid full-vector reconstruction",
    },
    {
        "metric_name": "top_k_match",
        "formula": r"|\{\text{top-}k\ \text{indices}\}_{\mathrm{gate}} \cap "
        r"\{\text{top-}k\}_{\mathrm{ref}}| / k",
        "applies_to_workflow": "qsvt_observable_readout",
        "measures": "fraction of the largest-magnitude update entries identified correctly",
        "does_not_measure": "exact magnitudes; full vector",
        "used_in_sections": "10",
        "claim_boundary": "exact only when the leading magnitudes are well separated (margin)",
        "notes": "top-k exactness is conditioned on margin",
    },
    {
        "metric_name": "condition_number_weighted_jacobian",
        "formula": r"\kappa(\tilde H) = \sigma_{\max}(\tilde H)/\sigma_{\min}(\tilde H)",
        "applies_to_workflow": "all",
        "measures": "ill-conditioning of the weighted Jacobian / measurement system",
        "does_not_measure": "estimator accuracy by itself",
        "used_in_sections": "8",
        "claim_boundary": "controlled benchmark conditioning",
        "notes": "drives the need for regularized spectral filtering",
    },
    {
        "metric_name": "success_probability",
        "formula": r"\lVert \text{block-encoded output} \rVert^2 (amplitude success proxy)",
        "applies_to_workflow": "qsvt_resource_model",
        "measures": "amplitude success probability proxy for the block-encoded solve",
        "does_not_measure": "wall-clock speed; quantum advantage",
        "used_in_sections": "10",
        "claim_boundary": "resource-model proxy; not hardware execution",
        "notes": "feeds the success-amplification cost model",
    },
)

_USAGE: tuple[dict[str, str], ...] = (
    {
        "manuscript_section": "5. Regularized Spectral Filtering",
        "primary_metrics": "single_step_update_rmse; condition_number_weighted_jacobian",
        "notes": "alpha sensitivity and filter mechanism",
    },
    {
        "manuscript_section": "8. Classical and Conditioning Results",
        "primary_metrics": "single_step_update_rmse; nonlinear_full_state_rmse; angle_rmse; "
        "voltage_magnitude_rmse; residual_norm; weighted_residual_norm; "
        "condition_number_weighted_jacobian",
        "notes": "classical/conditioning, structured stress, nonlinear AC consistency",
    },
    {
        "manuscript_section": "9. QSVT Solver-Prototype Results",
        "primary_metrics": "gate_vs_polynomial_state_error; gate_vs_ridge_state_error; "
        "direction_error_vs_ridge",
        "notes": "selected-subproblem gate validation and overshoot",
    },
    {
        "manuscript_section": "10. Resource and Readout Analysis",
        "primary_metrics": "observable_relative_error; top_k_match; success_probability",
        "notes": "readout limitation and resource model",
    },
)


def build_metric_definitions(config: dict[str, Any]) -> dict[str, Any]:
    input_root = Path(config.get("input_root", "outputs"))
    package_root = Path(config.get("package_root", input_root / "final_manuscript_package"))
    output_dir = Path(config.get("output_dir", package_root / "metrics"))
    ensure_directory(output_dir)
    return _write_outputs(
        output_dir=output_dir,
        input_config={
            "input_root": str(input_root),
            "package_root": str(package_root),
            "output_dir": str(output_dir),
        },
    )


def _definitions_markdown() -> str:
    lines = [
        "# Metric Definitions and Interpretation Notes",
        "",
        PAPER_CLAIM_BOUNDARY,
        "",
        "## Why this document exists",
        "Several distinct metrics share similar names. This note fixes their meaning so the "
        "manuscript never conflates measurement fit with state accuracy, single-step with "
        "nonlinear RMSE, or circuit-implementation error with estimator superiority.",
        "",
        "## Core definitions",
        "",
        "The weighted-Jacobian condition number:",
        "",
        r"\[",
        r"\kappa(\tilde H) = \frac{\sigma_{\max}(\tilde H)}{\sigma_{\min}(\tilde H)}.",
        r"\]",
        "",
        "The Ridge/Tikhonov spectral filter (the QSVT target):",
        "",
        r"\[",
        r"P_{\alpha}(\sigma) = \frac{\sigma}{\sigma^2 + \alpha}.",
        r"\]",
        "",
        "The pseudoinverse spectral filter:",
        "",
        r"\[",
        r"P_{\mathrm{pinv}}(\sigma) = \frac{1}{\sigma}.",
        r"\]",
        "",
        "## Critical interpretation rules",
        "",
        "1. **Residual is measurement fit, not state accuracy.** A small `residual_norm` or "
        "`weighted_residual_norm` reflects how well the estimate fits the (possibly redundant or "
        "corrupted) measurements; it does not guarantee a small state error, especially in "
        "poorly observed or rank-deficient configurations.",
        "2. **Single-step update RMSE is not nonlinear full-state RMSE.** "
        "`single_step_update_rmse` measures one AC-linearized weighted update; "
        "`nonlinear_full_state_rmse` measures the converged iterative estimate. They are reported "
        "separately and must not be compared as if equal.",
        "3. **Gate-vs-polynomial error validates the circuit, not the estimator.** "
        "`gate_vs_polynomial_state_error` and `gate_vs_ridge_state_error` show that the QSVT "
        "circuit reproduces the classical filter; they are agreement metrics and never evidence "
        "of QSVT numerical superiority.",
        "4. **QSVT-target classical equals Ridge under matched alpha.** The QSVT target filter is "
        r"numerically identical to \(P_{\alpha}(\sigma)\); any reported gate-vs-ridge error is a "
        "circuit-synthesis residual, not an estimator difference.",
        "5. **Top-k exactness depends on margin.** `top_k_match` is exact only when the leading "
        "update magnitudes are well separated; with nearly tied magnitudes it degrades. Top-k "
        "identification does not recover exact magnitudes or the full vector.",
        "",
        "See `metric_definitions_table.csv` for the per-metric formula, workflow, what it "
        "measures, and what it does not, and `metric_usage_by_section.csv` for which metrics "
        "appear in which manuscript section.",
        "",
    ]
    return "\n".join(lines)


def _write_outputs(*, output_dir: Path, input_config: dict[str, Any]) -> dict[str, Any]:
    table_path = rows_to_table(
        list(_METRICS), output_dir / "metric_definitions_table.csv", METRIC_COLUMNS
    )
    usage_path = rows_to_table(
        list(_USAGE), output_dir / "metric_usage_by_section.csv", USAGE_COLUMNS
    )
    md_path = output_dir / "metric_definitions.md"
    md_path.write_text(_definitions_markdown(), encoding="utf-8")

    artifacts = {
        "metric_definitions": str(md_path),
        "metric_definitions_table": str(table_path),
        "metric_usage_by_section": str(usage_path),
    }
    write_manifest(
        output_dir,
        artifacts=artifacts,
        input_config=input_config,
        claim_boundary=PAPER_CLAIM_BOUNDARY,
    )
    return {
        "output_dir": output_dir,
        "metric_rows": list(_METRICS),
        "usage_rows": list(_USAGE),
        "n_metrics": len(_METRICS),
        "artifacts": artifacts,
    }
