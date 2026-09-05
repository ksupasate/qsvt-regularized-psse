"""Phase 2: classical spectral-filtering core audit and consolidation.

Audits whether the classical spectral-filtering outputs are complete and
consolidates them into manuscript-ready estimator definitions, main-result and
singular-spectrum tables, and figure-ready data. The QSVT-target classical
filter is framed as numerically equivalent to Ridge/Tikhonov for matched alpha;
it is never framed as beating Ridge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.paper import PAPER_CLAIM_BOUNDARY
from robust_qsvt_se.paper._common import read_csv, rows_to_table
from robust_qsvt_se.qsvt import filters as classical_filters
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

ESTIMATOR_COLUMNS = [
    "estimator",
    "spectral_filter",
    "objective_or_update",
    "stabilizes_small_singular_values",
    "robust_to_outliers",
    "uses_normal_equations",
    "claim_boundary",
    "notes",
]

MAIN_RESULT_COLUMNS = [
    "case",
    "workflow",
    "experiment_group",
    "stress_type",
    "estimator",
    "alpha",
    "rmse",
    "residual_norm",
    "weighted_residual_norm",
    "condition_number",
    "seed",
    "source_artifact",
    "result_status",
    "notes",
]

SPECTRUM_COLUMNS = [
    "case",
    "workflow",
    "experiment_group",
    "subproblem_or_full",
    "matrix_shape",
    "sigma_min",
    "sigma_max",
    "condition_number",
    "numerical_rank",
    "effective_rank",
    "source_artifact",
    "notes",
]

FILTER_BEHAVIOR_COLUMNS = [
    "filter_name",
    "formula",
    "behavior_small_sigma",
    "behavior_large_sigma",
    "regularization_parameter",
    "equivalent_estimator",
    "claim_boundary",
    "notes",
]

MISSING_COLUMNS = [
    "missing_output",
    "needed_for",
    "importance",
    "reason_missing",
    "recommended_action",
]

SPECTRUM_FIGURE_COLUMNS = [
    "case",
    "workflow",
    "singular_index",
    "singular_value",
    "source_artifact",
]

FILTER_FIGURE_COLUMNS = [
    "sigma",
    "pseudoinverse_filter",
    "ridge_filter",
    "qsvt_regularized_filter",
    "truncated_svd_filter",
    "alpha",
    "tau",
    "note",
]

# Static estimator definitions traced to src/robust_qsvt_se/estimators and qsvt/filters.py.
ESTIMATOR_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "estimator": "pseudoinverse",
        "spectral_filter": "P(sigma)=1/sigma (eps cutoff)",
        "objective_or_update": "min ||H_tilde x - r_tilde||_2 via SVD pseudoinverse",
        "stabilizes_small_singular_values": "no (only eps cutoff)",
        "robust_to_outliers": "no",
        "uses_normal_equations": "no",
        "claim_boundary": "unregularized baseline",
        "notes": "Amplifies small singular values.",
    },
    {
        "estimator": "normal_equation_wls",
        "spectral_filter": "implicit (H_tilde^T H_tilde)^-1 H_tilde^T",
        "objective_or_update": "solve G x = H_tilde^T r_tilde, G = H_tilde^T H_tilde",
        "stabilizes_small_singular_values": "no",
        "robust_to_outliers": "no",
        "uses_normal_equations": "yes",
        "claim_boundary": "classical WLS baseline",
        "notes": "Fails when the normal matrix is too ill-conditioned.",
    },
    {
        "estimator": "ridge_tikhonov",
        "spectral_filter": "P_alpha(sigma)=sigma/(sigma^2+alpha)",
        "objective_or_update": "min ||H_tilde x - r_tilde||_2^2 + alpha ||x||_2^2",
        "stabilizes_small_singular_values": "yes (alpha damping)",
        "robust_to_outliers": "no",
        "uses_normal_equations": "no",
        "claim_boundary": "reference regularized spectral filter",
        "notes": "The Ridge/Tikhonov reference for the whole study.",
    },
    {
        "estimator": "truncated_svd",
        "spectral_filter": "P(sigma)=1/sigma if sigma>tau else 0",
        "objective_or_update": "pseudoinverse on retained singular directions",
        "stabilizes_small_singular_values": "yes (hard threshold)",
        "robust_to_outliers": "no",
        "uses_normal_equations": "no",
        "claim_boundary": "rank-truncation baseline",
        "notes": "Discards directions below tau.",
    },
    {
        "estimator": "huber_irls",
        "spectral_filter": "reweighted least squares (no fixed filter)",
        "objective_or_update": "IRLS with Huber weights w_i=min(1, delta/|r_i|)",
        "stabilizes_small_singular_values": "no",
        "robust_to_outliers": "yes",
        "uses_normal_equations": "no (lstsq)",
        "claim_boundary": "robust classical baseline",
        "notes": "Strong robust reference.",
    },
    {
        "estimator": "lav",
        "spectral_filter": "L1 minimization (no spectral filter)",
        "objective_or_update": "min ||H_tilde x - r_tilde||_1 via linprog (highs)",
        "stabilizes_small_singular_values": "no",
        "robust_to_outliers": "yes",
        "uses_normal_equations": "no",
        "claim_boundary": "robust classical baseline",
        "notes": "Least-absolute-value estimator.",
    },
    {
        "estimator": "hhl_style_proxy",
        "spectral_filter": "P(sigma)=1/max(sigma, cutoff)",
        "objective_or_update": "unstable inverse ablation; reports HHL resource proxy",
        "stabilizes_small_singular_values": "no (unstable ablation)",
        "robust_to_outliers": "no",
        "uses_normal_equations": "no",
        "claim_boundary": "unstable ablation, NOT the proposed method",
        "notes": "Used only to expose instability of unregularized inversion.",
    },
    {
        "estimator": "qsvt_target_classical",
        "spectral_filter": "P_alpha(sigma)=sigma/(sigma^2+alpha)",
        "objective_or_update": "classical simulation of the QSVT singular-value transform target",
        "stabilizes_small_singular_values": "yes (alpha damping)",
        "robust_to_outliers": "no",
        "uses_normal_equations": "no",
        "claim_boundary": "numerically identical to Ridge/Tikhonov for matched alpha; "
        "QSVT-compatible implementation pathway, NOT superior to Ridge",
        "notes": "qsvt_regularized_filter returns the same values as ridge_filter.",
    },
)

FILTER_BEHAVIORS: tuple[dict[str, Any], ...] = (
    {
        "filter_name": "pseudoinverse",
        "formula": "P_pinv(sigma)=1/sigma",
        "behavior_small_sigma": "diverges (amplifies noise)",
        "behavior_large_sigma": "-> 0",
        "regularization_parameter": "eps cutoff",
        "equivalent_estimator": "pseudoinverse",
        "claim_boundary": "unregularized baseline",
        "notes": "Unstable on ill-conditioned H.",
    },
    {
        "filter_name": "ridge_tikhonov",
        "formula": "P_alpha(sigma)=sigma/(sigma^2+alpha)",
        "behavior_small_sigma": "-> sigma/alpha (damped)",
        "behavior_large_sigma": "-> 1/sigma",
        "regularization_parameter": "alpha",
        "equivalent_estimator": "ridge_tikhonov",
        "claim_boundary": "reference regularized spectral filter",
        "notes": "Reference filter for the study.",
    },
    {
        "filter_name": "qsvt_target_classical",
        "formula": "P_alpha(sigma)=sigma/(sigma^2+alpha)",
        "behavior_small_sigma": "-> sigma/alpha (damped)",
        "behavior_large_sigma": "-> 1/sigma",
        "regularization_parameter": "alpha",
        "equivalent_estimator": "ridge_tikhonov",
        "claim_boundary": "numerically identical to Ridge for matched alpha; not superior",
        "notes": "QSVT-compatible implementation pathway for the same filter.",
    },
    {
        "filter_name": "truncated_svd",
        "formula": "P(sigma)=1/sigma if sigma>tau else 0",
        "behavior_small_sigma": "0 below tau",
        "behavior_large_sigma": "-> 1/sigma",
        "regularization_parameter": "tau",
        "equivalent_estimator": "truncated_svd",
        "claim_boundary": "rank-truncation baseline",
        "notes": "Hard spectral threshold.",
    },
    {
        "filter_name": "qsvt_unregularized_inverse",
        "formula": "P(sigma)=1/max(sigma, cutoff)",
        "behavior_small_sigma": "clipped at 1/cutoff (still large)",
        "behavior_large_sigma": "-> 0",
        "regularization_parameter": "cutoff",
        "equivalent_estimator": "hhl_style_proxy",
        "claim_boundary": "unstable ablation, not the proposed filter",
        "notes": "Ablation target only.",
    },
)

# Curated classical-result directories to consolidate (read-only).
_RESULT_DIRS: tuple[tuple[str, str, str], ...] = (
    ("real_ieee14_seed10", "ieee14", "PYPOWER AC-linearized"),
    ("real_ieee30_seed10", "ieee30", "PYPOWER AC-linearized"),
    ("real_ieee57_seed10", "ieee57", "PYPOWER AC-linearized"),
    ("real_ieee118_seed10", "ieee118", "PYPOWER AC-linearized"),
    ("real_ieee300_seed10", "ieee300", "PYPOWER AC-linearized"),
    ("nonlinear_ac_ieee14_seed10", "ieee14", "Nonlinear AC"),
    ("nonlinear_ac_ieee30_seed10", "ieee30", "Nonlinear AC"),
    ("nonlinear_ac_ieee57_seed10", "ieee57", "Nonlinear AC"),
    ("nonlinear_ac_ieee118_seed10", "ieee118", "Nonlinear AC"),
    ("diagnostic_missing_baselines", "synthetic", "Synthetic diagnostic"),
)


def build_classical_spectral_filtering_audit(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "input_root": "outputs",
        "output_dir": "outputs/final_manuscript_package/phase2_classical_spectral_filtering",
    }
    resolved.update(config)
    input_root = Path(resolved["input_root"])
    output_dir = ensure_directory(resolved["output_dir"])

    estimator_rows = [dict(e) for e in ESTIMATOR_DEFINITIONS]
    main_rows = classical_main_result_rows(input_root)
    spectrum_rows, spectrum_figure_rows = singular_spectrum_rows(input_root)
    behavior_rows = [dict(b) for b in FILTER_BEHAVIORS]
    filter_figure_rows = filter_comparison_rows(input_root)
    missing_rows = missing_classical_rows(main_rows, spectrum_rows)

    artifacts = _write_outputs(
        output_dir,
        resolved,
        estimator_rows=estimator_rows,
        main_rows=main_rows,
        spectrum_rows=spectrum_rows,
        behavior_rows=behavior_rows,
        missing_rows=missing_rows,
        spectrum_figure_rows=spectrum_figure_rows,
        filter_figure_rows=filter_figure_rows,
    )
    return {
        "output_dir": output_dir,
        "estimator_rows": estimator_rows,
        "main_rows": main_rows,
        "spectrum_rows": spectrum_rows,
        "missing_rows": missing_rows,
        "artifacts": artifacts,
    }


def classical_main_result_rows(input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel_dir, case, workflow in _RESULT_DIRS:
        summary = read_csv(input_root / rel_dir / "summary_metrics.csv")
        if summary.empty:
            continue
        source = f"outputs/{rel_dir}/summary_metrics.csv"
        for _, record in summary.iterrows():
            parameter = str(record.get("sweep_parameter", ""))
            value = record.get("sweep_value", "")
            failure_rate = float(record.get("failure_rate", 0) or 0)
            rows.append(
                {
                    "case": case,
                    "workflow": workflow,
                    "experiment_group": rel_dir,
                    "stress_type": f"{parameter}={value}",
                    "estimator": str(record.get("estimator", "")),
                    "alpha": _alpha_from_sweep(parameter, value),
                    "rmse": _num(record, "rmse_median"),
                    "residual_norm": _num(record, "residual_norm_median"),
                    "weighted_residual_norm": _num(record, "weighted_residual_norm_median"),
                    "condition_number": _num(record, "condition_number_median"),
                    "seed": "aggregated",
                    "source_artifact": source,
                    "result_status": "has_failures" if failure_rate > 0 else "completed",
                    "notes": "median over seeds; benchmark network model",
                }
            )
    return rows


def singular_spectrum_rows(
    input_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spectrum_rows: list[dict[str, Any]] = []
    figure_rows: list[dict[str, Any]] = []
    for rel_dir, case, workflow in _RESULT_DIRS:
        singular = read_csv(input_root / rel_dir / "singular_values.csv")
        if singular.empty or "singular_value" not in singular.columns:
            continue
        source = f"outputs/{rel_dir}/singular_values.csv"
        baseline = _baseline_trial(singular)
        sigma = baseline["singular_value"].to_numpy(dtype=float)
        sigma = sigma[np.isfinite(sigma) & (sigma > 0)]
        if sigma.size == 0:
            continue
        sigma_sorted = np.sort(sigma)[::-1]
        sigma_max = float(sigma_sorted[0])
        sigma_min = float(sigma_sorted[-1])
        rank = int(sigma_sorted.size)
        condition = _sig(sigma_max / sigma_min) if sigma_min else ""
        effective = _effective_rank(input_root / rel_dir / "aggregate_metrics.csv", case)
        spectrum_rows.append(
            {
                "case": case,
                "workflow": workflow,
                "experiment_group": rel_dir,
                "subproblem_or_full": "full",
                "matrix_shape": _matrix_shape(input_root / rel_dir / "aggregate_metrics.csv"),
                "sigma_min": _sig(sigma_min),
                "sigma_max": _sig(sigma_max),
                "condition_number": condition,
                "numerical_rank": rank,
                "effective_rank": effective,
                "source_artifact": source,
                "notes": "baseline (unperturbed) weighted-Jacobian spectrum",
            }
        )
        for index, value in enumerate(sigma_sorted):
            figure_rows.append(
                {
                    "case": case,
                    "workflow": workflow,
                    "singular_index": index,
                    "singular_value": round(float(value), 6),
                    "source_artifact": source,
                }
            )
    return spectrum_rows, figure_rows


def filter_comparison_rows(input_root: Path) -> list[dict[str, Any]]:
    """Evaluate the classical filter functions over a representative real spectrum grid."""

    singular = read_csv(input_root / "real_ieee14_seed10" / "singular_values.csv")
    if not singular.empty and "singular_value" in singular.columns:
        baseline = _baseline_trial(singular)
        sigma = baseline["singular_value"].to_numpy(dtype=float)
        sigma = sigma[np.isfinite(sigma) & (sigma > 0)]
    else:
        sigma = np.array([])
    if sigma.size == 0:
        sigma = np.logspace(0, 3.5, 64)
    lo, hi = float(np.min(sigma)), float(np.max(sigma))
    grid = np.logspace(np.log10(max(lo, 1e-3)), np.log10(hi), 128)

    alpha = round(hi**2 * 1e-4, 6)  # alpha scaled to the spectrum so damping is visible
    tau = round(lo * 5.0, 6)
    pinv = classical_filters.inverse_filter(grid, 1e-9)
    ridge = classical_filters.ridge_filter(grid, alpha)
    qsvt = classical_filters.qsvt_regularized_filter(grid, alpha)
    trunc = classical_filters.truncated_inverse_filter(grid, tau)
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(grid):
        rows.append(
            {
                "sigma": round(float(value), 6),
                "pseudoinverse_filter": round(float(pinv[index]), 9),
                "ridge_filter": round(float(ridge[index]), 9),
                "qsvt_regularized_filter": round(float(qsvt[index]), 9),
                "truncated_svd_filter": round(float(trunc[index]), 9),
                "alpha": alpha,
                "tau": tau,
                "note": "ridge_filter == qsvt_regularized_filter (numerically identical)",
            }
        )
    return rows


def missing_classical_rows(
    main_rows: list[dict[str, Any]], spectrum_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not main_rows:
        rows.append(
            {
                "missing_output": "consolidated classical main results",
                "needed_for": "Classical and Conditioning Results section",
                "importance": "critical",
                "reason_missing": "no per-case summary_metrics found",
                "recommended_action": "run src/robust_qsvt_se/cli/run_experiment.py per case",
            }
        )
    if not spectrum_rows:
        rows.append(
            {
                "missing_output": "singular-spectrum summary",
                "needed_for": "conditioning figure and table",
                "importance": "high",
                "reason_missing": "no singular_values.csv found",
                "recommended_action": "run benchmark experiments that export singular_values.csv",
            }
        )
    # Genuine consolidation gaps even when the per-run artifacts exist.
    rows.append(
        {
            "missing_output": "alpha-resolved main-result table (explicit alpha column per row)",
            "needed_for": "alpha-sensitivity discussion (Phase 3, out of scope here)",
            "importance": "medium",
            "reason_missing": "summary_metrics does not record per-row regularization alpha",
            "recommended_action": "consolidate from alpha_sensitivity_summary in a Phase 3 task",
        }
    )
    lav_present = any(r["estimator"] == "lav" for r in main_rows)
    if not lav_present:
        rows.append(
            {
                "missing_output": "LAV estimator main results across cases",
                "needed_for": "complete robust-baseline comparison",
                "importance": "low",
                "reason_missing": "LAV not included in the consolidated seed-expanded sweeps",
                "recommended_action": "add LAV to the benchmark estimator set if needed",
            }
        )
    return rows


def _alpha_from_sweep(parameter: str, value: Any) -> Any:
    return value if "alpha" in str(parameter).lower() else ""


def _num(record: pd.Series, column: str) -> Any:
    if column not in record:
        return ""
    value = pd.to_numeric(record[column], errors="coerce")
    return "" if pd.isna(value) else round(float(value), 8)


def _baseline_trial(singular: pd.DataFrame) -> pd.DataFrame:
    """Isolate a single matrix spectrum (one trial, one estimator) at the baseline stress."""

    frame = singular.copy()
    if "value" in frame.columns:
        numeric = pd.to_numeric(frame["value"], errors="coerce")
        if numeric.notna().any():
            frame = frame[numeric == numeric.min()]
    # Nonlinear AC spectra carry an estimator column and per-iteration repeats; pin to one
    # estimator so the reported numerical rank matches a single weighted Jacobian.
    if "estimator" in frame.columns and not frame.empty:
        frame = frame[frame["estimator"] == frame["estimator"].iloc[0]]
    if "trial_id" in frame.columns and not frame.empty:
        frame = frame[frame["trial_id"] == frame["trial_id"].iloc[0]]
    if "singular_index" in frame.columns and not frame.empty:
        frame = frame.drop_duplicates(subset="singular_index", keep="first")
    return frame


def _sig(value: float, digits: int = 6) -> float:
    if not np.isfinite(value) or value == 0:
        return float(value)
    from math import floor, log10

    return round(value, -floor(log10(abs(value))) + (digits - 1))


def _effective_rank(path: Path, case: str) -> Any:
    frame = read_csv(path)
    if frame.empty or "effective_rank" not in frame.columns:
        return ""
    if "case_name" in frame.columns:
        match = frame[frame["case_name"].astype(str).str.lower() == case.lower()]
        if not match.empty:
            frame = match
    value = pd.to_numeric(frame["effective_rank"], errors="coerce").dropna()
    return int(value.iloc[0]) if not value.empty else ""


def _matrix_shape(path: Path) -> str:
    frame = read_csv(path)
    if frame.empty:
        return ""
    measurements = pd.to_numeric(frame.get("n_measurements"), errors="coerce").dropna()
    states = pd.to_numeric(frame.get("n_states"), errors="coerce").dropna()
    if measurements.empty or states.empty:
        return ""
    return f"{int(measurements.iloc[0])}x{int(states.iloc[0])}"


def _write_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    *,
    estimator_rows: list[dict[str, Any]],
    main_rows: list[dict[str, Any]],
    spectrum_rows: list[dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
    spectrum_figure_rows: list[dict[str, Any]],
    filter_figure_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    estimator_path = rows_to_table(
        estimator_rows, output_dir / "paper_table_estimator_definitions.csv", ESTIMATOR_COLUMNS
    )
    main_path = rows_to_table(
        main_rows, output_dir / "paper_table_classical_main_results.csv", MAIN_RESULT_COLUMNS
    )
    spectrum_path = rows_to_table(
        spectrum_rows, output_dir / "paper_table_singular_spectrum_summary.csv", SPECTRUM_COLUMNS
    )
    behavior_path = rows_to_table(
        behavior_rows, output_dir / "paper_table_filter_behavior.csv", FILTER_BEHAVIOR_COLUMNS
    )
    missing_path = rows_to_table(
        missing_rows, output_dir / "missing_classical_outputs.csv", MISSING_COLUMNS
    )
    spectrum_figure_path = rows_to_table(
        spectrum_figure_rows,
        output_dir / "figure_data_singular_spectrum.csv",
        SPECTRUM_FIGURE_COLUMNS,
    )
    filter_figure_path = rows_to_table(
        filter_figure_rows, output_dir / "figure_data_filter_comparison.csv", FILTER_FIGURE_COLUMNS
    )
    status_path = output_dir / "classical_core_status.md"
    status_path.write_text(
        _status_markdown(main_rows, spectrum_rows, missing_rows), encoding="utf-8"
    )

    manifest = write_manifest(
        output_dir,
        artifacts={
            "paper_table_estimator_definitions": str(estimator_path),
            "paper_table_classical_main_results": str(main_path),
            "paper_table_singular_spectrum_summary": str(spectrum_path),
            "paper_table_filter_behavior": str(behavior_path),
            "missing_classical_outputs": str(missing_path),
            "figure_data_singular_spectrum": str(spectrum_figure_path),
            "figure_data_filter_comparison": str(filter_figure_path),
            "classical_core_status": str(status_path),
        },
        input_config=resolved,
        claim_boundary=PAPER_CLAIM_BOUNDARY,
    )
    return {
        "manifest": manifest,
        "paper_table_estimator_definitions": estimator_path,
        "paper_table_classical_main_results": main_path,
        "paper_table_singular_spectrum_summary": spectrum_path,
        "paper_table_filter_behavior": behavior_path,
        "missing_classical_outputs": missing_path,
        "figure_data_singular_spectrum": spectrum_figure_path,
        "figure_data_filter_comparison": filter_figure_path,
        "classical_core_status": status_path,
    }


def _status_markdown(
    main_rows: list[dict[str, Any]],
    spectrum_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
) -> str:
    complete = bool(main_rows) and bool(spectrum_rows)
    cases = sorted({r["case"] for r in main_rows})
    return "\n".join(
        [
            "# Classical Spectral Filtering Core Audit",
            "",
            PAPER_CLAIM_BOUNDARY,
            "",
            "## Estimator and filter definitions",
            "- P_pinv(sigma)=1/sigma; P_alpha(sigma)=sigma/(sigma^2+alpha); "
            "kappa(H_tilde)=sigma_max/sigma_min on the weighted Jacobian.",
            "- The QSVT-target classical filter is numerically identical to Ridge/Tikhonov for "
            "matched alpha (qsvt_regularized_filter == ridge_filter); it is not superior to Ridge.",
            "",
            "## Completeness",
            f"- Final classical outputs complete: {'yes' if complete else 'no'}.",
            f"- Main-result rows consolidated: {len(main_rows)} across cases {cases}.",
            f"- Singular-spectrum rows consolidated: {len(spectrum_rows)}.",
            f"- Missing/consolidation gaps recorded: {len(missing_rows)}.",
            "",
            "## Conclusion",
            "Estimator definitions, main results, singular-spectrum summary, and filter-comparison "
            "figure data are manuscript-ready. Remaining gaps are recorded in "
            "missing_classical_outputs.csv and do not assert any QSVT-over-Ridge advantage.",
            "",
        ]
    )
