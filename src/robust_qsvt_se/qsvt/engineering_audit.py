from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory, write_json

FORBIDDEN_PHRASES = [
    "quantum speedup",
    "quantum advantage",
    "full IEEE-scale hardware execution",
    "PMU/SCADA field data",
    "field-data validation",
    "QSVT outperforms Ridge",
    "QSVT numerically outperforms Ridge",
    "QSVT numerically beating Ridge",
    "validated on PMU/SCADA field data",
]

SAFE_CONTEXT_MARKERS = [
    "avoid",
    "do not",
    "does not",
    "not demonstrate",
    "not claim",
    "not claimed",
    "without claiming",
    "no quantum",
    "not a hardware",
    "not hardware",
    "not full",
    "not used",
    "not use",
    "not a quantum",
    "not qsvt phase",
    "no pmu/scada",
    "not pmu/scada",
    "not field data",
    "proxy diagnostics only",
    "distinction between",
    "is not",
    "claims to avoid",
    "claim boundaries",
    "boundary",
    "caveat",
    "limitation",
]

REQUIRED_CLAIMS_14 = {
    "Dense block-encoding prototype was validated on small normalized matrices.",
    "Exact QSVT-target spectral filtering matches Ridge/Tikhonov under the same alpha.",
    "Selected-alpha bounded polynomial/phase approximations were validated.",
    "QSVT resource estimates support feasibility discussion only.",
    "Shot-level readout analysis quantifies sampling cost for selected observables.",
    "Full-vector readout remains a limitation.",
    "Hardware-aware analysis is simulation/proxy only.",
    "Dense block encoding is not a scalable oracle.",
    "Multi-case resource diagnostics extend beyond IEEE14 where feasible.",
    "The extension does not demonstrate quantum speedup.",
    "The extension does not demonstrate quantum advantage.",
    "The extension does not execute full IEEE-scale QSVT on quantum hardware.",
    "The extension does not use real PMU/SCADA field data.",
    "QSVT does not numerically outperform Ridge/Tikhonov under the same alpha/filter.",
    "Selected-alpha polynomial approximation diagnostics were implemented.",
    "Degree sweep quantifies approximation error versus resource cost.",
    "Adaptive degree selection identifies whether target tolerances can be met.",
    "Optional phase synthesis is performed only if dependencies are available.",
    "Polynomial fallback is not full QSP/QSVT phase synthesis.",
    "Passing/failing 1e-3 tolerance is reported explicitly.",
    "Query count increases with polynomial degree.",
    "Approximation diagnostics support feasibility discussion only.",
    "Phase-response convention diagnostics validate the PennyLane scalar response convention.",
    "Known sanity-polynomial QSP/QSVT responses are checked before Ridge-target validation.",
    "Full phase-level Ridge/Tikhonov target validation remains unresolved when reported failed.",
    "Phase backend capabilities were audited.",
    "Stable polynomial candidates were tested.",
    "Bounded Ridge/Tikhonov target phase validation passed only if all gates passed.",
    "Sanity-polynomial phase response passed.",
    "Chebyshev-to-monomial conversion instability was measured.",
    "No unstable polynomial was forced into phase synthesis.",
    "No tolerance relaxation was used.",
    "No quantum speedup or hardware execution is claimed.",
    "External QSP/QSVT phase backends were audited.",
    "pyqsp/QSPPACK/PennyLane/local optimization backend availability was tested.",
    "Backend sanity regression was performed.",
    (
        "Target-level bounded Ridge/Tikhonov phase validation passed only if "
        "full-domain error <= 1e-3."
    ),
    "Actual-singular-value-only validation is not full-domain validation.",
    "No unsafe monomial candidate was forced into phase synthesis.",
    "Adaptive multicase degree search quantifies larger-case degree and query requirements.",
    "Some larger IEEE cases require higher degree than IEEE14 under the same tolerance.",
}


def build_engineering_audit(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    root_dir = Path(resolved["root_dir"])
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    specs = resolved.get("artifact_specs") or default_artifact_specs()
    rows = audit_artifacts(root_dir, specs)
    rows.extend(audit_docs(root_dir, list(resolved["docs_paths"])))
    rows.extend(audit_claim_matrix(root_dir / "outputs/qsvt_engineering_extension"))
    verdict = _audit_verdict(rows)
    table = pd.DataFrame(rows)
    table_path = output_dir / "audit_table.csv"
    json_path = output_dir / "audit_results.json"
    summary_path = output_dir / "audit_summary.md"
    table.to_csv(table_path, index=False)
    write_json(
        json_path,
        {
            "verdict": verdict,
            "row_count": len(rows),
            "status_counts": _status_counts(rows),
            "rows": rows,
        },
    )
    summary_path.write_text(_summary_markdown(verdict, rows), encoding="utf-8")
    manifest_path = write_manifest(
        output_dir,
        artifacts={
            "audit_summary_md": str(summary_path),
            "audit_results_json": str(json_path),
            "audit_table_csv": str(table_path),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": table,
        "artifacts": {
            "audit_summary_md": summary_path,
            "audit_results_json": json_path,
            "audit_table_csv": table_path,
            "manifest": manifest_path,
        },
    }


def audit_artifacts(root_dir: Path, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        relative_path = Path(spec["path"])
        path = root_dir / relative_path
        group = str(spec.get("group", relative_path.parent))
        rows.append(_file_row(group, relative_path, path))
        if not path.exists():
            continue
        if path.suffix == ".csv":
            rows.extend(_audit_csv(path, relative_path, spec))
        elif path.name == "manifest.json":
            rows.extend(_audit_manifest(path, relative_path, group))
        elif path.suffix == ".json":
            rows.extend(_audit_json(path, relative_path, group))
        elif path.suffix == ".md":
            rows.append(
                {
                    "group": group,
                    "path": str(relative_path),
                    "check": "markdown_nonempty",
                    "status": "pass" if path.read_text(encoding="utf-8").strip() else "fail",
                    "details": "markdown has content",
                    "value": "",
                    "threshold": "",
                    "classification": "",
                }
            )
    return rows


def audit_docs(root_dir: Path, docs_paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in docs_paths:
        path = root_dir / relative
        rows.append(_file_row("documentation_claim_safety", Path(relative), path))
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for hit in classify_forbidden_wording(text):
            status = "pass" if hit["classification"] == "safe_context" else "warn"
            if hit["classification"] == "unsafe_context":
                status = "fail"
            rows.append(
                {
                    "group": "documentation_claim_safety",
                    "path": relative,
                    "check": f"forbidden_phrase:{hit['phrase']}",
                    "status": status,
                    "details": hit["context"],
                    "value": hit["phrase"],
                    "threshold": "safe_context",
                    "classification": hit["classification"],
                }
            )
    return rows


def audit_claim_matrix(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = output_dir / "claim_support_matrix.csv"
    relative = Path("outputs/qsvt_engineering_extension/claim_support_matrix.csv")
    rows.append(_file_row("claim_matrix", relative, path))
    if not path.exists():
        return rows
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return [_fail_row("claim_matrix", relative, "read_csv", str(exc))]
    missing_columns = _missing_columns(
        frame,
        [
            "claim",
            "support_status",
            "supporting_files",
            "supporting_outputs",
            "strength",
            "limitations",
            "recommended_wording",
            "avoid_wording",
        ],
    )
    rows.append(_columns_row("claim_matrix", relative, missing_columns))
    claims = set(frame.get("claim", pd.Series(dtype=str)).astype(str))
    missing_claims = sorted(REQUIRED_CLAIMS_14 - claims)
    rows.append(
        {
            "group": "claim_matrix",
            "path": str(relative),
            "check": "required_14_claims",
            "status": "pass" if not missing_claims else "fail",
            "details": "" if not missing_claims else "; ".join(missing_claims),
            "value": len(claims),
            "threshold": len(REQUIRED_CLAIMS_14),
            "classification": "",
        }
    )
    return rows


def classify_forbidden_wording(text: str) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    lower_text = text.lower()
    for phrase in FORBIDDEN_PHRASES:
        phrase_lower = phrase.lower()
        start = 0
        while True:
            index = lower_text.find(phrase_lower, start)
            if index < 0:
                break
            context_start = max(0, index - 180)
            context_end = min(len(text), index + len(phrase) + 180)
            context = " ".join(text[context_start:context_end].split())
            hits.append(
                {
                    "phrase": phrase,
                    "classification": classify_forbidden_context(context, phrase),
                    "context": context,
                }
            )
            start = index + len(phrase)
    return hits


def classify_forbidden_context(context: str, phrase: str) -> str:
    lowered = context.lower()
    if any(marker in lowered for marker in SAFE_CONTEXT_MARKERS):
        return "safe_context"
    phrase_lower = phrase.lower()
    if "demonstrate" in lowered and phrase_lower in lowered:
        return "unsafe_context"
    if "outperform" in lowered or "superior" in lowered:
        return "unsafe_context"
    return "needs_manual_review"


def default_artifact_specs() -> list[dict[str, Any]]:
    return [
        {
            "group": "block_encoding",
            "path": "outputs/qsvt_block_encoding/block_encoding_summary.csv",
            "required_columns": [
                "matrix_shape",
                "beta",
                "spectral_norm_normalized",
                "encoded_block_error",
                "unitarity_error",
                "passed",
            ],
            "max_columns": {"encoded_block_error": 1.0e-8, "unitarity_error": 1.0e-8},
            "true_columns": ["passed"],
        },
        {
            "group": "block_encoding",
            "path": "outputs/qsvt_block_encoding/block_encoding_summary.json",
        },
        {"group": "block_encoding", "path": "outputs/qsvt_block_encoding/manifest.json"},
        {
            "group": "state_demo",
            "path": "outputs/qsvt_end_to_end_state_demo/state_demo_summary.csv",
            "required_columns": [
                "relative_error_vs_ridge",
                "cosine_similarity_vs_ridge",
                "state_fidelity_vs_ridge_direction",
                "passed_equivalence_check",
            ],
            "max_columns": {"relative_error_vs_ridge": 1.0e-8},
            "min_columns": {
                "cosine_similarity_vs_ridge": 1.0 - 1.0e-8,
                "state_fidelity_vs_ridge_direction": 1.0 - 1.0e-8,
            },
            "true_columns": ["passed_equivalence_check"],
        },
        {"group": "state_demo", "path": "outputs/qsvt_end_to_end_state_demo/manifest.json"},
        {
            "group": "resource_readout",
            "path": "outputs/qsvt_resource_readout/resource_summary.csv",
            "required_columns": [
                "m",
                "n",
                "kappa",
                "qsvt_degree_estimate",
                "query_count_estimate",
                "readout_caveat",
                "claim_strength",
            ],
            "nonnegative_columns": [
                "m",
                "n",
                "qsvt_degree_estimate",
                "query_count_estimate",
                "depth_estimate",
            ],
        },
        {"group": "resource_readout", "path": "outputs/qsvt_resource_readout/readout_summary.md"},
        {"group": "resource_readout", "path": "outputs/qsvt_resource_readout/manifest.json"},
        {
            "group": "alpha_sensitivity",
            "path": "outputs/qsvt_alpha_resource_sensitivity/alpha_resource_sensitivity.csv",
            "required_columns": [
                "alpha",
                "max_filter_gain",
                "bounded_scaling_C",
                "estimated_qsvt_degree",
                "estimated_query_count",
            ],
            "nonnegative_columns": [
                "alpha",
                "max_filter_gain",
                "bounded_scaling_C",
                "estimated_qsvt_degree",
                "estimated_query_count",
            ],
        },
        {
            "group": "preconditioning",
            "path": "outputs/qsvt_preconditioning_diagnostics/preconditioning_summary.csv",
            "required_columns": [
                "preconditioner_type",
                "kappa_before",
                "kappa_after",
                "estimated_qsvt_degree_before",
                "estimated_qsvt_degree_after",
                "claim_strength",
            ],
        },
        {
            "group": "selected_alpha_phase_validation",
            "path": "outputs/qsvt_selected_alpha_phase_validation/phase_validation_summary.csv",
            "required_columns": [
                "alpha",
                "bounded_scaling_C",
                "polynomial_degree",
                "max_pointwise_target_error",
                "query_count",
                "passed",
            ],
            "nonnegative_columns": [
                "alpha",
                "bounded_scaling_C",
                "polynomial_degree",
                "max_pointwise_target_error",
                "query_count",
            ],
        },
        {
            "group": "shot_readout",
            "path": "outputs/qsvt_shot_readout/shot_readout_summary.csv",
            "required_columns": [
                "observable_name",
                "shots",
                "standard_error",
                "required_shots_estimate",
                "readout_caveat",
            ],
            "nonnegative_columns": ["shots", "standard_error", "required_shots_estimate"],
        },
        {
            "group": "hardware_aware",
            "path": "outputs/qsvt_hardware_aware/hardware_aware_summary.csv",
            "required_columns": [
                "logical_qubits_estimate",
                "total_qubits_estimate",
                "estimated_depth",
                "estimated_two_qubit_gates",
                "hardware_caveat",
            ],
            "nonnegative_columns": [
                "logical_qubits_estimate",
                "total_qubits_estimate",
                "estimated_depth",
                "estimated_two_qubit_gates",
            ],
        },
        {
            "group": "block_encoding_scalability",
            "path": "outputs/qsvt_block_encoding_scalability/scalability_summary.csv",
            "required_columns": [
                "case_name",
                "m",
                "n",
                "nonzeros",
                "estimated_dense_encoding_dimension",
                "estimated_index_qubits",
                "scalability_caveat",
            ],
        },
        {
            "group": "multicase_resource",
            "path": "outputs/qsvt_multicase_resource_diagnostics/multicase_resource_summary.csv",
            "required_columns": [
                "case_name",
                "status",
                "m",
                "n",
                "qsvt_degree_estimate",
                "query_count_estimate",
                "readout_caveat",
                "failure_reason_if_any",
            ],
        },
        {
            "group": "multicase_resource",
            "path": "outputs/qsvt_multicase_resource_diagnostics/failure_log.csv",
            "allow_empty": True,
        },
        {
            "group": "approximation_degree_sweep",
            "path": "outputs/qsvt_approximation_degree_sweep/degree_sweep_summary.csv",
            "required_columns": [
                "alpha",
                "degree",
                "max_pointwise_error",
                "query_count_estimate",
                "passed_tol_1e_minus_3",
                "caveat",
            ],
            "nonnegative_columns": ["degree", "max_pointwise_error", "query_count_estimate"],
        },
        {
            "group": "approximation_degree_sweep",
            "path": "outputs/qsvt_approximation_degree_sweep/manifest.json",
        },
        {
            "group": "adaptive_degree_selection",
            "path": "outputs/qsvt_adaptive_degree_selection/adaptive_degree_summary.csv",
            "required_columns": [
                "alpha",
                "target_tolerance",
                "selected_degree",
                "selected_query_count",
                "achieved_max_error",
                "status",
            ],
            "nonnegative_columns": [
                "target_tolerance",
                "selected_degree",
                "selected_query_count",
                "achieved_max_error",
            ],
        },
        {
            "group": "adaptive_degree_selection",
            "path": "outputs/qsvt_adaptive_degree_selection/manifest.json",
        },
        {
            "group": "polynomial_method_comparison",
            "path": "outputs/qsvt_polynomial_method_comparison/method_comparison_summary.csv",
            "required_columns": [
                "method",
                "alpha",
                "degree",
                "max_pointwise_error",
                "passed_1e_minus_3",
                "caveat",
            ],
            "nonnegative_columns": ["degree", "max_pointwise_error", "query_count_estimate"],
        },
        {
            "group": "polynomial_method_comparison",
            "path": "outputs/qsvt_polynomial_method_comparison/manifest.json",
        },
        {
            "group": "optional_phase_synthesis",
            "path": (
                "outputs/qsvt_optional_phase_synthesis_validation/phase_synthesis_summary.csv"
            ),
            "required_columns": [
                "alpha",
                "phase_method",
                "dependency_available",
                "degree",
                "phase_count",
                "status",
                "caveat",
            ],
            "nonnegative_columns": ["degree", "phase_count", "query_count_estimate"],
        },
        {
            "group": "optional_phase_synthesis",
            "path": "outputs/qsvt_optional_phase_synthesis_validation/manifest.json",
        },
        {
            "group": "phase_response_convention",
            "path": (
                "outputs/qsvt_phase_response_convention_diagnostics/convention_search_summary.csv"
            ),
            "required_columns": [
                "polynomial_name",
                "target_type",
                "degree",
                "phase_method",
                "phase_count",
                "phase_order",
                "phase_sign",
                "phase_offset_rule",
                "signal_operator_convention",
                "response_component",
                "coefficient_basis_input",
                "coefficient_basis_expected",
                "max_pointwise_error",
                "status",
                "caveat",
            ],
            "nonnegative_columns": ["degree", "phase_count", "max_pointwise_error"],
        },
        {
            "group": "phase_response_convention",
            "path": (
                "outputs/qsvt_phase_response_convention_diagnostics/sanity_polynomial_results.csv"
            ),
            "required_columns": [
                "polynomial_name",
                "degree",
                "best_max_pointwise_error",
                "best_status",
                "best_convention",
                "sanity_tolerance",
            ],
            "max_columns": {"best_max_pointwise_error": 1.0e-6},
        },
        {
            "group": "phase_response_convention",
            "path": (
                "outputs/qsvt_phase_response_convention_diagnostics/best_convention_report.md"
            ),
        },
        {
            "group": "phase_response_convention",
            "path": "outputs/qsvt_phase_response_convention_diagnostics/manifest.json",
        },
        {
            "group": "adaptive_multicase_degree_search",
            "path": (
                "outputs/qsvt_adaptive_multicase_degree_search/adaptive_multicase_summary.csv"
            ),
            "required_columns": [
                "case_name",
                "status",
                "alpha",
                "m",
                "n",
                "kappa",
                "target_tolerance",
                "selected_degree",
                "selected_query_count",
                "achieved_max_error",
                "best_degree_tested",
                "best_max_error",
                "failure_reason_if_any",
                "interpretation",
            ],
            "nonnegative_columns": [
                "alpha",
                "m",
                "n",
                "kappa",
                "target_tolerance",
                "selected_degree",
                "selected_query_count",
                "achieved_max_error",
                "best_degree_tested",
                "best_max_error",
            ],
        },
        {
            "group": "adaptive_multicase_degree_search",
            "path": (
                "outputs/qsvt_adaptive_multicase_degree_search/adaptive_multicase_search_trace.csv"
            ),
            "required_columns": [
                "case_name",
                "alpha",
                "degree",
                "query_count",
                "max_pointwise_error",
                "passed_1e_minus_3",
                "status",
            ],
            "nonnegative_columns": ["alpha", "degree", "query_count", "max_pointwise_error"],
        },
        {
            "group": "adaptive_multicase_degree_search",
            "path": (
                "outputs/qsvt_adaptive_multicase_degree_search/adaptive_multicase_failure_log.csv"
            ),
            "allow_empty": True,
        },
        {
            "group": "adaptive_multicase_degree_search",
            "path": "outputs/qsvt_adaptive_multicase_degree_search/manifest.json",
        },
        {
            "group": "phase_and_multicase_summary",
            "path": "outputs/qsvt_phase_and_multicase_summary/phase_and_multicase_summary.md",
        },
        {
            "group": "phase_and_multicase_summary",
            "path": "outputs/qsvt_phase_and_multicase_summary/phase_and_multicase_summary.csv",
            "required_columns": [
                "summary_area",
                "item",
                "status",
                "max_error",
                "degree",
                "query_count",
                "details",
            ],
        },
        {
            "group": "phase_and_multicase_summary",
            "path": "outputs/qsvt_phase_and_multicase_summary/manifest.json",
        },
        {
            "group": "approximation_tradeoff",
            "path": "outputs/qsvt_approximation_tradeoff/tradeoff_summary.csv",
            "required_columns": [
                "alpha",
                "target_tolerance",
                "status",
                "query_count_estimate",
                "claim_safe_interpretation",
            ],
            "nonnegative_columns": ["target_tolerance", "query_count_estimate"],
        },
        {
            "group": "approximation_tradeoff",
            "path": "outputs/qsvt_approximation_tradeoff/tradeoff_report.md",
        },
        {
            "group": "approximation_tradeoff",
            "path": "outputs/qsvt_approximation_tradeoff/manifest.json",
        },
        {
            "group": "multicase_approximation",
            "path": (
                "outputs/qsvt_multicase_approximation_diagnostics/"
                "multicase_approximation_summary.csv"
            ),
            "required_columns": [
                "case_name",
                "status",
                "alpha",
                "degree",
                "max_pointwise_error",
                "query_count_estimate",
                "failure_reason_if_any",
            ],
        },
        {
            "group": "multicase_approximation",
            "path": "outputs/qsvt_multicase_approximation_diagnostics/failure_log.csv",
            "allow_empty": True,
        },
        {
            "group": "multicase_approximation",
            "path": "outputs/qsvt_multicase_approximation_diagnostics/manifest.json",
        },
        {
            "group": "engineering_extension",
            "path": "outputs/qsvt_engineering_extension/claim_support_matrix.csv",
        },
        {
            "group": "engineering_extension",
            "path": "outputs/qsvt_engineering_extension/manifest.json",
        },
    ]


def _audit_csv(path: Path, relative_path: Path, spec: dict[str, Any]) -> list[dict[str, Any]]:
    group = str(spec.get("group", relative_path.parent))
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return [_fail_row(group, relative_path, "read_csv", str(exc))]
    rows = [
        {
            "group": group,
            "path": str(relative_path),
            "check": "row_count_positive",
            "status": "pass" if len(frame) > 0 or bool(spec.get("allow_empty")) else "fail",
            "details": "",
            "value": len(frame),
            "threshold": ">0",
            "classification": "",
        }
    ]
    missing_columns = _missing_columns(frame, spec.get("required_columns", []))
    rows.append(_columns_row(group, relative_path, missing_columns))
    rows.extend(_finite_numeric_rows(group, relative_path, frame))
    rows.extend(_threshold_rows(group, relative_path, frame, spec.get("max_columns", {}), "max"))
    rows.extend(_threshold_rows(group, relative_path, frame, spec.get("min_columns", {}), "min"))
    rows.extend(_nonnegative_rows(group, relative_path, frame, spec.get("nonnegative_columns", [])))
    rows.extend(_true_column_rows(group, relative_path, frame, spec.get("true_columns", [])))
    return rows


def _audit_json(path: Path, relative_path: Path, group: str) -> list[dict[str, Any]]:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [_fail_row(group, relative_path, "valid_json", str(exc))]
    return [
        {
            "group": group,
            "path": str(relative_path),
            "check": "valid_json",
            "status": "pass",
            "details": "",
            "value": "",
            "threshold": "",
            "classification": "",
        }
    ]


def _audit_manifest(path: Path, relative_path: Path, group: str) -> list[dict[str, Any]]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [_fail_row(group, relative_path, "valid_manifest_json", str(exc))]
    required = {"command", "generated_at", "git_commit", "input_config", "artifacts"}
    missing = sorted(required - set(manifest))
    return [
        {
            "group": group,
            "path": str(relative_path),
            "check": "manifest_required_fields",
            "status": "pass" if not missing else "fail",
            "details": "" if not missing else "; ".join(missing),
            "value": len(required) - len(missing),
            "threshold": len(required),
            "classification": "",
        }
    ]


def _finite_numeric_rows(group: str, path: Path, frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for column in frame.select_dtypes(include=[np.number]).columns:
        values = frame[column].dropna().to_numpy(dtype=float)
        finite = bool(np.all(np.isfinite(values)))
        rows.append(
            {
                "group": group,
                "path": str(path),
                "check": f"finite_numeric:{column}",
                "status": "pass" if finite else "fail",
                "details": "",
                "value": "",
                "threshold": "finite",
                "classification": "",
            }
        )
    return rows


def _threshold_rows(
    group: str,
    path: Path,
    frame: pd.DataFrame,
    thresholds: dict[str, float],
    kind: str,
) -> list[dict[str, Any]]:
    rows = []
    for column, threshold in thresholds.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        observed = float(values.max() if kind == "max" else values.min()) if len(values) else np.nan
        passed = observed <= threshold if kind == "max" else observed >= threshold
        rows.append(
            {
                "group": group,
                "path": str(path),
                "check": f"{kind}_threshold:{column}",
                "status": "pass" if passed else "fail",
                "details": "",
                "value": observed,
                "threshold": threshold,
                "classification": "",
            }
        )
    return rows


def _nonnegative_rows(
    group: str,
    path: Path,
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> list[dict[str, Any]]:
    rows = []
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        observed = float(values.min()) if len(values) else 0.0
        rows.append(
            {
                "group": group,
                "path": str(path),
                "check": f"nonnegative:{column}",
                "status": "pass" if observed >= 0.0 else "fail",
                "details": "",
                "value": observed,
                "threshold": ">=0",
                "classification": "",
            }
        )
    return rows


def _true_column_rows(
    group: str,
    path: Path,
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> list[dict[str, Any]]:
    rows = []
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column].astype(str).str.lower().isin({"true", "1"})
        rows.append(
            {
                "group": group,
                "path": str(path),
                "check": f"all_true:{column}",
                "status": "pass" if bool(values.all()) else "fail",
                "details": "",
                "value": int(values.sum()),
                "threshold": len(values),
                "classification": "",
            }
        )
    return rows


def _missing_columns(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return sorted(set(columns) - set(frame.columns))


def _columns_row(group: str, path: Path, missing_columns: list[str]) -> dict[str, Any]:
    return {
        "group": group,
        "path": str(path),
        "check": "required_columns",
        "status": "pass" if not missing_columns else "fail",
        "details": "" if not missing_columns else "; ".join(missing_columns),
        "value": "",
        "threshold": "",
        "classification": "",
    }


def _file_row(group: str, relative_path: Path, path: Path) -> dict[str, Any]:
    return {
        "group": group,
        "path": str(relative_path),
        "check": "file_exists",
        "status": "pass" if path.exists() else "missing",
        "details": "" if path.exists() else "file missing",
        "value": "",
        "threshold": "",
        "classification": "",
    }


def _fail_row(group: str, path: Path, check: str, details: str) -> dict[str, Any]:
    return {
        "group": group,
        "path": str(path),
        "check": check,
        "status": "fail",
        "details": details,
        "value": "",
        "threshold": "",
        "classification": "",
    }


def _audit_verdict(rows: list[dict[str, Any]]) -> str:
    statuses = {str(row["status"]) for row in rows}
    if "fail" in statuses:
        return "FAIL"
    if "missing" in statuses or "warn" in statuses:
        return "PARTIAL PASS"
    return "PASS"


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    return counts


def _summary_markdown(verdict: str, rows: list[dict[str, Any]]) -> str:
    counts = _status_counts(rows)
    lines = "\n".join(f"- `{status}`: {count}" for status, count in sorted(counts.items()))
    return f"""# QSVT Engineering Extension Audit

Artifact audit verdict: **{verdict}**

This audit checks generated engineering-extension artifacts, manifests, selected
thresholds, finite numeric metrics, the expanded claim matrix, and conservative
claim wording in documentation. Missing files are reported as audit findings; the
script does not generate the missing science artifacts.

## Status Counts

{lines}

The audit itself does not demonstrate quantum speedup, quantum advantage, full
IEEE-scale hardware execution, field-data validation, or QSVT superiority over
Ridge/Tikhonov under the same alpha.
"""


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_engineering_extension_audit",
        "root_dir": ".",
        "docs_paths": [
            "docs/QSVT_ENGINEERING_EXTENSION.md",
            "docs/QSVT_EXTERNAL_PHASE_BACKENDS.md",
            "docs/QSVT_BLOCK_ENCODING_SCALABILITY.md",
            "docs/QSVT_APPROXIMATION_VALIDATION.md",
            "docs/QSVT_PHASE_RESPONSE_CONVENTIONS.md",
            "docs/QSVT_MULTICASE_DEGREE_SEARCH.md",
            "README.md",
        ],
        "artifact_specs": None,
    }
    if config:
        resolved.update(config)
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit QSVT engineering-extension outputs")
    parser.parse_args(argv)
    run = build_engineering_audit()
    print(f"QSVT engineering audit complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
