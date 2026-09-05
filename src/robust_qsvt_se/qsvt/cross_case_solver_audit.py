from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.subproblem_selection_policy import generate_candidate_subproblems
from robust_qsvt_se.qsvt.subproblem_sweep import _build_system
from robust_qsvt_se.utils.io import ensure_directory

CROSS_CASE_AUDIT_CLAIM = (
    "Cross-case readiness audit that inspects the IEEE14 selected-subproblem QSVT "
    "solver-prototype outputs and the shared pipeline before extending the study to "
    "IEEE30/IEEE57 selected 4x4 subproblems. It records which functions and selection modes "
    "are case-agnostic, which IEEE14 prototype outputs supply the cross-case defaults, and "
    "which assumptions and claim boundaries must be preserved. This remains a "
    "selected-subproblem study. Ridge/Tikhonov remains the reference filter; QSVT is an "
    "implementation pathway for the same regularized spectral filter. No full IEEE-scale "
    "QSVT solver, quantum speedup, quantum advantage, QSVT superiority over Ridge/Tikhonov, "
    "or hardware execution is claimed."
)

DEFAULT_CASES = ("ieee14", "ieee30", "ieee57")
CROSS_CASE_SELECTION_MODES = (
    "high_leverage",
    "metadata_mapped",
    "residual_supported",
    "best_conditioned",
    "random_seeded_pool",
    "worst_conditioned_control",
)

PRESERVED_ASSUMPTIONS = (
    "Readout of the full QSVT output direction is assumed; only selected observables are read.",
    "Block encoding uses B = H^T with beta = sigma_max(B); the known-C update is independent "
    "of the bounding constant C.",
    "The success amplitude is the normalized postselection probability of |r/||r||>.",
    "Subproblems are chosen by numerical/metadata criteria, never by post hoc QSVT performance.",
    "Each AC case is a single weighted linearized update, not a full iterative AC solve.",
)

PRESERVED_CLAIM_BOUNDARIES = (
    "cross-case selected-subproblem robustness (not a full IEEE-scale QSVT solver)",
    "Ridge/Tikhonov remains the reference; QSVT implements the same regularized filter",
    "no quantum speedup or quantum advantage",
    "no QSVT superiority over Ridge/Tikhonov",
    "no hardware execution or real PMU/SCADA validation",
)

REUSABLE_COMPONENT_COLUMNS = [
    "component",
    "module",
    "current_default_case",
    "supports_cross_case",
    "reuse_for_cross_case",
    "note",
]

CANDIDATE_CASE_COLUMNS = [
    "case",
    "model",
    "case_source",
    "available",
    "matrix_shape",
    "n_measurements",
    "n_states",
    "min_dim",
    "supports_submatrix_size",
    "candidate_modes_available",
    "role",
    "note",
]


def run_qsvt_cross_case_solver_audit(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "input_root": "outputs",
        "cases": list(DEFAULT_CASES),
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_size": 4,
        "selection_modes": list(CROSS_CASE_SELECTION_MODES),
        "seed": 123,
        "output_dir": "outputs/qsvt_cross_case_solver_audit",
    }
    resolved.update(config)
    input_root = Path(resolved["input_root"])
    output_dir = ensure_directory(resolved["output_dir"])

    components = reusable_pipeline_components()
    candidates = cross_case_candidate_cases(
        cases=[str(value) for value in resolved["cases"]],
        model=str(resolved["model"]),
        case_source=str(resolved["case_source"]),
        submatrix_size=int(resolved["submatrix_size"]),
        selection_modes=[str(value) for value in resolved["selection_modes"]],
        seed=int(resolved["seed"]),
    )
    defaults = ieee14_prototype_defaults(input_root)
    artifacts = write_cross_case_audit_outputs(
        output_dir, resolved, components, candidates, defaults
    )
    return {
        "output_dir": output_dir,
        "components": components,
        "candidates": candidates,
        "defaults": defaults,
        "artifacts": artifacts,
    }


def reusable_pipeline_components() -> pd.DataFrame:
    """Static map of the shared pipeline functions and whether they are case-agnostic.

    Cross-case support is determined by whether a function takes a ``case`` argument and
    routes through the pypower case loader (``_build_system`` -> ``load_ac_case_from_pypower``),
    which already supports IEEE14/30/57/118/300.
    """

    rows = [
        (
            "_build_system",
            "subproblem_sweep",
            "ieee14",
            True,
            "reuse_directly",
            "Routes any case through the pypower AC weighted-Jacobian loader.",
        ),
        (
            "generate_candidate_subproblems",
            "subproblem_selection_policy",
            "ieee14",
            True,
            "reuse_directly",
            "Criteria-based candidate selection takes case/model and works for any system.",
        ),
        (
            "build_selected_subproblem_from_policy_row",
            "subproblem_selection_policy",
            "ieee14",
            True,
            "reuse_directly",
            "Reconstructs a selected block from row/col indices for any case.",
        ),
        (
            "evaluate_subproblem_robustness",
            "codesigned_subproblem_robustness",
            "ieee14",
            True,
            "reuse_per_case",
            "Per-case loop wraps this single-case evaluator for IEEE14/30/57.",
        ),
        (
            "build_codesigned_solution",
            "codesigned_bounded_targets",
            "case_agnostic",
            True,
            "reuse_directly",
            "Builds the co-designed bounded target from a SelectedSubproblem only.",
        ),
        (
            "ridge_tikhonov_update",
            "gate_level_state_estimation_solver",
            "case_agnostic",
            True,
            "reuse_directly",
            "Reference Ridge filter; matrix-only, case-agnostic.",
        ),
        (
            "validate_robustness_gate_config",
            "robustness_gate_validation",
            "ieee14",
            True,
            "reuse_per_case",
            "Reconstructs the actual subproblem; cross-case driver supplies a per-case system.",
        ),
        (
            "_build_codesigned_gate_circuit",
            "codesigned_gate_validation",
            "case_agnostic",
            True,
            "reuse_directly",
            "Synthesizes phases and simulates the structured QSVT circuit from H, r, solution.",
        ),
        (
            "build_observable_protocols / evaluate_observable_protocol",
            "power_observable_protocols",
            "case_agnostic",
            True,
            "reuse_directly",
            "Observable protocols read functionals of the update for any subproblem metadata.",
        ),
        (
            "detect_overshoot / _off_support_peak",
            "overshoot_mechanism_diagnostic",
            "case_agnostic",
            True,
            "reuse_directly",
            "Off-support and singular-support diagnostics operate on the polynomial only.",
        ),
        (
            "extract_state_estimation_subproblem",
            "gate_level_state_estimation_solver",
            "ieee14",
            True,
            "reuse_per_case",
            "Extracts a case's high-leverage block; takes case/case_source arguments.",
        ),
    ]
    return pd.DataFrame(rows, columns=REUSABLE_COMPONENT_COLUMNS)


def cross_case_candidate_cases(
    *,
    cases: list[str],
    model: str,
    case_source: str,
    submatrix_size: int,
    selection_modes: list[str],
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in cases:
        role = "reference" if case == "ieee14" else "cross_case_test"
        record = {column: np.nan for column in CANDIDATE_CASE_COLUMNS}
        record.update({"case": case, "model": model, "case_source": case_source, "role": role})
        try:
            system, matrix_source = _build_system(
                case=case, model=model, case_source=case_source, seed=int(seed)
            )
            H = np.asarray(system.H_tilde, dtype=np.float64)
            min_dim = int(min(H.shape))
            candidates = generate_candidate_subproblems(
                system=system,
                matrix_source=matrix_source,
                case=case,
                model=model,
                submatrix_size=int(submatrix_size),
                candidate_modes=list(selection_modes),
                seed=int(seed),
            )
            modes = sorted({str(c.selection_source) for c in candidates})
            record.update(
                {
                    "available": True,
                    "matrix_shape": f"{H.shape[0]}x{H.shape[1]}",
                    "n_measurements": int(H.shape[0]),
                    "n_states": int(H.shape[1]),
                    "min_dim": min_dim,
                    "supports_submatrix_size": bool(min_dim >= int(submatrix_size)),
                    "candidate_modes_available": ";".join(modes),
                    "note": f"matrix_source={matrix_source}",
                }
            )
        except Exception as exc:  # pragma: no cover - depends on optional pypower data
            record.update(
                {
                    "available": False,
                    "supports_submatrix_size": False,
                    "candidate_modes_available": "",
                    "note": f"load_failed:{type(exc).__name__}",
                }
            )
        rows.append(record)
    return pd.DataFrame(rows, columns=CANDIDATE_CASE_COLUMNS)


def ieee14_prototype_defaults(input_root: Path) -> dict[str, Any]:
    """Read the IEEE14 prototype outputs that supply cross-case defaults."""

    degree_window = _read_csv(
        input_root / "qsvt_degree_window_overshoot" / "feasible_degree_window.csv"
    )
    feasible_degrees = (
        sorted({int(v) for v in pd.to_numeric(degree_window["degree"], errors="coerce").dropna()})
        if not degree_window.empty
        else []
    )
    feasible_families = (
        sorted(set(degree_window["target_family"].astype(str)))
        if not degree_window.empty and "target_family" in degree_window.columns
        else []
    )
    boundary = _read_csv(input_root / "qsvt_degree_window_overshoot" / "overshoot_boundary.csv")
    overshoot_onset = (
        int(pd.to_numeric(boundary["overshoot_onset_degree"], errors="coerce").dropna().min())
        if not boundary.empty and boundary["overshoot_onset_degree"].notna().any()
        else None
    )
    robustness_gate = _read_csv(
        input_root / "qsvt_robustness_gate_validation" / "robustness_gate_results.csv"
    )
    surviving_modes: list[str] = []
    if not robustness_gate.empty and "residual_feasible_after_gate" in robustness_gate.columns:
        mask = robustness_gate["residual_feasible_after_gate"].astype(str).str.lower().eq("true")
        key = "selection_mode" if "selection_mode" in robustness_gate.columns else "subproblem_id"
        surviving_modes = sorted(
            set(robustness_gate[mask].get(key, pd.Series(dtype=str)).astype(str))
        )
    prototype = _read_csv(
        input_root / "qsvt_selected_solver_prototype" / "solver_prototype_summary.csv"
    )
    ieee14_classification = "unknown"
    if not prototype.empty and {"metric", "value"}.issubset(prototype.columns):
        match = prototype[prototype["metric"].astype(str) == "classification"]
        if not match.empty:
            ieee14_classification = str(match.iloc[0]["value"])
    return {
        "feasible_degrees": feasible_degrees,
        "recommended_degrees": [15, 25, 35, 45, 47],
        "feasible_families": feasible_families or ["weighted_support_ls", "residual_aware"],
        "overshoot_onset_degree": overshoot_onset,
        "ieee14_surviving_modes": surviving_modes,
        "ieee14_prototype_classification": ieee14_classification,
        "recommended_alphas": [1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2],
    }


def write_cross_case_audit_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    components: pd.DataFrame,
    candidates: pd.DataFrame,
    defaults: dict[str, Any],
) -> dict[str, Path]:
    components_path = output_dir / "reusable_pipeline_components.csv"
    candidates_path = output_dir / "cross_case_candidate_cases.csv"
    audit_path = output_dir / "cross_case_audit.md"

    components.to_csv(components_path, index=False)
    candidates.to_csv(candidates_path, index=False)
    audit_path.write_text(_audit_markdown(components, candidates, defaults), encoding="utf-8")

    manifest = write_manifest(
        output_dir,
        artifacts={
            "reusable_pipeline_components": str(components_path),
            "cross_case_candidate_cases": str(candidates_path),
            "cross_case_audit": str(audit_path),
        },
        input_config=resolved,
        claim_boundary=CROSS_CASE_AUDIT_CLAIM,
    )
    return {
        "manifest": manifest,
        "reusable_pipeline_components": components_path,
        "cross_case_candidate_cases": candidates_path,
        "cross_case_audit": audit_path,
    }


def _audit_markdown(
    components: pd.DataFrame, candidates: pd.DataFrame, defaults: dict[str, Any]
) -> str:
    cross_case_components = sorted(
        set(components[components["supports_cross_case"].astype(bool)]["component"].astype(str))
    )
    available_cases = (
        sorted(set(candidates[candidates["available"].astype(bool)]["case"].astype(str)))
        if not candidates.empty
        else []
    )
    cross_case_ready = (
        sorted(
            set(
                candidates[
                    candidates["available"].astype(bool)
                    & candidates["supports_submatrix_size"].astype(bool)
                    & (candidates["role"].astype(str) == "cross_case_test")
                ]["case"].astype(str)
            )
        )
        if not candidates.empty
        else []
    )
    return "\n".join(
        [
            "# Cross-Case Readiness Audit",
            "",
            CROSS_CASE_AUDIT_CLAIM,
            "",
            "## Required Answers",
            f"1. Functions that support cases beyond IEEE14: {', '.join(cross_case_components)}.",
            "2. Selection modes reusable for IEEE30/IEEE57: "
            f"{', '.join(CROSS_CASE_SELECTION_MODES)} (criteria-based only; the "
            "worst-conditioned control never counts as positive evidence).",
            "3. IEEE14 prototype outputs used as cross-case defaults: feasible degree window "
            f"{defaults['feasible_degrees'] or 'none'} (recommended cross-case degrees "
            f"{defaults['recommended_degrees']}, families {defaults['feasible_families']}, "
            f"overshoot onset {defaults['overshoot_onset_degree']}); IEEE14 gate-surviving modes "
            f"{defaults['ieee14_surviving_modes'] or 'none'}.",
            "4. Assumptions that remain unchanged:",
            *[f"   - {item}" for item in PRESERVED_ASSUMPTIONS],
            "5. Claim boundaries that must be preserved:",
            *[f"   - {item}" for item in PRESERVED_CLAIM_BOUNDARIES],
            "",
            "## Case Availability",
            f"- Cases that load and support the selected submatrix size: {available_cases}.",
            f"- Cross-case test cases ready (IEEE30/IEEE57): {cross_case_ready or 'none'}.",
            "",
            "## Notes",
            "- The shared pipeline is case-agnostic because `_build_system` routes every case "
            "through the pypower AC weighted-Jacobian loader, which supports IEEE14/30/57/118/300.",
            "- Cross-case robustness reuses the single-case evaluators inside a per-case loop; no "
            "selection is done by post hoc QSVT performance.",
            "",
        ]
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
