from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.subproblem_selection_policy import generate_candidate_subproblems
from robust_qsvt_se.qsvt.subproblem_sweep import _build_system
from robust_qsvt_se.utils.io import ensure_directory

IEEE118_AUDIT_CLAIM = (
    "Readiness audit for extending the cross-case selected-subproblem QSVT solver prototype to "
    "IEEE118 selected 4x4 AC-linearized weighted subproblems. It confirms that IEEE118 loads "
    "through the same pypower AC pipeline, that criteria-selected 4x4 blocks can be extracted, "
    "which selection modes and target families/degrees to reuse, and which assumptions and "
    "claim boundaries must be preserved. This remains a selected-subproblem study, not a full "
    "IEEE118-scale QSVT solver. Ridge/Tikhonov remains the reference filter; QSVT is an "
    "implementation pathway for the same regularized spectral filter. No full IEEE-scale QSVT "
    "solver, quantum speedup, quantum advantage, QSVT superiority over Ridge/Tikhonov, or "
    "hardware execution is claimed."
)

DEFAULT_CASE = "ieee118"
IEEE118_SELECTION_MODES = (
    "high_leverage",
    "metadata_mapped",
    "residual_supported",
    "best_conditioned",
    "random_seeded_pool",
    "worst_conditioned_control",
)
REUSE_TARGET_FAMILIES = ("weighted_support_ls", "residual_aware")
REUSE_DEGREES = (15, 25, 35, 45, 47)
REUSE_ALPHAS = (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2)

PRESERVED_ASSUMPTIONS = (
    "Readout of the full QSVT output direction is assumed; only selected observables are read.",
    "Block encoding uses B = H^T with beta = sigma_max(B); the known-C update is independent "
    "of the bounding constant C.",
    "The success amplitude is the normalized postselection probability of |r/||r||>.",
    "Subproblems are chosen by numerical/metadata criteria, never by post hoc QSVT performance.",
    "Each AC case is a single weighted linearized update, not a full iterative AC solve.",
)

PRESERVED_CLAIM_BOUNDARIES = (
    "IEEE118 selected (4x4) subproblem extension (not a full IEEE118 QSVT solver)",
    "larger benchmark selected-subproblem test (not full IEEE-scale solving)",
    "Ridge/Tikhonov remains the reference; QSVT implements the same regularized filter",
    "no quantum speedup or quantum advantage",
    "no QSVT superiority over Ridge/Tikhonov",
    "no hardware execution or real PMU/SCADA validation",
)

REUSABLE_COMPONENT_COLUMNS = [
    "component",
    "module",
    "reuse_for_ieee118",
    "supports_ieee118",
    "note",
]

SELECTION_MODE_COLUMNS = [
    "case",
    "selection_mode",
    "criteria_based",
    "available",
    "n_candidates",
    "counts_as_positive_evidence",
    "note",
]


def run_qsvt_ieee118_extension_audit(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "input_root": "outputs",
        "case": DEFAULT_CASE,
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_size": 4,
        "selection_modes": list(IEEE118_SELECTION_MODES),
        "seed": 123,
        "output_dir": "outputs/qsvt_ieee118_extension_audit",
    }
    resolved.update(config)
    input_root = Path(resolved["input_root"])
    output_dir = ensure_directory(resolved["output_dir"])

    components = reusable_cross_case_components()
    selection_modes = ieee118_candidate_selection_modes(
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        case_source=str(resolved["case_source"]),
        submatrix_size=int(resolved["submatrix_size"]),
        selection_modes=[str(value) for value in resolved["selection_modes"]],
        seed=int(resolved["seed"]),
    )
    status = ieee118_load_status(selection_modes)
    artifacts = write_ieee118_audit_outputs(
        output_dir, resolved, components, selection_modes, status, input_root
    )
    return {
        "output_dir": output_dir,
        "components": components,
        "selection_modes": selection_modes,
        "status": status,
        "artifacts": artifacts,
    }


def reusable_cross_case_components() -> pd.DataFrame:
    """Cross-case pipeline functions reused unchanged for the IEEE118 extension.

    Every component is case-agnostic because ``_build_system`` routes IEEE118 through the same
    pypower AC weighted-Jacobian loader as IEEE14/30/57, so the cross-case modules accept
    ``case="ieee118"`` directly.
    """

    rows = [
        (
            "_build_system",
            "subproblem_sweep",
            "reuse_directly",
            True,
            "Routes IEEE118 through the pypower AC weighted-Jacobian loader.",
        ),
        (
            "generate_candidate_subproblems / build_selected_subproblem_from_policy_row",
            "subproblem_selection_policy",
            "reuse_directly",
            True,
            "Criteria-based 4x4 selection and reconstruction work for any case.",
        ),
        (
            "evaluate_case_robustness",
            "cross_case_codesigned_robustness",
            "reuse_per_case",
            True,
            "Single-case evaluator already parameterized by case; call with ieee118.",
        ),
        (
            "build_codesigned_solution",
            "codesigned_bounded_targets",
            "reuse_directly",
            True,
            "Builds the co-designed bounded target from a SelectedSubproblem only.",
        ),
        (
            "validate_robustness_gate_config",
            "robustness_gate_validation",
            "reuse_per_case",
            True,
            "Reconstructs the subproblem and runs the dense gate circuit for any case system.",
        ),
        (
            "evaluate_cross_case_observable_readout",
            "cross_case_gate_observable_readout",
            "reuse_directly",
            True,
            "Builds per-case systems from gate-result rows; ieee118 rows are handled identically.",
        ),
        (
            "compute_singular_support",
            "weighted_singular_support",
            "reuse_directly",
            True,
            "Per-direction singular-support weights from the SVD of any selected block.",
        ),
        (
            "classify_overshoot_failure_mode / detect_off_support_peak",
            "overshoot_mechanism_diagnostic",
            "reuse_directly",
            True,
            "Operate on the polynomial and SVD only; reused by the direction-resolved phase.",
        ),
    ]
    return pd.DataFrame(rows, columns=REUSABLE_COMPONENT_COLUMNS)


def ieee118_candidate_selection_modes(
    *,
    case: str,
    model: str,
    case_source: str,
    submatrix_size: int,
    selection_modes: list[str],
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    try:
        system, matrix_source = _build_system(
            case=case, model=model, case_source=case_source, seed=int(seed)
        )
        candidates = generate_candidate_subproblems(
            system=system,
            matrix_source=matrix_source,
            case=case,
            model=model,
            submatrix_size=int(submatrix_size),
            candidate_modes=list(selection_modes),
            seed=int(seed),
        )
        counts: dict[str, int] = {}
        for candidate in candidates:
            mode = str(candidate.selection_source)
            counts[mode] = counts.get(mode, 0) + 1
    except Exception as exc:  # pragma: no cover - depends on optional pypower data
        for mode in selection_modes:
            rows.append(
                {
                    "case": case,
                    "selection_mode": mode,
                    "criteria_based": True,
                    "available": False,
                    "n_candidates": 0,
                    "counts_as_positive_evidence": mode != "worst_conditioned_control",
                    "note": f"load_failed:{type(exc).__name__}",
                }
            )
        return pd.DataFrame(rows, columns=SELECTION_MODE_COLUMNS)

    for mode in selection_modes:
        count = counts.get(mode, 0)
        rows.append(
            {
                "case": case,
                "selection_mode": mode,
                "criteria_based": True,
                "available": bool(count > 0),
                "n_candidates": int(count),
                "counts_as_positive_evidence": mode != "worst_conditioned_control",
                "note": (
                    "control: never counts as positive evidence"
                    if mode == "worst_conditioned_control"
                    else f"matrix_source={matrix_source}"
                ),
            }
        )
    return pd.DataFrame(rows, columns=SELECTION_MODE_COLUMNS)


def ieee118_load_status(selection_modes: pd.DataFrame) -> dict[str, Any]:
    """Summarize whether IEEE118 loads and supports selected 4x4 extraction."""

    if selection_modes.empty:
        return {
            "loads": False,
            "supports_4x4_extraction": False,
            "available_modes": [],
            "positive_evidence_modes": [],
        }
    available = selection_modes[selection_modes["available"].astype(bool)]
    positive = available[available["counts_as_positive_evidence"].astype(bool)]
    loads = bool(not available.empty)
    return {
        "loads": loads,
        "supports_4x4_extraction": loads,
        "available_modes": sorted(set(available["selection_mode"].astype(str))),
        "positive_evidence_modes": sorted(set(positive["selection_mode"].astype(str))),
    }


def write_ieee118_audit_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    components: pd.DataFrame,
    selection_modes: pd.DataFrame,
    status: dict[str, Any],
    input_root: Path,
) -> dict[str, Path]:
    components_path = output_dir / "reusable_cross_case_components.csv"
    selection_path = output_dir / "ieee118_candidate_selection_modes.csv"
    audit_path = output_dir / "ieee118_extension_audit.md"

    components.to_csv(components_path, index=False)
    selection_modes.to_csv(selection_path, index=False)
    audit_path.write_text(
        _audit_markdown(components, selection_modes, status, input_root), encoding="utf-8"
    )

    manifest = write_manifest(
        output_dir,
        artifacts={
            "reusable_cross_case_components": str(components_path),
            "ieee118_candidate_selection_modes": str(selection_path),
            "ieee118_extension_audit": str(audit_path),
        },
        input_config=resolved,
        claim_boundary=IEEE118_AUDIT_CLAIM,
    )
    return {
        "manifest": manifest,
        "reusable_cross_case_components": components_path,
        "ieee118_candidate_selection_modes": selection_path,
        "ieee118_extension_audit": audit_path,
    }


def _audit_markdown(
    components: pd.DataFrame,
    selection_modes: pd.DataFrame,
    status: dict[str, Any],
    input_root: Path,
) -> str:
    cross_case_present = (input_root / "qsvt_cross_case_solver_prototype").exists()
    reusable = sorted(
        set(components[components["supports_ieee118"].astype(bool)]["component"].astype(str))
    )
    return "\n".join(
        [
            "# IEEE118 Selected-Subproblem Extension Audit",
            "",
            IEEE118_AUDIT_CLAIM,
            "",
            "## Required Answers",
            f"1. Does IEEE118 load through the same AC/pypower pipeline? "
            f"{'yes' if status['loads'] else 'no'} "
            "(`_build_system` routes IEEE118 through the pypower AC weighted-Jacobian loader, "
            "which supports IEEE14/30/57/118/300).",
            f"2. Can selected 4x4 subproblems be extracted? "
            f"{'yes' if status['supports_4x4_extraction'] else 'no'} (available "
            f"criteria-selected modes: {', '.join(status['available_modes']) or 'none'}).",
            "3. Which criteria-based selection modes are valid for IEEE118? "
            f"{', '.join(status['positive_evidence_modes']) or 'none'} count as positive evidence; "
            "the worst-conditioned control never counts as positive evidence.",
            "4. Which target families and degree window should be reused? "
            f"families {list(REUSE_TARGET_FAMILIES)}, degrees {list(REUSE_DEGREES)} "
            f"(degree 47 as the boundary/failure test), alphas {list(REUSE_ALPHAS)}.",
            "5. Claim boundaries that must be preserved:",
            *[f"   - {item}" for item in PRESERVED_CLAIM_BOUNDARIES],
            "",
            "## Preserved Assumptions",
            *[f"- {item}" for item in PRESERVED_ASSUMPTIONS],
            "",
            "## Reusable Cross-Case Components",
            *[f"- {item}" for item in reusable],
            "",
            "## Notes",
            "- The cross-case IEEE14/30/57 selected-subproblem prototype is the immediate "
            f"predecessor ({'present' if cross_case_present else 'not found'} under "
            "outputs/qsvt_cross_case_solver_prototype).",
            "- IEEE118 is a strictly larger benchmark; this audit only establishes that the same "
            "selected-subproblem pipeline applies, not that a full IEEE118-scale solver exists.",
            "",
        ]
    )
