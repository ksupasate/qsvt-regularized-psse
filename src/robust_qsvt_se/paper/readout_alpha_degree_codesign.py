"""Readout-aware alpha/degree co-design and two-view reporting.

The selected-subproblem gate-level full-vector readout pipeline reconstructs the Ridge/
Tikhonov update from an actual QSP circuit. Under a *common* matched ``alpha=1e-4`` benchmark,
some subproblems show a large gate-vs-Ridge vector error whose dominant source is the
finite-degree polynomial approximation of the steep regularized target
(``exact_target_vs_polynomial``), **not** the readout protocol.

This module preserves the matched-alpha diagnostic view and adds a per-subproblem
QSVT-validated view obtained from a targeted alpha/degree sweep. The validated view selects,
per subproblem, the bounded in-window configuration that minimizes gate-vs-Ridge error.
Where co-design reduces the error it strengthens the selected-subproblem readout claim; where
the subproblem's singular-value structure keeps the finite-degree polynomial limiting, the
case is reported honestly as ``qsvt_polynomial_limited``.

It does **not** establish efficient full IEEE-scale full-vector readout, quantum speedup, or
QSVT numerical superiority over Ridge/Tikhonov.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.paper.full_vector_readout import (
    FULL_VECTOR_READOUT_CLAIM,
    _global_sign_aligned,
    _protocol_global_sign,
    _rel_l2,
    _residual_ratio,
    _stable_base_seed,
    _top_k,
    _trial_seed,
    discover_validated_subproblems,
    gate_level_subproblem_readout,
    qsvt_target_readout,
    reconstruct_subproblem,
    recover_relative_signs,
    sample_basis_magnitudes,
)
from robust_qsvt_se.paper.full_vector_readout import (
    _SystemCache as SystemCache,
)
from robust_qsvt_se.utils.io import ensure_directory

CODESIGN_CLAIM = (
    "Readout-aware alpha/degree co-design strengthens selected-subproblem gate-level "
    "full-vector readout by reducing finite-degree QSVT polynomial approximation error where "
    "the subproblem singular-value structure permits. Matched-alpha high-error cases are "
    "diagnosed, not hidden. This does not demonstrate efficient full IEEE-scale full-vector "
    "readout, quantum speedup, or QSVT numerical superiority over Ridge/Tikhonov."
)

MATCHED_ALPHA = 1.0e-4
DEFAULT_ALPHA_GRID: tuple[float, ...] = (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2)
DEFAULT_DEGREE_GRID: tuple[int, ...] = (15, 21, 25, 29, 35, 41, 45)
DEFAULT_OVERSHOOT_DEGREES: tuple[int, ...] = (47, 49)
DEFAULT_SWEEP_SHOTS = 100_000
DEFAULT_SWEEP_TRIALS = 10
DEFAULT_SEED = 123

# gate_error_vs_ridge thresholds (fractions of ||Delta x_ridge||).
PASS_THRESHOLD = 0.05
WARNING_THRESHOLD = 0.20
GOOD_GATE_ERROR = 0.10
STRONG_IMPROVEMENT_FACTOR = 10.0
_MAX_DEGREE_FOR_PENALTY = 45.0
_INELIGIBLE_SCORE_PENALTY = 100.0

MATCHED_VIEW = "matched_alpha_common_reference"
VALIDATED_VIEW = "per_subproblem_qsvt_validated"

# --------------------------------------------------------------------------- column schemas
CLASSIFICATION_COLUMNS = [
    "case",
    "subproblem_id",
    "subproblem_type",
    "alpha",
    "degree",
    "gate_error_vs_ridge",
    "exact_target_vs_polynomial_error",
    "polynomial_vs_gate_error",
    "statevector_vs_shot_reconstruction_error",
    "residual_ratio_after_gate",
    "top_k_match",
    "dominant_error_source",
    "diagnostic_status",
    "recommended_action",
    "source_artifact",
    "notes",
]

MATCHED_ALPHA_VIEW_COLUMNS = [
    "case",
    "subproblem_id",
    "subproblem_type",
    "view",
    "alpha",
    "degree",
    "gate_error_vs_ridge",
    "exact_target_error_vs_ridge",
    "finite_polynomial_error",
    "gate_vs_polynomial_error",
    "shot_readout_error",
    "residual_ratio_after_gate",
    "residual_ratio_after_shot_reconstruction",
    "dominant_error_source",
    "diagnostic_status",
    "interpretation",
    "source_artifact",
    "notes",
]

SWEEP_COLUMNS = [
    "case",
    "subproblem_id",
    "subproblem_type",
    "alpha",
    "degree",
    "target_scale_C",
    "boundedness_ok",
    "overshoot_flag",
    "overshoot_diagnostic_only",
    "phase_synthesis_status",
    "gate_run_status",
    "exact_target_error_vs_ridge",
    "polynomial_error_vs_exact_target",
    "gate_error_vs_polynomial",
    "gate_error_vs_ridge",
    "shot_reconstruction_error_vs_gate",
    "shot_reconstruction_error_vs_ridge",
    "residual_ratio_after_gate",
    "residual_ratio_after_shot_reconstruction",
    "sign_reliable_fraction",
    "top_k_match",
    "estimated_total_shots",
    "selection_score",
    "eligible_for_best_config",
    "status",
    "notes",
]

BEST_CONFIG_COLUMNS = [
    "case",
    "subproblem_id",
    "subproblem_type",
    "matched_alpha",
    "matched_degree",
    "matched_alpha_error",
    "best_alpha",
    "best_degree",
    "best_gate_error_vs_ridge",
    "best_polynomial_error_vs_exact_target",
    "best_gate_vs_polynomial_error",
    "best_shot_reconstruction_error_vs_ridge",
    "best_residual_ratio",
    "best_sign_reliable_fraction",
    "best_top_k_match",
    "improvement_factor",
    "selection_rule",
    "selected_for_main_paper",
    "status",
    "notes",
]

VALIDATED_VIEW_COLUMNS = [
    "case",
    "subproblem_id",
    "subproblem_type",
    "view",
    "alpha",
    "degree",
    "selection_rule",
    "gate_error_vs_ridge",
    "exact_target_error_vs_ridge",
    "finite_polynomial_error",
    "gate_vs_polynomial_error",
    "shot_readout_error",
    "residual_ratio_after_gate",
    "residual_ratio_after_shot_reconstruction",
    "sign_reliable_fraction",
    "top_k_match",
    "status",
    "interpretation",
    "source_sweep_artifact",
    "notes",
]

TWO_VIEW_SUMMARY_COLUMNS = [
    "case",
    "subproblem_id",
    "subproblem_type",
    "matched_alpha",
    "matched_degree",
    "matched_alpha_gate_error",
    "validated_alpha",
    "validated_degree",
    "validated_gate_error",
    "improvement_factor",
    "matched_dominant_error_source",
    "validated_dominant_error_source",
    "final_status",
    "manuscript_interpretation",
    "notes",
]

FIGURE_TWO_VIEW_COLUMNS = [
    "case",
    "subproblem_type",
    "view",
    "alpha",
    "degree",
    "vector_relative_error_vs_ridge",
    "status",
    "plot_group",
    "notes",
]

FIGURE_DECOMP_COLUMNS = [
    "case",
    "subproblem_type",
    "alpha",
    "degree",
    "error_component",
    "error_value",
    "view",
    "dominant_error_source",
    "notes",
]

_ERROR_COMPONENTS = (
    "ridge_vs_exact_target",
    "exact_target_vs_polynomial",
    "polynomial_vs_gate",
    "gate_vs_shot_reconstruction",
    "shot_reconstruction_vs_ridge",
)


# --------------------------------------------------------------------------- io helpers
def _read_csv(path: Path) -> pd.DataFrame:
    if not Path(path).is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame()


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def _write_table(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> Path:
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- Phase 1
def _diagnostic_status(gate_error: float) -> str:
    if not np.isfinite(gate_error):
        return "missing_gate"
    if gate_error <= PASS_THRESHOLD:
        return "pass"
    if gate_error <= WARNING_THRESHOLD:
        return "warning"
    return "fail_diagnostic"


def _recommended_action(diagnostic_status: str, dominant: str) -> str:
    if diagnostic_status == "pass":
        return "none"
    if diagnostic_status == "missing_gate":
        return "rerun_gate_level_readout"
    if dominant == "exact_target_vs_polynomial":
        return "alpha_degree_codesign_sweep"
    if dominant == "polynomial_vs_gate":
        return "gate_synthesis_review"
    if dominant == "statevector_vs_shot_reconstruction":
        return "increase_shot_budget"
    return "alpha_degree_codesign_sweep"


def classify_readout_configs(readout_dir: Path) -> list[dict[str, Any]]:
    """Classify existing matched-alpha gate-level configs into pass/warning/fail diagnostics."""

    readout_dir = Path(readout_dir)
    gate = _read_csv(readout_dir / "gate_level_full_vector_summary.csv")
    decomposition = _read_csv(readout_dir / "readout_error_decomposition.csv")
    signed = _read_csv(readout_dir / "signed_vector_reconstruction_summary.csv")
    if gate.empty:
        return []

    decomp_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not decomposition.empty:
        for _, row in decomposition.iterrows():
            key = (str(row["case"]), str(row["subproblem_id"]), str(row["subproblem_type"]))
            decomp_by_key[key] = row.to_dict()

    top_k_by_key: dict[tuple[str, str, str], float] = {}
    if not signed.empty and "top_k_match" in signed.columns:
        best_shots = int(signed["shots"].max())
        for _, row in signed[signed["shots"] == best_shots].iterrows():
            key = (str(row["case"]), str(row["subproblem_id"]), str(row["subproblem_type"]))
            top_k_by_key[key] = _safe_float(row.get("top_k_match"))

    rows: list[dict[str, Any]] = []
    for _, summary in gate.iterrows():
        key = (
            str(summary["case"]),
            str(summary["subproblem_id"]),
            str(summary["subproblem_type"]),
        )
        gate_available = str(summary.get("gate_available", "")).strip().lower() in {"true", "1"}
        gate_error = _safe_float(summary.get("vector_relative_l2_error_vs_ridge"))
        if not gate_available:
            gate_error = float("nan")
        decomp = decomp_by_key.get(key, {})
        dominant = str(decomp.get("dominant_error_source", "")) if decomp else ""
        diagnostic_status = _diagnostic_status(gate_error)
        rows.append(
            {
                "case": key[0],
                "subproblem_id": key[1],
                "subproblem_type": key[2],
                "alpha": _safe_float(summary.get("alpha")),
                "degree": int(_safe_float(summary.get("degree"))),
                "gate_error_vs_ridge": gate_error,
                "exact_target_vs_polynomial_error": _safe_float(
                    decomp.get("exact_target_vs_polynomial_error")
                ),
                "polynomial_vs_gate_error": _safe_float(decomp.get("polynomial_vs_gate_error")),
                "statevector_vs_shot_reconstruction_error": _safe_float(
                    decomp.get("statevector_vs_shot_reconstruction_error")
                ),
                "residual_ratio_after_gate": _safe_float(
                    summary.get("residual_ratio_after_reconstruction")
                ),
                "top_k_match": top_k_by_key.get(key, float("nan")),
                "dominant_error_source": dominant,
                "diagnostic_status": diagnostic_status,
                "recommended_action": _recommended_action(diagnostic_status, dominant),
                "source_artifact": "full_vector_readout/gate_level_full_vector_summary.csv",
                "notes": (
                    "matched-alpha diagnostic; preserved, not overwritten"
                    if diagnostic_status == "pass"
                    else "matched-alpha high-error candidate for alpha/degree co-design"
                ),
            }
        )
    return rows


# --------------------------------------------------------------------------- Phase 2
def _matched_interpretation(gate_error: float, dominant: str) -> str:
    if not np.isfinite(gate_error):
        return "matched_alpha_warning"
    if gate_error <= PASS_THRESHOLD:
        return "matched_alpha_pass"
    if dominant == "exact_target_vs_polynomial":
        return "finite_degree_limited_common_alpha"
    if dominant == "polynomial_vs_gate":
        return "gate_implementation_limited"
    if dominant == "statevector_vs_shot_reconstruction":
        return "readout_limited"
    return "matched_alpha_warning"


_FINITE_DEGREE_NOTE = (
    "The matched-alpha view is a diagnostic common-reference setting. Large error indicates "
    "finite-degree approximation mismatch under the common alpha, not readout protocol failure."
)


def build_matched_alpha_view(readout_dir: Path) -> list[dict[str, Any]]:
    """Re-present the existing matched-alpha gate-level results as the diagnostic view."""

    readout_dir = Path(readout_dir)
    gate = _read_csv(readout_dir / "gate_level_full_vector_summary.csv")
    decomposition = _read_csv(readout_dir / "readout_error_decomposition.csv")
    signed = _read_csv(readout_dir / "signed_vector_reconstruction_summary.csv")
    classification = {
        (r["case"], r["subproblem_id"], r["subproblem_type"]): r
        for r in classify_readout_configs(readout_dir)
    }
    if gate.empty:
        return []

    decomp_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not decomposition.empty:
        for _, row in decomposition.iterrows():
            decomp_by_key[
                (str(row["case"]), str(row["subproblem_id"]), str(row["subproblem_type"]))
            ] = row.to_dict()

    signed_residual: dict[tuple[str, str, str], float] = {}
    if not signed.empty and "residual_ratio_after_reconstruction" in signed.columns:
        best_shots = int(signed["shots"].max())
        for _, row in signed[signed["shots"] == best_shots].iterrows():
            signed_residual[
                (str(row["case"]), str(row["subproblem_id"]), str(row["subproblem_type"]))
            ] = _safe_float(row.get("residual_ratio_after_reconstruction"))

    rows: list[dict[str, Any]] = []
    for _, summary in gate.iterrows():
        key = (
            str(summary["case"]),
            str(summary["subproblem_id"]),
            str(summary["subproblem_type"]),
        )
        decomp = decomp_by_key.get(key, {})
        gate_available = str(summary.get("gate_available", "")).strip().lower() in {"true", "1"}
        gate_error = (
            _safe_float(summary.get("vector_relative_l2_error_vs_ridge"))
            if gate_available
            else float("nan")
        )
        dominant = str(decomp.get("dominant_error_source", "")) if decomp else ""
        diagnostic = classification.get(key, {}).get("diagnostic_status", "")
        interpretation = _matched_interpretation(gate_error, dominant)
        note = (
            _FINITE_DEGREE_NOTE
            if interpretation == "finite_degree_limited_common_alpha"
            else "matched-alpha common-reference diagnostic (preserved verbatim)"
        )
        rows.append(
            {
                "case": key[0],
                "subproblem_id": key[1],
                "subproblem_type": key[2],
                "view": MATCHED_VIEW,
                "alpha": _safe_float(summary.get("alpha")),
                "degree": int(_safe_float(summary.get("degree"))),
                "gate_error_vs_ridge": gate_error,
                "exact_target_error_vs_ridge": _safe_float(
                    decomp.get("ridge_vs_exact_target_error")
                ),
                "finite_polynomial_error": _safe_float(
                    decomp.get("exact_target_vs_polynomial_error")
                ),
                "gate_vs_polynomial_error": _safe_float(decomp.get("polynomial_vs_gate_error")),
                "shot_readout_error": _safe_float(
                    decomp.get("statevector_vs_shot_reconstruction_error")
                ),
                "residual_ratio_after_gate": _safe_float(
                    summary.get("residual_ratio_after_reconstruction")
                ),
                "residual_ratio_after_shot_reconstruction": signed_residual.get(key, float("nan")),
                "dominant_error_source": dominant,
                "diagnostic_status": diagnostic,
                "interpretation": interpretation,
                "source_artifact": "full_vector_readout/gate_level_full_vector_summary.csv",
                "notes": note,
            }
        )
    return rows


# --------------------------------------------------------------------------- Phase 3
def _estimated_total_shots(dimension: int, success_probability: float, target_error: float) -> int:
    inv_eps_sq = int(np.ceil(1.0 / float(target_error) ** 2))
    overhead = (
        int(np.ceil(1.0 / success_probability))
        if np.isfinite(success_probability) and success_probability > 0.0
        else 1
    )
    basis = dimension * inv_eps_sq
    sign = max(dimension - 1, 1) * inv_eps_sq
    norm = inv_eps_sq
    return int((basis + sign + norm) * overhead)


def _gate_shot_reconstruction(
    sample_state: np.ndarray,
    recovered_norm: float,
    ridge: np.ndarray,
    *,
    shots: int,
    trials: int,
    base_seed: int,
) -> tuple[np.ndarray, float]:
    """Mean signed reconstruction of the actual gate state from finite-shot sampling."""

    reference_index = int(np.argmax(np.abs(sample_state)))
    accumulated = np.zeros_like(sample_state, dtype=np.float64)
    reliable_fraction = float("nan")
    for trial in range(int(trials)):
        seed = _trial_seed(base_seed, int(shots), trial)
        rng = np.random.default_rng(seed)
        sampling = sample_basis_magnitudes(sample_state, shots=int(shots), rng=rng)
        sign = recover_relative_signs(
            sample_state, shots=int(shots), rng=rng, reference_index=reference_index
        )
        recon = _protocol_global_sign(
            sign["estimated_signs"] * recovered_norm * sampling["estimated_magnitude"],
            reference_index,
        )
        accumulated += _global_sign_aligned(recon, ridge)
        reliable_fraction = float(
            np.mean([status == "reliable" for status in sign["reliability_status"]])
        )
    return accumulated / float(trials), reliable_fraction


def _failed_sweep_row(
    context: dict[str, Any],
    *,
    overshoot_diagnostic_only: bool,
    boundedness_ok: bool,
    target_scale_C: float,
    phase_status: str,
    gate_status: str,
    status: str,
    notes: str,
) -> dict[str, Any]:
    overshoot_flag = bool(overshoot_diagnostic_only or not boundedness_ok)
    return {
        **context,
        "target_scale_C": target_scale_C,
        "boundedness_ok": boundedness_ok,
        "overshoot_flag": overshoot_flag,
        "overshoot_diagnostic_only": bool(overshoot_diagnostic_only),
        "phase_synthesis_status": phase_status,
        "gate_run_status": gate_status,
        "exact_target_error_vs_ridge": float("nan"),
        "polynomial_error_vs_exact_target": float("nan"),
        "gate_error_vs_polynomial": float("nan"),
        "gate_error_vs_ridge": float("nan"),
        "shot_reconstruction_error_vs_gate": float("nan"),
        "shot_reconstruction_error_vs_ridge": float("nan"),
        "residual_ratio_after_gate": float("nan"),
        "residual_ratio_after_shot_reconstruction": float("nan"),
        "sign_reliable_fraction": float("nan"),
        "top_k_match": float("nan"),
        "estimated_total_shots": -1,
        "selection_score": float("inf"),
        "eligible_for_best_config": "no",
        "status": status,
        "notes": notes,
    }


def evaluate_sweep_cell(
    entry: dict[str, Any],
    H: np.ndarray,
    r: np.ndarray,
    *,
    alpha: float,
    degree: int,
    seed: int,
    shots: int,
    trials: int,
    target_error: float,
    overshoot_diagnostic_only: bool = False,
) -> dict[str, Any]:
    """Evaluate one (alpha, degree) cell: exact target, polynomial, gate, shot reconstruction."""

    context = {
        "case": entry["case"],
        "subproblem_id": entry["subproblem_id"],
        "subproblem_type": entry["subproblem_type"],
        "alpha": float(alpha),
        "degree": int(degree),
    }
    try:
        state = qsvt_target_readout(H, r, alpha=float(alpha), degree=int(degree))
    except Exception as exc:  # pragma: no cover - guarded degenerate fits
        return _failed_sweep_row(
            context,
            overshoot_diagnostic_only=overshoot_diagnostic_only,
            boundedness_ok=False,
            target_scale_C=float("nan"),
            phase_status="exact_target_failed",
            gate_status="not_run",
            status="exact_target_failed",
            notes=f"{type(exc).__name__}: {exc}",
        )

    boundedness_ok = bool(state.bounded_target_ok)
    target_scale_C = float(state.scale_factor_C)
    ridge = state.ridge_update
    exact_target = state.readout_state * state.recovered_norm
    exact_target_error_vs_ridge = _rel_l2(exact_target, ridge)
    polynomial_available = bool(state.polynomial_available)
    polynomial_error_vs_exact_target = (
        _rel_l2(state.polynomial_update, exact_target) if polynomial_available else float("nan")
    )

    gate = gate_level_subproblem_readout(
        H, r, alpha=float(alpha), degree=int(degree), seed=int(seed), case=entry["case"]
    )
    overshoot_flag = bool(overshoot_diagnostic_only or not boundedness_ok)
    if not gate.get("available"):
        row = _failed_sweep_row(
            context,
            overshoot_diagnostic_only=overshoot_diagnostic_only,
            boundedness_ok=boundedness_ok,
            target_scale_C=target_scale_C,
            phase_status="synthesis_failed",
            gate_status="failed",
            status="missing_gate_config",
            notes=gate.get("reason", "gate synthesis/simulation unavailable"),
        )
        row["exact_target_error_vs_ridge"] = exact_target_error_vs_ridge
        row["polynomial_error_vs_exact_target"] = polynomial_error_vs_exact_target
        return row

    gate_update = gate["gate_update"]
    gate_state = gate["gate_state"]
    gate_norm = float(np.linalg.norm(gate_update))
    gate_error_vs_polynomial = (
        _rel_l2(gate_update, state.polynomial_update) if polynomial_available else float("nan")
    )
    gate_error_vs_ridge = _rel_l2(gate_update, ridge)
    residual_ratio_after_gate = _residual_ratio(H, gate_update, r)

    base_seed = _stable_base_seed(
        (
            entry["case"],
            entry["subproblem_id"],
            entry["subproblem_type"],
            f"alpha={alpha:.3e}",
            f"degree={degree}",
        ),
        seed,
    )
    mean_recon, sign_reliable_fraction = _gate_shot_reconstruction(
        gate_state, gate_norm, ridge, shots=int(shots), trials=int(trials), base_seed=base_seed
    )
    shot_reconstruction_error_vs_gate = _rel_l2(mean_recon, gate_update)
    shot_reconstruction_error_vs_ridge = _rel_l2(mean_recon, ridge)
    residual_ratio_after_shot = _residual_ratio(H, mean_recon, r)
    k = max(1, ridge.size // 2)
    top_k_match = len(_top_k(np.abs(mean_recon), k) & _top_k(np.abs(ridge), k)) / float(k)

    overshoot_penalty = 1 if overshoot_flag else 0
    invalid_boundedness_penalty = 0 if boundedness_ok else 1
    eligible = bool(boundedness_ok and not overshoot_flag)
    selection_score = (
        gate_error_vs_ridge
        + 0.1 * float(degree) / _MAX_DEGREE_FOR_PENALTY
        + 10.0 * overshoot_penalty
        + 10.0 * invalid_boundedness_penalty
    )
    estimated_total_shots = _estimated_total_shots(
        int(ridge.size), float(gate["success_probability"]), float(target_error)
    )
    return {
        **context,
        "target_scale_C": target_scale_C,
        "boundedness_ok": boundedness_ok,
        "overshoot_flag": overshoot_flag,
        "overshoot_diagnostic_only": bool(overshoot_diagnostic_only),
        "phase_synthesis_status": "synthesized",
        "gate_run_status": "computed",
        "exact_target_error_vs_ridge": exact_target_error_vs_ridge,
        "polynomial_error_vs_exact_target": polynomial_error_vs_exact_target,
        "gate_error_vs_polynomial": gate_error_vs_polynomial,
        "gate_error_vs_ridge": gate_error_vs_ridge,
        "shot_reconstruction_error_vs_gate": shot_reconstruction_error_vs_gate,
        "shot_reconstruction_error_vs_ridge": shot_reconstruction_error_vs_ridge,
        "residual_ratio_after_gate": residual_ratio_after_gate,
        "residual_ratio_after_shot_reconstruction": residual_ratio_after_shot,
        "sign_reliable_fraction": sign_reliable_fraction,
        "top_k_match": top_k_match,
        "estimated_total_shots": estimated_total_shots,
        "selection_score": float(selection_score),
        "eligible_for_best_config": "yes" if eligible else "no",
        "status": "computed",
        "notes": (
            "synthesized_degree="
            f"{gate['synthesized_degree']}; gate tracks the finite-degree polynomial"
        ),
    }


def select_targeted_configs(
    classification_rows: list[dict[str, Any]],
    *,
    mandatory: tuple[tuple[str, str], ...] = (("ieee14", "metadata_mapped"),),
    top_k_high_error: int = 3,
) -> list[tuple[str, str]]:
    """Pick targeted (case, subproblem_type) pairs: mandatory plus the top-k highest-error."""

    ranked = sorted(
        (r for r in classification_rows if np.isfinite(_safe_float(r["gate_error_vs_ridge"]))),
        key=lambda r: _safe_float(r["gate_error_vs_ridge"]),
        reverse=True,
    )
    available = {(r["case"], r["subproblem_type"]) for r in classification_rows}
    selected: list[tuple[str, str]] = []
    for row in ranked[: max(0, int(top_k_high_error))]:
        pair = (row["case"], row["subproblem_type"])
        if pair not in selected:
            selected.append(pair)
    for pair in mandatory:
        if pair in available and pair not in selected:
            selected.append(pair)
    return selected


# --------------------------------------------------------------------------- Phase 4
def select_best_configs(
    sweep_rows: list[dict[str, Any]],
    matched_view_rows: list[dict[str, Any]],
    targeted: list[tuple[str, str]],
    *,
    available_keys: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Select the minimum-selection-score eligible config per targeted subproblem."""

    matched_by_pair = {(row["case"], row["subproblem_type"]): row for row in matched_view_rows}
    sweep_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in sweep_rows:
        sweep_by_pair.setdefault((row["case"], row["subproblem_type"]), []).append(row)

    rows: list[dict[str, Any]] = []
    for pair in targeted:
        matched = matched_by_pair.get(pair, {})
        subproblem_id = matched.get("subproblem_id", "")
        matched_alpha_error = _safe_float(matched.get("gate_error_vs_ridge", float("nan")))
        matched_degree = int(_safe_float(matched.get("degree", float("nan")))) if matched else -1
        candidates = sweep_by_pair.get(pair, [])
        if subproblem_id == "" and candidates:
            subproblem_id = candidates[0]["subproblem_id"]
        eligible = [c for c in candidates if c["eligible_for_best_config"] == "yes"]
        if not eligible:
            status = (
                "missing_phase_data" if pair not in available_keys else "missing_valid_gate_config"
            )
            rows.append(
                _empty_best_row(pair, subproblem_id, matched_alpha_error, matched_degree, status)
            )
            continue

        best = min(eligible, key=lambda c: (c["selection_score"], c["degree"]))
        best_error = _safe_float(best["gate_error_vs_ridge"])
        improvement_factor = (
            matched_alpha_error / best_error
            if np.isfinite(matched_alpha_error) and best_error > 0.0
            else float("nan")
        )
        status, selected_for_main = _best_status(
            matched_alpha_error, best_error, improvement_factor
        )
        rows.append(
            {
                "case": pair[0],
                "subproblem_id": subproblem_id,
                "subproblem_type": pair[1],
                "matched_alpha": MATCHED_ALPHA,
                "matched_degree": matched_degree,
                "matched_alpha_error": matched_alpha_error,
                "best_alpha": _safe_float(best["alpha"]),
                "best_degree": int(best["degree"]),
                "best_gate_error_vs_ridge": best_error,
                "best_polynomial_error_vs_exact_target": _safe_float(
                    best["polynomial_error_vs_exact_target"]
                ),
                "best_gate_vs_polynomial_error": _safe_float(best["gate_error_vs_polynomial"]),
                "best_shot_reconstruction_error_vs_ridge": _safe_float(
                    best["shot_reconstruction_error_vs_ridge"]
                ),
                "best_residual_ratio": _safe_float(best["residual_ratio_after_gate"]),
                "best_sign_reliable_fraction": _safe_float(best["sign_reliable_fraction"]),
                "best_top_k_match": _safe_float(best["top_k_match"]),
                "improvement_factor": improvement_factor,
                "selection_rule": (
                    "min selection_score over bounded, in-window, gate-computed configs "
                    "(score = gate_error_vs_ridge + 0.1*degree/45 + penalties)"
                ),
                "selected_for_main_paper": selected_for_main,
                "status": status,
                "notes": _best_notes(
                    status,
                    best,
                    matched_error=matched_alpha_error,
                    best_error=best_error,
                    improvement_factor=improvement_factor,
                ),
            }
        )
    return rows


def _empty_best_row(
    pair: tuple[str, str],
    subproblem_id: str,
    matched_alpha_error: float,
    matched_degree: int,
    status: str,
) -> dict[str, Any]:
    return {
        "case": pair[0],
        "subproblem_id": subproblem_id,
        "subproblem_type": pair[1],
        "matched_alpha": MATCHED_ALPHA,
        "matched_degree": matched_degree,
        "matched_alpha_error": matched_alpha_error,
        "best_alpha": float("nan"),
        "best_degree": -1,
        "best_gate_error_vs_ridge": float("nan"),
        "best_polynomial_error_vs_exact_target": float("nan"),
        "best_gate_vs_polynomial_error": float("nan"),
        "best_shot_reconstruction_error_vs_ridge": float("nan"),
        "best_residual_ratio": float("nan"),
        "best_sign_reliable_fraction": float("nan"),
        "best_top_k_match": float("nan"),
        "improvement_factor": float("nan"),
        "selection_rule": "no eligible bounded in-window gate-computed config",
        "selected_for_main_paper": "no",
        "status": status,
        "notes": (
            "no validated config available; recorded honestly, not fabricated"
            if status == "missing_valid_gate_config"
            else "no validated phase/gate artifact for this (case, subproblem-type)"
        ),
    }


def _best_status(
    matched_alpha_error: float, best_error: float, improvement_factor: float
) -> tuple[str, str]:
    if np.isfinite(matched_alpha_error) and matched_alpha_error <= PASS_THRESHOLD:
        return "matched_alpha_already_good", "yes"
    strong = (np.isfinite(best_error) and best_error < GOOD_GATE_ERROR) or (
        np.isfinite(improvement_factor) and improvement_factor > STRONG_IMPROVEMENT_FACTOR
    )
    if strong:
        return "improved_by_codesign", "yes"
    return "qsvt_polynomial_limited", "reported_as_limitation"


def _best_notes(
    status: str,
    best: dict[str, Any],
    *,
    matched_error: float,
    best_error: float,
    improvement_factor: float,
) -> str:
    gate_vs_poly = _safe_float(best.get("gate_error_vs_polynomial"))
    synthesis_limited = np.isfinite(gate_vs_poly) and gate_vs_poly > 0.05
    if status == "qsvt_polynomial_limited":
        if np.isfinite(improvement_factor) and improvement_factor > 1.1:
            reduction = (
                f"co-design reduced gate-vs-Ridge error {improvement_factor:.1f}x "
                f"({matched_error:.3f} -> {best_error:.3f}) but it stays above the "
                f"{GOOD_GATE_ERROR:.2f} threshold; "
            )
        else:
            reduction = "co-design did not materially reduce the error; "
        limit = (
            "at the degree required for further reduction the QSP phase synthesis itself "
            "degrades (gate-vs-polynomial error grows), so the case is reported as "
            "polynomial/synthesis-limited"
            if synthesis_limited
            else "finite-degree polynomial approximation of the steep regularized target remains "
            "limiting for this subproblem's singular-value structure"
        )
        return reduction + limit
    if status == "matched_alpha_already_good":
        return "matched-alpha already within the pass threshold; best config confirms it"
    return (
        "per-subproblem alpha/degree co-design reduces gate-level readout error "
        f"{improvement_factor:.1f}x (best degree={best['degree']}, "
        f"alpha={_safe_float(best['alpha']):.1e})"
    )


# --------------------------------------------------------------------------- Phase 5
def _validated_interpretation(status: str, gate_error: float) -> str:
    if status in {"missing_valid_gate_config", "missing_phase_data"}:
        return "missing_validated_config"
    if np.isfinite(gate_error) and gate_error <= PASS_THRESHOLD:
        return "validated_readout_pass"
    if status == "improved_by_codesign":
        return "improved_by_codesign"
    return "still_polynomial_limited"


def build_validated_view(
    best_rows: list[dict[str, Any]], sweep_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Per-subproblem QSVT-validated view from the selected best configs."""

    sweep_by_cell: dict[tuple[str, str, float, int], dict[str, Any]] = {}
    for row in sweep_rows:
        sweep_by_cell[
            (
                row["case"],
                row["subproblem_type"],
                round(float(row["alpha"]), 12),
                int(row["degree"]),
            )
        ] = row

    rows: list[dict[str, Any]] = []
    for best in best_rows:
        status = best["status"]
        gate_error = _safe_float(best["best_gate_error_vs_ridge"])
        interpretation = _validated_interpretation(status, gate_error)
        cell = sweep_by_cell.get(
            (
                best["case"],
                best["subproblem_type"],
                round(_safe_float(best["best_alpha"]), 12),
                int(best["best_degree"]) if best["best_degree"] != -1 else -1,
            ),
            {},
        )
        rows.append(
            {
                "case": best["case"],
                "subproblem_id": best["subproblem_id"],
                "subproblem_type": best["subproblem_type"],
                "view": VALIDATED_VIEW,
                "alpha": _safe_float(best["best_alpha"]),
                "degree": int(best["best_degree"]),
                "selection_rule": best["selection_rule"],
                "gate_error_vs_ridge": gate_error,
                "exact_target_error_vs_ridge": _safe_float(cell.get("exact_target_error_vs_ridge")),
                "finite_polynomial_error": _safe_float(
                    best["best_polynomial_error_vs_exact_target"]
                ),
                "gate_vs_polynomial_error": _safe_float(best["best_gate_vs_polynomial_error"]),
                "shot_readout_error": _safe_float(best["best_shot_reconstruction_error_vs_ridge"]),
                "residual_ratio_after_gate": _safe_float(best["best_residual_ratio"]),
                "residual_ratio_after_shot_reconstruction": _safe_float(
                    cell.get("residual_ratio_after_shot_reconstruction")
                ),
                "sign_reliable_fraction": _safe_float(best["best_sign_reliable_fraction"]),
                "top_k_match": _safe_float(best["best_top_k_match"]),
                "status": status,
                "interpretation": interpretation,
                "source_sweep_artifact": "full_vector_readout/readout_alpha_degree_sweep.csv",
                "notes": best["notes"],
            }
        )
    return rows


# --------------------------------------------------------------------------- Phase 6
def _final_status(best: dict[str, Any]) -> str:
    status = best["status"]
    matched_error = _safe_float(best["matched_alpha_error"])
    if status == "matched_alpha_already_good" or (
        np.isfinite(matched_error) and matched_error <= PASS_THRESHOLD
    ):
        return "matched_alpha_pass"
    if status == "improved_by_codesign":
        return "improved_by_codesign"
    if status in {"missing_valid_gate_config", "missing_phase_data"}:
        return "missing_validated_config"
    return "polynomial_limited_after_sweep"


_FINAL_STATUS_INTERPRETATION = {
    "matched_alpha_pass": (
        "Matched-alpha gate-level readout already reconstructs the Ridge update within the pass "
        "threshold; co-design confirms it."
    ),
    "improved_by_codesign": (
        "Per-subproblem QSVT-aware alpha/degree co-design reduces gate-level readout error "
        "relative to the common matched-alpha setting."
    ),
    "polynomial_limited_after_sweep": (
        "Gate-level readout error is dominated by finite-degree polynomial approximation of the "
        "steep regularized target; the readout protocol is valid but the polynomial remains "
        "limiting for this subproblem after the sweep."
    ),
    "missing_validated_config": (
        "No bounded, in-window, gate-computed configuration is available; recorded honestly."
    ),
}


def build_two_view_summary(
    matched_view_rows: list[dict[str, Any]],
    validated_view_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    matched_by_pair = {(r["case"], r["subproblem_type"]): r for r in matched_view_rows}
    validated_by_pair = {(r["case"], r["subproblem_type"]): r for r in validated_view_rows}
    rows: list[dict[str, Any]] = []
    for best in best_rows:
        pair = (best["case"], best["subproblem_type"])
        matched = matched_by_pair.get(pair, {})
        validated = validated_by_pair.get(pair, {})
        final_status = _final_status(best)
        rows.append(
            {
                "case": best["case"],
                "subproblem_id": best["subproblem_id"],
                "subproblem_type": best["subproblem_type"],
                "matched_alpha": MATCHED_ALPHA,
                "matched_degree": int(_safe_float(best["matched_degree"])),
                "matched_alpha_gate_error": _safe_float(best["matched_alpha_error"]),
                "validated_alpha": _safe_float(best["best_alpha"]),
                "validated_degree": int(best["best_degree"]),
                "validated_gate_error": _safe_float(best["best_gate_error_vs_ridge"]),
                "improvement_factor": _safe_float(best["improvement_factor"]),
                "matched_dominant_error_source": str(matched.get("dominant_error_source", "")),
                "validated_dominant_error_source": _validated_dominant(validated),
                "final_status": final_status,
                "manuscript_interpretation": _FINAL_STATUS_INTERPRETATION[final_status],
                "notes": best["notes"],
            }
        )
    return rows


def _validated_dominant(validated: dict[str, Any]) -> str:
    if not validated:
        return ""
    components = {
        "exact_target_vs_polynomial": _safe_float(validated.get("finite_polynomial_error")),
        "polynomial_vs_gate": _safe_float(validated.get("gate_vs_polynomial_error")),
        "shot_reconstruction_vs_ridge": _safe_float(validated.get("shot_readout_error")),
        "ridge_vs_exact_target": _safe_float(validated.get("exact_target_error_vs_ridge")),
    }
    finite = {k: v for k, v in components.items() if np.isfinite(v)}
    if not finite:
        return ""
    return max(finite, key=lambda key: finite[key])


def two_view_summary_markdown(summary_rows: list[dict[str, Any]]) -> str:
    improved = [r for r in summary_rows if r["final_status"] == "improved_by_codesign"]
    limited = [r for r in summary_rows if r["final_status"] == "polynomial_limited_after_sweep"]
    matched_errors = [
        _safe_float(r["matched_alpha_gate_error"])
        for r in summary_rows
        if np.isfinite(_safe_float(r["matched_alpha_gate_error"]))
    ]
    validated_errors = [
        _safe_float(r["validated_gate_error"])
        for r in summary_rows
        if np.isfinite(_safe_float(r["validated_gate_error"]))
    ]
    lines = [
        "# Two-View Gate-Level Readout Summary",
        "",
        CODESIGN_CLAIM,
        "",
        "## Two views",
        "The matched-alpha view fixes alpha=1e-4 to align with the classical benchmark and is "
        "used as a diagnostic common-reference setting.",
        "",
        "The per-subproblem validated view selects alpha and degree using a QSVT-aware sweep and "
        "is used to assess achievable gate-level readout accuracy.",
        "",
        "Large matched-alpha errors are attributed to finite-degree polynomial approximation when "
        "the dominant error source is exact_target_vs_polynomial, not to the readout protocol.",
        "",
        "Efficient full IEEE-scale full-vector readout remains outside the demonstrated scope.",
        "",
        "## Headline",
        f"- Targeted subproblems: {len(summary_rows)}",
        f"- Improved by co-design: {len(improved)}",
        f"- Polynomial-limited after sweep: {len(limited)}",
        (
            f"- Matched-alpha max gate error: {max(matched_errors):.3f}"
            if matched_errors
            else "- Matched-alpha max gate error: n/a"
        ),
        (
            f"- Validated-view max gate error: {max(validated_errors):.3f}"
            if validated_errors
            else "- Validated-view max gate error: n/a"
        ),
        "",
        "## Per-subproblem",
        "| case | subproblem | matched (alpha,deg)->err | validated (alpha,deg)->err | "
        "improvement | final status |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['case']} | {row['subproblem_type']} | "
            f"(1e-4, {int(_safe_float(row['matched_degree']))}) -> "
            f"{_safe_float(row['matched_alpha_gate_error']):.3f} | "
            f"({_safe_float(row['validated_alpha']):.0e}, {int(row['validated_degree'])}) -> "
            f"{_safe_float(row['validated_gate_error']):.3f} | "
            f"{_safe_float(row['improvement_factor']):.2f}x | {row['final_status']} |"
        )
    lines.extend(
        [
            "",
            "## What is not claimed",
            "- Not all gate-level configs accurately reconstruct Ridge; high-error cases are "
            "reported as polynomial-limited.",
            "- No efficient full IEEE-scale full-vector readout, quantum speedup, or QSVT "
            "superiority over Ridge/Tikhonov.",
            "",
        ]
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- Phase 7
def build_two_view_figure_rows(
    matched_view_rows: list[dict[str, Any]],
    validated_view_rows: list[dict[str, Any]],
    targeted: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    targeted_set = set(targeted)
    rows: list[dict[str, Any]] = []
    for source in (matched_view_rows, validated_view_rows):
        for row in source:
            pair = (row["case"], row["subproblem_type"])
            if pair not in targeted_set:
                continue
            rows.append(
                {
                    "case": row["case"],
                    "subproblem_type": row["subproblem_type"],
                    "view": row["view"],
                    "alpha": _safe_float(row["alpha"]),
                    "degree": int(_safe_float(row["degree"])),
                    "vector_relative_error_vs_ridge": _safe_float(row["gate_error_vs_ridge"]),
                    "status": row.get("interpretation", row.get("status", "")),
                    "plot_group": f"{row['case']}/{row['subproblem_type']}",
                    "notes": "matched vs validated gate-level vector error",
                }
            )
    return rows


def build_error_decomposition_figure_rows(
    matched_view_rows: list[dict[str, Any]],
    validated_view_rows: list[dict[str, Any]],
    targeted: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    targeted_set = set(targeted)
    rows: list[dict[str, Any]] = []
    for source in (matched_view_rows, validated_view_rows):
        for view_row in source:
            pair = (view_row["case"], view_row["subproblem_type"])
            if pair not in targeted_set:
                continue
            components = {
                "ridge_vs_exact_target": _safe_float(view_row.get("exact_target_error_vs_ridge")),
                "exact_target_vs_polynomial": _safe_float(view_row.get("finite_polynomial_error")),
                "polynomial_vs_gate": _safe_float(view_row.get("gate_vs_polynomial_error")),
                "gate_vs_shot_reconstruction": float("nan"),
                "shot_reconstruction_vs_ridge": _safe_float(view_row.get("shot_readout_error")),
            }
            dominant = _dominant_component(components)
            for component in _ERROR_COMPONENTS:
                value = components[component]
                missing = not np.isfinite(value)
                rows.append(
                    {
                        "case": view_row["case"],
                        "subproblem_type": view_row["subproblem_type"],
                        "alpha": _safe_float(view_row["alpha"]),
                        "degree": int(_safe_float(view_row["degree"])),
                        "error_component": component,
                        "error_value": abs(value) if not missing else float("nan"),
                        "view": view_row["view"],
                        "dominant_error_source": dominant,
                        "notes": (
                            "component not separately recorded in this view"
                            if missing
                            else "relative L2 error component"
                        ),
                    }
                )
    return rows


def _dominant_component(components: dict[str, float]) -> str:
    finite = {k: v for k, v in components.items() if np.isfinite(v)}
    if not finite:
        return ""
    return max(finite, key=lambda key: finite[key])


# --------------------------------------------------------------------------- Phase 8
def readout_codesign_scope_note_markdown(summary_rows: list[dict[str, Any]]) -> str:
    improved = [
        r["case"] + "/" + r["subproblem_type"]
        for r in summary_rows
        if r["final_status"] == "improved_by_codesign"
    ]
    limited = [
        r["case"] + "/" + r["subproblem_type"]
        for r in summary_rows
        if r["final_status"] == "polynomial_limited_after_sweep"
    ]
    return "\n".join(
        [
            "# Readout Alpha/Degree Co-Design Scope Note",
            "",
            CODESIGN_CLAIM,
            "",
            "## Two-view analysis",
            "The two-view analysis separates common-alpha diagnostic behavior from per-subproblem "
            "QSVT-validated behavior.",
            "",
            "The high matched-alpha gate-level errors are not hidden. They are diagnosed through "
            "error decomposition.",
            "",
            "Where co-design improves the error, this supports the selected-subproblem readout "
            "claim. Where co-design does not improve the error, the case is reported as "
            "polynomial-limited.",
            "",
            "The result does not establish efficient full IEEE-scale full-vector readout.",
            "",
            "## Subclaim C10a.1 (selected-subproblem readout dependence)",
            "Gate-level readout accuracy depends on the finite-degree QSVT polynomial "
            "approximation of the regularized target. Per-subproblem QSVT-aware alpha/degree "
            "selection reduces this error where the subproblem singular-value structure permits. "
            "This subclaim refines C10a and is not an independent readout claim; it does not "
            "promote the full-scale assumption C10b.",
            "",
            "## Outcome",
            *(
                [f"- Improved by co-design: {name}" for name in improved]
                or ["- Improved by co-design: none"]
            ),
            *(
                [f"- Polynomial-limited after sweep: {name}" for name in limited]
                or ["- Polynomial-limited after sweep: none"]
            ),
            "",
            "## What is not claimed",
            "- It is not claimed that gate-level readout accurately reconstructs Ridge for all "
            "selected subproblems.",
            "- No efficient full IEEE-scale full-vector readout, quantum speedup, or QSVT "
            "numerical superiority over Ridge/Tikhonov.",
            "",
        ]
    )


# --------------------------------------------------------------------------- orchestration
def run_alpha_degree_sweep(config: dict[str, Any]) -> dict[str, Any]:
    """Phases 1-5: classify, matched-alpha view, targeted sweep, best configs, validated view."""

    resolved: dict[str, Any] = {
        "input_root": "outputs",
        "output_dir": "outputs/full_vector_readout",
        "alpha_grid": list(DEFAULT_ALPHA_GRID),
        "degree_grid": list(DEFAULT_DEGREE_GRID),
        "overshoot_degrees": [],
        "shots": DEFAULT_SWEEP_SHOTS,
        "trials": DEFAULT_SWEEP_TRIALS,
        "seed": DEFAULT_SEED,
        "target_error": 1.0e-2,
        "mandatory": [("ieee14", "metadata_mapped")],
        "top_k_high_error": 3,
    }
    resolved.update(config)
    input_root = Path(resolved["input_root"])
    output_dir = ensure_directory(resolved["output_dir"])
    seed = int(resolved["seed"])
    shots = int(resolved["shots"])
    trials = int(resolved["trials"])
    target_error = float(resolved["target_error"])
    alpha_grid = [float(a) for a in resolved["alpha_grid"]]
    degree_grid = [int(d) for d in resolved["degree_grid"]]
    overshoot_degrees = [int(d) for d in resolved["overshoot_degrees"]]
    mandatory = tuple((str(c), str(t)) for c, t in resolved["mandatory"])

    classification_rows = classify_readout_configs(output_dir)
    matched_view_rows = build_matched_alpha_view(output_dir)
    discovered, _missing = discover_validated_subproblems(input_root)
    entry_by_pair = {(e["case"], e["subproblem_type"]): e for e in discovered}
    available_keys = set(entry_by_pair.keys())

    targeted = select_targeted_configs(
        classification_rows,
        mandatory=mandatory,
        top_k_high_error=int(resolved["top_k_high_error"]),
    )

    system_cache = SystemCache()
    sweep_rows: list[dict[str, Any]] = []
    for pair in targeted:
        entry = entry_by_pair.get(pair)
        if entry is None:
            continue
        sub = reconstruct_subproblem(
            case=entry["case"],
            row_indices=entry["row_indices"],
            col_indices=entry["col_indices"],
            system_cache=system_cache,
            seed=seed,
        )
        H, r = sub["H"], sub["r"]
        for degree in degree_grid:
            for alpha in alpha_grid:
                sweep_rows.append(
                    evaluate_sweep_cell(
                        entry,
                        H,
                        r,
                        alpha=alpha,
                        degree=degree,
                        seed=seed,
                        shots=shots,
                        trials=trials,
                        target_error=target_error,
                        overshoot_diagnostic_only=False,
                    )
                )
        for degree in overshoot_degrees:
            for alpha in alpha_grid:
                sweep_rows.append(
                    evaluate_sweep_cell(
                        entry,
                        H,
                        r,
                        alpha=alpha,
                        degree=degree,
                        seed=seed,
                        shots=shots,
                        trials=trials,
                        target_error=target_error,
                        overshoot_diagnostic_only=True,
                    )
                )

    best_rows = select_best_configs(
        sweep_rows, matched_view_rows, list(targeted), available_keys=available_keys
    )
    validated_view_rows = build_validated_view(best_rows, sweep_rows)

    artifacts = {
        "readout_config_diagnostic_classification": _write_table(
            classification_rows,
            output_dir / "readout_config_diagnostic_classification.csv",
            CLASSIFICATION_COLUMNS,
        ),
        "readout_matched_alpha_view": _write_table(
            matched_view_rows,
            output_dir / "readout_matched_alpha_view.csv",
            MATCHED_ALPHA_VIEW_COLUMNS,
        ),
        "readout_alpha_degree_sweep": _write_table(
            sweep_rows, output_dir / "readout_alpha_degree_sweep.csv", SWEEP_COLUMNS
        ),
        "readout_selected_best_configs": _write_table(
            best_rows, output_dir / "readout_selected_best_configs.csv", BEST_CONFIG_COLUMNS
        ),
        "readout_per_subproblem_validated_view": _write_table(
            validated_view_rows,
            output_dir / "readout_per_subproblem_validated_view.csv",
            VALIDATED_VIEW_COLUMNS,
        ),
    }
    return {
        "output_dir": output_dir,
        "targeted": list(targeted),
        "classification_rows": classification_rows,
        "matched_view_rows": matched_view_rows,
        "sweep_rows": sweep_rows,
        "best_rows": best_rows,
        "validated_view_rows": validated_view_rows,
        "artifacts": artifacts,
    }


def run_readout_two_view_summary(config: dict[str, Any]) -> dict[str, Any]:
    """Phases 6-8: two-view summary, figure data, and the co-design scope note."""

    resolved = {
        "input_dir": "outputs/full_vector_readout",
        "output_dir": "outputs/full_vector_readout",
    }
    resolved.update(config)
    input_dir = Path(resolved["input_dir"])
    output_dir = ensure_directory(resolved["output_dir"])

    matched = _read_csv(input_dir / "readout_matched_alpha_view.csv").to_dict("records")
    validated = _read_csv(input_dir / "readout_per_subproblem_validated_view.csv").to_dict(
        "records"
    )
    best = _read_csv(input_dir / "readout_selected_best_configs.csv").to_dict("records")
    targeted = [(str(r["case"]), str(r["subproblem_type"])) for r in best]

    summary_rows = build_two_view_summary(matched, validated, best)
    figure_two_view = build_two_view_figure_rows(matched, validated, targeted)
    figure_decomp = build_error_decomposition_figure_rows(matched, validated, targeted)
    scope_note = readout_codesign_scope_note_markdown(summary_rows)
    summary_markdown = two_view_summary_markdown(summary_rows)

    artifacts = {
        "readout_two_view_summary": _write_table(
            summary_rows, output_dir / "readout_two_view_summary.csv", TWO_VIEW_SUMMARY_COLUMNS
        ),
        "figure_data_two_view_gate_error": _write_table(
            figure_two_view,
            output_dir / "figure_data_two_view_gate_error.csv",
            FIGURE_TWO_VIEW_COLUMNS,
        ),
        "figure_data_error_decomposition_high_error": _write_table(
            figure_decomp,
            output_dir / "figure_data_error_decomposition_high_error.csv",
            FIGURE_DECOMP_COLUMNS,
        ),
    }
    summary_md_path = output_dir / "readout_two_view_summary.md"
    summary_md_path.write_text(summary_markdown, encoding="utf-8")
    artifacts["readout_two_view_summary_md"] = summary_md_path
    scope_path = output_dir / "readout_codesign_scope_note.md"
    scope_path.write_text(scope_note, encoding="utf-8")
    artifacts["readout_codesign_scope_note"] = scope_path

    return {
        "output_dir": output_dir,
        "summary_rows": summary_rows,
        "figure_two_view_rows": figure_two_view,
        "figure_decomposition_rows": figure_decomp,
        "artifacts": artifacts,
    }


__all__ = [
    "CODESIGN_CLAIM",
    "FULL_VECTOR_READOUT_CLAIM",
    "build_error_decomposition_figure_rows",
    "build_matched_alpha_view",
    "build_two_view_figure_rows",
    "build_two_view_summary",
    "build_validated_view",
    "classify_readout_configs",
    "evaluate_sweep_cell",
    "readout_codesign_scope_note_markdown",
    "run_alpha_degree_sweep",
    "run_readout_two_view_summary",
    "select_best_configs",
    "select_targeted_configs",
    "two_view_summary_markdown",
]
