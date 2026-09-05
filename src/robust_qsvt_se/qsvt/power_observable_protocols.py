from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.full_matrix_qsvt_demo import rescale_bounded_target_to_original
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    extract_state_estimation_subproblem,
    ridge_tikhonov_update,
)
from robust_qsvt_se.qsvt.polynomial import fit_odd_regularized_polynomial
from robust_qsvt_se.qsvt.power_observable_mapping import build_power_observables
from robust_qsvt_se.qsvt.shot_readout_model import required_shots_for_additive_error
from robust_qsvt_se.utils.io import ensure_directory

CLAIM = (
    "Application-specific observable protocols classify which power-system quantities "
    "are practical to read out under the QSVT-compatible pathway for the same "
    "regularized spectral filter used by Ridge/Tikhonov. This is a readout-aware "
    "observable protocol study on a selected IEEE-derived subproblem. It does not "
    "claim quantum speedup, QSVT superiority over Ridge/Tikhonov, hardware execution, "
    "a solved readout bottleneck, or full-vector tomography."
)

# Protocol vocabulary; each observable is classified with exactly one of these.
PROBABILITY_READOUT = "probability_readout"
SIGNED_OVERLAP = "signed_overlap"
HADAMARD_TEST_PROXY = "hadamard_test_proxy"
NORM_SCALED_OBSERVABLE = "norm_scaled_observable"
AMPLITUDE_ESTIMATION_PROXY = "amplitude_estimation_proxy"
FULL_VECTOR_REQUIRED = "full_vector_required"

_SIGNED_FUNCTIONAL = "signed_linear_functional"
_ENERGY = "subset_energy"
_TOPK = "topk_identification"
_FULL_VECTOR = "full_vector_reconstruction"

SUMMARY_COLUMNS = [
    "observable_name",
    "physical_meaning",
    "state_indices",
    "observable_vector_norm",
    "requires_norm_recovery",
    "requires_signed_overlap",
    "requires_full_vector_readout",
    "protocol_type",
    "ridge_value",
    "qsvt_value_raw",
    "qsvt_value_scaled",
    "absolute_error_scaled",
    "relative_error_scaled",
    "target_tolerance",
    "shot_cost_proxy",
    "query_cost_proxy",
    "claim_allowed",
    "claim_disallowed",
]

CLAIM_ALLOWED = "readout-aware observable protocol on a selected IEEE-derived subproblem"
CLAIM_DISALLOWED = "no quantum speedup, QSVT-over-Ridge superiority, or solved readout bottleneck"


@dataclass(frozen=True, slots=True)
class ObservableProtocol:
    """A power-system observable together with its readout-protocol classification."""

    observable_name: str
    physical_meaning: str
    state_indices: tuple[int, ...]
    kind: str
    coefficients: np.ndarray | None
    observable_vector_norm: float
    protocol_type: str
    requires_norm_recovery: bool
    requires_signed_overlap: bool
    requires_full_vector_readout: bool


@dataclass(frozen=True, slots=True)
class ObservableEvaluation:
    """Resolved observable values on Ridge vs QSVT updates."""

    protocol: ObservableProtocol
    ridge_value: float
    qsvt_value_raw: float
    qsvt_value_scaled: float
    absolute_error_scaled: float
    relative_error_scaled: float


def run_qsvt_power_observable_protocols(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "case": "ieee14",
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_size": 4,
        "alpha": 1.0e-4,
        "degree": 51,
        "target_tolerances": [1.0e-1, 5.0e-2, 1.0e-2, 1.0e-3],
        "topk": 2,
        "seed": 123,
        "output_dir": "outputs/qsvt_power_observable_protocols",
    }
    resolved.update(config)
    resolved["target_tolerances"] = [float(value) for value in resolved["target_tolerances"]]
    if any(value <= 0.0 for value in resolved["target_tolerances"]):
        raise ValueError("target_tolerances must be positive")
    output_dir = ensure_directory(resolved["output_dir"])

    subproblem = extract_state_estimation_subproblem(
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        submatrix_size=int(resolved["submatrix_size"]),
        seed=int(resolved["seed"]),
        case_source=str(resolved["case_source"]),
    )
    H = np.asarray(subproblem.H_tilde, dtype=np.float64)
    r = np.asarray(subproblem.r_tilde, dtype=np.float64)
    ridge_update = ridge_tikhonov_update(H, r, alpha=float(resolved["alpha"]))
    qsvt_update = polynomial_action_update(
        H, r, alpha=float(resolved["alpha"]), degree=int(resolved["degree"])
    )

    protocols = build_observable_protocols(
        H_tilde=H,
        ridge_update=ridge_update,
        metadata=subproblem.metadata,
        topk=int(resolved["topk"]),
    )
    evaluations = [
        evaluate_observable_protocol(
            protocol,
            ridge_update=ridge_update,
            qsvt_update=qsvt_update,
        )
        for protocol in protocols
    ]
    summary_rows = _summary_rows(evaluations, resolved["target_tolerances"])
    tradeoff_rows = _tradeoff_rows(evaluations, resolved["target_tolerances"])
    artifacts = write_qsvt_power_observable_protocols_outputs(
        output_dir, resolved, summary_rows, tradeoff_rows
    )
    return {
        "output_dir": output_dir,
        "summary_rows": summary_rows,
        "tradeoff_rows": tradeoff_rows,
        "artifacts": artifacts,
    }


def polynomial_action_update(
    H_tilde: np.ndarray,
    r_tilde: np.ndarray,
    *,
    alpha: float,
    degree: int,
) -> np.ndarray:
    """Standard bounded QSVT polynomial-action update (known-C rescaled to physical units)."""

    H = np.asarray(H_tilde, dtype=np.float64)
    r = np.asarray(r_tilde, dtype=np.float64)
    B = H.T
    beta = max(float(np.linalg.svd(B, compute_uv=False)[0]), np.finfo(float).eps)
    A = B / beta
    singular_values_A = np.linalg.svd(A, compute_uv=False)
    positive = singular_values_A[singular_values_A > 1.0e-14]
    domain_min = min(max(1.0e-6, 0.9 * float(np.min(positive))), 0.95)
    alpha_norm = float(alpha) / beta**2
    approximation = fit_odd_regularized_polynomial(
        alpha=alpha_norm,
        block_encoding_normalization=1.0,
        degree=int(degree),
        domain_min=domain_min,
        domain_max=1.0,
        grid_size=max(2048, int(degree) + 2),
    )
    polynomial = Polynomial(
        np.asarray(approximation.power_coefficients, dtype=np.float64)
        / float(approximation.scale_factor)
    )
    U, singular_values, Vh = np.linalg.svd(A, full_matrices=False)
    polynomial_operator = U @ (polynomial(singular_values)[:, None] * Vh)
    return rescale_bounded_target_to_original(
        polynomial_operator @ r,
        beta=beta,
        C=float(approximation.scale_factor),
    )


def build_observable_protocols(
    *,
    H_tilde: np.ndarray,
    ridge_update: np.ndarray,
    metadata: dict[str, Any],
    topk: int = 2,
) -> list[ObservableProtocol]:
    """Construct power-system observables and classify each readout protocol.

    Reuses ``build_power_observables`` for component/difference/energy/row observables,
    then adds a top-k identification observable and a full-vector reconstruction
    pseudo-observable that is explicitly marked as not-to-be-emphasized.
    """

    dimension = int(np.asarray(H_tilde).shape[1])
    protocols: list[ObservableProtocol] = []
    for observable in build_power_observables(H_tilde, metadata):
        protocols.append(_classify_power_observable(observable, dimension=dimension))

    k = max(1, min(int(topk), dimension))
    top_indices = _topk_indices(ridge_update, k)
    protocols.append(
        ObservableProtocol(
            observable_name="topk_update_component_identification",
            physical_meaning=(
                f"identify the {k} largest-magnitude update components (relative magnitudes)"
            ),
            state_indices=top_indices,
            kind=_TOPK,
            coefficients=None,
            observable_vector_norm=float(np.sqrt(len(top_indices))),
            protocol_type=PROBABILITY_READOUT,
            requires_norm_recovery=False,
            requires_signed_overlap=False,
            requires_full_vector_readout=False,
        )
    )
    protocols.append(
        ObservableProtocol(
            observable_name="full_state_vector_reconstruction",
            physical_meaning=("full update-vector reconstruction (all components); not emphasized"),
            state_indices=tuple(range(dimension)),
            kind=_FULL_VECTOR,
            coefficients=None,
            observable_vector_norm=float(np.sqrt(dimension)),
            protocol_type=FULL_VECTOR_REQUIRED,
            requires_norm_recovery=True,
            requires_signed_overlap=True,
            requires_full_vector_readout=True,
        )
    )
    return protocols


def evaluate_observable_protocol(
    protocol: ObservableProtocol,
    *,
    ridge_update: np.ndarray,
    qsvt_update: np.ndarray,
) -> ObservableEvaluation:
    ridge = np.asarray(ridge_update, dtype=np.float64)
    qsvt = np.asarray(qsvt_update, dtype=np.float64)
    ridge_value = _observable_value(protocol, ridge)
    qsvt_value_scaled = _observable_value(protocol, qsvt)
    qsvt_value_raw = _observable_value(protocol, _normalize(qsvt))
    absolute = abs(qsvt_value_scaled - ridge_value)
    relative = absolute / abs(ridge_value) if abs(ridge_value) > 1.0e-15 else float("nan")
    return ObservableEvaluation(
        protocol=protocol,
        ridge_value=float(ridge_value),
        qsvt_value_raw=float(qsvt_value_raw),
        qsvt_value_scaled=float(qsvt_value_scaled),
        absolute_error_scaled=float(absolute),
        relative_error_scaled=float(relative),
    )


def write_qsvt_power_observable_protocols_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    tradeoff_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    summary_path = output_dir / "observable_protocol_summary.csv"
    tradeoff_path = output_dir / "observable_accuracy_cost_tradeoff.csv"
    interpretation_path = output_dir / "observable_protocol_interpretation.md"
    pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS).to_csv(summary_path, index=False)
    pd.DataFrame(tradeoff_rows).to_csv(tradeoff_path, index=False)
    interpretation_path.write_text(
        _interpretation_markdown(summary_rows, tradeoff_rows),
        encoding="utf-8",
    )
    manifest = write_manifest(
        output_dir,
        artifacts={
            "observable_protocol_summary": str(summary_path),
            "observable_accuracy_cost_tradeoff": str(tradeoff_path),
            "observable_protocol_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=CLAIM,
    )
    return {
        "manifest": manifest,
        "observable_protocol_summary": summary_path,
        "observable_accuracy_cost_tradeoff": tradeoff_path,
        "observable_protocol_interpretation": interpretation_path,
    }


def _classify_power_observable(observable: Any, *, dimension: int) -> ObservableProtocol:
    name = observable.observable_name
    observable_type = observable.observable_type
    indices = tuple(int(index) for index in observable.state_indices)
    coefficients = (
        np.asarray(observable.coefficients, dtype=np.float64)
        if observable.coefficients is not None
        else None
    )
    if observable_type == "selected_subset_update_energy":
        return ObservableProtocol(
            observable_name=name,
            physical_meaning=observable.physical_interpretation,
            state_indices=indices,
            kind=_ENERGY,
            coefficients=None,
            observable_vector_norm=float(np.sqrt(len(indices))),
            protocol_type=NORM_SCALED_OBSERVABLE,
            requires_norm_recovery=True,
            requires_signed_overlap=False,
            requires_full_vector_readout=False,
        )
    # All remaining build_power_observables entries are signed linear functionals
    # o.x: component (angle/voltage), branch angle-difference, or measurement row.
    coeff_norm = float(np.linalg.norm(coefficients)) if coefficients is not None else 1.0
    protocol_type = HADAMARD_TEST_PROXY if observable_type.endswith("proxy") else SIGNED_OVERLAP
    return ObservableProtocol(
        observable_name=name,
        physical_meaning=observable.physical_interpretation,
        state_indices=indices,
        kind=_SIGNED_FUNCTIONAL,
        coefficients=coefficients,
        observable_vector_norm=coeff_norm,
        protocol_type=protocol_type,
        requires_norm_recovery=True,
        requires_signed_overlap=True,
        requires_full_vector_readout=False,
    )


def _observable_value(protocol: ObservableProtocol, update: np.ndarray) -> float:
    vector = np.asarray(update, dtype=np.float64)
    if protocol.kind == _ENERGY:
        return float(np.sum(vector[list(protocol.state_indices)] ** 2))
    if protocol.kind == _TOPK:
        identified = _topk_indices(vector, len(protocol.state_indices))
        return _jaccard(identified, protocol.state_indices)
    if protocol.kind == _FULL_VECTOR:
        return float(np.linalg.norm(vector))
    coefficients = protocol.coefficients
    if coefficients is None:
        raise ValueError("signed-functional protocol requires coefficients")
    return float(np.dot(coefficients, vector))


def _summary_rows(
    evaluations: list[ObservableEvaluation],
    target_tolerances: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        protocol = evaluation.protocol
        for tolerance in target_tolerances:
            rows.append(
                {
                    "observable_name": protocol.observable_name,
                    "physical_meaning": protocol.physical_meaning,
                    "state_indices": " ".join(str(index) for index in protocol.state_indices),
                    "observable_vector_norm": float(protocol.observable_vector_norm),
                    "requires_norm_recovery": bool(protocol.requires_norm_recovery),
                    "requires_signed_overlap": bool(protocol.requires_signed_overlap),
                    "requires_full_vector_readout": bool(protocol.requires_full_vector_readout),
                    "protocol_type": protocol.protocol_type,
                    "ridge_value": evaluation.ridge_value,
                    "qsvt_value_raw": evaluation.qsvt_value_raw,
                    "qsvt_value_scaled": evaluation.qsvt_value_scaled,
                    "absolute_error_scaled": evaluation.absolute_error_scaled,
                    "relative_error_scaled": evaluation.relative_error_scaled,
                    "target_tolerance": float(tolerance),
                    "shot_cost_proxy": required_shots_for_additive_error(float(tolerance)),
                    "query_cost_proxy": _query_cost_proxy(float(tolerance)),
                    "claim_allowed": CLAIM_ALLOWED,
                    "claim_disallowed": CLAIM_DISALLOWED,
                }
            )
    return rows


def _tradeoff_rows(
    evaluations: list[ObservableEvaluation],
    target_tolerances: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        protocol = evaluation.protocol
        for tolerance in target_tolerances:
            shot_cost = required_shots_for_additive_error(float(tolerance))
            query_cost = _query_cost_proxy(float(tolerance))
            rows.append(
                {
                    "observable_name": protocol.observable_name,
                    "protocol_type": protocol.protocol_type,
                    "physical_meaning": protocol.physical_meaning,
                    "requires_full_vector_readout": bool(protocol.requires_full_vector_readout),
                    "target_tolerance": float(tolerance),
                    "absolute_error_scaled": evaluation.absolute_error_scaled,
                    "relative_error_scaled": evaluation.relative_error_scaled,
                    "shot_cost_proxy": shot_cost,
                    "query_cost_proxy": query_cost,
                    "shot_to_query_ratio": float(shot_cost) / float(max(query_cost, 1)),
                    "dominant_cost": "readout_shots" if shot_cost >= query_cost else "qsvt_queries",
                }
            )
    return rows


def _interpretation_markdown(
    summary_rows: list[dict[str, Any]],
    tradeoff_rows: list[dict[str, Any]],
) -> str:
    by_name: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        by_name.setdefault(row["observable_name"], row)
    practical = [
        row["observable_name"]
        for row in by_name.values()
        if not row["requires_full_vector_readout"]
    ]
    full_vector = [
        row["observable_name"] for row in by_name.values() if row["requires_full_vector_readout"]
    ]
    signed = [row["observable_name"] for row in by_name.values() if row["requires_signed_overlap"]]
    readout_dominant = sum(1 for row in tradeoff_rows if row["dominant_cost"] == "readout_shots")
    query_dominant = len(tradeoff_rows) - readout_dominant
    power_terms = ("angle", "voltage", "theta", "branch", "energy", "v ", "measurement")
    interpretable = [
        row["observable_name"]
        for row in by_name.values()
        if not row["requires_full_vector_readout"]
        and any(term in row["physical_meaning"].lower() for term in power_terms)
    ]
    lines = [
        "# Application-Specific Observable Protocols",
        "",
        CLAIM,
        "",
        "## 1. Observables Practical Under the Current QSVT Pathway",
        "",
        "Practical observables avoid full-vector reconstruction; signed scalar functionals "
        "require both norm recovery and signed-overlap (Hadamard-test-style) readout, "
        "while energy/top-k observables avoid signed overlap.",
        "",
        *[f"- `{name}`" for name in practical],
        "",
        "## 2. Observables Requiring Full-Vector Reconstruction (Not Emphasized)",
        "",
        "These require estimating every component and should not be emphasized as a "
        "cheap-readout advantage of the pathway:",
        "",
        *[f"- `{name}`" for name in full_vector],
        "",
        "## 3. Power-System Interpretability of Selected Observables",
        "",
        "The selected observables preserve power-system meaning: bus angle and voltage "
        "magnitude updates, branch angle-difference proxies, selected-area update energy, "
        "and linearized measurement-row corrections. Signed functionals (which need signed "
        "overlap) include:",
        "",
        *[f"- `{name}`" for name in signed],
        "",
        f"Power-interpretable, practical observables: {', '.join(interpretable) or 'none'}.",
        "",
        "## 4. Readout Cost vs QSVT Query Cost",
        "",
        "Shot-based readout cost scales as O(1/eps^2); the amplitude-estimation-style query "
        "cost scales as O(1/eps). As the target tolerance tightens, shot-based readout cost "
        "dominates the QSVT query cost.",
        "",
        f"- Tolerance/observable rows where readout shots dominate: {readout_dominant}",
        f"- Tolerance/observable rows where QSVT queries dominate: {query_dominant}",
        "- Conclusion: readout cost dominates over QSVT query cost at the tighter tolerances, "
        "which is why selected cheap observables (not full-vector tomography) are emphasized.",
        "",
    ]
    return "\n".join(lines)


def _query_cost_proxy(tolerance: float) -> int:
    return ceil(1.0 / float(tolerance))


def _topk_indices(values: np.ndarray, k: int) -> tuple[int, ...]:
    vector = np.asarray(values, dtype=np.float64)
    indices = np.arange(vector.size)
    order = np.lexsort((indices, -np.abs(vector)))
    return tuple(sorted(int(index) for index in order[: int(k)]))


def _jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    if not union:
        return 1.0
    return float(len(left_set & right_set) / len(union))


def _normalize(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-15:
        return vector
    return vector / norm
