"""Raw-versus-Ridge conditioning diagnostics over frozen benchmark matrices."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .canonical_registry import (
    RESULT_FIELDS,
    _canonical_record,
    _read_csv,
    atomic_write_csv,
    load_json,
    stable_array_fingerprint,
)


@dataclass(frozen=True, slots=True)
class MatrixAuditSpec:
    matrix_id: str
    experiment_family: str
    ieee_case: str
    structural_group_id: str | None
    matrix_fingerprint: str
    source_artifact: Path
    source_row_locator: str
    configuration_id: str
    matrix: np.ndarray
    alpha: float
    residual_paths: tuple[Path, ...]


def singular_threshold(matrix: np.ndarray, sigma_max: float) -> float:
    """Return max(m,n)*eps*sigma_max, the frozen numerical-rank threshold."""

    return max(matrix.shape) * np.finfo(np.float64).eps * sigma_max


def effective_rank_entropy(singular_values: np.ndarray) -> float:
    """Effective rank from Shannon entropy of squared singular-value weights."""

    energy = np.square(np.asarray(singular_values, dtype=float))
    total = float(energy.sum())
    if total == 0.0:
        return 0.0
    probabilities = energy[energy > 0.0] / total
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def regularized_condition_number(sigma_max: float, sigma_min: float, alpha: float) -> float:
    return float((sigma_max**2 + alpha) / (sigma_min**2 + alpha))


def global_ridge_filter_bound(alpha: float) -> float:
    return float(1.0 / (2.0 * math.sqrt(alpha)))


def actual_ridge_filter_response(singular_values: np.ndarray, alpha: float) -> float:
    values = np.asarray(singular_values, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.max(values / (values**2 + alpha)))


def ridge_solve_residual(
    matrix: np.ndarray, residual: np.ndarray, alpha: float, epsilon: float = 1e-30
) -> float:
    """Compute the relative normal-equation residual for the Ridge solve."""

    h = np.asarray(matrix, dtype=float)
    r = np.asarray(residual, dtype=float).reshape(-1)
    normal = h.T @ h + alpha * np.eye(h.shape[1], dtype=float)
    rhs = h.T @ r
    solution = np.linalg.solve(normal, rhs)
    numerator = np.linalg.norm(normal @ solution - rhs)
    denominator = max(float(np.linalg.norm(rhs)), epsilon)
    return float(numerator / denominator)


def _residual_paths_from_registry(family: Path, instance_id: str) -> tuple[Path, ...]:
    registry = _read_csv(family / "residual_registry.csv")
    status_column = "status" if "status" in registry.columns else "generation_status"
    rows = registry[
        (registry["instance_id"] == instance_id) & (registry[status_column] == "completed")
    ]
    paths: list[Path] = []
    for value in rows["residual_file"].astype(str):
        path = Path(value)
        paths.append(path if path.is_absolute() else family / path)
    return tuple(paths)


def _embedded_instance_specs(
    root: Path, family_name: str, alpha_fallback: float
) -> list[MatrixAuditSpec]:
    family = root / "outputs" / family_name
    specs: list[MatrixAuditSpec] = []
    for path in sorted((family / "instances").glob("*.json")):
        payload = load_json(path)
        matrix = np.asarray(payload["matrix"], dtype=float)
        instance_id = str(payload["instance_id"])
        specs.append(
            MatrixAuditSpec(
                matrix_id=f"{family_name}:{instance_id}",
                experiment_family=family_name,
                ieee_case=str(payload["ieee_case"]),
                structural_group_id=payload.get("structural_group_id"),
                matrix_fingerprint=str(payload["matrix_fingerprint"]),
                source_artifact=path.relative_to(root),
                source_row_locator="$.matrix",
                configuration_id=f"cfg:{family_name}:study",
                matrix=matrix,
                alpha=float(payload.get("regularization_alpha", alpha_fallback)),
                residual_paths=_residual_paths_from_registry(family, instance_id),
            )
        )
    return specs


def collect_matrix_specs(root: Path, output_dir: Path) -> list[MatrixAuditSpec]:
    """Collect every frozen final matrix without copying its payload."""

    integrated = root / "outputs/sparse_integrated_chain"
    iconfig = load_json(integrated / "configuration.json")
    specs: list[MatrixAuditSpec] = [
        MatrixAuditSpec(
            matrix_id="integrated_sparse_chain:quantized_8x8",
            experiment_family="integrated_sparse_chain",
            ieee_case="ieee14",
            structural_group_id=None,
            matrix_fingerprint=iconfig["matrix_fingerprint"],
            source_artifact=Path("outputs/sparse_integrated_chain/matrix_quantized.npy"),
            source_row_locator="array",
            configuration_id=(f"cfg:integrated:{iconfig['configuration_id']}:statevector"),
            matrix=np.load(integrated / "matrix_quantized.npy"),
            alpha=float(iconfig["alpha"]),
            residual_paths=(integrated / "residual.npy",),
        )
    ]

    precision = root / "outputs/sparse_error_precision_study"
    matrix_registry = _read_csv(precision / "matrix_precision_registry.csv")
    configurations = _read_csv(output_dir / "canonical_configuration_registry.csv", dtype=str)
    precision_configs = configurations[
        configurations["experiment_family"] == "precision_sensitivity"
    ]
    for _, row in matrix_registry.iterrows():
        if pd.isna(row["matrix_file"]):
            source_name = (
                "matrix_original.npy"
                if str(row["value_bits"]) == "original"
                else "matrix_sparse_exact.npy"
            )
            path = Path("outputs/sparse_error_precision_study") / source_name
        else:
            path = Path(str(row["matrix_file"]))
        source = path if path.is_absolute() else root / path
        matching = precision_configs[
            precision_configs["matrix_fingerprint"] == str(row["matrix_fingerprint"])
        ]
        configuration_id = (
            str(matching.iloc[0]["configuration_id"])
            if not matching.empty
            else str(precision_configs.iloc[0]["configuration_id"])
        )
        specs.append(
            MatrixAuditSpec(
                matrix_id=f"precision_sensitivity:value_bits_{row['value_bits']}",
                experiment_family="precision_sensitivity",
                ieee_case="ieee14",
                structural_group_id=None,
                matrix_fingerprint=str(row["matrix_fingerprint"]),
                source_artifact=source.relative_to(root),
                source_row_locator="array",
                configuration_id=configuration_id,
                matrix=np.load(source),
                alpha=float(iconfig["alpha"]),
                residual_paths=(precision / "matrices/residual.npy",),
            )
        )

    development = root / "outputs/output_aware_sparse_selection"
    dconfig = load_json(development / "study_configuration.json")
    development_matrix_path = root / "outputs/sparse_error_precision_study/matrix_original.npy"
    dmatrix = np.load(development_matrix_path)
    split = load_json(development / "residual_split.json")
    dresiduals = tuple(
        development / str(item["residual_file"])
        for item in split["records"]
        if item["status"] == "completed"
    )
    specs.append(
        MatrixAuditSpec(
            matrix_id="output_aware_sparse_selection:development_ieee14_seed123_8x8",
            experiment_family="output_aware_sparse_selection",
            ieee_case="ieee14",
            structural_group_id=None,
            matrix_fingerprint=str(dconfig["matrix_fingerprint"]),
            source_artifact=development_matrix_path.relative_to(root),
            source_row_locator="array",
            configuration_id="cfg:output_aware_sparse_selection:study",
            matrix=dmatrix,
            alpha=float(dconfig["physical_alpha"]),
            residual_paths=dresiduals,
        )
    )

    specs.extend(
        _embedded_instance_specs(root, "output_aware_generalization", float(iconfig["alpha"]))
    )
    specs.extend(
        _embedded_instance_specs(
            root, "output_aware_structural_generalization", float(iconfig["alpha"])
        )
    )
    ids = [spec.matrix_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate matrix audit IDs")
    return specs


def audit_matrix_conditioning(
    spec: MatrixAuditSpec, solve_epsilon: float = 1e-30
) -> dict[str, object]:
    matrix = np.asarray(spec.matrix, dtype=float)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    threshold = singular_threshold(matrix, sigma_max)
    numerical_rank = int(np.sum(singular_values > threshold))
    full_rank = numerical_rank == min(matrix.shape)
    smallest_computed = float(singular_values[-1]) if singular_values.size else 0.0
    sigma_min = smallest_computed if full_rank else 0.0
    positive = singular_values[singular_values > threshold]
    thresholded_condition = (
        float(sigma_max / positive[-1]) if positive.size and positive[-1] > 0 else math.inf
    )
    raw_condition = float(sigma_max / sigma_min) if full_rank and sigma_min > 0 else math.inf
    regularized = regularized_condition_number(sigma_max, sigma_min, spec.alpha)
    residual_values = [
        ridge_solve_residual(matrix, np.load(path), spec.alpha, solve_epsilon)
        for path in spec.residual_paths
        if path.exists()
    ]
    if residual_values:
        residual_array = np.asarray(residual_values)
        residual_median = float(np.median(residual_array))
        residual_p90 = float(np.quantile(residual_array, 0.9))
        residual_max = float(residual_array.max())
    else:
        residual_median = residual_p90 = residual_max = math.nan
    stable_array_fp = stable_array_fingerprint(matrix)
    return {
        "matrix_id": spec.matrix_id,
        "experiment_family": spec.experiment_family,
        "ieee_case": spec.ieee_case,
        "structural_group_id": spec.structural_group_id or "",
        "matrix_fingerprint": spec.matrix_fingerprint,
        "recomputed_matrix_fingerprint": stable_array_fp,
        "matrix_fingerprint_matches": stable_array_fp == spec.matrix_fingerprint,
        "source_artifact": spec.source_artifact.as_posix(),
        "source_row_locator": spec.source_row_locator,
        "configuration_id": spec.configuration_id,
        "matrix_shape": f"{matrix.shape[0]}x{matrix.shape[1]}",
        "numerical_rank": numerical_rank,
        "effective_rank": effective_rank_entropy(singular_values),
        "rank_deficient": not full_rank,
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "smallest_computed_singular_value": smallest_computed,
        "singular_threshold": threshold,
        "singular_threshold_rule": "max(m,n)*float64_epsilon*sigma_max",
        "raw_condition_number": raw_condition,
        "thresholded_condition_number": thresholded_condition,
        "alpha": spec.alpha,
        "regularized_condition_number": regularized,
        "max_ridge_filter_response_global": global_ridge_filter_bound(spec.alpha),
        "max_ridge_filter_response_actual": actual_ridge_filter_response(
            singular_values, spec.alpha
        ),
        "effective_dimension": float(
            np.sum(singular_values**2 / (singular_values**2 + spec.alpha))
        ),
        "residual_count": len(residual_values),
        "ridge_solve_residual_median": residual_median,
        "ridge_solve_residual_p90": residual_p90,
        "ridge_solve_residual_max": residual_max,
        "raw_conditioning_label": "raw_matrix_conditioning",
        "regularized_conditioning_label": "regularized_normal_system_conditioning",
        "ridge_filter_label": "ridge_filter_amplification",
    }


def build_conditioning_audit(root: Path, output_dir: Path) -> pd.DataFrame:
    specs = collect_matrix_specs(root, output_dir)
    rows = [audit_matrix_conditioning(spec) for spec in specs]
    frame = pd.DataFrame(rows).sort_values("matrix_id").reset_index(drop=True)
    atomic_write_csv(
        output_dir / "regularized_conditioning_audit.csv",
        frame.to_dict(orient="records"),
        frame.columns.tolist(),
    )
    summary = _conditioning_summary(frame)
    atomic_write_csv(
        output_dir / "regularized_conditioning_summary.csv",
        summary.to_dict(orient="records"),
        summary.columns.tolist(),
    )
    violations = build_conditioning_guards(root, output_dir, frame)
    unresolved = violations[violations["status"] == "unresolved"] if len(violations) else violations
    if not unresolved.empty:
        raise ValueError(f"unresolved conditioning interpretation guards: {len(unresolved)}")
    _append_conditioning_results(output_dir, frame)
    return frame


def _finite_range(values: pd.Series) -> tuple[float, float, int]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    return (
        float(finite.min()) if finite.size else math.nan,
        float(finite.max()) if finite.size else math.nan,
        int(np.isinf(numeric).sum()),
    )


def _conditioning_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groupings: Iterable[tuple[tuple[str, str], pd.DataFrame]] = frame.groupby(
        ["experiment_family", "ieee_case"], sort=True
    )
    for (family, case), group in groupings:
        raw_min, raw_max, raw_infinite = _finite_range(group["raw_condition_number"])
        rows.append(
            {
                "experiment_family": family,
                "ieee_case": case,
                "matrices": len(group),
                "rank_deficient": int(group["rank_deficient"].astype(bool).sum()),
                "raw_condition_number_finite_min": raw_min,
                "raw_condition_number_finite_max": raw_max,
                "raw_condition_number_infinite_count": raw_infinite,
                "regularized_condition_number_min": float(
                    group["regularized_condition_number"].min()
                ),
                "regularized_condition_number_median": float(
                    group["regularized_condition_number"].median()
                ),
                "regularized_condition_number_max": float(
                    group["regularized_condition_number"].max()
                ),
                "ridge_solve_residual_median_min": float(
                    group["ridge_solve_residual_median"].min()
                ),
                "ridge_solve_residual_median_max": float(
                    group["ridge_solve_residual_median"].max()
                ),
                "ridge_solve_residual_max": float(group["ridge_solve_residual_max"].max()),
                "raw_conditioning_label": "raw_matrix_conditioning",
                "regularized_conditioning_label": "regularized_normal_system_conditioning",
                "ridge_filter_label": "ridge_filter_amplification",
            }
        )
    return pd.DataFrame(rows)


def build_conditioning_guards(root: Path, output_dir: Path, audit: pd.DataFrame) -> pd.DataFrame:
    """Record source interpretation hazards and reject canonical inconsistencies."""

    fields = [
        "guard_id",
        "matrix_id",
        "source_artifact",
        "violation_type",
        "observed_value",
        "required_interpretation",
        "status",
        "blocking",
    ]
    rows: list[dict[str, object]] = []
    for _, row in audit.iterrows():
        matrix_id = str(row["matrix_id"])
        rank_deficient = bool(row["rank_deficient"])
        raw = float(row["raw_condition_number"])
        sigma_max = float(row["sigma_max"])
        sigma_min = float(row["sigma_min"])
        alpha = float(row["alpha"])
        regularized = float(row["regularized_condition_number"])
        required = regularized_condition_number(sigma_max, sigma_min, alpha)
        if rank_deficient and not math.isinf(raw):
            rows.append(
                {
                    "guard_id": "finite_raw_condition_for_rank_deficiency",
                    "matrix_id": matrix_id,
                    "source_artifact": row["source_artifact"],
                    "violation_type": "rank_threshold_inconsistency",
                    "observed_value": raw,
                    "required_interpretation": (
                        "rank-deficient raw condition number must be infinity"
                    ),
                    "status": "unresolved",
                    "blocking": True,
                }
            )
        if not math.isclose(regularized, required, rel_tol=1e-12, abs_tol=1e-12):
            rows.append(
                {
                    "guard_id": "regularized_formula",
                    "matrix_id": matrix_id,
                    "source_artifact": row["source_artifact"],
                    "violation_type": "conditioning_inconsistency",
                    "observed_value": regularized,
                    "required_interpretation": "(sigma_max^2+alpha)/(sigma_min^2+alpha)",
                    "status": "unresolved",
                    "blocking": True,
                }
            )
        if row["raw_conditioning_label"] != "raw_matrix_conditioning":
            rows.append(
                {
                    "guard_id": "raw_label",
                    "matrix_id": matrix_id,
                    "source_artifact": row["source_artifact"],
                    "violation_type": "conditioning_label_conflation",
                    "observed_value": row["raw_conditioning_label"],
                    "required_interpretation": "raw_matrix_conditioning",
                    "status": "unresolved",
                    "blocking": True,
                }
            )
    # Frozen instance registries used np.linalg.cond and can contain finite values
    # below the canonical numerical-rank threshold.  Record this as a resolved
    # source-interpretation hazard; prior artifacts remain untouched.
    for family_name in (
        "output_aware_generalization",
        "output_aware_structural_generalization",
    ):
        source = root / "outputs" / family_name / "instance_registry.csv"
        source_rows = _read_csv(source)
        family_audit = audit[audit["experiment_family"] == family_name]
        source_map = dict(
            zip(source_rows["instance_id"], source_rows["condition_number"], strict=False)
        )
        for _, item in family_audit[family_audit["rank_deficient"]].iterrows():
            instance_id = str(item["matrix_id"]).split(":", 1)[1]
            source_value = source_map.get(instance_id, math.nan)
            if np.isfinite(float(source_value)):
                rows.append(
                    {
                        "guard_id": "source_finite_library_condition_on_rank_deficient_matrix",
                        "matrix_id": item["matrix_id"],
                        "source_artifact": source.relative_to(root).as_posix(),
                        "violation_type": "source_interpretation_hazard",
                        "observed_value": source_value,
                        "required_interpretation": (
                            "canonical raw condition is infinity; finite source library value "
                            "is retained but not used as Ridge-system conditioning"
                        ),
                        "status": "resolved_in_canonical_export",
                        "blocking": False,
                    }
                )
    frame = pd.DataFrame(rows, columns=fields)
    atomic_write_csv(
        output_dir / "conditioning_interpretation_violations.csv",
        frame.to_dict(orient="records"),
        fields,
    )
    return frame


def _append_conditioning_results(output_dir: Path, audit: pd.DataFrame) -> None:
    registry_path = output_dir / "canonical_result_registry.csv"
    registry = _read_csv(registry_path, dtype=str)
    existing = set(registry["result_id"])
    rows: list[dict[str, object]] = []
    for _, item in audit.iterrows():
        slug = str(item["matrix_id"]).replace(":", ":matrix:")
        limitation = "rank_deficient_blocks" if bool(item["rank_deficient"]) else "small_scale_8x8"
        for metric, unit in (
            ("raw_condition_number", "condition_number"),
            ("regularized_condition_number", "condition_number"),
            ("max_ridge_filter_response_global", "inverse_singular_value"),
            ("ridge_solve_residual_max", "relative_residual"),
        ):
            result_id = f"res:conditioning:{slug}:{metric}"
            if result_id in existing:
                continue
            rows.append(
                _canonical_record(
                    result_id=result_id,
                    claim_family="heldout_generalization",
                    experiment_family="regularized_conditioning_audit",
                    configuration_id=str(item["configuration_id"]),
                    source_artifact=Path(
                        "outputs/final_contribution_evidence/regularized_conditioning_audit.csv"
                    ),
                    source_row_locator=f"row[matrix_id={item['matrix_id']}]",
                    evidence_tier="diagnostic_only",
                    matrix_fingerprint=str(item["matrix_fingerprint"]),
                    ieee_case=str(item["ieee_case"]),
                    structural_group_id=(
                        str(item["structural_group_id"])
                        if str(item["structural_group_id"])
                        else None
                    ),
                    value=float(item[metric]),
                    unit=unit,
                    limitation_code=limitation + ";no_quantum_speedup",
                    notes="Diagnostic only; raw and regularized conditioning labels are separate.",
                ).csv_row()
            )
    if rows:
        combined = pd.concat([registry, pd.DataFrame(rows)], ignore_index=True)
        combined = combined.sort_values("result_id").reset_index(drop=True)
        atomic_write_csv(registry_path, combined.to_dict(orient="records"), RESULT_FIELDS)
