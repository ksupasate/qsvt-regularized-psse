"""Canonical registries and independent checks over frozen evidence artifacts.

The functions in this module are deliberately read-only with respect to all
scientific source directories.  They aggregate row-level artifacts, attach
typed provenance, and write only to the caller-supplied closure directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

EvidenceTier = Literal[
    "classical_exact",
    "polynomial_evaluation",
    "exact_svt",
    "qsvt_matrix_action",
    "qsvt_statevector",
    "finite_shot_simulation",
    "transpiled_executed_circuit",
    "modeled_resource",
    "diagnostic_only",
    "excluded",
]

EVIDENCE_TIERS: tuple[EvidenceTier, ...] = (
    "classical_exact",
    "polynomial_evaluation",
    "exact_svt",
    "qsvt_matrix_action",
    "qsvt_statevector",
    "finite_shot_simulation",
    "transpiled_executed_circuit",
    "modeled_resource",
    "diagnostic_only",
    "excluded",
)


@dataclass(frozen=True, slots=True)
class CanonicalEvidenceRecord:
    """One typed result with exact source and limitation provenance."""

    result_id: str
    claim_family: str
    experiment_family: str
    configuration_id: str
    source_artifact: Path
    source_row_locator: str
    evidence_tier: EvidenceTier
    matrix_fingerprint: str | None
    support_fingerprint: str | None
    residual_split_fingerprint: str | None
    phase_fingerprint: str | None
    polynomial_fingerprint: str | None
    functional_id: str | None
    ieee_case: str | None
    structural_group_id: str | None
    value: float | str
    unit: str
    status: str
    manuscript_eligible: bool
    limitation_code: str | None
    notes: str

    def csv_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["source_artifact"] = self.source_artifact.as_posix()
        return row


CONFIGURATION_FIELDS = [
    "configuration_id",
    "experiment_family",
    "ieee_case",
    "structural_group_id",
    "matrix_shape",
    "matrix_fingerprint",
    "support_fingerprint",
    "residual_split_fingerprint",
    "functional_set_fingerprint",
    "alpha",
    "beta",
    "lambda",
    "C",
    "polynomial_degree",
    "phase_count",
    "phase_fingerprint",
    "quantization_bits",
    "nonzeros",
    "slot_budget",
    "actual_slots",
    "evidence_tier",
    "source_configuration_file",
    "source_manifest",
    "status",
]

RESULT_FIELDS = [field for field in CanonicalEvidenceRecord.__dataclass_fields__]

CLAIM_FIELDS = [
    "claim_id",
    "claim_family",
    "claim_text_neutral",
    "support_status",
    "supporting_result_ids",
    "contradicting_or_limiting_result_ids",
    "evidence_tier",
    "manuscript_eligible",
    "required_qualifier",
    "prohibited_overclaim",
]

LIMITATION_FIELDS = [
    "limitation_id",
    "category",
    "description",
    "affected_result_ids",
    "required_manuscript_qualifier",
    "prohibited_claim",
    "evidence_source",
    "status",
]

HEADLINE_CHECK_FIELDS = [
    "check_id",
    "result_id",
    "expected_value",
    "recomputed_value",
    "absolute_difference",
    "relative_difference",
    "tolerance",
    "status",
    "source_artifact",
]


def repository_root(start: Path | None = None) -> Path:
    """Find the repository root without mutating Git state."""

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError(f"repository root not found from {current}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_array_fingerprint(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def stable_json_fingerprint(value: Any) -> str:
    payload = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "infinity" if value > 0 else "-infinity"
    return value


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})
    os.replace(temporary, path)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "infinity" if value > 0 else "-infinity"
    return value


def parse_shape(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            value = json.loads(stripped)
        elif "x" in stripped:
            return stripped
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return "x".join(str(int(item)) for item in value)
    return str(value)


def _present(value: Any) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def _clean(value: Any) -> Any:
    return value if _present(value) else ""


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs)


def _source_path(root: Path, value: str | Path, family_dir: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if (root / path).exists():
        return root / path
    if family_dir is not None and (family_dir / path).exists():
        return family_dir / path
    return root / path


def _support_mask_fingerprint(matrix: np.ndarray) -> str:
    return stable_array_fingerprint((np.asarray(matrix) != 0.0).astype(np.float64))


def _functional_set_fingerprint(frame: pd.DataFrame) -> str:
    columns = [
        column
        for column in ("functional_id", "functional_family", "functional_vector")
        if column in frame.columns
    ]
    rows = frame[columns].sort_values(columns).to_dict(orient="records") if columns else []
    return stable_json_fingerprint(rows)


def _residual_split_fingerprint(path: Path) -> str:
    return sha256_file(path) if path.exists() else ""


def _base_configuration_record(**overrides: Any) -> dict[str, Any]:
    row = {field: "" for field in CONFIGURATION_FIELDS}
    row.update(overrides)
    return row


def _instance_context(
    root: Path, family_dir: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    registry = _read_csv(family_dir / "instance_registry.csv")
    contexts: dict[str, dict[str, Any]] = {}
    split_fingerprints: dict[str, str] = {}
    for row in registry.to_dict(orient="records"):
        instance_id = str(row["instance_id"])
        contexts[instance_id] = row
        split_path = family_dir / "residual_splits" / f"{instance_id}.json"
        if not split_path.exists():
            # Multi-instance studies also use this exact naming convention, but retain
            # a deterministic registry fallback if a split payload is unavailable.
            residual_rows = family_dir / "residual_registry.csv"
            if residual_rows.exists():
                subset = _read_csv(residual_rows)
                subset = subset[subset["instance_id"] == instance_id]
                split_fingerprints[instance_id] = stable_json_fingerprint(
                    subset[["residual_seed", "split", "residual_fingerprint"]]
                    .sort_values(["split", "residual_seed"])
                    .to_dict(orient="records")
                )
        else:
            split_fingerprints[instance_id] = _residual_split_fingerprint(split_path)
    return contexts, split_fingerprints


def build_configuration_registry(root: Path, output_dir: Path) -> pd.DataFrame:
    """Build a non-collapsing configuration registry across frozen campaigns."""

    rows: list[dict[str, Any]] = []
    integrated = root / "outputs/sparse_integrated_chain"
    iconfig = load_json(integrated / "configuration.json")
    imeta = load_json(integrated / "matrix_metadata.json")
    matrix = np.load(integrated / "matrix_quantized.npy")
    phases = np.load(integrated / "phases.npy")
    functionals_payload = load_json(integrated / "selected_functionals.json")
    support_fp = _support_mask_fingerprint(matrix)
    functional_fp = stable_json_fingerprint(functionals_payload)
    common_integrated = dict(
        experiment_family="integrated_sparse_chain",
        ieee_case="ieee14",
        matrix_shape=parse_shape(matrix.shape),
        matrix_fingerprint=iconfig["matrix_fingerprint"],
        support_fingerprint=support_fp,
        residual_split_fingerprint=iconfig["residual_fingerprint"],
        functional_set_fingerprint=functional_fp,
        alpha=iconfig["alpha"],
        beta=iconfig["beta"],
        C=iconfig["C"],
        polynomial_degree=iconfig["polynomial_degree"],
        phase_count=len(phases),
        phase_fingerprint=stable_array_fingerprint(phases),
        quantization_bits=iconfig["matrix_value_bits"],
        nonzeros=int(np.count_nonzero(matrix)),
        slot_budget=imeta["slots"],
        actual_slots=imeta["slots"],
        source_configuration_file=(integrated / "configuration.json").relative_to(root),
        source_manifest=(integrated / "manifest.json").relative_to(root),
        status="completed",
    )
    common_integrated["lambda"] = iconfig["lambda"]
    rows.append(
        _base_configuration_record(
            configuration_id=f"cfg:integrated:{iconfig['configuration_id']}:statevector",
            evidence_tier="qsvt_statevector",
            **common_integrated,
        )
    )
    rows.append(
        _base_configuration_record(
            configuration_id=f"cfg:integrated:{iconfig['configuration_id']}:finite-shot",
            evidence_tier="finite_shot_simulation",
            **common_integrated,
        )
    )
    rows.append(
        _base_configuration_record(
            configuration_id=f"cfg:integrated:{iconfig['configuration_id']}:resource",
            evidence_tier="transpiled_executed_circuit",
            **common_integrated,
        )
    )
    integrated_resources = _read_csv(integrated / "resource_ledger.csv")
    for _, resource_row in integrated_resources.iterrows():
        category = str(resource_row["resource_category"])
        if category == "executed_small_scale_sparse_integrated":
            continue
        modeled = category.startswith("modeled_")
        resource_common = dict(common_integrated)
        resource_common["status"] = str(resource_row["execution_status"])
        if "dense" in category or modeled:
            resource_common["support_fingerprint"] = ""
        rows.append(
            _base_configuration_record(
                configuration_id=(
                    f"cfg:integrated:{iconfig['configuration_id']}:resource:{category}"
                ),
                evidence_tier=("modeled_resource" if modeled else "transpiled_executed_circuit"),
                **resource_common,
            )
        )
    convention_dir = root / "outputs/rectangular_convention_fix"
    convention = load_json(convention_dir / "convention_target_configuration.json")
    decision = load_json(convention_dir / "final_rectangular_decision.json")
    rows.append(
        _base_configuration_record(
            configuration_id="cfg:qsvt_convention:rectangular_fix_final",
            experiment_family="qsvt_convention_validation",
            ieee_case="ieee14",
            matrix_shape="27x82",
            matrix_fingerprint=decision["stage_summary"]["production"]["qsvt_operator_summary"][
                "sha256"
            ],
            polynomial_degree=_clean(convention.get("degree", 255)),
            phase_count=_clean(convention.get("phase_count", 256)),
            evidence_tier="qsvt_statevector",
            source_configuration_file=(
                convention_dir / "convention_target_configuration.json"
            ).relative_to(root),
            source_manifest=(convention_dir / "manifest.json").relative_to(root),
            status="completed",
        )
    )

    precision = root / "outputs/sparse_error_precision_study"
    pconfig = load_json(precision / "study_configuration.json")
    grid = _read_csv(precision / "statevector_precision_grid.csv")
    precision_support = load_json(precision / "sparse_support.json")
    precision_support_fp = stable_json_fingerprint(precision_support)
    for config_id, group in grid.groupby("configuration_id", sort=True):
        row = group.iloc[0]
        rows.append(
            _base_configuration_record(
                configuration_id=f"cfg:precision:{config_id}",
                experiment_family="precision_sensitivity",
                ieee_case="ieee14",
                matrix_shape="8x8",
                matrix_fingerprint=row["matrix_fingerprint"],
                support_fingerprint=precision_support_fp,
                residual_split_fingerprint=pconfig.get("baseline_residual_fingerprint", ""),
                functional_set_fingerprint=functional_fp,
                alpha=row["alpha"],
                beta=row["beta"],
                C=row["C"],
                polynomial_degree=int(row["degree"]),
                phase_count=32,
                phase_fingerprint=row["phase_fingerprint"],
                quantization_bits=f"value={row['value_bits']};phase={row['phase_bits']}",
                nonzeros=16,
                slot_budget=3,
                actual_slots=3,
                evidence_tier="qsvt_statevector",
                source_configuration_file=(precision / "study_configuration.json").relative_to(
                    root
                ),
                source_manifest=(precision / "manifest.json").relative_to(root),
                status=str(row["status"]),
                **{"lambda": row["lambda"]},
            )
        )

    rows.extend(_selector_configuration_rows(root, "output_aware_sparse_selection"))
    rows.extend(_selector_configuration_rows(root, "output_aware_generalization"))
    rows.extend(_selector_configuration_rows(root, "output_aware_structural_generalization"))

    frame = pd.DataFrame(rows)
    if frame["configuration_id"].duplicated().any():
        duplicates = frame.loc[frame["configuration_id"].duplicated(), "configuration_id"]
        raise ValueError(f"duplicate configuration IDs: {duplicates.tolist()[:10]}")
    invalid_tiers = sorted(set(frame["evidence_tier"]) - set(EVIDENCE_TIERS))
    if invalid_tiers:
        raise ValueError(f"invalid evidence tiers: {invalid_tiers}")
    frame = frame[CONFIGURATION_FIELDS].sort_values("configuration_id").reset_index(drop=True)
    atomic_write_csv(
        output_dir / "canonical_configuration_registry.csv",
        frame.to_dict(orient="records"),
        CONFIGURATION_FIELDS,
    )
    return frame


def _selector_configuration_rows(root: Path, family_name: str) -> list[dict[str, Any]]:
    family_dir = root / "outputs" / family_name
    study = load_json(family_dir / "study_configuration.json")
    manifest = (family_dir / "manifest.json").relative_to(root)
    source_config = (family_dir / "study_configuration.json").relative_to(root)
    structural = family_name == "output_aware_structural_generalization"
    development = family_name == "output_aware_sparse_selection"
    if development:
        contexts = {
            "development_ieee14_seed123_8x8": {
                "instance_id": "development_ieee14_seed123_8x8",
                "ieee_case": "ieee14",
                "matrix_shape": study["matrix_shape"],
                "matrix_fingerprint": study["matrix_fingerprint"],
                "regularization_alpha": study["physical_alpha"],
                "reference_beta": "",
                "lambda_ref": "",
                "structural_group_id": "",
            }
        }
        residual_fps = {
            "development_ieee14_seed123_8x8": _residual_split_fingerprint(
                family_dir / "residual_split.json"
            )
        }
        functional_fp = stable_json_fingerprint(study["functionals_values"])
    else:
        contexts, residual_fps = _instance_context(root, family_dir)
        functions = _read_csv(family_dir / "functional_registry.csv")
        functional_fp = ""
    support_registry = _read_csv(family_dir / "support_registry.csv")
    resource_path = family_dir / "resource_registry.csv"
    resource_map: dict[str, dict[str, Any]] = {}
    if resource_path.exists():
        for row in _read_csv(resource_path).to_dict(orient="records"):
            resource_map[str(row["support_id"])] = row
    rows: list[dict[str, Any]] = [
        _base_configuration_record(
            configuration_id=f"cfg:{family_name}:study",
            experiment_family=family_name,
            ieee_case="ieee14" if development else "multi_case",
            matrix_shape="8x8",
            matrix_fingerprint=(
                study.get("matrix_fingerprint", "")
                if development
                else study.get("instance_registry_fingerprint", "")
            ),
            residual_split_fingerprint=(
                residual_fps["development_ieee14_seed123_8x8"] if development else "multiple"
            ),
            functional_set_fingerprint=(functional_fp if development else "multiple"),
            alpha=study.get("physical_alpha", "multiple"),
            evidence_tier="diagnostic_only",
            source_configuration_file=source_config,
            source_manifest=manifest,
            status="completed",
        )
    ]
    for support in support_registry.to_dict(orient="records"):
        instance_id = (
            "development_ieee14_seed123_8x8" if development else str(support["instance_id"])
        )
        context = contexts[instance_id]
        if development:
            ffp = functional_fp
        else:
            ffp = _functional_set_fingerprint(functions[functions["instance_id"] == instance_id])
        status = str(support.get("status", ""))
        tier: EvidenceTier = "classical_exact" if status == "completed" else "excluded"
        resource = resource_map.get(str(support["support_id"]), {})
        sparse_fp = support.get("sparse_matrix_fingerprint", support.get("matrix_fingerprint", ""))
        row = _base_configuration_record(
            configuration_id=f"cfg:{family_name}:support:{support['support_id']}",
            experiment_family=family_name,
            ieee_case=context.get("ieee_case", "ieee14"),
            structural_group_id=context.get("structural_group_id", "") if structural else "",
            matrix_shape=parse_shape(context.get("matrix_shape", [8, 8])),
            matrix_fingerprint=context.get(
                "matrix_fingerprint", study.get("matrix_fingerprint", "")
            ),
            support_fingerprint=_clean(support.get("support_fingerprint", "")),
            residual_split_fingerprint=residual_fps.get(instance_id, ""),
            functional_set_fingerprint=ffp,
            alpha=context.get("regularization_alpha", study.get("physical_alpha", "")),
            beta=_clean(resource.get("native_beta", context.get("reference_beta", ""))),
            C="",
            polynomial_degree="",
            phase_count="",
            phase_fingerprint="",
            quantization_bits="exact_selected_values",
            nonzeros=_clean(support.get("actual_nonzeros", "")),
            slot_budget=_clean(support.get("slot_budget", "")),
            actual_slots=_clean(support.get("slot_count", resource.get("slot_count", ""))),
            evidence_tier=tier,
            source_configuration_file=source_config,
            source_manifest=manifest,
            status=status,
            **{"lambda": context.get("lambda_ref", "")},
        )
        row["_sparse_matrix_fingerprint"] = sparse_fp
        rows.append(row)

    qsvt_path = family_dir / "qsvt_validation_results.csv"
    if qsvt_path.exists():
        qsvt = _read_csv(qsvt_path)
        key_columns = (
            ["support_id", "common_design_fingerprint"]
            if development
            else ["instance_id", "support_id", "common_design_fingerprint"]
        )
        for _, group in qsvt.groupby(key_columns, dropna=False, sort=True):
            item = group.iloc[0]
            instance_id = (
                "development_ieee14_seed123_8x8" if development else str(item["instance_id"])
            )
            context = contexts[instance_id]
            source_support = support_registry[support_registry["support_id"] == item["support_id"]]
            support_fp_value = (
                source_support.iloc[0]["support_fingerprint"]
                if not source_support.empty
                else item.get("support_fingerprint", "")
            )
            if development:
                ffp = functional_fp
            else:
                ffp = _functional_set_fingerprint(
                    functions[functions["instance_id"] == instance_id]
                )
            rows.append(
                _base_configuration_record(
                    configuration_id=(
                        f"cfg:{family_name}:qsvt:"
                        f"{item.get('instance_id', instance_id)}:{item['support_id']}:"
                        f"{str(item['common_design_fingerprint'])[:16]}"
                    ),
                    experiment_family=(
                        "common_design_qsvt"
                        if not development
                        else "output_aware_common_design_qsvt"
                    ),
                    ieee_case=item.get("ieee_case", context.get("ieee_case", "ieee14")),
                    structural_group_id=(
                        context.get("structural_group_id", "") if structural else ""
                    ),
                    matrix_shape=parse_shape(context.get("matrix_shape", [8, 8])),
                    matrix_fingerprint=context.get(
                        "matrix_fingerprint", study.get("matrix_fingerprint", "")
                    ),
                    support_fingerprint=support_fp_value,
                    residual_split_fingerprint=residual_fps.get(instance_id, ""),
                    functional_set_fingerprint=ffp,
                    alpha=context.get("regularization_alpha", study.get("physical_alpha", "")),
                    beta=item["beta"],
                    C=item["C"],
                    polynomial_degree=int(item["degree"]),
                    phase_count=int(item["phase_count"]),
                    phase_fingerprint=item["phase_fingerprint"],
                    quantization_bits="exact_selected_values",
                    nonzeros="",
                    slot_budget=int(item["slot_budget"]),
                    actual_slots=int(item["slot_budget"]),
                    evidence_tier="qsvt_statevector",
                    source_configuration_file=source_config,
                    source_manifest=manifest,
                    status=str(item["status"]),
                    **{"lambda": item["lambda"]},
                )
            )

    if resource_path.exists():
        resource = _read_csv(resource_path)
        for item in resource.to_dict(orient="records"):
            instance_id = (
                "development_ieee14_seed123_8x8" if development else str(item["instance_id"])
            )
            context = contexts[instance_id]
            status = str(item.get("status", ""))
            executed = status == "completed" and str(
                item.get("resource_measurement", "")
            ).startswith("actual_")
            tier = "transpiled_executed_circuit" if executed else "excluded"
            rows.append(
                _base_configuration_record(
                    configuration_id=f"cfg:{family_name}:resource:{item['support_id']}",
                    experiment_family="resource_accounting",
                    ieee_case=context.get("ieee_case", "ieee14"),
                    structural_group_id=(
                        context.get("structural_group_id", "") if structural else ""
                    ),
                    matrix_shape=parse_shape(context.get("matrix_shape", [8, 8])),
                    matrix_fingerprint=context.get(
                        "matrix_fingerprint", study.get("matrix_fingerprint", "")
                    ),
                    support_fingerprint=_clean(item.get("support_fingerprint", "")),
                    residual_split_fingerprint=residual_fps.get(instance_id, ""),
                    functional_set_fingerprint=(
                        functional_fp
                        if development
                        else _functional_set_fingerprint(
                            functions[functions["instance_id"] == instance_id]
                        )
                    ),
                    alpha=context.get("regularization_alpha", study.get("physical_alpha", "")),
                    beta=_clean(item.get("native_beta", "")),
                    nonzeros=_clean(item.get("actual_nonzeros", "")),
                    slot_budget=_clean(item.get("slot_budget", "")),
                    actual_slots=_clean(item.get("slot_count", "")),
                    evidence_tier=tier,
                    source_configuration_file=source_config,
                    source_manifest=manifest,
                    status=status,
                    **{"lambda": context.get("lambda_ref", "")},
                )
            )

    finite_path = family_dir / "finite_shot_results.csv"
    if finite_path.exists():
        finite = _read_csv(finite_path)
        for index, item in finite.iterrows():
            instance_id = (
                "development_ieee14_seed123_8x8"
                if development
                else str(item.get("instance_id", ""))
            )
            context = contexts.get(instance_id, next(iter(contexts.values())))
            support_id = str(item.get("support_id", f"row{index}"))
            source_support = support_registry[support_registry["support_id"] == support_id]
            support_fp_value = (
                source_support.iloc[0].get("support_fingerprint", "")
                if not source_support.empty
                else ""
            )
            rows.append(
                _base_configuration_record(
                    configuration_id=f"cfg:{family_name}:finite-shot:{support_id}:row{index}",
                    experiment_family="finite_shot_selected_output",
                    ieee_case=context.get("ieee_case", item.get("ieee_case", "ieee14")),
                    structural_group_id=(
                        context.get("structural_group_id", "") if structural else ""
                    ),
                    matrix_shape=parse_shape(context.get("matrix_shape", [8, 8])),
                    matrix_fingerprint=context.get(
                        "matrix_fingerprint", study.get("matrix_fingerprint", "")
                    ),
                    support_fingerprint=_clean(support_fp_value),
                    residual_split_fingerprint=residual_fps.get(instance_id, ""),
                    functional_set_fingerprint=(
                        functional_fp
                        if development
                        else _functional_set_fingerprint(
                            functions[functions["instance_id"] == instance_id]
                        )
                    ),
                    alpha=context.get("regularization_alpha", study.get("physical_alpha", "")),
                    evidence_tier="diagnostic_only",
                    source_configuration_file=source_config,
                    source_manifest=manifest,
                    status=str(item.get("status", "skipped")),
                )
            )
    return rows


def _canonical_record(
    *,
    result_id: str,
    claim_family: str,
    experiment_family: str,
    configuration_id: str,
    source_artifact: Path,
    source_row_locator: str,
    evidence_tier: EvidenceTier,
    value: float | str,
    unit: str,
    limitation_code: str,
    matrix_fingerprint: str | None = None,
    support_fingerprint: str | None = None,
    residual_split_fingerprint: str | None = None,
    phase_fingerprint: str | None = None,
    polynomial_fingerprint: str | None = None,
    functional_id: str | None = None,
    ieee_case: str | None = None,
    structural_group_id: str | None = None,
    status: str = "completed",
    manuscript_eligible: bool = True,
    notes: str = "",
) -> CanonicalEvidenceRecord:
    return CanonicalEvidenceRecord(
        result_id=result_id,
        claim_family=claim_family,
        experiment_family=experiment_family,
        configuration_id=configuration_id,
        source_artifact=source_artifact,
        source_row_locator=source_row_locator,
        evidence_tier=evidence_tier,
        matrix_fingerprint=matrix_fingerprint,
        support_fingerprint=support_fingerprint,
        residual_split_fingerprint=residual_split_fingerprint,
        phase_fingerprint=phase_fingerprint,
        polynomial_fingerprint=polynomial_fingerprint,
        functional_id=functional_id,
        ieee_case=ieee_case,
        structural_group_id=structural_group_id,
        value=value,
        unit=unit,
        status=status,
        manuscript_eligible=manuscript_eligible,
        limitation_code=limitation_code,
        notes=notes,
    )


def _locate(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    return "row[" + ",".join(f"{key}={row.get(key)}" for key in keys) + "]"


def _integrated_result_records(root: Path) -> list[CanonicalEvidenceRecord]:
    family = root / "outputs/sparse_integrated_chain"
    config = load_json(family / "configuration.json")
    matrix = np.load(family / "matrix_quantized.npy")
    phases = np.load(family / "phases.npy")
    coefficients = np.load(family / "polynomial_coefficients.npy")
    state = _read_csv(family / "statevector_validation.csv")
    finite = _read_csv(family / "finite_shot_results.csv")
    resource = _read_csv(family / "resource_ledger.csv")
    sparse_resource = resource[
        resource["resource_category"] == "executed_small_scale_sparse_integrated"
    ].iloc[0]
    matrix_fp = config["matrix_fingerprint"]
    support_fp = _support_mask_fingerprint(matrix)
    phase_fp = stable_array_fingerprint(phases)
    polynomial_fp = stable_array_fingerprint(coefficients)
    limitations = (
        "small_scale_8x8;simulator_only;selected_output_only;no_hardware;"
        "no_quantum_speedup;no_ieee_scale_oracle;direct_rotation_architecture"
    )
    base = dict(
        claim_family="integrated_sparse_chain",
        experiment_family="integrated_sparse_chain",
        configuration_id=f"cfg:integrated:{config['configuration_id']}:statevector",
        evidence_tier=cast(EvidenceTier, "qsvt_statevector"),
        matrix_fingerprint=matrix_fp,
        support_fingerprint=support_fp,
        residual_split_fingerprint=config["residual_fingerprint"],
        phase_fingerprint=phase_fp,
        polynomial_fingerprint=polynomial_fp,
        ieee_case="ieee14",
        limitation_code=limitations,
    )
    records = [
        _canonical_record(
            result_id="res:integrated:matrix_shape",
            source_artifact=Path("outputs/sparse_integrated_chain/matrix_quantized.npy"),
            source_row_locator="array.shape",
            value=parse_shape(matrix.shape),
            unit="rows_x_columns",
            **base,
        ),
        _canonical_record(
            result_id="res:integrated:matrix_nonzeros",
            source_artifact=Path("outputs/sparse_integrated_chain/matrix_quantized.npy"),
            source_row_locator="count_nonzero(array)",
            value=float(np.count_nonzero(matrix)),
            unit="nonzeros",
            **base,
        ),
        _canonical_record(
            result_id="res:integrated:polynomial_degree",
            source_artifact=Path("outputs/sparse_integrated_chain/polynomial_coefficients.npy"),
            source_row_locator="array.size-1",
            value=float(coefficients.size - 1),
            unit="degree",
            **base,
        ),
        _canonical_record(
            result_id="res:integrated:phase_count",
            source_artifact=Path("outputs/sparse_integrated_chain/phases.npy"),
            source_row_locator="array.size",
            value=float(phases.size),
            unit="phases",
            **base,
        ),
        _canonical_record(
            result_id="res:integrated:postselection_probability",
            source_artifact=Path("outputs/sparse_integrated_chain/statevector_validation.csv"),
            source_row_locator="column[sparse_postselection_probability],unique",
            value=float(state["sparse_postselection_probability"].median()),
            unit="probability",
            **base,
        ),
        _canonical_record(
            result_id="res:integrated:sparse_dense_action_error",
            source_artifact=Path("outputs/sparse_integrated_chain/statevector_validation.csv"),
            source_row_locator="column[sparse_dense_action_relative_l2_error],max",
            value=float(state["sparse_dense_action_relative_l2_error"].max()),
            unit="relative_l2_error",
            **base,
        ),
        _canonical_record(
            result_id="res:integrated:qsvt_exact_svt_action_error",
            source_artifact=Path("outputs/sparse_integrated_chain/statevector_validation.csv"),
            source_row_locator="column[qsvt_exact_polynomial_svt_relative_l2_error],max",
            value=float(state["qsvt_exact_polynomial_svt_relative_l2_error"].max()),
            unit="relative_l2_error",
            **base,
        ),
        _canonical_record(
            result_id="res:integrated:selected_output_qsvt_error",
            source_artifact=Path("outputs/sparse_integrated_chain/statevector_validation.csv"),
            source_row_locator="column[selected_output_absolute_error_vs_quantized_ridge],max",
            value=float(state["selected_output_absolute_error_vs_quantized_ridge"].max()),
            unit="selected_output_absolute_error",
            **base,
        ),
    ]
    for _, row in state.iterrows():
        records.append(
            _canonical_record(
                result_id=f"res:integrated:selected_output:{row['functional_id']}",
                source_artifact=Path("outputs/sparse_integrated_chain/statevector_validation.csv"),
                source_row_locator=_locate(row, ["configuration_id", "functional_id"]),
                functional_id=str(row["functional_id"]),
                value=float(row["sparse_statevector_selected_output"]),
                unit="regularized_linear_update",
                **base,
            )
        )
    finite_subset = finite[
        (finite["chain_type"] == "sparse")
        & (finite["functional_id"] == "coordinate_e0")
        & (finite["shots_attempted"] == 1_000_000)
    ]
    records.append(
        _canonical_record(
            result_id="res:integrated:finite_shot_coordinate_1e6",
            claim_family="finite_shot_selected_output",
            experiment_family="integrated_sparse_chain",
            configuration_id=f"cfg:integrated:{config['configuration_id']}:finite-shot",
            source_artifact=Path("outputs/sparse_integrated_chain/finite_shot_results.csv"),
            source_row_locator="rows[chain_type=sparse,functional_id=coordinate_e0,shots_attempted=1000000],mean(selected_output_estimate)",
            evidence_tier="finite_shot_simulation",
            matrix_fingerprint=matrix_fp,
            support_fingerprint=support_fp,
            residual_split_fingerprint=config["residual_fingerprint"],
            phase_fingerprint=phase_fp,
            polynomial_fingerprint=polynomial_fp,
            functional_id="coordinate_e0",
            ieee_case="ieee14",
            value=float(finite_subset["selected_output_estimate"].mean()),
            unit="regularized_linear_update",
            limitation_code=limitations,
            notes="One million attempted shots per seed; ten simulator seeds.",
        )
    )
    for column, unit in (
        ("transpiled_gate_count", "gates"),
        ("transpiled_depth", "depth"),
        ("toffoli_count", "toffoli_gates"),
        ("controlled_rotation_count", "controlled_rotations"),
    ):
        records.append(
            _canonical_record(
                result_id=f"res:integrated:resource:{column}",
                claim_family="resource_accounting",
                experiment_family="integrated_sparse_chain",
                configuration_id=f"cfg:integrated:{config['configuration_id']}:resource",
                source_artifact=Path("outputs/sparse_integrated_chain/resource_ledger.csv"),
                source_row_locator="row[resource_category=executed_small_scale_sparse_integrated]",
                evidence_tier="transpiled_executed_circuit",
                matrix_fingerprint=matrix_fp,
                support_fingerprint=support_fp,
                residual_split_fingerprint=config["residual_fingerprint"],
                phase_fingerprint=phase_fp,
                polynomial_fingerprint=polynomial_fp,
                ieee_case="ieee14",
                value=float(sparse_resource[column]),
                unit=unit,
                limitation_code=limitations,
            )
        )
    return records


def _convention_result_records(root: Path) -> list[CanonicalEvidenceRecord]:
    path = root / "outputs/rectangular_convention_fix/final_rectangular_decision.json"
    payload = load_json(path)
    production = payload["stage_summary"]["production"]
    return [
        _canonical_record(
            result_id="res:qsvt_convention:final_decision",
            claim_family="qsvt_convention_validation",
            experiment_family="qsvt_convention_validation",
            configuration_id="cfg:qsvt_convention:rectangular_fix_final",
            source_artifact=path.relative_to(root),
            source_row_locator="$.decision",
            evidence_tier="qsvt_statevector",
            matrix_fingerprint=production["qsvt_operator_summary"]["sha256"],
            value=str(payload["decision"]),
            unit="status",
            status="completed",
            manuscript_eligible=True,
            limitation_code="simulator_only;no_hardware;no_quantum_speedup",
            notes="Convention-correct production statevector and shot validation; not hardware.",
        ),
        _canonical_record(
            result_id="res:qsvt_convention:selected_relative_error",
            claim_family="qsvt_convention_validation",
            experiment_family="qsvt_convention_validation",
            configuration_id="cfg:qsvt_convention:rectangular_fix_final",
            source_artifact=path.relative_to(root),
            source_row_locator="$.stage_summary.production.selected_rel_exact",
            evidence_tier="qsvt_statevector",
            matrix_fingerprint=production["qsvt_operator_summary"]["sha256"],
            value=float(production["selected_rel_exact"]),
            unit="relative_error",
            limitation_code="simulator_only;selected_output_only;no_hardware;no_quantum_speedup",
        ),
    ]


def _precision_result_records(root: Path) -> list[CanonicalEvidenceRecord]:
    family = root / "outputs/sparse_error_precision_study"
    grid = _read_csv(family / "statevector_precision_grid.csv")
    config = load_json(family / "study_configuration.json")
    support_fp = stable_json_fingerprint(load_json(family / "sparse_support.json"))
    records: list[CanonicalEvidenceRecord] = []
    limitations = (
        "small_scale_8x8;simulator_only;selected_output_only;no_hardware;"
        "no_quantum_speedup;direct_rotation_architecture"
    )
    metrics = (
        ("sparsification_absolute_error", "error_source_decomposition"),
        ("quantization_absolute_error", "error_source_decomposition"),
        ("qsvt_absolute_error", "error_source_decomposition"),
        ("phase_absolute_error", "precision_sensitivity"),
        ("total_statevector_error_vs_original", "precision_sensitivity"),
        ("postselection_probability", "precision_sensitivity"),
    )
    for _, row in grid.iterrows():
        config_id = f"cfg:precision:{row['configuration_id']}"
        locator = _locate(row, ["configuration_id", "value_bits", "phase_bits", "functional_id"])
        for metric, claim in metrics:
            records.append(
                _canonical_record(
                    result_id=(
                        f"res:precision:{row['configuration_id']}:{row['functional_id']}:{metric}"
                    ),
                    claim_family=claim,
                    experiment_family="precision_sensitivity",
                    configuration_id=config_id,
                    source_artifact=Path(
                        "outputs/sparse_error_precision_study/statevector_precision_grid.csv"
                    ),
                    source_row_locator=locator,
                    evidence_tier="qsvt_statevector",
                    matrix_fingerprint=str(row["matrix_fingerprint"]),
                    support_fingerprint=support_fp,
                    residual_split_fingerprint=config.get("baseline_residual_fingerprint", ""),
                    phase_fingerprint=str(row["phase_fingerprint"]),
                    polynomial_fingerprint=None,
                    functional_id=str(row["functional_id"]),
                    ieee_case="ieee14",
                    value=float(row[metric]),
                    unit="probability"
                    if metric == "postselection_probability"
                    else "absolute_error",
                    status=str(row["status"]),
                    manuscript_eligible=str(row["status"]) == "completed",
                    limitation_code=limitations,
                )
            )
    decomposition = _read_csv(family / "error_decomposition.csv")
    completed = decomposition[decomposition["status"] == "completed"]
    max_residual = float(completed["cumulative_identity_residual"].abs().max())
    records.extend(
        [
            _canonical_record(
                result_id="res:error_decomposition:max_identity_residual",
                claim_family="error_source_decomposition",
                experiment_family="precision_sensitivity",
                configuration_id=f"cfg:precision:{grid.iloc[0]['configuration_id']}",
                source_artifact=Path(
                    "outputs/sparse_error_precision_study/error_decomposition.csv"
                ),
                source_row_locator="rows[status=completed],max(abs(cumulative_identity_residual))",
                evidence_tier="classical_exact",
                matrix_fingerprint=str(grid.iloc[0]["matrix_fingerprint"]),
                support_fingerprint=support_fp,
                residual_split_fingerprint=config.get("baseline_residual_fingerprint", ""),
                value=max_residual,
                unit="absolute_residual",
                limitation_code=limitations,
            ),
            _canonical_record(
                result_id="res:error_decomposition:triangle_bound_all",
                claim_family="error_source_decomposition",
                experiment_family="precision_sensitivity",
                configuration_id=f"cfg:precision:{grid.iloc[0]['configuration_id']}",
                source_artifact=Path(
                    "outputs/sparse_error_precision_study/error_decomposition.csv"
                ),
                source_row_locator="rows[status=completed],all(triangle_bound_satisfied)",
                evidence_tier="classical_exact",
                matrix_fingerprint=str(grid.iloc[0]["matrix_fingerprint"]),
                support_fingerprint=support_fp,
                residual_split_fingerprint=config.get("baseline_residual_fingerprint", ""),
                value=str(bool(completed["triangle_bound_satisfied"].all())).lower(),
                unit="boolean",
                limitation_code=limitations,
            ),
        ]
    )
    source_columns = {
        "sparsification": "sparsification_absolute_error",
        "value_quantization": "quantization_absolute_error",
        "qsvt": "qsvt_absolute_error",
        "sampling": "sampling_absolute_error",
    }
    for functional_id, group in completed.groupby("functional_id", sort=True):
        medians = {
            name: float(pd.to_numeric(group[column], errors="coerce").median())
            for name, column in source_columns.items()
        }
        dominant = max(medians, key=medians.__getitem__)
        records.append(
            _canonical_record(
                result_id=f"res:error_decomposition:dominant:{functional_id}",
                claim_family="error_source_decomposition",
                experiment_family="precision_sensitivity",
                configuration_id=f"cfg:precision:{grid.iloc[0]['configuration_id']}",
                source_artifact=Path(
                    "outputs/sparse_error_precision_study/error_decomposition.csv"
                ),
                source_row_locator=(
                    f"rows[functional_id={functional_id},status=completed],"
                    "argmax(median absolute component)"
                ),
                evidence_tier="classical_exact",
                matrix_fingerprint=str(grid.iloc[0]["matrix_fingerprint"]),
                support_fingerprint=support_fp,
                residual_split_fingerprint=config.get("baseline_residual_fingerprint", ""),
                functional_id=str(functional_id),
                ieee_case="ieee14",
                value=dominant,
                unit="error_source",
                limitation_code=limitations,
            )
        )
    return records


def _read_primary_heldout(
    path: Path,
    baseline: str,
    candidate: str,
    k_budget: int,
    slot_budget: int,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=200_000, low_memory=False):
        mask = (
            chunk["selector"].isin([baseline, candidate])
            & (chunk["k_budget"] == k_budget)
            & (chunk["slot_budget"] == slot_budget)
        )
        if "status" in chunk.columns:
            mask &= chunk["status"] == "completed"
        if mask.any():
            chunks.append(chunk.loc[mask].copy())
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _selector_result_records(root: Path, family_name: str) -> list[CanonicalEvidenceRecord]:
    family = root / "outputs" / family_name
    study = load_json(family / "study_configuration.json")
    development = family_name == "output_aware_sparse_selection"
    structural = family_name == "output_aware_structural_generalization"
    limitations = (
        "small_scale_8x8;simulator_only;selected_output_only;controlled_generated_residuals;"
        "no_field_data;no_hardware;no_quantum_speedup;no_ieee_scale_oracle;"
        "local_sensitivity_model;local_refinement_only;no_iterative_nonlinear_integration"
    )
    if not development:
        limitations += ";fixed_three_topologies;case_dependence;functional_dependence"
    if structural:
        limitations += ";aggregate_weakness;ieee57_mixed_result;rank_deficient_blocks"
    records: list[CanonicalEvidenceRecord] = []
    study_config_id = f"cfg:{family_name}:study"
    study_matrix_fp = str(
        study.get("matrix_fingerprint", study.get("instance_registry_fingerprint", ""))
    )

    if development:
        primary = _read_primary_heldout(
            family / "heldout_results.csv",
            "balanced_magnitude",
            "sensitivity_initial_mean",
            16,
            3,
        )
        for selector, group in primary.groupby("selector", sort=True):
            records.append(
                _canonical_record(
                    result_id=f"res:development:heldout:{selector}:median_normalized_error",
                    claim_family="output_aware_selection",
                    experiment_family=family_name,
                    configuration_id=study_config_id,
                    source_artifact=Path(
                        "outputs/output_aware_sparse_selection/heldout_results.csv"
                    ),
                    source_row_locator=(
                        f"rows[selector={selector},k_budget=16,slot_budget=3,status=completed],"
                        "median(normalized_error)"
                    ),
                    evidence_tier="classical_exact",
                    matrix_fingerprint=study["matrix_fingerprint"],
                    residual_split_fingerprint=_residual_split_fingerprint(
                        family / "residual_split.json"
                    ),
                    ieee_case="ieee14",
                    value=float(group["normalized_error"].median()),
                    unit="median_normalized_error",
                    limitation_code=limitations,
                )
            )
        functional = (
            primary.groupby(["selector", "functional_id"])["normalized_error"]
            .median()
            .unstack("selector")
        )
        negative_transfer = int(
            (functional["sensitivity_initial_mean"] > functional["balanced_magnitude"] * 1.01).sum()
        )
        records.append(
            _canonical_record(
                result_id="res:development:negative_transfer:functional_count",
                claim_family="functional_dependence",
                experiment_family=family_name,
                configuration_id=study_config_id,
                source_artifact=Path("outputs/output_aware_sparse_selection/heldout_results.csv"),
                source_row_locator=(
                    "primary rows,functional medians,count(candidate > 1.01*baseline)"
                ),
                evidence_tier="classical_exact",
                matrix_fingerprint=study["matrix_fingerprint"],
                residual_split_fingerprint=_residual_split_fingerprint(
                    family / "residual_split.json"
                ),
                ieee_case="ieee14",
                value=float(negative_transfer),
                unit="functionals",
                limitation_code=limitations + ";functional_dependence",
            )
        )
    else:
        instance_registry = _read_csv(family / "instance_registry.csv")
        contexts = {
            str(row["instance_id"]): row for row in instance_registry.to_dict(orient="records")
        }
        if structural:
            pairs_path = family / "structural_primary_matched_pairs.csv"
            pairs = _read_csv(pairs_path)
            group_fingerprints = {
                group_id: stable_json_fingerprint(
                    sorted(
                        instance_registry.loc[
                            instance_registry["structural_group_id"] == group_id,
                            "matrix_fingerprint",
                        ].astype(str)
                    )
                )
                for group_id in pairs["structural_group_id"].astype(str)
            }
            for _, row in pairs.iterrows():
                group_id = str(row["structural_group_id"])
                records.append(
                    _canonical_record(
                        result_id=f"res:structural:primary:{group_id}:paired_difference",
                        claim_family="structural_generalization",
                        experiment_family=family_name,
                        configuration_id=study_config_id,
                        source_artifact=pairs_path.relative_to(root),
                        source_row_locator=_locate(row, ["structural_group_id"]),
                        evidence_tier="classical_exact",
                        matrix_fingerprint=group_fingerprints[group_id],
                        ieee_case=str(row["ieee_case"]),
                        structural_group_id=group_id,
                        value=float(row["paired_difference_candidate_minus_baseline"]),
                        unit="paired_normalized_error_difference",
                        limitation_code=limitations,
                        notes=f"Frozen outcome={row['outcome']}; ties remain in primary analysis.",
                    )
                )
            functional_path = family / "structural_primary_functional_pairs.csv"
            functional = _read_csv(functional_path)
            for _, row in functional.iterrows():
                group_id = str(row["structural_group_id"])
                records.append(
                    _canonical_record(
                        result_id=(
                            f"res:structural:functional:{group_id}:{row['functional_id']}:"
                            "paired_difference"
                        ),
                        claim_family="functional_dependence",
                        experiment_family=family_name,
                        configuration_id=study_config_id,
                        source_artifact=functional_path.relative_to(root),
                        source_row_locator=_locate(row, ["structural_group_id", "functional_id"]),
                        evidence_tier="classical_exact",
                        matrix_fingerprint=group_fingerprints[group_id],
                        functional_id=str(row["functional_id"]),
                        ieee_case=str(row["ieee_case"]),
                        structural_group_id=group_id,
                        value=float(row["paired_difference_candidate_minus_baseline"]),
                        unit="paired_normalized_error_difference",
                        limitation_code=limitations,
                        notes=f"Frozen functional outcome={row['outcome']}.",
                    )
                )
            stability_path = family / "support_stability_summary.csv"
            stability = _read_csv(stability_path)
            for _, row in stability.iterrows():
                records.append(
                    _canonical_record(
                        result_id=f"res:structural:stability:{row['selector']}:median_jaccard",
                        claim_family="support_stability",
                        experiment_family=family_name,
                        configuration_id=study_config_id,
                        source_artifact=stability_path.relative_to(root),
                        source_row_locator=_locate(row, ["selector"]),
                        evidence_tier="diagnostic_only",
                        matrix_fingerprint=study_matrix_fp,
                        value=float(row["median_jaccard"]),
                        unit="jaccard_similarity",
                        limitation_code=limitations + ";local_sensitivity_model",
                    )
                )
        else:
            pairs_path = family / "generalization_matched_pairs.csv"
            pairs = _read_csv(pairs_path)
            pairs = pairs[pairs["comparison_label"] == "primary"]
            for _, row in pairs.iterrows():
                instance_id = str(row["instance_id"])
                context = contexts[instance_id]
                records.append(
                    _canonical_record(
                        result_id=f"res:generalization:primary:{instance_id}:paired_difference",
                        claim_family="heldout_generalization",
                        experiment_family=family_name,
                        configuration_id=study_config_id,
                        source_artifact=pairs_path.relative_to(root),
                        source_row_locator=_locate(row, ["instance_id"]),
                        evidence_tier="classical_exact",
                        matrix_fingerprint=str(context["matrix_fingerprint"]),
                        ieee_case=str(row["ieee_case"]),
                        value=float(row["paired_difference_candidate_minus_baseline"]),
                        unit="paired_normalized_error_difference",
                        limitation_code=limitations,
                        notes=f"Frozen outcome={row['outcome']}.",
                    )
                )

    qsvt_path = family / "qsvt_validation_results.csv"
    if qsvt_path.exists():
        qsvt = _read_csv(qsvt_path)
        context_map: dict[str, Mapping[str, Any]] = {}
        if not development:
            instance_registry = _read_csv(family / "instance_registry.csv")
            context_map = {
                str(row["instance_id"]): row for row in instance_registry.to_dict(orient="records")
            }
        for _, row in qsvt.iterrows():
            iid = str(row.get("instance_id", "development_ieee14_seed123_8x8"))
            context = context_map.get(iid, {})
            if development:
                source_support = _read_csv(family / "support_registry.csv")
                source_support = source_support[source_support["support_id"] == row["support_id"]]
                support_fp_value = (
                    str(source_support.iloc[0]["support_fingerprint"])
                    if not source_support.empty
                    else ""
                )
                qsvt_error_column = "selected_output_qsvt_error_vs_sparse_ridge"
                locator_keys = ["support_id", "functional_id"]
            else:
                support_fp_value = str(row["support_fingerprint"])
                qsvt_error_column = "qsvt_error_on_sparse_matrix"
                locator_keys = ["instance_id", "support_id", "functional_id"]
            configuration_id = (
                f"cfg:{family_name}:qsvt:{row.get('instance_id', iid)}:{row['support_id']}:"
                f"{str(row['common_design_fingerprint'])[:16]}"
            )
            records.append(
                _canonical_record(
                    result_id=(
                        f"res:{family_name}:qsvt:{iid}:{row['support_id']}:"
                        f"{row['functional_id']}:error"
                    ),
                    claim_family="common_design_qsvt",
                    experiment_family=family_name,
                    configuration_id=configuration_id,
                    source_artifact=qsvt_path.relative_to(root),
                    source_row_locator=_locate(row, locator_keys),
                    evidence_tier="qsvt_statevector",
                    matrix_fingerprint=str(
                        context.get("matrix_fingerprint", study.get("matrix_fingerprint", ""))
                    ),
                    support_fingerprint=support_fp_value,
                    residual_split_fingerprint=(
                        _residual_split_fingerprint(family / "residual_split.json")
                        if development
                        else _residual_split_fingerprint(family / "residual_splits" / f"{iid}.json")
                    ),
                    phase_fingerprint=str(row["phase_fingerprint"]),
                    polynomial_fingerprint=str(row["polynomial_fingerprint"]),
                    functional_id=str(row["functional_id"]),
                    ieee_case=str(row.get("ieee_case", "ieee14")),
                    structural_group_id=(
                        str(context.get("structural_group_id", "")) or None if structural else None
                    ),
                    value=float(row[qsvt_error_column]),
                    unit="selected_output_absolute_error",
                    status=str(row["status"]),
                    manuscript_eligible=str(row["status"]) == "completed",
                    limitation_code=limitations
                    + ";simulator_only;selected_output_only;no_hardware",
                    notes="Common design; no per-support phase refit.",
                )
            )

    certificate_path = (
        family / "certificate_summary.csv"
        if development
        else family / "certificate_case_summary.csv"
    )
    if certificate_path.exists():
        certificate = _read_csv(certificate_path)
        coverage_column = "coverage"
        if coverage_column not in certificate.columns:
            coverage_column = "certificate_coverage"
        records.append(
            _canonical_record(
                result_id=f"res:{family_name}:certificate:coverage",
                claim_family="certificate_validity",
                experiment_family=family_name,
                configuration_id=study_config_id,
                source_artifact=certificate_path.relative_to(root),
                source_row_locator=f"column[{coverage_column}],minimum",
                evidence_tier="classical_exact",
                matrix_fingerprint=study_matrix_fp,
                value=float(certificate[coverage_column].min()),
                unit="coverage_fraction",
                limitation_code=limitations + ";certificate_looseness",
            )
        )

    resource_path = family / "resource_registry.csv"
    if resource_path.exists():
        resources = _read_csv(resource_path)
        completed = resources[resources["status"] == "completed"]
        for column, unit in (
            ("signal_unitary_gate_count", "gates"),
            ("signal_unitary_depth", "depth"),
            ("controlled_rotations", "controlled_rotations"),
            ("wrapper_reconstruction_error", "absolute_error"),
        ):
            if column not in completed:
                continue
            records.extend(
                [
                    _canonical_record(
                        result_id=f"res:{family_name}:resource:{column}:minimum",
                        claim_family="resource_accounting",
                        experiment_family=family_name,
                        configuration_id=study_config_id,
                        source_artifact=resource_path.relative_to(root),
                        source_row_locator=f"rows[status=completed],min({column})",
                        evidence_tier="transpiled_executed_circuit",
                        matrix_fingerprint=study_matrix_fp,
                        value=float(pd.to_numeric(completed[column]).min()),
                        unit=unit,
                        limitation_code=limitations
                        + ";direct_rotation_architecture;no_ieee_scale_oracle",
                    ),
                    _canonical_record(
                        result_id=f"res:{family_name}:resource:{column}:maximum",
                        claim_family="resource_accounting",
                        experiment_family=family_name,
                        configuration_id=study_config_id,
                        source_artifact=resource_path.relative_to(root),
                        source_row_locator=f"rows[status=completed],max({column})",
                        evidence_tier="transpiled_executed_circuit",
                        matrix_fingerprint=study_matrix_fp,
                        value=float(pd.to_numeric(completed[column]).max()),
                        unit=unit,
                        limitation_code=limitations
                        + ";direct_rotation_architecture;no_ieee_scale_oracle",
                    ),
                ]
            )

    finite_path = family / "finite_shot_results.csv"
    if finite_path.exists() and not development:
        finite = _read_csv(finite_path)
        status_value = ";".join(sorted(finite["status"].astype(str).unique()))
        records.append(
            _canonical_record(
                result_id=f"res:{family_name}:finite_shot:status",
                claim_family="negative_scalability_boundary",
                experiment_family=family_name,
                configuration_id=study_config_id,
                source_artifact=finite_path.relative_to(root),
                source_row_locator="column[status],unique",
                evidence_tier="diagnostic_only",
                matrix_fingerprint=study_matrix_fp,
                value=status_value,
                unit="status",
                status="negative_evidence",
                manuscript_eligible=True,
                limitation_code=limitations
                + (";finite_shot_structural_campaign_skipped" if structural else ";simulator_only"),
                notes="Skipped rows are not finite-shot measurements.",
            )
        )
    return records


def _headline_summary_result_records(root: Path) -> list[CanonicalEvidenceRecord]:
    records: list[CanonicalEvidenceRecord] = []
    general = root / "outputs/output_aware_generalization"
    gpairs = _read_csv(general / "generalization_matched_pairs.csv")
    gpairs = gpairs[gpairs["comparison_label"] == "primary"]
    gstudy = load_json(general / "study_configuration.json")
    glim = (
        "small_scale_8x8;controlled_generated_residuals;fixed_three_topologies;"
        "case_dependence;functional_dependence;no_field_data;no_hardware;"
        "no_quantum_speedup"
    )
    for outcome in ("win", "tie", "loss"):
        records.append(
            _canonical_record(
                result_id=f"res:generalization:primary:{outcome}_count",
                claim_family="heldout_generalization",
                experiment_family="output_aware_generalization",
                configuration_id="cfg:output_aware_generalization:study",
                source_artifact=Path(
                    "outputs/output_aware_generalization/generalization_matched_pairs.csv"
                ),
                source_row_locator=f"count(outcome={outcome})",
                evidence_tier="classical_exact",
                matrix_fingerprint=str(gstudy["instance_registry_fingerprint"]),
                value=float((gpairs["outcome"] == outcome).sum()),
                unit="instances",
                limitation_code=glim,
            )
        )
    structural = root / "outputs/output_aware_structural_generalization"
    spairs = _read_csv(structural / "structural_primary_matched_pairs.csv")
    stest = load_json(structural / "structural_primary_test.json")
    slim = glim + ";aggregate_weakness;ieee57_mixed_result;rank_deficient_blocks"
    for outcome in ("win", "tie", "loss"):
        records.append(
            _canonical_record(
                result_id=f"res:structural:primary:{outcome}_count",
                claim_family="structural_generalization",
                experiment_family="output_aware_structural_generalization",
                configuration_id="cfg:output_aware_structural_generalization:study",
                source_artifact=Path(
                    "outputs/output_aware_structural_generalization/structural_primary_matched_pairs.csv"
                ),
                source_row_locator=f"count(outcome={outcome})",
                evidence_tier="classical_exact",
                matrix_fingerprint=str(
                    load_json(structural / "study_configuration.json")[
                        "structural_group_registry_fingerprint"
                    ]
                ),
                value=float((spairs["outcome"] == outcome).sum()),
                unit="structural_groups",
                limitation_code=slim,
            )
        )
    functional = stest["functional_case_win_tie_loss"]["ieee57:aggregate_e0_to_e3"]
    records.append(
        _canonical_record(
            result_id="res:structural:ieee57:aggregate:win_tie_loss",
            claim_family="functional_dependence",
            experiment_family="output_aware_structural_generalization",
            configuration_id="cfg:output_aware_structural_generalization:study",
            source_artifact=Path(
                "outputs/output_aware_structural_generalization/structural_primary_test.json"
            ),
            source_row_locator="$.functional_case_win_tie_loss['ieee57:aggregate_e0_to_e3']",
            evidence_tier="classical_exact",
            matrix_fingerprint=str(
                load_json(structural / "study_configuration.json")[
                    "structural_group_registry_fingerprint"
                ]
            ),
            functional_id="aggregate_e0_to_e3",
            ieee_case="ieee57",
            value=f"{functional['win']}/{functional['tie']}/{functional['loss']}",
            unit="wins_ties_losses",
            limitation_code=slim,
            notes="Aggregate-functional weakness remains visible.",
        )
    )
    return records


def build_result_registry(root: Path, output_dir: Path) -> pd.DataFrame:
    """Build the canonical result registry from authoritative machine-readable rows."""

    records: list[CanonicalEvidenceRecord] = []
    records.extend(_convention_result_records(root))
    records.extend(_integrated_result_records(root))
    records.extend(_precision_result_records(root))
    for family in (
        "output_aware_sparse_selection",
        "output_aware_generalization",
        "output_aware_structural_generalization",
    ):
        records.extend(_selector_result_records(root, family))
    records.extend(_headline_summary_result_records(root))
    # Explicitly retain an ineligible superseded summary example so eligibility
    # rules are auditable rather than implicit.
    records.append(
        _canonical_record(
            result_id="res:excluded:structural_markdown_summary",
            claim_family="excluded_claims",
            experiment_family="output_aware_structural_generalization",
            configuration_id="cfg:output_aware_structural_generalization:study",
            source_artifact=Path("outputs/output_aware_structural_generalization/summary.md"),
            source_row_locator="entire_markdown_summary",
            evidence_tier="excluded",
            matrix_fingerprint=None,
            value="superseded_as_canonical_numeric_source",
            unit="status",
            status="excluded",
            manuscript_eligible=False,
            limitation_code="missing_low_level_row_provenance",
            notes="The summary is useful for orientation but is not a canonical numeric source.",
        )
    )
    ids = [record.result_id for record in records]
    if len(ids) != len(set(ids)):
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        raise ValueError(f"duplicate result IDs: {duplicates[:10]}")
    rows = [record.csv_row() for record in sorted(records, key=lambda item: item.result_id)]
    atomic_write_csv(output_dir / "canonical_result_registry.csv", rows, RESULT_FIELDS)
    return pd.DataFrame(rows, columns=RESULT_FIELDS)


_LIMITATION_DEFINITIONS: dict[str, tuple[str, str, str]] = {
    "small_scale_8x8": (
        "Evidence is restricted to explicitly enumerated 8x8 selected blocks.",
        "small_scale",
        "scalable_ieee_execution",
    ),
    "simulator_only": (
        "Quantum-circuit evidence was obtained on classical simulators.",
        "simulator_only",
        "hardware_demonstrated",
    ),
    "selected_output_only": (
        "Validation targets selected linear functionals rather than full-vector readout.",
        "selected_output",
        "full_state_readout_advantage",
    ),
    "controlled_generated_residuals": (
        "Residual tasks are controlled generated PSSE-derived workloads.",
        "controlled_psse_derived_workload",
        "field_validated",
    ),
    "no_field_data": (
        "No field PMU or SCADA dataset is used in these campaigns.",
        "no_field_data",
        "field_validated",
    ),
    "no_hardware": (
        "No quantum-hardware execution was performed.",
        "no_hardware_execution",
        "hardware_demonstrated",
    ),
    "no_quantum_speedup": (
        "No quantum speedup is claimed or measured.",
        "no_quantum_speedup",
        "quantum_advantage",
    ),
    "no_ieee_scale_oracle": (
        "No end-to-end sparse oracle was executed at full IEEE case scale.",
        "no_ieee_scale_sparse_oracle",
        "scalable_ieee_execution",
    ),
    "direct_rotation_architecture": (
        "Resource counts describe the enumerated direct-rotation wrapper architecture.",
        "direct_rotation_architecture",
        "architecture_independent_resource_scaling",
    ),
    "finite_shot_structural_campaign_skipped": (
        "The structural finite-shot campaign was skipped under its frozen cost ceiling.",
        "finite_shot_skipped",
        "structural_finite_shot_measured",
    ),
    "case_dependence": (
        "Selector performance varies by IEEE case.",
        "case_dependent",
        "universal_generalization",
    ),
    "functional_dependence": (
        "Selector performance varies across selected functionals.",
        "functional_dependent",
        "always_outperforms_magnitude",
    ),
    "aggregate_weakness": (
        "Aggregate-output preservation is the weakest structural functional family.",
        "functional_dependent",
        "uniform_functional_superiority",
    ),
    "ieee57_mixed_result": (
        "IEEE-57 structural outcomes are mixed.",
        "case_dependent",
        "universal_selector_superiority",
    ),
    "certificate_looseness": (
        "Perturbation certificates cover observed errors but can be extremely loose.",
        "conservative_certificate",
        "tight_certificate",
    ),
    "local_sensitivity_model": (
        "Scores use a local analytical Ridge selected-output sensitivity model.",
        "local_sensitivity_model",
        "global_nonlinear_optimality",
    ),
    "local_refinement_only": (
        "Refined supports use bounded local exchanges rather than global refinement guarantees.",
        "local_refinement",
        "globally_optimal_refinement",
    ),
    "rank_deficient_blocks": (
        "Some selected blocks are numerically rank deficient before Ridge regularization.",
        "regularized_linear_update",
        "finite_raw_condition_number_for_all_blocks",
    ),
    "fixed_three_topologies": (
        "Topology coverage is limited to IEEE-14, IEEE-30, and IEEE-57.",
        "fixed_three_topologies",
        "universal_topology_generalization",
    ),
    "no_iterative_nonlinear_integration": (
        "The selected-output update was not integrated into a complete nonlinear AC iteration.",
        "regularized_linear_update",
        "full_nonlinear_psse_solution",
    ),
}


def build_limitation_registry(root: Path, output_dir: Path) -> pd.DataFrame:
    results = _read_csv(output_dir / "canonical_result_registry.csv", dtype=str)
    rows: list[dict[str, Any]] = []
    for limitation_id, (description, qualifier, prohibited) in sorted(
        _LIMITATION_DEFINITIONS.items()
    ):
        affected = results[
            results["limitation_code"]
            .fillna("")
            .map(lambda value, target=limitation_id: target in str(value).split(";"))
        ]["result_id"].tolist()
        rows.append(
            {
                "limitation_id": limitation_id,
                "category": limitation_id,
                "description": description,
                "affected_result_ids": ";".join(affected),
                "required_manuscript_qualifier": qualifier,
                "prohibited_claim": prohibited,
                "evidence_source": "canonical_result_registry.csv",
                "status": "active",
            }
        )
    frame = pd.DataFrame(rows, columns=LIMITATION_FIELDS)
    atomic_write_csv(
        output_dir / "canonical_limitation_registry.csv",
        frame.to_dict(orient="records"),
        LIMITATION_FIELDS,
    )
    return frame


_CLAIM_DEFINITIONS: dict[str, tuple[str, str, str]] = {
    "qsvt_convention_validation": (
        "The real rectangular-QSVT convention is validated against exact low-degree "
        "and production references.",
        "simulator_only;selected_output",
        "hardware_demonstrated;quantum_advantage",
    ),
    "integrated_sparse_chain": (
        "The frozen 8x8 sparse selected-output chain reproduces its exact sparse "
        "references within reported tolerances.",
        "small_scale;simulator_only;selected_output",
        "scalable_ieee_execution;quantum_advantage",
    ),
    "finite_shot_selected_output": (
        "Finite-shot simulator readout is available for the frozen integrated "
        "selected-output chain.",
        "small_scale;simulator_only;selected_output",
        "hardware_demonstrated",
    ),
    "error_source_decomposition": (
        "The frozen selected-output error is additively decomposed into recorded "
        "numerical components.",
        "small_scale;selected_output",
        "universal_error_hierarchy",
    ),
    "precision_sensitivity": (
        "Value and phase precision sensitivity is quantified for the frozen 8x8 configuration.",
        "small_scale;simulator_only",
        "hardware_precision_requirement",
    ),
    "output_aware_selection": (
        "Output-aware support selection is evaluated on controlled PSSE-derived residual tasks.",
        "controlled_psse_derived_workload;local_sensitivity_model",
        "always_outperforms_magnitude",
    ),
    "heldout_generalization": (
        "Held-out evidence favors sensitivity selection in the frozen multi-instance "
        "comparison while retaining losses.",
        "case_dependent;functional_dependent;no_field_data",
        "universal_generalization;always_outperforms_magnitude",
    ),
    "structural_generalization": (
        "Structurally distinct 8x8 blocks show more frozen primary wins than losses "
        "with ties retained.",
        "case_dependent;functional_dependent;controlled_psse_derived_workload",
        "universal_selector_superiority;always_outperforms_magnitude",
    ),
    "functional_dependence": (
        "Selected functional families exhibit different support-preservation outcomes.",
        "functional_dependent",
        "uniform_functional_superiority",
    ),
    "support_stability": (
        "Training-subset support stability is reported as a descriptive diagnostic.",
        "local_sensitivity_model",
        "universal_support_stability",
    ),
    "certificate_validity": (
        "Perturbation certificates cover the frozen evaluated rows but remain conservative.",
        "conservative_certificate",
        "tight_certificate",
    ),
    "resource_accounting": (
        "Sparse-wrapper resources are reported with executed and modeled tiers kept separate.",
        "direct_rotation_architecture;no_ieee_scale_sparse_oracle",
        "scalable_ieee_execution;quantum_advantage",
    ),
    "common_design_qsvt": (
        "Common-design statevector validation separates support error from QSVT error "
        "without per-support phase refits.",
        "small_scale;simulator_only;selected_output",
        "hardware_demonstrated;scalable_ieee_execution",
    ),
    "negative_scalability_boundary": (
        "Skipped finite-shot and infeasible resource rows remain explicit negative "
        "boundary evidence.",
        "no_hardware_execution;no_ieee_scale_sparse_oracle",
        "structural_finite_shot_measured;scalable_ieee_execution",
    ),
    "excluded_claims": (
        "Superseded summaries and incomplete rows are excluded from canonical numerical support.",
        "provenance_required",
        "unsupported_claim",
    ),
}


def _strongest_tier(tiers: Iterable[str]) -> str:
    priority = {
        "transpiled_executed_circuit": 9,
        "finite_shot_simulation": 8,
        "qsvt_statevector": 7,
        "qsvt_matrix_action": 6,
        "exact_svt": 5,
        "polynomial_evaluation": 4,
        "classical_exact": 3,
        "modeled_resource": 2,
        "diagnostic_only": 1,
        "excluded": 0,
    }
    values = list(tiers)
    return max(values, key=lambda item: priority.get(item, -1)) if values else "excluded"


def build_claim_registry(root: Path, output_dir: Path) -> pd.DataFrame:
    del root  # Sources are already exact in the canonical result rows.
    results = _read_csv(output_dir / "canonical_result_registry.csv", dtype=str)
    rows: list[dict[str, Any]] = []
    for family, (text, qualifier, prohibited) in _CLAIM_DEFINITIONS.items():
        subset = results[results["claim_family"] == family]
        supporting = subset[subset["manuscript_eligible"] == "true"]["result_id"].tolist()
        limiting = subset[subset["manuscript_eligible"] != "true"]["result_id"].tolist()
        eligible = bool(supporting) and family != "excluded_claims"
        rows.append(
            {
                "claim_id": f"claim:{family}",
                "claim_family": family,
                "claim_text_neutral": text,
                "support_status": "supported_with_qualifiers" if eligible else "excluded",
                "supporting_result_ids": ";".join(supporting),
                "contradicting_or_limiting_result_ids": ";".join(limiting),
                "evidence_tier": _strongest_tier(subset["evidence_tier"].tolist()),
                "manuscript_eligible": eligible,
                "required_qualifier": qualifier,
                "prohibited_overclaim": prohibited,
            }
        )
    frame = pd.DataFrame(rows, columns=CLAIM_FIELDS)
    atomic_write_csv(
        output_dir / "canonical_claim_evidence_registry.csv",
        frame.to_dict(orient="records"),
        CLAIM_FIELDS,
    )
    return frame


def _difference(
    expected: Any, recomputed: Any, absolute_tolerance: float, relative_tolerance: float
) -> tuple[float, float, float, str]:
    try:
        left = float(expected)
        right = float(recomputed)
    except (TypeError, ValueError):
        equal = str(expected) == str(recomputed)
        return (
            0.0 if equal else math.inf,
            0.0 if equal else math.inf,
            0.0,
            "pass" if equal else "fail",
        )
    if math.isinf(left) or math.isinf(right):
        equal = left == right
        return (
            0.0 if equal else math.inf,
            0.0 if equal else math.inf,
            0.0,
            "pass" if equal else "fail",
        )
    absolute = abs(left - right)
    scale = max(abs(left), abs(right), np.finfo(float).tiny)
    relative = absolute / scale
    tolerance = max(absolute_tolerance, relative_tolerance * scale)
    return absolute, relative, tolerance, "pass" if absolute <= tolerance else "fail"


def _check_row(
    *,
    check_id: str,
    result_id: str,
    expected: Any,
    recomputed: Any,
    source_artifact: str,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> dict[str, Any]:
    absolute, relative, tolerance, status = _difference(
        expected, recomputed, absolute_tolerance, relative_tolerance
    )
    return {
        "check_id": check_id,
        "result_id": result_id,
        "expected_value": expected,
        "recomputed_value": recomputed,
        "absolute_difference": absolute,
        "relative_difference": relative,
        "tolerance": tolerance,
        "status": status,
        "source_artifact": source_artifact,
    }


def _count_csv_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _certificate_stats(path: Path) -> tuple[int, float, int]:
    rows = 0
    holds = 0
    violations = 0
    for chunk in pd.read_csv(
        path,
        usecols=lambda column: column in {"certificate_holds", "status"},
        chunksize=250_000,
        low_memory=False,
    ):
        rows += len(chunk)
        if "certificate_holds" in chunk:
            truth = chunk["certificate_holds"].astype(str).str.lower() == "true"
            holds += int(truth.sum())
            violations += int((~truth).sum())
        else:
            completed = chunk["status"].astype(str) == "completed"
            holds += int(completed.sum())
            violations += int((~completed).sum())
    return rows, holds / rows if rows else math.nan, violations


def _outcome_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        outcome: int((frame["outcome"].astype(str) == outcome).sum())
        for outcome in ("win", "tie", "loss")
    }


def _append_headline_records(output_dir: Path, checks: Sequence[Mapping[str, Any]]) -> None:
    path = output_dir / "canonical_result_registry.csv"
    frame = _read_csv(path, dtype=str)
    configurations = _read_csv(
        output_dir / "canonical_configuration_registry.csv", dtype=str
    ).set_index("configuration_id")
    existing = set(frame["result_id"])
    new_rows: list[dict[str, Any]] = []
    for check in checks:
        result_id = str(check["result_id"])
        if result_id in existing:
            continue
        check_id = str(check["check_id"])
        if check_id.startswith("integrated"):
            family = "integrated_sparse_chain"
            experiment = "integrated_sparse_chain"
            configuration = "cfg:integrated:ieee14_sparse_quantized_8x8_d31_selected_v1:statevector"
            limitation = (
                "small_scale_8x8;simulator_only;selected_output_only;no_hardware;no_quantum_speedup"
            )
        elif check_id.startswith("precision") or check_id.startswith("decomposition"):
            family = "error_source_decomposition"
            experiment = "precision_sensitivity"
            precision_configs = _read_csv(
                output_dir / "canonical_configuration_registry.csv", dtype=str
            )
            configuration = str(
                precision_configs[
                    precision_configs["experiment_family"] == "precision_sensitivity"
                ].iloc[0]["configuration_id"]
            )
            limitation = "small_scale_8x8;simulator_only;selected_output_only"
        elif check_id.startswith("development"):
            family = "output_aware_selection"
            experiment = "output_aware_sparse_selection"
            configuration = "cfg:output_aware_sparse_selection:study"
            limitation = "small_scale_8x8;controlled_generated_residuals;functional_dependence"
        elif check_id.startswith("generalization"):
            family = "heldout_generalization"
            experiment = "output_aware_generalization"
            configuration = "cfg:output_aware_generalization:study"
            limitation = (
                "small_scale_8x8;fixed_three_topologies;case_dependence;functional_dependence"
            )
        else:
            family = "structural_generalization"
            experiment = "output_aware_structural_generalization"
            configuration = "cfg:output_aware_structural_generalization:study"
            limitation = (
                "small_scale_8x8;fixed_three_topologies;case_dependence;"
                "functional_dependence;aggregate_weakness;ieee57_mixed_result"
            )
        source = Path(str(check["source_artifact"]))
        new_rows.append(
            _canonical_record(
                result_id=result_id,
                claim_family=family,
                experiment_family=experiment,
                configuration_id=configuration,
                source_artifact=source,
                source_row_locator=f"independent_headline_check[{check_id}]",
                evidence_tier="diagnostic_only",
                matrix_fingerprint=(
                    str(configurations.loc[configuration, "matrix_fingerprint"])
                    if configuration in configurations.index
                    and str(configurations.loc[configuration, "matrix_fingerprint"])
                    not in {"", "nan"}
                    else None
                ),
                value=cast(float | str, check["recomputed_value"]),
                unit="headline_check_value",
                status="completed" if check["status"] == "pass" else "failed",
                manuscript_eligible=check["status"] == "pass",
                limitation_code=limitation,
                notes="Independent aggregation from the authoritative artifact named by the check.",
            ).csv_row()
        )
        existing.add(result_id)
    if new_rows:
        combined = pd.concat([frame, pd.DataFrame(new_rows)], ignore_index=True)
        combined = combined.sort_values("result_id").reset_index(drop=True)
        atomic_write_csv(path, combined.to_dict(orient="records"), RESULT_FIELDS)


def build_headline_checks(root: Path, output_dir: Path) -> pd.DataFrame:
    """Independently rebuild all headline counts and numerical summaries."""

    checks: list[dict[str, Any]] = []
    abs_tol = 1e-12
    rel_tol = 1e-10
    integrated = root / "outputs/sparse_integrated_chain"
    iconfig = load_json(integrated / "configuration.json")
    imeta = load_json(integrated / "matrix_metadata.json")
    matrix = np.load(integrated / "matrix_quantized.npy")
    phases = np.load(integrated / "phases.npy")
    coefficients = np.load(integrated / "polynomial_coefficients.npy")
    state = _read_csv(integrated / "statevector_validation.csv")
    finite_summary = _read_csv(integrated / "finite_shot_summary.csv")
    finite_rows = _read_csv(integrated / "finite_shot_results.csv")
    resource = _read_csv(integrated / "resource_ledger.csv")
    sparse_resource = resource[
        resource["resource_category"] == "executed_small_scale_sparse_integrated"
    ].iloc[0]

    def add(
        check_id: str,
        result_id: str,
        expected: Any,
        recomputed: Any,
        source: Path,
        *,
        atol: float = 0.0,
        rtol: float = 0.0,
    ) -> None:
        checks.append(
            _check_row(
                check_id=check_id,
                result_id=result_id,
                expected=expected,
                recomputed=recomputed,
                source_artifact=source.relative_to(root).as_posix(),
                absolute_tolerance=atol,
                relative_tolerance=rtol,
            )
        )

    add(
        "integrated.configuration_id",
        "res:headline:integrated:configuration_id",
        iconfig["configuration_id"],
        resource.iloc[0]["configuration_id"],
        integrated / "resource_ledger.csv",
    )
    add(
        "integrated.matrix_fingerprint",
        "res:headline:integrated:matrix_fingerprint",
        iconfig["matrix_fingerprint"],
        stable_array_fingerprint(matrix),
        integrated / "matrix_quantized.npy",
    )
    add(
        "integrated.shape",
        "res:integrated:matrix_shape",
        parse_shape(iconfig["matrix_shape"]),
        parse_shape(matrix.shape),
        integrated / "matrix_quantized.npy",
    )
    add(
        "integrated.nonzeros",
        "res:integrated:matrix_nonzeros",
        imeta["nnz"],
        int(np.count_nonzero(matrix)),
        integrated / "matrix_quantized.npy",
    )
    add(
        "integrated.degree",
        "res:integrated:polynomial_degree",
        iconfig["polynomial_degree"],
        coefficients.size - 1,
        integrated / "polynomial_coefficients.npy",
    )
    add(
        "integrated.phase_count",
        "res:integrated:phase_count",
        imeta["phase_count"],
        phases.size,
        integrated / "phases.npy",
    )
    add(
        "integrated.postselection_probability",
        "res:integrated:postselection_probability",
        float(state.iloc[0]["sparse_postselection_probability"]),
        float(state["sparse_postselection_probability"].median()),
        integrated / "statevector_validation.csv",
        atol=abs_tol,
        rtol=rel_tol,
    )
    first_state = state.iloc[0]
    sparse_payload = json.loads(first_state["sparse_statevector_reference"])
    dense_payload = json.loads(first_state["dense_statevector_reference"])
    sparse_vector = np.asarray(sparse_payload["real"]) + 1j * np.asarray(sparse_payload["imag"])
    dense_vector = np.asarray(dense_payload["real"]) + 1j * np.asarray(dense_payload["imag"])
    action_error = float(
        np.linalg.norm(sparse_vector - dense_vector) / np.linalg.norm(dense_vector)
    )
    add(
        "integrated.sparse_dense_action_error",
        "res:integrated:sparse_dense_action_error",
        float(first_state["sparse_dense_action_relative_l2_error"]),
        action_error,
        integrated / "statevector_validation.csv",
        atol=abs_tol,
        rtol=rel_tol,
    )
    add(
        "integrated.qsvt_exact_svt_error",
        "res:integrated:qsvt_exact_svt_action_error",
        float(first_state["qsvt_exact_polynomial_svt_relative_l2_error"]),
        float(state["qsvt_exact_polynomial_svt_relative_l2_error"].max()),
        integrated / "statevector_validation.csv",
        atol=abs_tol,
        rtol=rel_tol,
    )
    selected_recomputed = (
        (state["sparse_statevector_selected_output"] - state["quantized_ridge_selected_output"])
        .abs()
        .max()
    )
    add(
        "integrated.selected_output_qsvt_error",
        "res:integrated:selected_output_qsvt_error",
        float(state["selected_output_absolute_error_vs_quantized_ridge"].max()),
        float(selected_recomputed),
        integrated / "statevector_validation.csv",
        atol=abs_tol,
        rtol=rel_tol,
    )
    fs_expected = finite_summary[
        (finite_summary["chain_type"] == "sparse")
        & (finite_summary["functional_id"] == "coordinate_e0")
        & (finite_summary["shots"] == 1_000_000)
    ].iloc[0]["mean_selected_output_estimate"]
    fs_recomputed = finite_rows[
        (finite_rows["chain_type"] == "sparse")
        & (finite_rows["functional_id"] == "coordinate_e0")
        & (finite_rows["shots_attempted"] == 1_000_000)
    ]["selected_output_estimate"].mean()
    add(
        "integrated.finite_shot_1e6",
        "res:integrated:finite_shot_coordinate_1e6",
        float(fs_expected),
        float(fs_recomputed),
        integrated / "finite_shot_results.csv",
        atol=abs_tol,
        rtol=rel_tol,
    )
    circuit_meta = load_json(integrated / "circuit_metadata.json")["transpilation"]["sparse"]
    for column, meta_key in (
        ("transpiled_gate_count", "gate_count"),
        ("transpiled_depth", "depth"),
        ("toffoli_count", "toffoli_count"),
    ):
        add(
            f"integrated.resource.{column}",
            f"res:integrated:resource:{column}",
            circuit_meta[meta_key],
            sparse_resource[column],
            integrated / "resource_ledger.csv",
        )
    add(
        "integrated.resource.controlled_rotations",
        "res:integrated:resource:controlled_rotation_count",
        load_json(integrated / "circuit_metadata.json")["operation_counts"][
            "value_rotations_per_attempt"
        ],
        sparse_resource["controlled_rotation_count"],
        integrated / "resource_ledger.csv",
    )

    precision = root / "outputs/sparse_error_precision_study"
    decomp = _read_csv(precision / "error_decomposition.csv")
    completed_decomp = decomp[decomp["status"] == "completed"].copy()
    signed_sum = (
        completed_decomp["sparsification_signed_delta"].fillna(0.0)
        + completed_decomp["quantization_signed_delta"].fillna(0.0)
        + completed_decomp["qsvt_signed_delta"].fillna(0.0)
        + completed_decomp["sampling_signed_delta"].fillna(0.0)
    )
    identity_residual = float((completed_decomp["total_signed_delta"] - signed_sum).abs().max())
    add(
        "decomposition.max_identity_residual",
        "res:error_decomposition:max_identity_residual",
        0.0,
        identity_residual,
        precision / "error_decomposition.csv",
        atol=abs_tol,
    )
    triangle_recomputed = bool(
        (
            completed_decomp["total_absolute_error"]
            <= completed_decomp["triangle_bound_sum"] + abs_tol
        ).all()
    )
    add(
        "decomposition.triangle_bound",
        "res:error_decomposition:triangle_bound_all",
        "true",
        str(triangle_recomputed).lower(),
        precision / "error_decomposition.csv",
    )

    development = root / "outputs/output_aware_sparse_selection"
    dcheckpoint = load_json(development / "checkpoint.json")
    dqsvt = _read_csv(development / "qsvt_validation_results.csv")
    add(
        "development.qsvt_rows",
        "res:headline:development:qsvt_rows",
        dcheckpoint["stages"]["qsvt"]["result"]["rows"],
        len(dqsvt),
        development / "qsvt_validation_results.csv",
    )
    add(
        "development.qsvt_failures",
        "res:headline:development:qsvt_failures",
        dcheckpoint["stages"]["qsvt"]["result"]["failures"],
        int((dqsvt["status"] != "completed").sum()),
        development / "qsvt_validation_results.csv",
    )
    dcert_rows, dcoverage, dviolations = _certificate_stats(
        development / "certificate_registry.csv"
    )
    add(
        "development.certificate_rows",
        "res:headline:development:certificate_rows",
        dcheckpoint["stages"]["certificates"]["result"]["certificate_rows"],
        dcert_rows,
        development / "certificate_registry.csv",
    )
    add(
        "development.certificate_coverage",
        "res:output_aware_sparse_selection:certificate:coverage",
        1.0,
        dcoverage,
        development / "certificate_registry.csv",
        atol=abs_tol,
    )
    add(
        "development.certificate_violations",
        "res:headline:development:certificate_violations",
        0,
        dviolations,
        development / "certificate_registry.csv",
    )

    general = root / "outputs/output_aware_generalization"
    gcheckpoint = load_json(general / "checkpoint.json")
    ginstances = _read_csv(general / "instance_registry.csv")
    gpairs = _read_csv(general / "generalization_matched_pairs.csv")
    gpairs = gpairs[gpairs["comparison_label"] == "primary"]
    goutcomes = _outcome_counts(gpairs)
    add(
        "generalization.instance_count",
        "res:headline:generalization:instances",
        gcheckpoint["stages"]["instances"]["result"]["included"],
        len(ginstances),
        general / "instance_registry.csv",
    )
    for case in ("ieee14", "ieee30", "ieee57"):
        add(
            f"generalization.case_count.{case}",
            f"res:headline:generalization:{case}:instances",
            5,
            int((ginstances["ieee_case"] == case).sum()),
            general / "instance_registry.csv",
        )
    for outcome, expected_key in (
        ("win", "primary_win"),
        ("tie", "primary_tie"),
        ("loss", "primary_loss"),
    ):
        add(
            f"generalization.primary.{outcome}",
            f"res:generalization:primary:{outcome}_count",
            gcheckpoint["stages"]["primary-test"]["result"][expected_key],
            goutcomes[outcome],
            general / "generalization_matched_pairs.csv",
        )
    gbootstrap = _read_csv(general / "generalization_bootstrap.csv")
    gvalues = gbootstrap["median_paired_difference_sensitivity_minus_magnitude"].to_numpy()
    for label, quantile, expected_key in (
        ("low", 0.025, "bootstrap_ci_low"),
        ("high", 0.975, "bootstrap_ci_high"),
    ):
        add(
            f"generalization.bootstrap_ci.{label}",
            f"res:headline:generalization:bootstrap_ci:{label}",
            gcheckpoint["stages"]["primary-test"]["result"][expected_key],
            float(np.quantile(gvalues, quantile)),
            general / "generalization_bootstrap.csv",
            atol=abs_tol,
            rtol=rel_tol,
        )
    gqsvt = _read_csv(general / "qsvt_validation_results.csv")
    for label, expected, recomputed in (
        ("rows", 72, len(gqsvt)),
        ("failures", 0, int((gqsvt["status"] != "completed").sum())),
        ("instances", 6, gqsvt["instance_id"].nunique()),
        ("cases", 3, gqsvt["ieee_case"].nunique()),
    ):
        add(
            f"generalization.qsvt.{label}",
            f"res:headline:generalization:qsvt:{label}",
            expected,
            recomputed,
            general / "qsvt_validation_results.csv",
        )
    gfunctional = _read_csv(general / "generalization_functional_pairs.csv")
    gfunctional = gfunctional[gfunctional["comparison_label"] == "primary"]
    gtest = load_json(general / "generalization_primary_test.json")
    for functional_id, expected_counts in gtest["primary_functional_win_tie_loss"].items():
        subset = gfunctional[gfunctional["functional_id"] == functional_id]
        observed = _outcome_counts(subset)
        for outcome in ("win", "tie", "loss"):
            add(
                f"generalization.functional.{functional_id}.{outcome}",
                f"res:headline:generalization:functional:{functional_id}:{outcome}",
                expected_counts[outcome],
                observed[outcome],
                general / "generalization_functional_pairs.csv",
            )

    structural = root / "outputs/output_aware_structural_generalization"
    checkpoint = load_json(structural / "checkpoint.json")
    simple_counts = [
        ("candidate_proposals", "candidates", "candidate_rows", "candidate_registry.csv"),
        (
            "structural_groups",
            "structural-selection",
            "selected_groups",
            "structural_group_registry.csv",
        ),
        ("matrix_realizations", "realizations", "realizations", "instance_registry.csv"),
        ("residuals", "residuals", "records", "residual_registry.csv"),
        ("selector_records", "supports", "support_records", "support_registry.csv"),
    ]
    for label, stage, key, filename in simple_counts:
        add(
            f"structural.{label}",
            f"res:headline:structural:{label}",
            checkpoint["stages"][stage]["result"][key],
            _count_csv_rows(structural / filename),
            structural / filename,
        )
    supports = _read_csv(structural / "support_registry.csv", usecols=["status"])
    add(
        "structural.selector_completed",
        "res:headline:structural:selector_completed",
        checkpoint["stages"]["supports"]["result"]["completed"],
        int((supports["status"] == "completed").sum()),
        structural / "support_registry.csv",
    )
    add(
        "structural.selector_infeasible",
        "res:headline:structural:selector_infeasible",
        checkpoint["stages"]["supports"]["result"]["failed_or_infeasible"],
        int((supports["status"] != "completed").sum()),
        structural / "support_registry.csv",
    )
    spairs = _read_csv(structural / "structural_primary_matched_pairs.csv")
    soutcomes = _outcome_counts(spairs)
    for outcome in ("win", "tie", "loss"):
        add(
            f"structural.primary.{outcome}",
            f"res:structural:primary:{outcome}_count",
            checkpoint["stages"]["primary-test"]["result"][outcome],
            soutcomes[outcome],
            structural / "structural_primary_matched_pairs.csv",
        )
    sbootstrap = _read_csv(structural / "structural_group_bootstrap.csv")
    svalues = sbootstrap["median_paired_difference_sensitivity_minus_magnitude"].to_numpy()
    for label, quantile, expected_key in (
        ("low", 0.025, "bootstrap_ci_low"),
        ("high", 0.975, "bootstrap_ci_high"),
    ):
        add(
            f"structural.bootstrap_ci.{label}",
            f"res:headline:structural:bootstrap_ci:{label}",
            checkpoint["stages"]["primary-test"]["result"][expected_key],
            float(np.quantile(svalues, quantile)),
            structural / "structural_group_bootstrap.csv",
            atol=abs_tol,
            rtol=rel_tol,
        )
    sfunctional = _read_csv(structural / "structural_primary_functional_pairs.csv")
    stest = load_json(structural / "structural_primary_test.json")
    for functional_id, expected_counts in stest["functional_win_tie_loss"].items():
        observed = _outcome_counts(sfunctional[sfunctional["functional_id"] == functional_id])
        for outcome in ("win", "tie", "loss"):
            add(
                f"structural.functional.{functional_id}.{outcome}",
                f"res:headline:structural:functional:{functional_id}:{outcome}",
                expected_counts[outcome],
                observed[outcome],
                structural / "structural_primary_functional_pairs.csv",
            )
    stability = _read_csv(structural / "support_stability_summary.csv")
    raw_stability = _read_csv(structural / "support_stability.csv")
    for _, row in stability.iterrows():
        selector = str(row["selector"])
        observed = float(
            raw_stability[
                (raw_stability["selector"] == selector) & (raw_stability["status"] == "completed")
            ]["jaccard_similarity"].median()
        )
        add(
            f"structural.stability.{selector}",
            f"res:structural:stability:{selector}:median_jaccard",
            float(row["median_jaccard"]),
            observed,
            structural / "support_stability.csv",
            atol=abs_tol,
            rtol=rel_tol,
        )
    cert_rows, cert_coverage, cert_violations = _certificate_stats(
        structural / "certificate_results.csv"
    )
    for label, expected, observed in (
        ("rows", checkpoint["stages"]["certificates"]["result"]["rows"], cert_rows),
        ("coverage", checkpoint["stages"]["certificates"]["result"]["coverage"], cert_coverage),
        (
            "violations",
            checkpoint["stages"]["certificates"]["result"]["violations"],
            cert_violations,
        ),
    ):
        add(
            f"structural.certificate.{label}",
            (
                "res:output_aware_structural_generalization:certificate:coverage"
                if label == "coverage"
                else f"res:headline:structural:certificate:{label}"
            ),
            expected,
            observed,
            structural / "certificate_results.csv",
            atol=abs_tol if label == "coverage" else 0.0,
        )
    resources = _read_csv(structural / "resource_registry.csv")
    completed_resources = resources[resources["status"] == "completed"]
    resource_checkpoint = checkpoint["stages"]["resources"]["result"]
    resource_checks = (
        ("records", resource_checkpoint["resource_records"], len(resources)),
        ("completed", 3204, len(completed_resources)),
        (
            "unavailable",
            resource_checkpoint["unavailable_support_resource_records"],
            int((resources["status"] != "completed").sum()),
        ),
        (
            "wrapper_failures",
            resource_checkpoint["executed_wrapper_validation_failures"],
            int(
                (
                    completed_resources["wrapper_reconstruction_holds"].astype(str).str.lower()
                    != "true"
                ).sum()
            ),
        ),
        ("gate_min", 567, int(completed_resources["signal_unitary_gate_count"].min())),
        ("gate_max", 25957, int(completed_resources["signal_unitary_gate_count"].max())),
        ("depth_min", 342, int(completed_resources["signal_unitary_depth"].min())),
        ("depth_max", 19324, int(completed_resources["signal_unitary_depth"].max())),
    )
    for label, expected, observed in resource_checks:
        add(
            f"structural.resource.{label}",
            f"res:headline:structural:resource:{label}",
            expected,
            observed,
            structural / "resource_registry.csv",
        )
    sqsvt = _read_csv(structural / "qsvt_validation_results.csv")
    for label, expected, observed in (
        ("rows", 72, len(sqsvt)),
        ("failures", 0, int((sqsvt["status"] != "completed").sum())),
        ("groups", 6, sqsvt["structural_group_id"].nunique()),
        ("cases", 3, sqsvt["ieee_case"].nunique()),
    ):
        add(
            f"structural.qsvt.{label}",
            f"res:headline:structural:qsvt:{label}",
            expected,
            observed,
            structural / "qsvt_validation_results.csv",
        )
    finite = _read_csv(structural / "finite_shot_results.csv")
    add(
        "structural.finite_shot_status",
        "res:output_aware_structural_generalization:finite_shot:status",
        "skipped_under_frozen_cost_ceiling",
        ";".join(sorted(finite["status"].astype(str).unique())),
        structural / "finite_shot_results.csv",
    )

    frame = pd.DataFrame(checks, columns=HEADLINE_CHECK_FIELDS)
    atomic_write_csv(
        output_dir / "headline_result_checks.csv",
        frame.to_dict(orient="records"),
        HEADLINE_CHECK_FIELDS,
    )
    failed = frame[frame["status"] != "pass"]
    summary = {
        "schema_version": 1,
        "checks": len(frame),
        "passed": int((frame["status"] == "pass").sum()),
        "failed": len(failed),
        "blocking_failures": failed["check_id"].tolist(),
        "status": "pass" if failed.empty else "blocking_failure",
    }
    atomic_write_json(output_dir / "headline_result_check_summary.json", summary)
    _append_headline_records(output_dir, checks)
    return frame


def verify_protected_sources(root: Path, output_dir: Path) -> dict[str, Any]:
    """Compare every snapshotted source by presence, size, and SHA-256."""

    snapshot_path = output_dir / "protected_source_snapshot.json"
    snapshot = load_json(snapshot_path)
    changed: list[dict[str, Any]] = []
    deleted: list[dict[str, Any]] = []
    for record in snapshot["files"]:
        path = root / record["path"]
        if not path.exists():
            deleted.append(
                {
                    "path": record["path"],
                    "category": record["category"],
                    "expected_sha256": record["sha256"],
                }
            )
            continue
        stat = path.stat()
        digest = sha256_file(path)
        if stat.st_size != record["size_bytes"] or digest != record["sha256"]:
            changed.append(
                {
                    "path": record["path"],
                    "category": record["category"],
                    "expected_size_bytes": record["size_bytes"],
                    "actual_size_bytes": stat.st_size,
                    "expected_sha256": record["sha256"],
                    "actual_sha256": digest,
                    "mtime_changed_only": digest == record["sha256"],
                }
            )
    manifest_or_checksum_changed = [
        item
        for item in changed + deleted
        if str(item["path"]).endswith(("manifest.json", "checksums.sha256"))
    ]
    report = {
        "schema_version": 1,
        "protected_files": int(snapshot["file_count"]),
        "verified_files": int(snapshot["file_count"]) - len(deleted),
        "changed_files": changed,
        "deleted_files": deleted,
        "changed_count": len(changed),
        "deleted_count": len(deleted),
        "changed_prior_manifests_or_checksums": manifest_or_checksum_changed,
        "status": "pass" if not changed and not deleted else "blocking_failure",
        "comparison_semantics": (
            "path existence, byte size, and SHA-256; mtime is not authoritative"
        ),
    }
    atomic_write_json(output_dir / "protected_source_comparison.json", report)
    return report


def build_eligibility_audit(root: Path, output_dir: Path) -> pd.DataFrame:
    """Apply deterministic manuscript-eligibility rules to every canonical row."""

    results = _read_csv(output_dir / "canonical_result_registry.csv", dtype=str)
    configurations = _read_csv(output_dir / "canonical_configuration_registry.csv", dtype=str)
    config_ids = set(configurations["configuration_id"])
    headline = _read_csv(output_dir / "headline_result_checks.csv", dtype=str)
    headline_status = dict(zip(headline["result_id"], headline["status"], strict=False))
    snapshot = load_json(output_dir / "protected_source_snapshot.json")
    protected = {row["path"]: row for row in snapshot["files"]}
    protected_status = verify_protected_sources(root, output_dir)
    protected_unchanged = protected_status["status"] == "pass"
    audit_rows: list[dict[str, Any]] = []
    for row in results.to_dict(orient="records"):
        source_relative = str(row["source_artifact"])
        source = root / source_relative
        source_exists = source.is_file()
        if source_exists and source_relative in protected:
            source_integrity = sha256_file(source) == protected[source_relative]["sha256"]
            integrity_basis = "protected_snapshot_sha256"
        elif source_exists and output_dir in source.parents:
            source_integrity = True
            integrity_basis = "new_closure_artifact_sha256_available"
        else:
            source_integrity = False
            integrity_basis = "not_in_protected_snapshot"
        configuration_resolves = str(row["configuration_id"]) in config_ids
        source_locator_complete = bool(str(row["source_row_locator"]).strip())
        tier_valid = str(row["evidence_tier"]) in EVIDENCE_TIERS
        status_valid = str(row["status"]) in {
            "completed",
            "pass",
            "negative_evidence",
            "skipped_under_frozen_cost_ceiling",
            "skipped_under_predeclared_cost_ceiling",
        }
        limitations_attached = (
            bool(str(row["limitation_code"]).strip())
            and str(row["limitation_code"]).lower() != "nan"
        )
        matrix_applicable = str(row["claim_family"]) not in {"excluded_claims"}
        matrix_present = not matrix_applicable or str(
            row["matrix_fingerprint"]
        ).strip().lower() not in {"", "nan"}
        support_applicable = str(row["result_id"]).startswith(
            ("res:integrated", "res:precision")
        ) or (
            str(row["evidence_tier"]) == "qsvt_statevector"
            and str(row["claim_family"]) == "common_design_qsvt"
        )
        support_present = not support_applicable or str(
            row["support_fingerprint"]
        ).strip().lower() not in {"", "nan"}
        not_evidence_upgrade = not (
            str(row["evidence_tier"]) == "finite_shot_simulation"
            and "skipped" in str(row["status"])
        )
        check_pass = headline_status.get(str(row["result_id"]), "pass") == "pass"
        no_contradiction = not results["result_id"].duplicated().any()
        recomputed = all(
            (
                source_exists,
                source_integrity,
                configuration_resolves,
                source_locator_complete,
                tier_valid,
                status_valid,
                limitations_attached,
                matrix_present,
                support_present,
                not_evidence_upgrade,
                check_pass,
                no_contradiction,
                protected_unchanged,
            )
        )
        declared = str(row["manuscript_eligible"]).lower() == "true"
        expected_recomputed = recomputed if declared else False
        reasons: list[str] = []
        rule_values = {
            "source_exists": source_exists,
            "source_integrity_valid": source_integrity,
            "configuration_resolves": configuration_resolves,
            "source_row_locator_complete": source_locator_complete,
            "evidence_tier_valid": tier_valid,
            "status_valid_or_negative": status_valid,
            "limitations_attached": limitations_attached,
            "matrix_fingerprint_present_if_applicable": matrix_present,
            "support_fingerprint_present_if_applicable": support_present,
            "evidence_tier_not_upgraded": not_evidence_upgrade,
            "headline_checks_pass": check_pass,
            "no_contradictory_duplicate": no_contradiction,
            "protected_sources_unchanged": protected_unchanged,
        }
        reasons.extend(key for key, value in rule_values.items() if not value)
        status = "pass" if declared == expected_recomputed else "fail"
        if not declared and str(row["evidence_tier"]) == "excluded":
            status = "pass"
        audit_rows.append(
            {
                "result_id": row["result_id"],
                "declared_manuscript_eligible": declared,
                "recomputed_manuscript_eligible": expected_recomputed,
                **rule_values,
                "source_integrity_basis": integrity_basis,
                "ineligibility_reasons": ";".join(reasons),
                "status": status,
            }
        )
    fields = [
        "result_id",
        "declared_manuscript_eligible",
        "recomputed_manuscript_eligible",
        "source_exists",
        "source_integrity_valid",
        "configuration_resolves",
        "source_row_locator_complete",
        "evidence_tier_valid",
        "status_valid_or_negative",
        "limitations_attached",
        "matrix_fingerprint_present_if_applicable",
        "support_fingerprint_present_if_applicable",
        "evidence_tier_not_upgraded",
        "headline_checks_pass",
        "no_contradictory_duplicate",
        "protected_sources_unchanged",
        "source_integrity_basis",
        "ineligibility_reasons",
        "status",
    ]
    frame = pd.DataFrame(audit_rows, columns=fields)
    atomic_write_csv(
        output_dir / "manuscript_eligibility_audit.csv",
        frame.to_dict(orient="records"),
        fields,
    )
    return frame


PROHIBITED_CLAIMS = (
    "quantum speedup",
    "quantum advantage",
    "hardware execution",
    "field pmu validation",
    "field scada validation",
    "full ieee-scale sparse circuit",
    "scalable qram",
    "universal selector superiority",
    "always outperforms",
    "tight certificate",
    "full nonlinear ac psse execution",
)

_NEGATIVE_MARKERS = (
    "no ",
    "not ",
    "was not",
    "were not",
    "without ",
    "prohibit",
    "prohibited",
    "excluded",
    "not claimed",
    "is claimed or measured",  # appears after a leading "No" in a full sentence
)


def _negative_context(text: str, start: int) -> bool:
    context = text[max(0, start - 100) : start].lower()
    sentence = re.split(r"[\n.!?]", context)[-1]
    return any(marker in sentence for marker in _NEGATIVE_MARKERS)


def run_claim_guards(root: Path, output_dir: Path) -> dict[str, Any]:
    """Reject positive prohibited language while allowing explicit negations."""

    del root
    scan_files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".csv", ".json", ".md"}
        and path.name
        not in {
            "checkpoint.json",
            "excluded_claim_guard_report.json",
            "manifest.json",
        }
        and not {"checkpoint_parts", "filewise"}.intersection(path.parts)
    )
    violations: list[dict[str, Any]] = []
    allowed: list[dict[str, Any]] = []
    for path in scan_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        lowered = text.lower()
        for phrase in PROHIBITED_CLAIMS:
            for match in re.finditer(re.escape(phrase), lowered):
                item = {
                    "path": path.relative_to(output_dir).as_posix(),
                    "phrase": phrase,
                    "offset": match.start(),
                    "context": text[max(0, match.start() - 80) : match.end() + 80]
                    .replace("\n", " ")
                    .strip(),
                }
                if _negative_context(lowered, match.start()):
                    allowed.append(item)
                else:
                    violations.append(item)
    report = {
        "schema_version": 1,
        "prohibited_claims_searched": list(PROHIBITED_CLAIMS),
        "files_scanned": len(scan_files),
        "negative_statements_allowed": len(allowed),
        "allowed_negative_occurrences": allowed,
        "violations": violations,
        "violation_count": len(violations),
        "status": "pass" if not violations else "blocking_failure",
    }
    atomic_write_json(output_dir / "excluded_claim_guard_report.json", report)
    return report
