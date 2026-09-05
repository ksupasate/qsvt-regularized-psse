from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robust_qsvt_se.data.cases import load_ac_case, load_dc_case  # noqa: E402
from robust_qsvt_se.measurement.ac_linear import (  # noqa: E402
    ac_measurements_and_jacobian,
    default_ac_state_vector,
)
from robust_qsvt_se.measurement.dc_linear import build_dc_measurement_matrix  # noqa: E402
from robust_qsvt_se.qsvt.research_matrix import (  # noqa: E402
    DEFAULT_MEASUREMENT_CONFIG,
)

DEFAULT_CASES = ("ieee14", "ieee30", "ieee57", "ieee118", "ieee300")
INVENTORY_COLUMNS = [
    "case_name",
    "case_source",
    "experiment_group",
    "model_type",
    "config_file",
    "measurement_type",
    "bus_i",
    "bus_j",
    "row_index",
    "sigma",
    "weight",
    "is_physical_measurement",
    "is_synthetic_row",
    "is_generated",
    "is_after_weighting",
    "pmu_scada_status",
    "notes",
    "state_dimension",
]
DC_TYPE_LABELS = {
    "branch_flow": "dc_branch_flow",
    "bus_injection": "dc_bus_injection",
    "angle": "dc_angle",
}
AC_TYPE_LABELS = {
    "voltage_magnitude": "ac_voltage_magnitude",
    "p_injection": "ac_active_injection",
    "q_injection": "ac_reactive_injection",
    "p_branch_flow": "ac_active_branch_flow",
    "q_branch_flow": "ac_reactive_branch_flow",
}
NONLINEAR_AC_TYPE_LABELS = {key: f"nonlinear_{value}" for key, value in AC_TYPE_LABELS.items()}
QSVT_RESOURCE_CONFIGS = {"qsvt_resource_full_ieee.yaml"}
QSVT_MATRIX_CONFIG_PREFIX = "qsvt_full_matrix_ieee"


def export_measurement_inventory(
    *,
    config_dir: str | Path = REPO_ROOT / "configs",
    output_dir: str | Path = REPO_ROOT / "outputs" / "measurement_inventory",
    config_paths: Sequence[str | Path] | None = None,
    case_names: Sequence[str] = DEFAULT_CASES,
    include_qsvt: bool = True,
) -> dict[str, Any]:
    """Export row-level measurement inventory without running experiments."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    configs = _config_paths(Path(config_dir), config_paths)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for config_path in configs:
        raw = _read_yaml(config_path)
        try:
            rows.extend(_rows_for_standard_experiment(raw, config_path))
            if include_qsvt:
                rows.extend(_rows_for_qsvt_experiment(raw, config_path, case_names))
        except Exception as exc:
            skipped.append(
                {
                    "config_file": _display_path(config_path),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    inventory = pd.DataFrame(rows, columns=INVENTORY_COLUMNS)
    if not inventory.empty:
        inventory = inventory.sort_values(
            ["experiment_group", "case_name", "config_file", "row_index"],
            kind="mergesort",
        ).reset_index(drop=True)
    by_case = _summary_by_case(inventory)
    by_type = _summary_by_type(inventory)
    exact_dc_counts, exact_ac_counts = _exact_row_count_tables(case_names)

    all_cases_path = output_path / "measurement_inventory_all_cases.csv"
    by_case_path = output_path / "measurement_inventory_by_case.csv"
    by_type_path = output_path / "measurement_inventory_by_type.csv"
    dc_counts_path = output_path / "exact_dc_row_counts.csv"
    ac_counts_path = output_path / "exact_ac_row_counts.csv"
    summary_path = output_path / "measurement_inventory_summary.md"
    manifest_path = output_path / "measurement_inventory_manifest.json"
    readme_path = output_path / "README.md"

    inventory.to_csv(all_cases_path, index=False)
    by_case.to_csv(by_case_path, index=False)
    by_type.to_csv(by_type_path, index=False)
    exact_dc_counts.to_csv(dc_counts_path, index=False)
    exact_ac_counts.to_csv(ac_counts_path, index=False)
    summary_path.write_text(
        _summary_markdown(inventory, by_case, by_type, exact_dc_counts, exact_ac_counts, skipped),
        encoding="utf-8",
    )
    readme_path.write_text(_readme_text(), encoding="utf-8")
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "script": "scripts/export_measurement_inventory.py",
        "config_count_scanned": len(configs),
        "row_count": len(inventory),
        "case_names_requested": list(case_names),
        "skipped_configs": skipped,
        "outputs": {
            "summary": str(summary_path),
            "all_cases": str(all_cases_path),
            "by_case": str(by_case_path),
            "by_type": str(by_type_path),
            "exact_dc_row_counts": str(dc_counts_path),
            "exact_ac_row_counts": str(ac_counts_path),
            "readme": str(readme_path),
        },
        "notes": [
            "Rows are generated from repository configs and case fixtures.",
            "Counts are before random missing-measurement row removal.",
            "IEEE/PYPOWER cases are network models, not PMU/SCADA field measurements.",
            "Sigma implies diagonal R through R_ii = sigma_i^2 where sigma is available.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "output_dir": output_path,
        "inventory": inventory,
        "by_case": by_case,
        "by_type": by_type,
        "exact_dc_counts": exact_dc_counts,
        "exact_ac_counts": exact_ac_counts,
        "manifest": manifest,
    }


def _config_paths(
    config_dir: Path,
    config_paths: Sequence[str | Path] | None,
) -> list[Path]:
    if config_paths is not None:
        return [Path(path) for path in config_paths]
    return sorted(config_dir.glob("*.yaml"))


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    return loaded if isinstance(loaded, dict) else {}


def _rows_for_standard_experiment(
    config: dict[str, Any],
    config_path: Path,
) -> list[dict[str, Any]]:
    system = config.get("system")
    if not isinstance(system, dict):
        return []

    mode = str(system.get("mode", "synthetic_linearized"))
    if mode == "synthetic_linearized":
        return _synthetic_rows(system, config_path)
    if mode == "dc_power_flow_linearized":
        return _dc_rows(system, config_path)
    if mode == "ac_power_flow_linearized":
        return _ac_rows(system, config_path, nonlinear=False)
    if mode in {"nonlinear_ac_state_estimation", "ac_iterative_state_estimation"}:
        return _ac_rows(system, config_path, nonlinear=True)
    return []


def _synthetic_rows(system: dict[str, Any], config_path: Path) -> list[dict[str, Any]]:
    n_measurements = int(system.get("n_measurements", 0))
    n_states = int(system.get("n_states", 0))
    rows = []
    for row_index in range(n_measurements):
        rows.append(
            _inventory_row(
                case_name=str(system.get("case_name", "ieee14")),
                case_source=str(system.get("case_source", "synthetic_generated")),
                experiment_group="Synthetic weighted linearized",
                model_type="synthetic_linearized",
                config_path=config_path,
                measurement_type="synthetic_weighted_row",
                row_index=row_index,
                state_dimension=n_states,
                sigma=None,
                buses=(),
                is_physical_measurement=False,
                is_after_weighting=True,
                notes=(
                    "Synthetic row generated directly in weighted residual space; "
                    "no bus mapping or raw measurement sigma is defined."
                ),
            )
        )
    return rows


def _dc_rows(system: dict[str, Any], config_path: Path) -> list[dict[str, Any]]:
    case_name = str(system.get("case_name", "ieee14"))
    case_source = str(system.get("case_source", "builtin"))
    case = load_dc_case(case_name, case_source=case_source)
    _, measurement_rows = build_dc_measurement_matrix(
        case=case,
        measurement_config=dict(system.get("measurement", {})),
    )
    rows = []
    for row_index, row in enumerate(measurement_rows):
        rows.append(
            _inventory_row(
                case_name=case.name,
                case_source=case_source,
                experiment_group=_experiment_group(
                    "dc_power_flow_linearized",
                    case_source,
                    config_path,
                ),
                model_type="dc_power_flow_linearized",
                config_path=config_path,
                measurement_type=DC_TYPE_LABELS.get(row.measurement_type, "unknown"),
                row_index=row_index,
                state_dimension=len(case.state_buses),
                sigma=row.std,
                buses=row.buses,
                is_physical_measurement=True,
                is_after_weighting=True,
                notes=(
                    "Generated DC measurement equation; sigma comes from "
                    "system.measurement and the row is weighted by 1/sigma."
                ),
            )
        )
    return rows


def _ac_rows(
    system: dict[str, Any],
    config_path: Path,
    *,
    nonlinear: bool,
) -> list[dict[str, Any]]:
    case_name = str(system.get("case_name", "ieee14"))
    case_source = str(system.get("case_source", "builtin"))
    case = load_ac_case(case_name, case_source=case_source)
    state = default_ac_state_vector(case)
    _, _, measurement_rows = ac_measurements_and_jacobian(
        case,
        state,
        dict(system.get("measurement", {})),
    )
    state_dimension = len(case.angle_state_buses) + len(case.voltage_state_buses)
    type_labels = NONLINEAR_AC_TYPE_LABELS if nonlinear else AC_TYPE_LABELS
    model_type = "nonlinear_ac_state_estimation" if nonlinear else "ac_power_flow_linearized"
    notes = (
        "Raw nonlinear AC measurement row; z=h(x_true)+e+b is formed before "
        "per-iteration row weighting by sigma."
        if nonlinear
        else "Generated AC single-step row; H_tilde and r_tilde are row-divided by sigma."
    )
    return [
        _inventory_row(
            case_name=case.name,
            case_source=case_source,
            experiment_group=_experiment_group(model_type, case_source, config_path),
            model_type=model_type,
            config_path=config_path,
            measurement_type=type_labels.get(row.measurement_type, "unknown"),
            row_index=row_index,
            state_dimension=state_dimension,
            sigma=row.std,
            buses=row.buses,
            is_physical_measurement=True,
            is_after_weighting=not nonlinear,
            notes=notes,
        )
        for row_index, row in enumerate(measurement_rows)
    ]


def _rows_for_qsvt_experiment(
    config: dict[str, Any],
    config_path: Path,
    case_names: Sequence[str],
) -> list[dict[str, Any]]:
    if config_path.name in QSVT_RESOURCE_CONFIGS:
        resource = config.get("resource", {})
        if not isinstance(resource, dict):
            return []
        requested_cases = [str(case_name) for case_name in resource.get("cases", case_names)]
        return [
            row
            for case_name in requested_cases
            for row in _qsvt_case_rows(
                case_name=case_name,
                case_source=str(resource.get("case_source", "pypower")),
                measurement_config={
                    **DEFAULT_MEASUREMENT_CONFIG,
                    **dict(resource.get("measurement", {})),
                },
                config_path=config_path,
            )
        ]

    matrix = config.get("matrix")
    if not isinstance(matrix, dict) or not config_path.name.startswith(QSVT_MATRIX_CONFIG_PREFIX):
        return []
    case_name = str(matrix.get("case_name", "ieee14"))
    if case_name.lower() not in {case.lower() for case in case_names}:
        return []
    return _qsvt_case_rows(
        case_name=case_name,
        case_source=str(matrix.get("case_source", "pypower")),
        measurement_config={**DEFAULT_MEASUREMENT_CONFIG, **dict(matrix.get("measurement", {}))},
        config_path=config_path,
    )


def _qsvt_case_rows(
    *,
    case_name: str,
    case_source: str,
    measurement_config: dict[str, Any],
    config_path: Path,
) -> list[dict[str, Any]]:
    case = load_ac_case(case_name, case_source=case_source)
    state = default_ac_state_vector(case)
    _, _, measurement_rows = ac_measurements_and_jacobian(
        case,
        state,
        measurement_config,
    )
    state_dimension = len(case.angle_state_buses) + len(case.voltage_state_buses)
    return [
        _inventory_row(
            case_name=case.name,
            case_source=case_source,
            experiment_group="QSVT matrix / resource",
            model_type="ac_weighted_jacobian",
            config_path=config_path,
            measurement_type="qsvt_matrix_row",
            row_index=row_index,
            state_dimension=state_dimension,
            sigma=row.std,
            buses=row.buses,
            is_physical_measurement=False,
            is_after_weighting=True,
            notes=(
                "Diagnostic QSVT matrix row extracted from generated weighted AC Jacobian; "
                f"source AC measurement type={row.measurement_type}. "
                "This is not a raw field measurement row."
            ),
        )
        for row_index, row in enumerate(measurement_rows)
    ]


def _inventory_row(
    *,
    case_name: str,
    case_source: str,
    experiment_group: str,
    model_type: str,
    config_path: Path,
    measurement_type: str,
    row_index: int,
    state_dimension: int,
    sigma: float | None,
    buses: Sequence[int],
    is_physical_measurement: bool,
    is_after_weighting: bool,
    notes: str,
) -> dict[str, Any]:
    bus_i = int(buses[0]) if len(buses) >= 1 else None
    bus_j = int(buses[1]) if len(buses) >= 2 else None
    weight = None if sigma is None else 1.0 / float(sigma)
    return {
        "case_name": _display_case(case_name),
        "case_source": case_source,
        "experiment_group": experiment_group,
        "model_type": model_type,
        "config_file": _display_path(config_path),
        "measurement_type": measurement_type,
        "bus_i": bus_i,
        "bus_j": bus_j,
        "row_index": int(row_index),
        "sigma": None if sigma is None else float(sigma),
        "weight": weight,
        "is_physical_measurement": bool(is_physical_measurement),
        "is_synthetic_row": measurement_type == "synthetic_weighted_row",
        "is_generated": True,
        "is_after_weighting": bool(is_after_weighting),
        "pmu_scada_status": "conceptual_label_only_no_field_records",
        "notes": notes,
        "state_dimension": int(state_dimension),
    }


def _experiment_group(mode: str, case_source: str, config_path: Path) -> str:
    if mode == "dc_power_flow_linearized":
        return "DC-linearized"
    if mode == "ac_power_flow_linearized" and case_source == "pypower":
        return "PYPOWER AC-linearized"
    if mode == "ac_power_flow_linearized":
        if "bad_data" in config_path.name:
            return "AC-linearized built-in IEEE14 / bad-data stress"
        return "AC-linearized built-in IEEE14"
    if mode == "nonlinear_ac_state_estimation" and case_source == "pypower":
        return "PYPOWER nonlinear AC"
    if mode in {"nonlinear_ac_state_estimation", "ac_iterative_state_estimation"}:
        return "Nonlinear AC built-in IEEE14"
    return mode


def _summary_by_case(inventory: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "case_name",
        "case_source",
        "experiment_group",
        "model_type",
        "config_file",
        "row_count_before_missing",
        "state_dimension",
        "redundancy_ratio",
        "physical_row_count",
        "synthetic_row_count",
        "generated_row_count",
        "after_weighting_row_count",
    ]
    if inventory.empty:
        return pd.DataFrame(columns=columns)
    grouped = inventory.groupby(
        ["case_name", "case_source", "experiment_group", "model_type", "config_file"],
        dropna=False,
        sort=True,
    )
    rows = []
    for group_key, frame in grouped:
        state_dimension = int(frame["state_dimension"].dropna().iloc[0])
        row_count = len(frame)
        rows.append(
            {
                **dict(
                    zip(
                        [
                            "case_name",
                            "case_source",
                            "experiment_group",
                            "model_type",
                            "config_file",
                        ],
                        group_key,
                        strict=True,
                    )
                ),
                "row_count_before_missing": row_count,
                "state_dimension": state_dimension,
                "redundancy_ratio": row_count / state_dimension if state_dimension else None,
                "physical_row_count": int(frame["is_physical_measurement"].sum()),
                "synthetic_row_count": int(frame["is_synthetic_row"].sum()),
                "generated_row_count": int(frame["is_generated"].sum()),
                "after_weighting_row_count": int(frame["is_after_weighting"].sum()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _summary_by_type(inventory: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "measurement_type",
        "experiment_group",
        "model_type",
        "row_count_before_missing",
        "case_count",
        "config_count",
        "min_sigma",
        "max_sigma",
        "min_weight",
        "max_weight",
        "physical_row_count",
        "synthetic_row_count",
        "generated_row_count",
        "after_weighting_row_count",
    ]
    if inventory.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    grouped = inventory.groupby(
        ["measurement_type", "experiment_group", "model_type"],
        dropna=False,
        sort=True,
    )
    for group_key, frame in grouped:
        rows.append(
            {
                **dict(
                    zip(
                        ["measurement_type", "experiment_group", "model_type"],
                        group_key,
                        strict=True,
                    )
                ),
                "row_count_before_missing": len(frame),
                "case_count": int(frame["case_name"].nunique()),
                "config_count": int(frame["config_file"].nunique()),
                "min_sigma": _numeric_min(frame["sigma"]),
                "max_sigma": _numeric_max(frame["sigma"]),
                "min_weight": _numeric_min(frame["weight"]),
                "max_weight": _numeric_max(frame["weight"]),
                "physical_row_count": int(frame["is_physical_measurement"].sum()),
                "synthetic_row_count": int(frame["is_synthetic_row"].sum()),
                "generated_row_count": int(frame["is_generated"].sum()),
                "after_weighting_row_count": int(frame["is_after_weighting"].sum()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _exact_row_count_tables(case_names: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    dc_rows = []
    dc_case = load_dc_case("ieee14", case_source="builtin")
    _, dc_measurements = build_dc_measurement_matrix(
        case=dc_case,
        measurement_config={
            "include_branch_flows": True,
            "include_bus_injections": True,
            "angle_buses": [2, 6, 9, 14],
            "flow_std": 0.02,
            "injection_std": 0.03,
            "angle_std": 0.005,
        },
    )
    dc_type_counts = pd.Series([row.measurement_type for row in dc_measurements]).value_counts()
    dc_rows.append(
        {
            "case": "IEEE14",
            "case_source": "builtin",
            "branch_flow": int(dc_type_counts.get("branch_flow", 0)),
            "bus_injection": int(dc_type_counts.get("bus_injection", 0)),
            "angle_rows": int(dc_type_counts.get("angle", 0)),
            "total_rows": len(dc_measurements),
            "state_dimension": len(dc_case.state_buses),
            "notes": "Built-in DC profile before random missing-row removal.",
        }
    )

    ac_rows = []
    if "ieee14" in {case.lower() for case in case_names}:
        ac_rows.append(
            _exact_ac_row_count(
                case_name="ieee14",
                case_source="builtin",
                experiment_group="AC-linearized built-in IEEE14",
                measurement_config={
                    "include_voltage_magnitudes": True,
                    "include_p_injections": True,
                    "include_q_injections": True,
                    "include_p_branch_flows": True,
                    "include_q_branch_flows": True,
                },
            )
        )
    for case_name in case_names:
        normalized = case_name.lower()
        ac_rows.append(
            _exact_ac_row_count(
                case_name=normalized,
                case_source="pypower",
                experiment_group="PYPOWER AC-linearized",
                measurement_config={
                    "include_voltage_magnitudes": True,
                    "include_p_injections": True,
                    "include_q_injections": False,
                    "include_p_branch_flows": True,
                    "include_q_branch_flows": False,
                },
            )
        )
        ac_rows.append(
            _exact_ac_row_count(
                case_name=normalized,
                case_source="pypower",
                experiment_group="PYPOWER nonlinear AC",
                measurement_config={
                    "include_voltage_magnitudes": True,
                    "include_p_injections": True,
                    "include_q_injections": True,
                    "include_p_branch_flows": True,
                    "include_q_branch_flows": True,
                },
            )
        )
    return pd.DataFrame(dc_rows), pd.DataFrame(ac_rows)


def _exact_ac_row_count(
    *,
    case_name: str,
    case_source: str,
    experiment_group: str,
    measurement_config: dict[str, Any],
) -> dict[str, Any]:
    case = load_ac_case(case_name, case_source=case_source)
    state = default_ac_state_vector(case)
    _, _, rows = ac_measurements_and_jacobian(case, state, measurement_config)
    counts = pd.Series([row.measurement_type for row in rows]).value_counts()
    state_dimension = len(case.angle_state_buses) + len(case.voltage_state_buses)
    return {
        "case": _display_case(case.name),
        "case_source": case_source,
        "experiment_group": experiment_group,
        "v_rows": int(counts.get("voltage_magnitude", 0)),
        "p_injection_rows": int(counts.get("p_injection", 0)),
        "q_injection_rows": int(counts.get("q_injection", 0)),
        "p_branch_flow_rows": int(counts.get("p_branch_flow", 0)),
        "q_branch_flow_rows": int(counts.get("q_branch_flow", 0)),
        "total_rows": len(rows),
        "state_dimension": state_dimension,
        "notes": "Exact generated row layout before random missing-row removal.",
    }


def _numeric_min(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return None if numeric.empty else float(numeric.min())


def _numeric_max(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return None if numeric.empty else float(numeric.max())


def _summary_markdown(
    inventory: pd.DataFrame,
    by_case: pd.DataFrame,
    by_type: pd.DataFrame,
    exact_dc_counts: pd.DataFrame,
    exact_ac_counts: pd.DataFrame,
    skipped: list[dict[str, str]],
) -> str:
    total_rows = len(inventory)
    physical_rows = int(inventory["is_physical_measurement"].sum()) if total_rows else 0
    synthetic_rows = total_rows - physical_rows
    lines = [
        "# Measurement Inventory Summary",
        "",
        "This inventory is generated from repository configs and code-defined measurement rows. "
        "Counts are before random missing-row removal.",
        "",
        f"- Total rows before missing removal: {total_rows}",
        f"- Generated physical measurement-equation rows: {physical_rows}",
        f"- Synthetic or diagnostic rows: {synthetic_rows}",
        "- PMU/SCADA labels are conceptual in this repository; no field PMU/SCADA "
        "records are loaded.",
        "- IEEE/PYPOWER cases provide network benchmark models. Measurement rows are "
        "generated by code.",
        "- Where sigma is available, implicit diagonal covariance is R_ii = sigma_i^2 "
        "and weight is 1/sigma_i.",
        "",
        "## Summary by Case and Experiment Group",
        "",
        _markdown_table(
            by_case[
                [
                    "case_name",
                    "experiment_group",
                    "model_type",
                    "config_file",
                    "row_count_before_missing",
                    "state_dimension",
                    "redundancy_ratio",
                    "physical_row_count",
                    "generated_row_count",
                    "after_weighting_row_count",
                ]
            ],
            max_rows=80,
        ),
        "",
        "## Summary by Measurement Type",
        "",
        _markdown_table(by_type, max_rows=120),
        "",
        "## Exact DC Row Counts",
        "",
        _markdown_table(exact_dc_counts, max_rows=20),
        "",
        "## Exact AC Row Counts",
        "",
        _markdown_table(exact_ac_counts, max_rows=40),
        "",
        "## Interpretation Notes",
        "",
        "- `synthetic_weighted_row` rows are abstract weighted rows and have no bus index.",
        "- DC rows are branch-flow, bus-injection, and configured angle equations.",
        "- AC single-step rows use generated voltage magnitude, P/Q injection, and P/Q "
        "branch-flow equations and are reported after row weighting.",
        "- Nonlinear AC rows describe raw generated measurements; weighting occurs inside "
        "each iterative update system.",
        "- `qsvt_matrix_row` rows are diagnostic weighted-Jacobian rows used for matrix "
        "or resource-estimation evidence, not raw physical measurements.",
        "",
        "## Coverage and Gaps",
        "",
        "- PYPOWER-backed AC and nonlinear AC inventory covers IEEE14, IEEE30, IEEE57, "
        "IEEE118, and IEEE300 when the corresponding configs are present.",
        "- Built-in AC and DC fixture configs cover IEEE14 only because the built-in "
        "fixture loader only supports IEEE14.",
        "- Synthetic configs are reported as IEEE14-labeled only where the configs use "
        "`case_name: ieee14`; the rows are not physical IEEE measurements.",
        "",
    ]
    if skipped:
        lines.extend(
            [
                "## Skipped Relevant Configs",
                "",
                _markdown_table(pd.DataFrame(skipped), max_rows=120),
                "",
            ]
        )
    return "\n".join(lines)


def _readme_text() -> str:
    return """# Measurement Inventory

Generated by `scripts/export_measurement_inventory.py`.

The CSV files list generated measurement rows before missing-row removal. IEEE and
PYPOWER cases are benchmark network models; this inventory does not represent
field PMU/SCADA records.

Key files:

- `measurement_inventory_all_cases.csv`: row-level inventory.
- `measurement_inventory_by_case.csv`: case/config-level counts and redundancy ratios.
- `measurement_inventory_by_type.csv`: measurement-type counts and sigma/weight ranges.
- `exact_dc_row_counts.csv`: paper-ready DC row-count table before missing-row removal.
- `exact_ac_row_counts.csv`: paper-ready AC row-count table before missing-row removal.
- `measurement_inventory_summary.md`: human-readable summary for paper review.
- `measurement_inventory_manifest.json`: provenance for this export.
"""


def _markdown_table(frame: pd.DataFrame, *, max_rows: int) -> str:
    if frame.empty:
        return "_No rows available._"
    visible = frame.head(max_rows).copy()
    for column in visible.columns:
        visible[column] = visible[column].map(_format_cell)
    header = "| " + " | ".join(str(column) for column in visible.columns) + " |"
    divider = "| " + " | ".join("---" for _ in visible.columns) + " |"
    body = [
        "| " + " | ".join(str(row[column]) for column in visible.columns) + " |"
        for _, row in visible.iterrows()
    ]
    if len(frame) > max_rows:
        body.append(f"| ... | {' | '.join(['...'] * (len(visible.columns) - 1))} |")
    return "\n".join([header, divider, *body])


def _format_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    text = str(value)
    return text.replace("\n", " ").replace("|", "\\|")


def _display_case(case_name: str) -> str:
    normalized = str(case_name).lower()
    if normalized.startswith("ieee"):
        return normalized.upper()
    return normalized


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=REPO_ROOT / "configs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "measurement_inventory",
    )
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--no-qsvt", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_measurement_inventory(
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        case_names=tuple(args.cases or DEFAULT_CASES),
        include_qsvt=not args.no_qsvt,
    )
    print(f"Measurement inventory written to {result['output_dir']}")
    print(f"Rows exported: {len(result['inventory'])}")


if __name__ == "__main__":
    main()
