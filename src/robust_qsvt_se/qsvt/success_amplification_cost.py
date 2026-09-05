from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

SUCCESS_COST_CLAIM = (
    "This study reports postselection and amplitude-amplification cost proxies. "
    "It does not implement amplitude amplification as a hardware circuit."
)


def bottleneck_severity(p_success: float) -> str:
    p = _clip_success_probability(p_success)
    if p < 1.0e-6:
        return "severe"
    if p < 1.0e-3:
        return "high"
    if p < 1.0e-1:
        return "moderate"
    return "low"


def cost_row(
    source: str,
    p_success: float,
    qsvt_query_count: float | int | None = None,
) -> dict[str, Any]:
    p = _clip_success_probability(p_success)
    post_cost = 1.0 / p
    amp_cost = 1.0 / np.sqrt(p)
    query_count = np.nan if qsvt_query_count is None else float(qsvt_query_count)
    return {
        "source": source,
        "success_probability": p,
        "postselection_cost": post_cost,
        "amplitude_amplification_cost_proxy": amp_cost,
        "qsvt_query_count": query_count,
        "amplified_qsvt_query_proxy": (
            amp_cost * query_count if np.isfinite(query_count) else np.nan
        ),
        "bottleneck_severity": bottleneck_severity(p),
    }


def run_success_amplification_cost_study(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "input_dirs": [
            "outputs/gate_level_qsvt_ieee14_solver",
            "outputs/qsvt_matrix_free_ieee_experiments",
            "outputs/qsvt_matrix_free_ieee_resource_only",
        ],
        "output_dir": "outputs/qsvt_success_amplification_cost",
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])
    rows = collect_success_probability_rows([Path(value) for value in resolved["input_dirs"]])
    if not rows:
        rows = [
            {
                "source": "no_success_probability_inputs_found",
                "success_probability": 1.0,
                "qsvt_query_count": np.nan,
                "alpha": np.nan,
                "degree": np.nan,
                "case": "",
            }
        ]
    probability_rows = []
    for row in rows:
        cost = cost_row(row["source"], row["success_probability"], row.get("qsvt_query_count"))
        probability_rows.append({**row, **cost})
    artifacts = _write_outputs(output_dir, resolved, probability_rows)
    return {"output_dir": output_dir, "rows": probability_rows, "artifacts": artifacts}


def collect_success_probability_rows(input_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for input_dir in input_dirs:
        if not input_dir.exists():
            continue
        for path in sorted(input_dir.rglob("*")):
            if path.suffix == ".json":
                rows.extend(_rows_from_json(path))
            elif path.suffix == ".csv":
                rows.extend(_rows_from_csv(path))
    return _deduplicate_rows(rows)


def _rows_from_json(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    p = _extract_first(
        data,
        ["success_probability", "postselection_probability", "success_probability_proxy"],
    )
    if p is None:
        return []
    return [
        {
            "source": str(path),
            "success_probability": float(p),
            "qsvt_query_count": _extract_first(data, ["qsvt_query_count", "query_count_estimate"]),
            "alpha": _extract_first(data, ["alpha"]),
            "degree": _extract_first(data, ["requested_degree", "degree", "synthesized_degree"]),
            "case": str(_extract_first(data, ["case", "case_name"]) or ""),
        }
    ]


def _rows_from_csv(path: Path) -> list[dict[str, Any]]:
    try:
        frame = pd.read_csv(path)
    except Exception:
        return []
    p_column = next(
        (
            column
            for column in [
                "success_probability",
                "postselection_probability",
                "success_probability_proxy",
            ]
            if column in frame.columns
        ),
        None,
    )
    if p_column is None:
        return []
    rows = []
    for index, row in frame.iterrows():
        if not np.isfinite(float(row[p_column])):
            continue
        rows.append(
            {
                "source": f"{path}#row={index}",
                "success_probability": float(row[p_column]),
                "qsvt_query_count": _column_value(row, ["qsvt_query_count", "query_count"]),
                "alpha": _column_value(row, ["alpha"]),
                "degree": _column_value(row, ["degree", "requested_degree", "synthesized_degree"]),
                "case": str(_column_value(row, ["case", "case_name"]) or ""),
            }
        )
    return rows


def _write_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    sweep_path = output_dir / "success_probability_sweep.csv"
    post_path = output_dir / "postselection_cost.csv"
    amp_path = output_dir / "amplitude_amplification_cost.csv"
    tradeoff_path = output_dir / "alpha_degree_success_tradeoff.csv"
    interpretation_path = output_dir / "success_bottleneck_interpretation.md"
    frame = pd.DataFrame(rows)
    frame.to_csv(sweep_path, index=False)
    frame[["source", "success_probability", "postselection_cost", "bottleneck_severity"]].to_csv(
        post_path,
        index=False,
    )
    frame[
        [
            "source",
            "success_probability",
            "amplitude_amplification_cost_proxy",
            "qsvt_query_count",
            "amplified_qsvt_query_proxy",
            "bottleneck_severity",
        ]
    ].to_csv(amp_path, index=False)
    frame[
        [
            "source",
            "case",
            "alpha",
            "degree",
            "success_probability",
            "bottleneck_severity",
        ]
    ].to_csv(tradeoff_path, index=False)
    interpretation_path.write_text(_interpretation_markdown(rows), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "success_probability_sweep": str(sweep_path),
            "postselection_cost": str(post_path),
            "amplitude_amplification_cost": str(amp_path),
            "alpha_degree_success_tradeoff": str(tradeoff_path),
            "success_bottleneck_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=SUCCESS_COST_CLAIM,
    )
    return {
        "manifest": manifest,
        "success_probability_sweep": sweep_path,
        "postselection_cost": post_path,
        "amplitude_amplification_cost": amp_path,
        "alpha_degree_success_tradeoff": tradeoff_path,
        "success_bottleneck_interpretation": interpretation_path,
    }


def _interpretation_markdown(rows: list[dict[str, Any]]) -> str:
    probabilities = [float(row["success_probability"]) for row in rows]
    severities = sorted({str(row["bottleneck_severity"]) for row in rows})
    worst_post = max((float(row["postselection_cost"]) for row in rows), default=float("nan"))
    worst_amp = max(
        (float(row["amplitude_amplification_cost_proxy"]) for row in rows),
        default=float("nan"),
    )
    return "\n".join(
        [
            "# QSVT Success and Amplification Cost",
            "",
            SUCCESS_COST_CLAIM,
            "",
            f"- Success probability range: {min(probabilities):.17g} to {max(probabilities):.17g}",
            f"- Worst postselection cost: {worst_post:.17g}",
            f"- Worst amplitude-amplification proxy cost: {worst_amp:.17g}",
            f"- Bottleneck severities observed: {', '.join(severities)}",
            "",
        ]
    )


def _deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, float]] = set()
    unique = []
    for row in rows:
        key = (str(row["source"]), float(row["success_probability"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _extract_first(data: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _column_value(row: pd.Series, columns: list[str]) -> Any:
    for column in columns:
        if column in row.index:
            value = row[column]
            if pd.isna(value):
                return None
            return value
    return None


def _clip_success_probability(value: float) -> float:
    if not np.isfinite(float(value)):
        raise ValueError("success probability must be finite")
    return float(np.clip(float(value), 1.0e-300, 1.0))
