"""Phase 6: readout-limitation formalization.

Consolidates the existing gate-observable readout, amplitude-estimation, and
norm-recovery artifacts into a single observable-vs-full-vector readout table, a cost
table, and a top-k margin-sensitivity table. It formalizes which observables avoid
full-vector tomography and records that full update-vector recovery still requires
full-vector readout (an assumption, not an executed result). This phase reads existing
artifacts only and never fabricates readout results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.paper import PAPER_CLAIM_BOUNDARY
from robust_qsvt_se.paper._common import read_csv, rows_to_table
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

SOURCE_SCRIPT = "scripts/build_readout_limitation_formalization.py"

OBSERVABLE_COLUMNS = [
    "observable_name",
    "requires_full_vector_readout",
    "requires_norm_recovery",
    "requires_signed_overlap",
    "requires_topk_margin",
    "readout_protocol",
    "readout_cost_proxy",
    "supported_by_gate_values",
    "claim_boundary",
    "notes",
]
COST_COLUMNS = [
    "observable_name",
    "readout_cost_proxy",
    "practical_status",
    "median_absolute_error_gate_vs_ridge",
    "median_relative_error_gate_vs_ridge",
    "n_source_rows",
    "source_artifact",
]
TOPK_COLUMNS = [
    "case",
    "alpha",
    "degree",
    "top_k_match_if_applicable",
    "margin_condition",
    "exact_when_well_separated",
    "source_artifact",
]

_ACCURACY_FILES = (
    (
        "cross_case",
        "qsvt_cross_case_gate_observable_readout/cross_case_gate_observable_accuracy_cost.csv",
    ),
    ("ieee118", "qsvt_ieee118_gate_observable_readout/ieee118_gate_observable_accuracy_cost.csv"),
    ("gate", "qsvt_gate_observable_readout/gate_observable_accuracy_cost.csv"),
)
_VALUES_FILES = (
    "qsvt_cross_case_gate_observable_readout/cross_case_gate_observable_values.csv",
    "qsvt_ieee118_gate_observable_readout/ieee118_gate_observable_values.csv",
    "qsvt_gate_observable_readout/gate_observable_values.csv",
)

# Observable semantics: signed directional updates need a signed overlap; the energy and
# top-k observables do not; only top-k identification needs a magnitude-separation margin.
_SIGNED_OVERLAP = {
    "bus_angle_update",
    "bus_voltage_update",
    "branch_angle_difference_proxy",
    "measurement_row_correction_proxy",
}
_PROTOCOL = {
    "bus_angle_update": "selected_overlap_with_norm_recovery",
    "bus_voltage_update": "selected_overlap_with_norm_recovery",
    "branch_angle_difference_proxy": "selected_overlap_with_norm_recovery",
    "measurement_row_correction_proxy": "selected_overlap_with_norm_recovery",
    "selected_area_update_energy": "amplitude_estimation_energy",
    "top_k_update_identification": "top_k_amplitude_ranking",
}
_OBSERVABLE_CLAIM = (
    "observable-first readout of selected physical observables; not full-vector tomography"
)


def build_readout_limitation_formalization(config: dict[str, Any]) -> dict[str, Any]:
    input_root = Path(config.get("input_root", "outputs"))
    output_dir = Path(config.get("output_dir", input_root / "readout_limitation_formalization"))

    accuracy = _load_accuracy(input_root)
    gate_observables = _gate_value_observables(input_root)
    observable_rows = _observable_rows(accuracy, gate_observables)
    cost_rows = _cost_rows(accuracy)
    topk_rows = _topk_rows(accuracy)

    return _write_outputs(
        output_dir=output_dir,
        input_root=input_root,
        observable_rows=observable_rows,
        cost_rows=cost_rows,
        topk_rows=topk_rows,
        input_config={"input_root": str(input_root), "output_dir": str(output_dir)},
    )


def _load_accuracy(input_root: Path) -> pd.DataFrame:
    frames = []
    for tag, rel in _ACCURACY_FILES:
        frame = read_csv(input_root / rel)
        if frame.empty or "observable_name" not in frame.columns:
            continue
        frame = frame.copy()
        frame["source_artifact"] = rel
        frame["source_tag"] = tag
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _gate_value_observables(input_root: Path) -> set[str]:
    observables: set[str] = set()
    for rel in _VALUES_FILES:
        frame = read_csv(input_root / rel)
        if not frame.empty and "observable_name" in frame.columns:
            observables.update(frame["observable_name"].astype(str).unique())
    return observables


def _truthy(series: pd.Series) -> bool:
    if series.empty:
        return False
    value = series.iloc[0]
    return str(value).strip().lower() in ("true", "1", "yes")


def _observable_rows(accuracy: pd.DataFrame, gate_observables: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not accuracy.empty:
        for observable, group in accuracy.groupby("observable_name", sort=True):
            name = str(observable)
            requires_norm = _truthy(group.get("requires_norm_recovery", pd.Series(dtype=object)))
            requires_full = _truthy(
                group.get("requires_full_vector_readout", pd.Series(dtype=object))
            )
            requires_topk = name == "top_k_update_identification"
            rows.append(
                {
                    "observable_name": name,
                    "requires_full_vector_readout": "yes" if requires_full else "no",
                    "requires_norm_recovery": "yes" if requires_norm else "no",
                    "requires_signed_overlap": "yes" if name in _SIGNED_OVERLAP else "no",
                    "requires_topk_margin": "yes" if requires_topk else "no",
                    "readout_protocol": _PROTOCOL.get(name, "selected_observable_measurement"),
                    "readout_cost_proxy": _first_numeric(group.get("readout_cost_proxy")),
                    "supported_by_gate_values": "yes" if name in gate_observables else "no",
                    "claim_boundary": _OBSERVABLE_CLAIM,
                    "notes": (
                        "observable-first route avoids full-vector reconstruction"
                        if not requires_full
                        else "requires full-vector readout"
                    ),
                }
            )
    # Formalize the residual limitation: the full update vector still needs full-vector readout.
    rows.append(
        {
            "observable_name": "full_update_vector",
            "requires_full_vector_readout": "yes",
            "requires_norm_recovery": "yes",
            "requires_signed_overlap": "yes",
            "requires_topk_margin": "no",
            "readout_protocol": "full_vector_tomography_assumption",
            "readout_cost_proxy": "",
            "supported_by_gate_values": "no",
            "claim_boundary": (
                "full update-vector recovery requires full-vector readout; assumption-only, "
                "not executed"
            ),
            "notes": (
                "assumption_only: the observable-first route is a partial workaround; recovering "
                "the entire update vector is not solved"
            ),
        }
    )
    return rows


def _first_numeric(series: pd.Series | None) -> Any:
    if series is None or series.empty:
        return ""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.iloc[0]) if not numeric.empty else ""


def _median_numeric(series: pd.Series | None) -> float:
    if series is None:
        return float("nan")
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(numeric.median()) if not numeric.empty else float("nan")


def _cost_rows(accuracy: pd.DataFrame) -> list[dict[str, Any]]:
    if accuracy.empty:
        return []
    rows = []
    for observable, group in accuracy.groupby("observable_name", sort=True):
        practical = group.get("practical_status")
        rows.append(
            {
                "observable_name": str(observable),
                "readout_cost_proxy": _first_numeric(group.get("readout_cost_proxy")),
                "practical_status": (
                    str(practical.iloc[0]) if practical is not None and not practical.empty else ""
                ),
                "median_absolute_error_gate_vs_ridge": _median_numeric(
                    group.get("absolute_error_gate_vs_ridge")
                ),
                "median_relative_error_gate_vs_ridge": _median_numeric(
                    group.get("relative_error_gate_vs_ridge")
                ),
                "n_source_rows": len(group),
                "source_artifact": str(group["source_artifact"].iloc[0]),
            }
        )
    return rows


def _topk_rows(accuracy: pd.DataFrame) -> list[dict[str, Any]]:
    if accuracy.empty:
        return []
    topk = accuracy[accuracy["observable_name"] == "top_k_update_identification"]
    if topk.empty or "top_k_match_if_applicable" not in topk.columns:
        return []
    rows = []
    for _, item in topk.iterrows():
        match = pd.to_numeric(pd.Series([item.get("top_k_match_if_applicable")]), errors="coerce")
        match_value = float(match.iloc[0]) if not match.isna().all() else float("nan")
        rows.append(
            {
                "case": item.get("case", ""),
                "alpha": item.get("alpha", ""),
                "degree": item.get("degree", ""),
                "top_k_match_if_applicable": match_value,
                "margin_condition": "leading-magnitude separation must exceed readout precision",
                "exact_when_well_separated": "yes" if match_value >= 1.0 else "no",
                "source_artifact": item.get("source_artifact", ""),
            }
        )
    return rows


def _summary_markdown(
    observable_rows: list[dict[str, Any]],
    cost_rows: list[dict[str, Any]],
    topk_rows: list[dict[str, Any]],
) -> str:
    avoid_full = [
        r["observable_name"] for r in observable_rows if r["requires_full_vector_readout"] == "no"
    ]
    need_norm = [
        r["observable_name"] for r in observable_rows if r["requires_norm_recovery"] == "yes"
    ]
    topk_exact = (
        all(r["exact_when_well_separated"] == "yes" for r in topk_rows) if topk_rows else False
    )
    lines = [
        "# Readout-Limitation Formalization (Phase 6)",
        "",
        PAPER_CLAIM_BOUNDARY,
        "",
        "This phase consolidates existing gate-observable readout, amplitude-estimation, and "
        "norm-recovery artifacts. It does not run new circuits and does not solve full-vector "
        "tomography.",
        "",
        "## Interpretation",
        "",
        f"1. **Which observables avoid full-vector reconstruction?** {', '.join(avoid_full)} "
        "(observable-first readout).",
        f"2. **Which observables still require norm/sign recovery?** {', '.join(need_norm)} "
        "require norm recovery; signed directional updates additionally require a signed overlap.",
        "3. **When does top-k identification fail?** Top-k identification is "
        + (
            "exact in the consolidated runs but only when the leading-magnitude separation "
            "exceeds the readout precision (margin condition in `topk_margin_sensitivity.csv`); "
            "it degrades when magnitudes are nearly tied."
            if topk_exact
            else "conditioned on magnitude separation; see `topk_margin_sensitivity.csv`."
        ),
        "4. **What full-vector readout assumption remains?** Recovering the entire update vector "
        "still requires full-vector readout (the `full_update_vector` row is "
        "`assumption_only`, not executed).",
        "5. **How should this limitation be worded in the manuscript?** Observable-first readout "
        "recovers selected physical observables without full-vector tomography; full "
        "output-direction readout of the complete update vector remains an assumption for the "
        "full IEEE-scale pathway. Top-k exactness is conditioned on magnitude separation.",
        "",
        "Full-vector readout is **not** claimed solved. The observable-first route is a partial "
        "workaround. Top-k exactness is conditioned on margin.",
    ]
    return "\n".join(lines)


def _write_outputs(
    *,
    output_dir: Path,
    input_root: Path,
    observable_rows: list[dict[str, Any]],
    cost_rows: list[dict[str, Any]],
    topk_rows: list[dict[str, Any]],
    input_config: dict[str, Any],
) -> dict[str, Any]:
    ensure_directory(output_dir)
    observable_path = output_dir / "observable_vs_full_vector_readout.csv"
    cost_path = output_dir / "readout_cost_table.csv"
    topk_path = output_dir / "topk_margin_sensitivity.csv"
    summary_path = output_dir / "readout_limitation_summary.md"

    rows_to_table(observable_rows, observable_path, OBSERVABLE_COLUMNS)
    rows_to_table(cost_rows, cost_path, COST_COLUMNS)
    rows_to_table(topk_rows, topk_path, TOPK_COLUMNS)
    summary_path.write_text(
        _summary_markdown(observable_rows, cost_rows, topk_rows), encoding="utf-8"
    )

    artifacts = {
        "readout_cost_table": str(cost_path),
        "observable_vs_full_vector_readout": str(observable_path),
        "topk_margin_sensitivity": str(topk_path),
        "readout_limitation_summary": str(summary_path),
    }
    paper_dir = _index_into_package(input_root, artifacts)
    write_manifest(
        output_dir,
        artifacts=artifacts,
        input_config=input_config,
        claim_boundary=PAPER_CLAIM_BOUNDARY,
    )
    full_vector_assumption = any(
        r["observable_name"] == "full_update_vector" and r["requires_full_vector_readout"] == "yes"
        for r in observable_rows
    )
    return {
        "output_dir": output_dir,
        "paper_dir": paper_dir,
        "observable_rows": observable_rows,
        "cost_rows": cost_rows,
        "topk_rows": topk_rows,
        "full_vector_assumption_recorded": full_vector_assumption,
        "artifacts": artifacts,
    }


def _index_into_package(input_root: Path, artifacts: dict[str, str]) -> Path | None:
    package = input_root / "final_manuscript_package" / "phase6_readout_limitation_formalization"
    try:
        ensure_directory(package)
    except Exception:
        return None
    for key in (
        "observable_vs_full_vector_readout",
        "readout_cost_table",
        "topk_margin_sensitivity",
        "readout_limitation_summary",
    ):
        target = package / Path(artifacts[key]).name
        try:
            target.write_text(Path(artifacts[key]).read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            continue
    return package
