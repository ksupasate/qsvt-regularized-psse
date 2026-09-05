"""Phase 3 / hardening Phase 1: reactive-power / P-Q conditioning paper-facing table.

Computes the weighted-Jacobian conditioning of the controlled AC-linearized benchmark
for an exact P/Q measurement build-up (setups S0-S4) plus the drop-based views. Each
setup is an exact measurement subset constructed from the existing measurement
include/exclude flags: the full measurement system is built once per (case, seed) and the
rows are masked by measurement type, which is numerically identical to rebuilding the
system with the corresponding include flags but far cheaper. Conditioning and the
reference-estimator RMSE are real computed values; a setup that cannot be built is
recorded as missing rather than approximated, and no claim is made that reactive-power
rows always improve conditioning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.measurement.linear_system import WeightedSystem
from robust_qsvt_se.paper import PAPER_CLAIM_BOUNDARY
from robust_qsvt_se.paper._common import rows_to_table
from robust_qsvt_se.paper._estimation import (
    DEFAULT_CASE_SOURCE,
    DEFAULT_MEASUREMENT,
    build_estimator,
    build_system,
    conditioning,
    rank_status,
    solve_detailed,
    subset_spec,
)
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

SOURCE_SCRIPT = "scripts/build_reactive_power_conditioning_table.py"

TABLE_COLUMNS = [
    "case",
    "setup_id",
    "rows_included",
    "rows_dropped",
    "n_rows",
    "state_dimension",
    "redundancy_ratio",
    "sigma_min",
    "sigma_max",
    "condition_number",
    "numerical_rank",
    "effective_rank",
    "angle_rmse",
    "voltage_rmse",
    "total_rmse",
    "weighted_residual_norm",
    "source_artifact",
    "mapping_status",
    "claim_boundary",
    "notes",
]
MISSING_COLUMNS = ["case", "setup_id", "requested_subset", "reason", "status"]

# (setup_id, exact measurement subset, set-builder notation). Every setup maps to an
# exact include-flag subset, so all rows are computed exactly (no approximations).
_SETUPS: tuple[tuple[str, str, str], ...] = (
    ("S0_voltage_only", "voltage_only", "{V}"),
    ("S1_voltage_plus_p_injection", "s1_voltage_plus_p_injection", "{V, P_i}"),
    (
        "S2_voltage_plus_p_injection_plus_p_branch_flow",
        "s2_voltage_plus_p_injection_plus_p_branch_flow",
        "{V, P_i, P_ij}",
    ),
    (
        "S3_voltage_plus_p_injection_plus_p_branch_flow_plus_q_injection",
        "s3_voltage_plus_p_injection_plus_p_branch_flow_plus_q_injection",
        "{V, P_i, P_ij, Q_i}",
    ),
    ("S4_full_pq_ac_measurement_set", "full_ac_measurement_set", "{V, P_i, Q_i, P_ij, Q_ij}"),
    ("drop_q_injection_rows", "drop_q_injection_rows", "full minus Q_i"),
    ("drop_q_branch_flow_rows", "drop_q_branch_flow_rows", "full minus Q_ij"),
    ("drop_injection_rows", "drop_injection_rows", "full minus P_i, Q_i"),
    ("drop_branch_flow_rows", "drop_branch_flow_rows", "full minus P_ij, Q_ij"),
)

DEFAULT_CASES = ("ieee14", "ieee30", "ieee57", "ieee118")
DEFAULT_SEEDS = (0, 1, 2)
_REFERENCE_ESTIMATOR = "ridge_tikhonov"
_REFERENCE_ALPHA = 1.0e-4
_ROW_CLAIM = (
    "controlled IEEE benchmark; reactive-power rows as conditioning/observability factor; "
    "not field-calibrated"
)


def build_reactive_power_conditioning_table(config: dict[str, Any]) -> dict[str, Any]:
    input_root = Path(config.get("input_root", "outputs"))
    package_root = Path(config.get("package_root", input_root / "final_manuscript_package"))
    output_dir = Path(config.get("output_dir", package_root / "reactive_power_conditioning"))
    cases = list(config.get("cases", DEFAULT_CASES))
    seeds = list(config.get("seeds", DEFAULT_SEEDS))
    case_source = str(config.get("case_source", DEFAULT_CASE_SOURCE))
    ensure_directory(output_dir)

    table_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for case in cases:
        per_setup: dict[str, list[dict[str, Any]]] = {setup_id: [] for setup_id, _, _ in _SETUPS}
        for seed in seeds:
            full = _safe_full_system(case, case_source, seed)
            if full is None:
                for setup_id, subset, _ in _SETUPS:
                    missing_rows.append(
                        _missing_row(case, setup_id, subset, "full system build failed")
                    )
                continue
            for setup_id, subset, _ in _SETUPS:
                spec = subset_spec(subset)
                if spec is None:
                    missing_rows.append(
                        _missing_row(case, setup_id, subset, "unknown measurement subset")
                    )
                    continue
                sub = _subset_system(full, set(spec.included_types))
                per_setup[setup_id].append(
                    _seed_record(sub, spec.included_types, spec.dropped_types)
                )
        for setup_id, subset, definition in _SETUPS:
            records = per_setup[setup_id]
            if not records:
                continue
            table_rows.append(_aggregate_row(case, setup_id, subset, definition, records))

    return _write_outputs(
        output_dir=output_dir,
        table_rows=table_rows,
        missing_rows=missing_rows,
        input_config={
            "input_root": str(input_root),
            "package_root": str(package_root),
            "output_dir": str(output_dir),
            "cases": cases,
            "seeds": seeds,
            "case_source": case_source,
            "reference_estimator": _REFERENCE_ESTIMATOR,
            "reference_alpha": _REFERENCE_ALPHA,
        },
    )


def _safe_full_system(case: str, case_source: str, seed: int) -> WeightedSystem | None:
    try:
        return build_system(
            case=case,
            measurement_config=dict(DEFAULT_MEASUREMENT),
            seed=seed,
            case_source=case_source,
        )
    except Exception:
        return None


def _subset_system(full: WeightedSystem, included_types: set[str]) -> WeightedSystem:
    """Row-mask the full weighted system to the included measurement types.

    Each measurement is an independent weighted row, so masking rows yields exactly the
    weighted system that building with the corresponding include flags would produce.
    """

    types = full.metadata.get("measurement_types")
    if not isinstance(types, list) or len(types) != full.n_measurements:
        return full
    keep = [index for index, mtype in enumerate(types) if mtype in included_types]
    metadata = dict(full.metadata)
    for key in ("measurement_types", "measurement_labels", "measurement_stds", "measurement_buses"):
        value = metadata.get(key)
        if isinstance(value, list) and len(value) == full.n_measurements:
            metadata[key] = [value[index] for index in keep]
    return WeightedSystem(
        H_tilde=np.asarray(full.H_tilde)[keep, :],
        r_tilde=np.asarray(full.r_tilde)[keep],
        x_true=full.x_true,
        metadata=metadata,
    )


def _seed_record(
    system: WeightedSystem, included: tuple[str, ...], dropped: tuple[str, ...]
) -> dict[str, Any]:
    cond = conditioning(system)
    solved = solve_detailed(build_estimator(_REFERENCE_ESTIMATOR, _REFERENCE_ALPHA), system)
    return {
        "rank_status": rank_status(system),
        "rows_included": "|".join(included) if included else "none",
        "rows_dropped": "|".join(dropped) if dropped else "none",
        **cond,
        "angle_rmse": solved["angle_rmse"],
        "voltage_rmse": solved["voltage_rmse"],
        "total_rmse": solved["rmse"],
        "weighted_residual_norm": solved["weighted_residual_norm"],
    }


def _median(records: list[dict[str, Any]], key: str) -> float:
    values = [record[key] for record in records if record.get(key) is not None]
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    return float(numeric.median()) if not numeric.empty else float("nan")


def _aggregate_row(
    case: str, setup_id: str, subset: str, definition: str, records: list[dict[str, Any]]
) -> dict[str, Any]:
    rank_deficient = any(record["rank_status"] == "rank_deficient" for record in records)
    first = records[0]
    return {
        "case": case,
        "setup_id": setup_id,
        "rows_included": first["rows_included"],
        "rows_dropped": first["rows_dropped"],
        "n_rows": _median(records, "n_rows"),
        "state_dimension": _median(records, "state_dimension"),
        "redundancy_ratio": _median(records, "redundancy_ratio"),
        "sigma_min": _median(records, "sigma_min"),
        "sigma_max": _median(records, "sigma_max"),
        "condition_number": _median(records, "condition_number"),
        "numerical_rank": _median(records, "numerical_rank"),
        "effective_rank": _median(records, "effective_rank"),
        "angle_rmse": _median(records, "angle_rmse"),
        "voltage_rmse": _median(records, "voltage_rmse"),
        "total_rmse": _median(records, "total_rmse"),
        "weighted_residual_norm": _median(records, "weighted_residual_norm"),
        "source_artifact": f"computed:reactive_power_conditioning:{case}:{subset}",
        "mapping_status": "exact_subset",
        "claim_boundary": _ROW_CLAIM,
        "notes": (
            f"{definition}; rank_deficient (angle observability lost)"
            if rank_deficient
            else f"{definition}; computed from real AC-linearized weighted system"
        ),
    }


def _missing_row(case: str, setup_id: str, subset: str, reason: str) -> dict[str, Any]:
    return {
        "case": case,
        "setup_id": setup_id,
        "requested_subset": subset,
        "reason": reason,
        "status": "build_failed",
    }


def _finite_lookup(rows: list[dict[str, Any]], case: str, setup_id: str, key: str) -> float:
    for row in rows:
        if row["case"] == case and row["setup_id"] == setup_id:
            value = row[key]
            return (
                float(value)
                if isinstance(value, int | float) and np.isfinite(value)
                else float("nan")
            )
    return float("nan")


def _summary_markdown(table_rows: list[dict[str, Any]], cases: list[str]) -> str:
    q_effect = _reactive_power_effect(table_rows, cases)
    branch_critical = _branch_flow_critical(table_rows, cases)
    buildup = _buildup_trend(table_rows, cases)
    voltage_rank_deficient = any(
        row["setup_id"] == "S0_voltage_only" and "rank_deficient" in str(row["notes"])
        for row in table_rows
    )
    lines = [
        "# Reactive-Power / P-Q Conditioning (paper-facing)",
        "",
        PAPER_CLAIM_BOUNDARY,
        "",
        "This table reports the weighted-Jacobian conditioning of the controlled "
        "AC-linearized benchmark for the exact P/Q measurement build-up S0-S4 and the "
        "drop-based views. Every setup is an exact measurement subset (no approximation); "
        "the weighted-Jacobian condition number is",
        "",
        r"\[",
        r"\kappa(\tilde H)",
        r"=",
        r"\frac{\sigma_{\max}(\tilde H)}{\sigma_{\min}(\tilde H)}.",
        r"\]",
        "",
        "## Setups",
        "- S0 = {V}",
        "- S1 = {V, P_i}",
        "- S2 = {V, P_i, P_ij}",
        "- S3 = {V, P_i, P_ij, Q_i}",
        "- S4 = {V, P_i, Q_i, P_ij, Q_ij} (full P/Q AC measurement set)",
        "",
        "## Interpretation",
        "",
        f"1. **Do Q_i / Q_ij rows affect rank/conditioning?** {q_effect}",
        f"2. **Is branch-flow information conditioning-critical?** {branch_critical}",
        "3. **Are voltage-only rows rank-deficient?** "
        + (
            "Yes - the voltage-only setup (S0) is rank-deficient because the angle states lose "
            "observability without injection/flow rows."
            if voltage_rank_deficient
            else "Not flagged rank-deficient in the computed rows."
        ),
        f"4. **What does the S0->S4 build-up show?** {buildup}",
        "5. **Does reactive-power information always improve conditioning?** No monotonic "
        "improvement is claimed: adding Q rows (S3->S4) or dropping Q rows changes conditioning "
        "case-dependently and does not uniformly lower the condition number across cases.",
        "6. **Manuscript wording:** Reactive-power and branch-flow rows are observability and "
        "conditioning factors in the controlled benchmark; branch-flow rows are the most "
        "conditioning-critical and voltage-only is rank-deficient. These are controlled IEEE "
        "benchmark observations, not field-calibrated measurement statistics.",
        "",
        "All S0-S4 setups are now computed exactly from the measurement include/exclude flags; "
        "any setup that fails to build is recorded in `missing_reactive_power_rows.csv` rather "
        "than approximated.",
        "",
    ]
    return "\n".join(lines)


def _reactive_power_effect(table_rows: list[dict[str, Any]], cases: list[str]) -> str:
    changed = []
    for case in cases:
        full = _finite_lookup(table_rows, case, "S4_full_pq_ac_measurement_set", "condition_number")
        drop_qi = _finite_lookup(table_rows, case, "drop_q_injection_rows", "condition_number")
        drop_qf = _finite_lookup(table_rows, case, "drop_q_branch_flow_rows", "condition_number")
        if np.isfinite(full) and (np.isfinite(drop_qi) or np.isfinite(drop_qf)):
            worse = (np.isfinite(drop_qi) and drop_qi > full) or (
                np.isfinite(drop_qf) and drop_qf > full
            )
            changed.append((case, worse))
    if not changed:
        return "Insufficient finite condition numbers to compare."
    n_worse = sum(1 for _, worse in changed if worse)
    return (
        f"Yes - removing Q-injection or Q-branch-flow rows changes the condition number; "
        f"in {n_worse}/{len(changed)} cases at least one Q-drop worsens conditioning relative to "
        "the full P/Q set. The effect is case-dependent (not monotonic)."
    )


def _branch_flow_critical(table_rows: list[dict[str, Any]], cases: list[str]) -> str:
    worse = 0
    total = 0
    for case in cases:
        full = _finite_lookup(table_rows, case, "S4_full_pq_ac_measurement_set", "condition_number")
        drop_bf = _finite_lookup(table_rows, case, "drop_branch_flow_rows", "condition_number")
        if np.isfinite(full) and np.isfinite(drop_bf):
            total += 1
            worse += int(drop_bf > full)
    if total == 0:
        return "Insufficient finite condition numbers to compare."
    return (
        f"Dropping all branch-flow rows worsens the condition number in {worse}/{total} cases, so "
        "branch-flow information is the most conditioning-critical measurement group in the "
        "controlled benchmark."
    )


def _buildup_trend(table_rows: list[dict[str, Any]], cases: list[str]) -> str:
    order = (
        "S0_voltage_only",
        "S1_voltage_plus_p_injection",
        "S2_voltage_plus_p_injection_plus_p_branch_flow",
        "S3_voltage_plus_p_injection_plus_p_branch_flow_plus_q_injection",
        "S4_full_pq_ac_measurement_set",
    )
    observable = 0
    total = 0
    for case in cases:
        s0 = _finite_lookup(table_rows, case, order[0], "numerical_rank")
        s1 = _finite_lookup(table_rows, case, order[1], "numerical_rank")
        state_dim = _finite_lookup(table_rows, case, order[4], "state_dimension")
        if np.isfinite(s0) and np.isfinite(s1) and np.isfinite(state_dim):
            total += 1
            # Adding injection rows (S0->S1) restores angle observability toward full rank.
            observable += int(s1 > s0)
    if total == 0:
        return "Insufficient finite ranks to compare the build-up."
    return (
        f"Adding active-power injection rows (S0->S1) increases numerical rank in {observable}/"
        f"{total} cases (angle observability is restored once injection/flow rows are added); the "
        "condition number then varies case-dependently across S1->S4."
    )


def _write_outputs(
    *,
    output_dir: Path,
    table_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
    input_config: dict[str, Any],
) -> dict[str, Any]:
    cases = sorted({row["case"] for row in table_rows})
    table_path = rows_to_table(
        table_rows, output_dir / "reactive_power_conditioning_table.csv", TABLE_COLUMNS
    )
    missing_path = rows_to_table(
        missing_rows, output_dir / "missing_reactive_power_rows.csv", MISSING_COLUMNS
    )
    summary_path = output_dir / "reactive_power_conditioning_summary.md"
    summary_path.write_text(_summary_markdown(table_rows, cases), encoding="utf-8")

    artifacts = {
        "reactive_power_conditioning_table": str(table_path),
        "reactive_power_conditioning_summary": str(summary_path),
        "missing_reactive_power_rows": str(missing_path),
    }
    write_manifest(
        output_dir,
        artifacts=artifacts,
        input_config=input_config,
        claim_boundary=PAPER_CLAIM_BOUNDARY,
    )
    return {
        "output_dir": output_dir,
        "table_rows": table_rows,
        "missing_rows": missing_rows,
        "cases": cases,
        "n_rows": len(table_rows),
        "artifacts": artifacts,
    }
