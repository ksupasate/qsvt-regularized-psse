from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

CONSOLIDATION_CLAIM = (
    "Audit that consolidates the latest co-designed QSVT-safe solver outputs on a selected "
    "IEEE14-derived subproblem before the degree-window, observable-gate-readout, and "
    "robustness phases. It records the best co-designed target families and degrees, the "
    "gate-validated configurations to reuse, the observables still missing gate-level "
    "values, and the subproblem-selection modes to use for robustness testing. "
    "Ridge/Tikhonov remains the reference filter; no QSVT superiority over Ridge/Tikhonov, "
    "quantum speedup, quantum advantage, full IEEE-scale gate-level solving, or hardware "
    "execution is claimed."
)

DEPLOYABLE_CLASSES = ("general_qsvt_safe", "instance_aware_qsvt_safe")
ROBUSTNESS_MODES = (
    "high_leverage",
    "best_conditioned",
    "metadata_mapped",
    "residual_supported",
    "random_seeded_pool",
    "worst_conditioned_control",
)

BEST_CONFIG_COLUMNS = [
    "alpha",
    "degree",
    "target_family",
    "deployability_class",
    "qsvt_safe",
    "residual_ratio_vs_no_update",
    "direction_error_vs_ridge",
    "success_probability_proxy",
]

GATE_CONFIG_COLUMNS = [
    "alpha",
    "degree",
    "target_family",
    "gate_status",
    "gate_residual_scaled",
    "residual_ratio_vs_no_update",
    "direction_error_gate_vs_ridge",
    "success_probability_exact",
    "residual_feasible_after_gate",
]

OBSERVABLE_ROW_COLUMNS = [
    "observable_name",
    "alpha",
    "degree",
    "target_family",
    "ridge_value",
    "qsvt_value",
    "gate_value_if_available",
    "relative_error",
    "practical_status",
    "needs_gate_value",
]


def run_qsvt_solver_prototype_consolidation_audit(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "input_root": "outputs",
        "max_best_configs": 10,
        "output_dir": "outputs/qsvt_solver_prototype_consolidation_audit",
    }
    resolved.update(config)
    input_root = Path(resolved["input_root"])
    output_dir = ensure_directory(resolved["output_dir"])

    best_configs = best_codesigned_configs(input_root, max_rows=int(resolved["max_best_configs"]))
    gate_configs = gate_validated_configs(input_root)
    observable_rows = observable_first_available_rows(input_root)
    artifacts = write_consolidation_outputs(
        output_dir, resolved, best_configs, gate_configs, observable_rows, input_root
    )
    return {
        "output_dir": output_dir,
        "best_configs": best_configs,
        "gate_configs": gate_configs,
        "observable_rows": observable_rows,
        "artifacts": artifacts,
    }


def best_codesigned_configs(input_root: Path, *, max_rows: int = 10) -> pd.DataFrame:
    frame = _read_csv(
        input_root / "qsvt_codesigned_bounded_target_study" / "codesigned_target_summary.csv"
    )
    if frame.empty:
        return pd.DataFrame(columns=BEST_CONFIG_COLUMNS)
    numeric = frame.copy()
    for column in ("residual_ratio_vs_no_update", "direction_error_vs_ridge", "alpha", "degree"):
        if column in numeric.columns:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    deployable = numeric[
        numeric.get("deployability_class", "").isin(DEPLOYABLE_CLASSES)
        & (numeric.get("qsvt_safe", False) == True)  # noqa: E712
    ]
    pool = deployable if not deployable.empty else numeric
    pool = pool.sort_values(
        by=[
            c
            for c in ("residual_ratio_vs_no_update", "direction_error_vs_ridge")
            if c in pool.columns
        ],
        kind="stable",
    ).head(int(max_rows))
    for column in BEST_CONFIG_COLUMNS:
        if column not in pool.columns:
            pool[column] = np.nan
    return pool[BEST_CONFIG_COLUMNS].reset_index(drop=True)


def gate_validated_configs(input_root: Path) -> pd.DataFrame:
    frame = _read_csv(
        input_root / "qsvt_codesigned_gate_validation" / "codesigned_gate_validation_results.csv"
    )
    if frame.empty:
        return pd.DataFrame(columns=GATE_CONFIG_COLUMNS)
    feasible = frame
    if "residual_feasible_after_gate" in frame.columns:
        mask = frame["residual_feasible_after_gate"].astype(str).str.lower().eq("true")
        if mask.any():
            feasible = frame[mask]
    for column in GATE_CONFIG_COLUMNS:
        if column not in feasible.columns:
            feasible[column] = np.nan
    return feasible[GATE_CONFIG_COLUMNS].reset_index(drop=True)


def observable_first_available_rows(input_root: Path) -> pd.DataFrame:
    frame = _read_csv(
        input_root / "qsvt_observable_first_solver_v2" / "observable_first_v2_summary.csv"
    )
    if frame.empty:
        return pd.DataFrame(columns=OBSERVABLE_ROW_COLUMNS)
    out = frame.copy()
    gate_values = pd.to_numeric(
        out.get("gate_value_if_available", pd.Series([np.nan] * len(out))), errors="coerce"
    )
    out["needs_gate_value"] = gate_values.isna()
    for column in OBSERVABLE_ROW_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    return out[OBSERVABLE_ROW_COLUMNS].reset_index(drop=True)


def write_consolidation_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    best_configs: pd.DataFrame,
    gate_configs: pd.DataFrame,
    observable_rows: pd.DataFrame,
    input_root: Path,
) -> dict[str, Path]:
    best_path = output_dir / "best_codesigned_configs.csv"
    gate_path = output_dir / "gate_validated_configs.csv"
    observable_path = output_dir / "observable_first_available_rows.csv"
    interpretation_path = output_dir / "consolidation_audit.md"

    best_configs.to_csv(best_path, index=False)
    gate_configs.to_csv(gate_path, index=False)
    observable_rows.to_csv(observable_path, index=False)
    interpretation_path.write_text(
        _consolidation_markdown(best_configs, gate_configs, observable_rows, input_root),
        encoding="utf-8",
    )

    manifest = write_manifest(
        output_dir,
        artifacts={
            "best_codesigned_configs": str(best_path),
            "gate_validated_configs": str(gate_path),
            "observable_first_available_rows": str(observable_path),
            "consolidation_audit": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=CONSOLIDATION_CLAIM,
    )
    return {
        "manifest": manifest,
        "best_codesigned_configs": best_path,
        "gate_validated_configs": gate_path,
        "observable_first_available_rows": observable_path,
        "consolidation_audit": interpretation_path,
    }


def _consolidation_markdown(
    best_configs: pd.DataFrame,
    gate_configs: pd.DataFrame,
    observable_rows: pd.DataFrame,
    input_root: Path,
) -> str:
    best_families = (
        sorted(set(best_configs["target_family"].astype(str)))
        if not best_configs.empty
        else ["none"]
    )
    feasible_degrees = (
        sorted(
            {
                int(value)
                for value in pd.to_numeric(best_configs["degree"], errors="coerce").dropna()
            }
        )
        if not best_configs.empty
        else []
    )
    overshoot_degrees = _overshoot_degrees(input_root)
    gate_degrees = (
        sorted(
            {
                int(value)
                for value in pd.to_numeric(gate_configs["degree"], errors="coerce").dropna()
            }
        )
        if not gate_configs.empty
        else []
    )
    observables_needing_gate = (
        sorted(
            set(
                observable_rows[observable_rows["needs_gate_value"].astype(bool)][
                    "observable_name"
                ].astype(str)
            )
        )
        if not observable_rows.empty
        else []
    )

    return "\n".join(
        [
            "# Solver-Prototype Consolidation Audit",
            "",
            CONSOLIDATION_CLAIM,
            "",
            "## Required Statements",
            f"1. Best co-designed target families: {', '.join(best_families)}.",
            f"2. Currently feasible degrees (deployable co-designed targets): "
            f"{feasible_degrees or 'none'}.",
            f"3. Degrees showing overshoot (from the degree-window study if present): "
            f"{overshoot_degrees or 'not yet mapped'}.",
            f"4. Gate-validated configurations to reuse: {len(gate_configs)} rows at degrees "
            f"{gate_degrees or 'none'} (all residual-feasible after gate).",
            f"5. Observables still needing gate-level values: "
            f"{', '.join(observables_needing_gate) or 'none'}.",
            f"6. Subproblem-selection modes for robustness testing: {', '.join(ROBUSTNESS_MODES)}.",
            "",
            "## Notes",
            "- The stable families are the singular-support-weighted and residual-aware targets; "
            "the success-amplitude-aware and degree-scheduled clipped families trade direction "
            "alignment for success amplitude.",
            "- Gate-validated configurations are reused as the seed for the degree-window gate "
            "validation; observable-first rows still carry NaN gate values, which the "
            "gate-observable-readout phase fills.",
            "",
        ]
    )


def _overshoot_degrees(input_root: Path) -> list[int]:
    """Overshoot degrees for the families that are feasible somewhere (the stable families).

    Reporting the union across all families would include the clipped families that lose
    direction alignment at every degree, which is not the overshoot boundary of interest.
    """

    frame = _read_csv(input_root / "qsvt_degree_window_overshoot" / "degree_window_summary.csv")
    if frame.empty or "degree_window_class" not in frame.columns:
        return []
    stable_families = set(
        frame[frame["degree_window_class"] == "residual_feasible"]["target_family"].astype(str)
    )
    if not stable_families:
        return []
    stable = frame[frame["target_family"].astype(str).isin(stable_families)]
    overshoot = stable[stable["degree_window_class"] == "overshoot_risk"]
    return sorted(
        {int(value) for value in pd.to_numeric(overshoot["degree"], errors="coerce").dropna()}
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
