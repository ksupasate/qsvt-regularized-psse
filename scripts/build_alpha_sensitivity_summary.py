from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

ALPHA_VALUES = [1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0]
ALPHA_CONFIGS = [
    "configs/alpha_sensitivity_real_ieee14.yaml",
    "configs/alpha_sensitivity_real_ieee118.yaml",
    "configs/alpha_sensitivity_real_ieee300_reduced.yaml",
    "configs/alpha_sensitivity_nonlinear_ac_ieee14.yaml",
]
ALPHA_COLUMNS = [
    "case_name",
    "config_file",
    "model_type",
    "scenario_name",
    "noise_std",
    "missing_ratio",
    "bad_data_ratio",
    "estimator",
    "alpha",
    "rmse_mean",
    "rmse_std",
    "residual_mean",
    "condition_number_mean",
    "runtime_mean",
    "num_trials",
    "notes",
]


def build_alpha_sensitivity_summary(
    *,
    outputs_dir: str | Path = REPO_ROOT / "outputs",
    config_dir: str | Path = REPO_ROOT / "configs",
    output_dir: str | Path = REPO_ROOT / "outputs" / "alpha_sensitivity_summary",
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figures_dir = output_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    config_index = _config_index(Path(config_dir))
    alpha_rows = _load_alpha_rows(Path(outputs_dir), config_index)
    best_by_case = _best_by(alpha_rows, ["case_name", "estimator"])
    best_by_scenario = _best_by(alpha_rows, ["case_name", "scenario_name", "estimator"])

    all_path = output_path / "alpha_sensitivity_all.csv"
    best_case_path = output_path / "alpha_best_by_case.csv"
    best_scenario_path = output_path / "alpha_best_by_scenario.csv"
    notes_path = output_path / "alpha_robustness_notes.md"
    summary_path = output_path / "alpha_sensitivity_summary.md"
    recommended_path = output_path / "recommended_alpha_runs.md"
    manifest_path = output_path / "alpha_sensitivity_manifest.json"

    alpha_rows.to_csv(all_path, index=False)
    best_by_case.to_csv(best_case_path, index=False)
    best_by_scenario.to_csv(best_scenario_path, index=False)
    recommended_path.write_text(_recommended_commands_text(), encoding="utf-8")
    notes_path.write_text(_robustness_notes(alpha_rows, best_by_case), encoding="utf-8")
    _write_alpha_figures(alpha_rows, figures_dir)
    summary_path.write_text(
        _summary_markdown(alpha_rows, best_by_case, best_by_scenario, recommended_path),
        encoding="utf-8",
    )

    missing_fields = [
        column
        for column in ALPHA_COLUMNS
        if column in alpha_rows.columns and alpha_rows[column].isna().any()
    ]
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "script": "scripts/build_alpha_sensitivity_summary.py",
        "alpha_values_expected": ALPHA_VALUES,
        "alpha_rows_found": len(alpha_rows),
        "recommended_configs": ALPHA_CONFIGS,
        "missing_or_nullable_fields": missing_fields,
        "outputs": {
            "summary": str(summary_path),
            "all": str(all_path),
            "best_by_case": str(best_case_path),
            "best_by_scenario": str(best_scenario_path),
            "robustness_notes": str(notes_path),
            "recommended_runs": str(recommended_path),
            "figures_dir": str(figures_dir),
        },
        "notes": [
            "The script reads existing output artifacts only.",
            "No alpha sensitivity experiment is run by this script.",
            "Empty CSV outputs indicate missing alpha sweep results.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "output_dir": output_path,
        "alpha_rows": alpha_rows,
        "best_by_case": best_by_case,
        "best_by_scenario": best_by_scenario,
        "manifest": manifest,
    }


def _config_index(config_dir: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for path in sorted(config_dir.glob("*.yaml")):
        config = _read_yaml(path)
        output = config.get("output", {}) if isinstance(config.get("output"), dict) else {}
        run = config.get("run", {}) if isinstance(config.get("run"), dict) else {}
        run_id = output.get("run_id") or run.get("run_id") or config.get("run_name") or path.stem
        index[str(run_id)] = {"path": path, "config": config}
    return index


def _load_alpha_rows(outputs_dir: Path, config_index: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if not outputs_dir.exists():
        return pd.DataFrame(columns=ALPHA_COLUMNS)
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in outputs_dir.iterdir() if path.is_dir()):
        config_info = _config_for_output(run_dir, config_index)
        aggregate_path = run_dir / "aggregate_metrics.csv"
        summary_path = run_dir / "summary_metrics.csv"
        if aggregate_path.is_file():
            rows.extend(_aggregate_alpha_rows(aggregate_path, run_dir, config_info))
        elif summary_path.is_file():
            rows.extend(_summary_alpha_rows(summary_path, run_dir, config_info))
    return pd.DataFrame(rows, columns=ALPHA_COLUMNS)


def _aggregate_alpha_rows(
    path: Path,
    run_dir: Path,
    config_info: dict[str, Any],
) -> list[dict[str, Any]]:
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return []
    if "sweep_parameter" not in frame.columns:
        return []
    frame = frame[frame["sweep_parameter"].astype(str).str.contains("alpha", case=False)]
    if frame.empty:
        return []

    group_columns = ["sweep_value", "estimator"]
    rows = []
    for group_key, group in frame.groupby(group_columns, dropna=False, sort=True):
        alpha, estimator = group_key
        first = group.iloc[0]
        rows.append(
            _alpha_row(
                config_info=config_info,
                run_dir=run_dir,
                estimator=str(estimator),
                alpha=_float_or_nan(alpha),
                case_name=first.get("case_name"),
                model_type=first.get("mode"),
                scenario_name=first.get("scenario_name"),
                noise_std=_first_numeric(group, "noise_std"),
                missing_ratio=_first_numeric(group, "missing_ratio"),
                bad_data_ratio=_first_numeric(group, "bad_data_ratio"),
                rmse_mean=_mean(group, "rmse"),
                rmse_std=_std(group, "rmse"),
                residual_mean=_mean_first_available(
                    group,
                    ["weighted_residual_norm", "weighted_residual", "residual_norm"],
                ),
                condition_number_mean=_mean(group, "condition_number"),
                runtime_mean=_mean(group, "runtime_seconds"),
                num_trials=_trial_count(group),
            )
        )
    return rows


def _summary_alpha_rows(
    path: Path,
    run_dir: Path,
    config_info: dict[str, Any],
) -> list[dict[str, Any]]:
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return []
    if "sweep_parameter" not in frame.columns:
        return []
    frame = frame[frame["sweep_parameter"].astype(str).str.contains("alpha", case=False)]
    if frame.empty:
        return []

    rows = []
    for _, row in frame.iterrows():
        rows.append(
            _alpha_row(
                config_info=config_info,
                run_dir=run_dir,
                estimator=str(row.get("estimator")),
                alpha=_float_or_nan(row.get("sweep_value")),
                case_name=None,
                model_type=None,
                scenario_name=None,
                noise_std=None,
                missing_ratio=None,
                bad_data_ratio=None,
                rmse_mean=_float_or_nan(row.get("rmse_mean")),
                rmse_std=_float_or_nan(row.get("rmse_std")),
                residual_mean=_first_row_value(
                    row,
                    ["weighted_residual_norm_mean", "weighted_residual_mean", "residual_norm_mean"],
                ),
                condition_number_mean=_float_or_nan(row.get("condition_number_mean")),
                runtime_mean=_float_or_nan(row.get("runtime_seconds_mean")),
                num_trials=_int_or_nan(row.get("n_trials")),
            )
        )
    return rows


def _alpha_row(
    *,
    config_info: dict[str, Any],
    run_dir: Path,
    estimator: str,
    alpha: float,
    case_name: Any,
    model_type: Any,
    scenario_name: Any,
    noise_std: Any,
    missing_ratio: Any,
    bad_data_ratio: Any,
    rmse_mean: float,
    rmse_std: float,
    residual_mean: float,
    condition_number_mean: float,
    runtime_mean: float,
    num_trials: int | float,
) -> dict[str, Any]:
    config = config_info.get("config", {})
    system = config.get("system", {}) if isinstance(config.get("system"), dict) else {}
    scenario = config.get("scenario", {}) if isinstance(config.get("scenario"), dict) else {}
    bad_data = scenario.get("bad_data", {}) if isinstance(scenario.get("bad_data"), dict) else {}
    config_path = config_info.get("path")
    return {
        "case_name": _clean_value(case_name) or system.get("case_name"),
        "config_file": _display_path(config_path)
        if isinstance(config_path, Path)
        else str(run_dir),
        "model_type": _clean_value(model_type) or system.get("mode"),
        "scenario_name": _clean_value(scenario_name) or scenario.get("name"),
        "noise_std": _clean_value(noise_std)
        if not pd.isna(noise_std)
        else scenario.get("noise_std"),
        "missing_ratio": (
            _clean_value(missing_ratio)
            if not pd.isna(missing_ratio)
            else scenario.get("missing_ratio")
        ),
        "bad_data_ratio": (
            _clean_value(bad_data_ratio) if not pd.isna(bad_data_ratio) else bad_data.get("ratio")
        ),
        "estimator": estimator,
        "alpha": alpha,
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
        "residual_mean": residual_mean,
        "condition_number_mean": condition_number_mean,
        "runtime_mean": runtime_mean,
        "num_trials": num_trials,
        "notes": _row_note(estimator),
    }


def _best_by(alpha_rows: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if alpha_rows.empty:
        return pd.DataFrame(columns=ALPHA_COLUMNS)
    frame = alpha_rows[alpha_rows["estimator"].isin(["ridge", "qsvt_regularized"])].copy()
    frame["rmse_mean"] = pd.to_numeric(frame["rmse_mean"], errors="coerce")
    frame = frame.dropna(subset=["rmse_mean"])
    if frame.empty:
        return pd.DataFrame(columns=ALPHA_COLUMNS)
    indices = frame.groupby(group_columns, dropna=False)["rmse_mean"].idxmin()
    return frame.loc[indices].sort_values(group_columns).reset_index(drop=True)


def _summary_markdown(
    alpha_rows: pd.DataFrame,
    best_by_case: pd.DataFrame,
    best_by_scenario: pd.DataFrame,
    recommended_path: Path,
) -> str:
    tested = _joined_values(alpha_rows, "alpha") or ", ".join(str(value) for value in ALPHA_VALUES)
    cases = _joined_values(alpha_rows, "case_name") or "No completed alpha outputs found."
    estimators = _joined_values(
        alpha_rows[alpha_rows["estimator"].isin(["ridge", "qsvt_regularized"])],
        "estimator",
    )
    lines = [
        "# Alpha Sensitivity Summary",
        "",
        "This summary reads existing output artifacts only. It does not run alpha sweeps.",
        "",
        "## 1. Alpha Values Tested",
        "",
        tested,
        "",
        "## 2. Cases Tested",
        "",
        cases,
        "",
        "## 3. Alpha-Sensitive Estimators",
        "",
        estimators or "No completed alpha sweep rows were found for ridge or qsvt_regularized.",
        "",
        "## 4. Best Alpha by Case and Scenario",
        "",
        _markdown_table(best_by_case[ALPHA_COLUMNS] if not best_by_case.empty else best_by_case),
        "",
        "Scenario-level best rows:",
        "",
        _markdown_table(
            best_by_scenario[ALPHA_COLUMNS] if not best_by_scenario.empty else best_by_scenario
        ),
        "",
        "## 5. Ridge/QSVT-Target Equivalence",
        "",
        _equivalence_text(alpha_rows),
        "",
        "## 6. Dependence on Alpha",
        "",
        _alpha_dependence_text(alpha_rows),
        "",
        "## 7. Recommended Paper-Level Alpha",
        "",
        _recommended_alpha_text(best_by_case),
        "",
        "## 8. Remaining Limitations",
        "",
        "- Completed alpha sweep outputs are required before making empirical alpha-selection "
        "claims.",
        "- Alpha sensitivity should be interpreted with noise, missing, bad-data, and "
        "conditioning stress.",
        f"- Recommended run commands are in `{_display_path(recommended_path)}`.",
    ]
    if alpha_rows.empty:
        lines.extend(
            [
                "",
                "## Missing Output Notice",
                "",
                "No existing alpha sensitivity rows were found. The CSV files are "
                "intentionally empty except for headers, and recommended commands were generated.",
            ]
        )
    return "\n".join(lines)


def _robustness_notes(alpha_rows: pd.DataFrame, best_by_case: pd.DataFrame) -> str:
    if alpha_rows.empty:
        return "\n".join(
            [
                "# Alpha Robustness Notes",
                "",
                "No alpha sweep result rows are available yet.",
                "",
                "Run the recommended alpha commands before using alpha-selection claims "
                "in the paper.",
            ]
        )
    return "\n".join(
        [
            "# Alpha Robustness Notes",
            "",
            f"Rows analyzed: {len(alpha_rows)}",
            f"Best-row count by case/estimator: {len(best_by_case)}",
            "",
            "Ridge and qsvt_regularized should match when the same alpha is used because the "
            "classical simulator uses the same spectral filter target.",
        ]
    )


def _recommended_commands_text() -> str:
    lines = [
        "# Recommended Alpha Sensitivity Runs",
        "",
        "Use the repository's current experiment CLI:",
        "",
        "```bash",
    ]
    for config in ALPHA_CONFIGS:
        lines.append(
            f".venv/bin/python -m robust_qsvt_se.experiments.run_benchmark --config {config}"
        )
    lines.extend(
        [
            "```",
            "",
            "The IEEE300 config is reduced to one seed and essential estimators. Do not use a "
            "full nonlinear IEEE300 alpha sweep as a default run.",
        ]
    )
    return "\n".join(lines)


def _write_alpha_figures(alpha_rows: pd.DataFrame, figures_dir: Path) -> None:
    if alpha_rows.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    frame = alpha_rows[alpha_rows["estimator"].isin(["ridge", "qsvt_regularized"])].copy()
    frame["alpha"] = pd.to_numeric(frame["alpha"], errors="coerce")
    frame["rmse_mean"] = pd.to_numeric(frame["rmse_mean"], errors="coerce")
    frame = frame.dropna(subset=["alpha", "rmse_mean"])
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for (case_name, estimator), group in frame.groupby(["case_name", "estimator"], dropna=False):
        ordered = group.sort_values("alpha")
        ax.plot(
            ordered["alpha"],
            ordered["rmse_mean"],
            marker="o",
            label=f"{case_name} {estimator}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Mean RMSE")
    ax.set_title("Alpha sensitivity")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize="x-small")
    fig.tight_layout()
    fig.savefig(figures_dir / "alpha_rmse_by_case.png", dpi=150)
    plt.close(fig)


def _config_for_output(run_dir: Path, config_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    config_path = run_dir / "config_resolved.yaml"
    if config_path.is_file():
        config = _read_yaml(config_path)
        run_id = _run_id_from_config(config, fallback=run_dir.name)
        indexed = config_index.get(run_id)
        return indexed or {"path": config_path, "config": config}
    return config_index.get(run_dir.name, {"path": None, "config": {}})


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
    except OSError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _run_id_from_config(config: dict[str, Any], *, fallback: str) -> str:
    output = config.get("output", {}) if isinstance(config.get("output"), dict) else {}
    return str(output.get("run_id") or config.get("run_name") or fallback)


def _row_note(estimator: str) -> str:
    if estimator in {"ridge", "qsvt_regularized"}:
        return "alpha-sensitive regularized spectral estimator"
    return "reference estimator included in alpha sweep; alpha does not configure this estimator"


def _equivalence_text(alpha_rows: pd.DataFrame) -> str:
    if alpha_rows.empty:
        return (
            "No completed alpha rows are available. Method definition still indicates numerical "
            "equivalence between Ridge and QSVT-target when alpha is the same."
        )
    subset = alpha_rows[alpha_rows["estimator"].isin(["ridge", "qsvt_regularized"])]
    pivot = subset.pivot_table(
        index=["case_name", "scenario_name", "alpha"],
        columns="estimator",
        values="rmse_mean",
        aggfunc="mean",
    )
    if not {"ridge", "qsvt_regularized"}.issubset(pivot.columns):
        return "Ridge/QSVT paired rows are incomplete in the current alpha outputs."
    diff = (pivot["ridge"] - pivot["qsvt_regularized"]).abs().dropna()
    if diff.empty:
        return "No paired Ridge/QSVT alpha rows are available."
    return f"Paired rows show maximum absolute RMSE difference {float(diff.max()):.6g}."


def _alpha_dependence_text(alpha_rows: pd.DataFrame) -> str:
    if alpha_rows.empty:
        return "Not assessable until alpha sweep outputs are generated."
    subset = alpha_rows[alpha_rows["estimator"].isin(["ridge", "qsvt_regularized"])].copy()
    if subset.empty:
        return "No ridge or qsvt_regularized alpha rows are available."
    spread = (
        subset.groupby(["case_name", "scenario_name", "estimator"], dropna=False)["rmse_mean"]
        .agg(["min", "max"])
        .reset_index()
    )
    spread["relative_spread"] = (spread["max"] - spread["min"]) / spread["min"].replace(0.0, np.nan)
    max_spread = spread["relative_spread"].replace([np.inf, -np.inf], np.nan).max()
    if pd.isna(max_spread):
        return "Alpha dependence could not be quantified from finite RMSE rows."
    return f"Maximum relative RMSE spread across tested alpha values is {float(max_spread):.6g}."


def _recommended_alpha_text(best_by_case: pd.DataFrame) -> str:
    if best_by_case.empty:
        return "Keep the current paper-level default alpha until alpha sweeps are run."
    regularized = best_by_case[best_by_case["estimator"].isin(["ridge", "qsvt_regularized"])]
    if regularized.empty:
        return "No regularized-estimator best alpha rows are available."
    counts = regularized["alpha"].value_counts(dropna=True)
    if counts.empty:
        return "No finite best alpha values are available."
    alpha = counts.index[0]
    return (
        f"The most frequent best alpha in completed rows is {alpha}; verify manually "
        "before paper use."
    )


def _mean(group: pd.DataFrame, column: str) -> float:
    if column not in group:
        return float("nan")
    numeric = pd.to_numeric(group[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(numeric.mean()) if numeric.notna().any() else float("nan")


def _std(group: pd.DataFrame, column: str) -> float:
    if column not in group:
        return float("nan")
    numeric = pd.to_numeric(group[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    numeric = numeric.dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0


def _mean_first_available(group: pd.DataFrame, columns: Sequence[str]) -> float:
    for column in columns:
        if column in group:
            return _mean(group, column)
    return float("nan")


def _first_numeric(group: pd.DataFrame, column: str) -> float:
    if column not in group:
        return float("nan")
    numeric = pd.to_numeric(group[column], errors="coerce").dropna()
    return float(numeric.iloc[0]) if not numeric.empty else float("nan")


def _first_row_value(row: pd.Series, columns: Sequence[str]) -> float:
    for column in columns:
        value = row.get(column)
        if value is not None and not pd.isna(value):
            return _float_or_nan(value)
    return float("nan")


def _trial_count(group: pd.DataFrame) -> int:
    if "trial_id" in group:
        return int(group["trial_id"].nunique())
    return len(group)


def _float_or_nan(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _int_or_nan(value: Any) -> int | float:
    try:
        if value is None or pd.isna(value):
            return float("nan")
        return int(value)
    except (TypeError, ValueError):
        return float("nan")


def _clean_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    return value


def _joined_values(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame:
        return ""
    values = sorted({str(value) for value in frame[column].dropna().unique()})
    return ", ".join(values)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    visible = frame.head(20).copy()
    for column in visible.columns:
        visible[column] = visible[column].map(_format_cell)
    header = "| " + " | ".join(str(column) for column in visible.columns) + " |"
    divider = "| " + " | ".join("---" for _ in visible.columns) + " |"
    body = [
        "| " + " | ".join(str(row[column]) for column in visible.columns) + " |"
        for _, row in visible.iterrows()
    ]
    return "\n".join([header, divider, *body])


def _format_cell(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("\n", " ").replace("|", "\\|")


def _display_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--config-dir", type=Path, default=REPO_ROOT / "configs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "alpha_sensitivity_summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_alpha_sensitivity_summary(
        outputs_dir=args.outputs_dir,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
    )
    print(f"Alpha sensitivity summary written to {result['output_dir']}")
    print(f"Alpha rows found: {len(result['alpha_rows'])}")


if __name__ == "__main__":
    main()
