from __future__ import annotations

import argparse
import json
import sys
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

SCRIPT_ROOT = REPO_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from export_measurement_inventory import export_measurement_inventory  # noqa: E402

REDUNDANCY_COLUMNS = [
    "case_name",
    "case_source",
    "experiment_group",
    "model_type",
    "config_file",
    "run_id",
    "measurement_rows_before_missing",
    "state_dimension",
    "redundancy_ratio",
    "missing_ratio",
    "expected_rows_after_missing",
    "redundancy_ratio_after_missing",
    "is_overdetermined_before_missing",
    "is_overdetermined_after_missing",
    "condition_number_before_missing",
    "condition_number_after_missing_if_available",
    "notes",
]


def build_measurement_redundancy_report(
    *,
    config_dir: str | Path = REPO_ROOT / "configs",
    outputs_dir: str | Path = REPO_ROOT / "outputs",
    inventory_dir: str | Path = REPO_ROOT / "outputs" / "measurement_inventory",
    output_dir: str | Path = REPO_ROOT / "outputs" / "measurement_redundancy",
) -> dict[str, Any]:
    """Build measurement-redundancy summaries without running experiments."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figures_dir = output_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    inventory_result = export_measurement_inventory(
        config_dir=config_dir,
        output_dir=inventory_dir,
    )
    by_case = inventory_result["by_case"]
    configs = _config_map(Path(config_dir))
    by_case_report = _redundancy_rows(by_case, configs, Path(outputs_dir))
    by_group = _group_summary(by_case_report)
    missing_effect = _missing_ratio_effect_rows(by_case, configs, Path(outputs_dir))

    by_case_path = output_path / "measurement_redundancy_by_case.csv"
    by_group_path = output_path / "measurement_redundancy_by_group.csv"
    missing_path = output_path / "missing_ratio_redundancy_effect.csv"
    summary_path = output_path / "measurement_redundancy_summary.md"
    manifest_path = output_path / "measurement_redundancy_manifest.json"

    by_case_report.to_csv(by_case_path, index=False)
    by_group.to_csv(by_group_path, index=False)
    missing_effect.to_csv(missing_path, index=False)
    figure_outputs = _write_figures(by_case_report, figures_dir)
    summary_path.write_text(
        _summary_markdown(by_case_report, by_group, missing_effect),
        encoding="utf-8",
    )
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "script": "scripts/build_measurement_redundancy_report.py",
        "source_inventory": str(Path(inventory_dir)),
        "rows_by_case": len(by_case_report),
        "rows_missing_ratio_effect": len(missing_effect),
        "outputs": {
            "summary": str(summary_path),
            "by_case": str(by_case_path),
            "by_group": str(by_group_path),
            "missing_ratio_effect": str(missing_path),
            "figures_dir": str(figures_dir),
        },
        "figures": figure_outputs,
        "notes": [
            "Redundancy ratio is measurement rows divided by state dimension.",
            "Redundancy is not a full observability proof.",
            "More rows do not guarantee low RMSE.",
            "Missing measurements can reduce redundancy and worsen conditioning.",
            "Condition-number fields are populated only when existing metrics artifacts "
            "expose them.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "output_dir": output_path,
        "by_case": by_case_report,
        "by_group": by_group,
        "missing_effect": missing_effect,
        "manifest": manifest,
    }


def _config_map(config_dir: Path) -> dict[str, dict[str, Any]]:
    configs = {}
    for path in sorted(config_dir.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        if isinstance(loaded, dict):
            configs[_display_path(path)] = loaded
    return configs


def _redundancy_rows(
    by_case: pd.DataFrame,
    configs: dict[str, dict[str, Any]],
    outputs_dir: Path,
) -> pd.DataFrame:
    rows = []
    for _, item in by_case.iterrows():
        config_file = str(item["config_file"])
        config = configs.get(config_file, {})
        missing_ratio = _base_missing_ratio(config)
        rows.append(
            _redundancy_row(
                item=item,
                config=config,
                outputs_dir=outputs_dir,
                missing_ratio=missing_ratio,
                notes=(
                    "Base scenario missing ratio from config; counts are expected "
                    "before RNG row choice."
                ),
            )
        )
    return pd.DataFrame(rows, columns=REDUNDANCY_COLUMNS)


def _missing_ratio_effect_rows(
    by_case: pd.DataFrame,
    configs: dict[str, dict[str, Any]],
    outputs_dir: Path,
) -> pd.DataFrame:
    rows = []
    for _, item in by_case.iterrows():
        config_file = str(item["config_file"])
        config = configs.get(config_file, {})
        ratios = _missing_ratios_from_config(config)
        for ratio in ratios:
            rows.append(
                _redundancy_row(
                    item=item,
                    config=config,
                    outputs_dir=outputs_dir,
                    missing_ratio=ratio,
                    notes=(
                        "Missing-ratio sweep value where configured; no estimator run is executed."
                    ),
                )
            )
    return pd.DataFrame(rows, columns=REDUNDANCY_COLUMNS)


def _redundancy_row(
    *,
    item: pd.Series,
    config: dict[str, Any],
    outputs_dir: Path,
    missing_ratio: float,
    notes: str,
) -> dict[str, Any]:
    row_count = int(item["row_count_before_missing"])
    state_dimension = int(item["state_dimension"])
    after_rows = row_count - round(row_count * float(missing_ratio))
    run_id = _run_id(config)
    before_condition, after_condition = _condition_numbers_if_available(
        outputs_dir=outputs_dir,
        run_id=run_id,
        missing_ratio=missing_ratio,
    )
    return {
        "case_name": item["case_name"],
        "case_source": item["case_source"],
        "experiment_group": item["experiment_group"],
        "model_type": item["model_type"],
        "config_file": item["config_file"],
        "run_id": run_id,
        "measurement_rows_before_missing": row_count,
        "state_dimension": state_dimension,
        "redundancy_ratio": row_count / state_dimension if state_dimension else np.nan,
        "missing_ratio": float(missing_ratio),
        "expected_rows_after_missing": after_rows,
        "redundancy_ratio_after_missing": (
            after_rows / state_dimension if state_dimension else np.nan
        ),
        "is_overdetermined_before_missing": row_count >= state_dimension,
        "is_overdetermined_after_missing": after_rows >= state_dimension,
        "condition_number_before_missing": before_condition,
        "condition_number_after_missing_if_available": after_condition,
        "notes": notes,
    }


def _missing_ratios_from_config(config: dict[str, Any]) -> list[float]:
    ratios = {_base_missing_ratio(config)}
    for sweep in config.get("sweeps", []) or []:
        if str(sweep.get("parameter")) == "scenario.missing_ratio":
            ratios.update(float(value) for value in sweep.get("values", []))
    return sorted(ratios)


def _base_missing_ratio(config: dict[str, Any]) -> float:
    scenario = config.get("scenario", {})
    if not isinstance(scenario, dict):
        return 0.0
    return float(scenario.get("missing_ratio", 0.0))


def _run_id(config: dict[str, Any]) -> str:
    output = config.get("output", {})
    if isinstance(output, dict) and output.get("run_id"):
        return str(output["run_id"])
    return str(config.get("run_name", ""))


def _condition_numbers_if_available(
    *,
    outputs_dir: Path,
    run_id: str,
    missing_ratio: float,
) -> tuple[float | None, float | None]:
    if not run_id:
        return None, None
    run_dir = outputs_dir / run_id
    frames = []
    for name in ("aggregate_metrics.csv", "metrics.csv"):
        path = run_dir / name
        if not path.exists():
            continue
        try:
            frames.append(pd.read_csv(path))
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
    if not frames:
        return None, None
    frame = pd.concat(frames, ignore_index=True, sort=False)
    if "missing_ratio" in frame.columns:
        numeric_missing = pd.to_numeric(frame["missing_ratio"], errors="coerce")
        frame = frame.loc[np.isclose(numeric_missing, float(missing_ratio), equal_nan=False)]
    before = _median_numeric(frame.get("achieved_condition_number"))
    after = _median_numeric(frame.get("condition_number"))
    return before, after


def _median_numeric(values: pd.Series | None) -> float | None:
    if values is None:
        return None
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return None if numeric.empty else float(numeric.median())


def _group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    group_columns = ["experiment_group", "case_source", "model_type"]
    rows = []
    for group_key, group in frame.groupby(group_columns, dropna=False, sort=True):
        rows.append(
            {
                **dict(zip(group_columns, group_key, strict=True)),
                "case_count": int(group["case_name"].nunique()),
                "config_count": int(group["config_file"].nunique()),
                "min_redundancy_ratio": float(group["redundancy_ratio"].min()),
                "median_redundancy_ratio": float(group["redundancy_ratio"].median()),
                "min_redundancy_ratio_after_missing": float(
                    group["redundancy_ratio_after_missing"].min()
                ),
                "all_overdetermined_before_missing": bool(
                    group["is_overdetermined_before_missing"].all()
                ),
                "all_overdetermined_after_missing": bool(
                    group["is_overdetermined_after_missing"].all()
                ),
            }
        )
    return pd.DataFrame(rows)


def _summary_markdown(
    by_case: pd.DataFrame,
    by_group: pd.DataFrame,
    missing_effect: pd.DataFrame,
) -> str:
    lines = [
        "# Measurement Redundancy Summary",
        "",
        "Redundancy ratio is `measurement rows / state dimension`.",
        "",
        "Important interpretation boundaries:",
        "",
        "- Redundancy is not a full observability proof.",
        "- More rows do not guarantee low RMSE.",
        "- Missing measurements can reduce redundancy and worsen conditioning.",
        "- Condition-number fields are included only where existing metrics expose them.",
        "",
        "## By Experiment Group",
        "",
        _markdown_table(by_group, max_rows=80),
        "",
        "## By Case And Config",
        "",
        _markdown_table(
            by_case[
                [
                    "case_name",
                    "experiment_group",
                    "config_file",
                    "measurement_rows_before_missing",
                    "state_dimension",
                    "redundancy_ratio",
                    "missing_ratio",
                    "expected_rows_after_missing",
                    "redundancy_ratio_after_missing",
                    "is_overdetermined_after_missing",
                ]
            ],
            max_rows=100,
        ),
        "",
        "## Missing-Ratio Effects",
        "",
        _markdown_table(
            missing_effect[
                [
                    "case_name",
                    "experiment_group",
                    "config_file",
                    "missing_ratio",
                    "expected_rows_after_missing",
                    "redundancy_ratio_after_missing",
                    "is_overdetermined_after_missing",
                ]
            ],
            max_rows=100,
        ),
    ]
    return "\n".join(lines)


def _write_figures(by_case: pd.DataFrame, figures_dir: Path) -> dict[str, str]:
    if by_case.empty:
        return {}
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {"status": "skipped: matplotlib unavailable"}

    outputs: dict[str, str] = {}
    rows_by_case = by_case.groupby("case_name")["measurement_rows_before_missing"].sum()
    fig, ax = plt.subplots(figsize=(7, 4))
    rows_by_case.plot(kind="bar", ax=ax)
    ax.set_xlabel("Case")
    ax.set_ylabel("Rows before missing")
    ax.set_title("Measurement rows by case")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = figures_dir / "measurement_rows_by_case.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    outputs["measurement_rows_by_case"] = str(path)

    ratio_by_case = by_case.groupby("case_name")["redundancy_ratio"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ratio_by_case.plot(kind="bar", ax=ax)
    ax.set_xlabel("Case")
    ax.set_ylabel("Mean redundancy ratio")
    ax.set_title("Redundancy ratio by case")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = figures_dir / "redundancy_ratio_by_case.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    outputs["redundancy_ratio_by_case"] = str(path)
    return outputs


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
    return str(value).replace("\n", " ").replace("|", "\\|")


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=REPO_ROOT / "configs")
    parser.add_argument("--outputs-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument(
        "--inventory-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "measurement_inventory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "measurement_redundancy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_measurement_redundancy_report(
        config_dir=args.config_dir,
        outputs_dir=args.outputs_dir,
        inventory_dir=args.inventory_dir,
        output_dir=args.output_dir,
    )
    print(f"Measurement redundancy report written to {result['output_dir']}")
    print(f"Rows by case: {len(result['by_case'])}")


if __name__ == "__main__":
    main()
