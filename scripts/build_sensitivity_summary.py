from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SENSITIVITY_COLUMNS = [
    "source_output",
    "source_file",
    "run_id",
    "sweep_name",
    "sweep_parameter",
    "sweep_value",
    "estimator",
    "n_trials",
    "n_successful_trials",
    "failure_rate",
    "rmse_mean",
    "rmse_median",
    "angle_rmse_mean",
    "voltage_magnitude_rmse_mean",
    "residual_norm_mean",
    "weighted_residual_mean",
    "weighted_residual_norm_mean",
    "condition_number_mean",
    "runtime_seconds_mean",
]
RECOMMENDED_CONFIGS = [
    "alpha_sensitivity_real_ieee14.yaml",
    "alpha_sensitivity_real_ieee118.yaml",
    "alpha_sensitivity_real_ieee300_reduced.yaml",
    "alpha_sensitivity_nonlinear_ac_ieee14.yaml",
    "noise_missing_bad_data_sensitivity_ieee14.yaml",
    "noise_missing_bad_data_sensitivity_ieee118.yaml",
]


def build_sensitivity_summary(
    *,
    outputs_dir: str | Path = REPO_ROOT / "outputs",
    output_dir: str | Path = REPO_ROOT / "outputs" / "sensitivity_summary",
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figures_dir = output_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    combined = _load_sweep_summaries(Path(outputs_dir))
    category_frames = {
        "alpha": _category_frame(combined, "alpha"),
        "noise": _category_frame(combined, "noise"),
        "missing": _category_frame(combined, "missing"),
        "bad_data": _category_frame(combined, "bad_data"),
    }
    paths = {
        "alpha": output_path / "alpha_sensitivity.csv",
        "noise": output_path / "noise_sensitivity.csv",
        "missing": output_path / "missing_sensitivity.csv",
        "bad_data": output_path / "bad_data_sensitivity.csv",
    }
    for category, frame in category_frames.items():
        frame.to_csv(paths[category], index=False)
        _write_category_plot(frame, category, figures_dir)

    missing_categories = [category for category, frame in category_frames.items() if frame.empty]
    alpha_rows_found = len(category_frames["alpha"])
    recommended_path = output_path / "recommended_sensitivity_runs.md"
    recommended_path.write_text(
        _recommended_runs_text(category_frames, missing_categories),
        encoding="utf-8",
    )
    summary_path = output_path / "sensitivity_summary.md"
    summary_path.write_text(
        _summary_markdown(category_frames, missing_categories, recommended_path),
        encoding="utf-8",
    )
    manifest_path = output_path / "sensitivity_manifest.json"
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "script": "scripts/build_sensitivity_summary.py",
        "source_rows_loaded": len(combined),
        "alpha_rows_found": alpha_rows_found,
        "alpha_sensitivity_status": (
            "available" if alpha_rows_found else "missing_results_needs_experiment_execution"
        ),
        "missing_categories": missing_categories,
        "outputs": {
            "summary": str(summary_path),
            "alpha": str(paths["alpha"]),
            "noise": str(paths["noise"]),
            "missing": str(paths["missing"]),
            "bad_data": str(paths["bad_data"]),
            "recommended_runs": str(recommended_path),
            "figures_dir": str(figures_dir),
        },
        "notes": [
            "The script reads existing output artifacts only and does not run experiments.",
            "Empty CSVs mean no matching existing sweep outputs were found.",
            "If alpha_rows_found is 0, alpha sensitivity configs are recommendations only "
            "until run.",
            "Recommended configs are suggestions and are not generated or run by this script.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "output_dir": output_path,
        "combined": combined,
        "category_frames": category_frames,
        "manifest": manifest,
    }


def _load_sweep_summaries(outputs_dir: Path) -> pd.DataFrame:
    if not outputs_dir.exists():
        return pd.DataFrame(columns=SENSITIVITY_COLUMNS)
    paths = sorted(outputs_dir.glob("*/summary_metrics.csv"))
    paths.extend(sorted(outputs_dir.glob("*/combined_summary_metrics.csv")))
    frames = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if "sweep_parameter" not in frame.columns:
            continue
        frame = frame.copy()
        source_output = path.parent
        frame["source_output"] = _display_path(source_output)
        frame["source_file"] = _display_path(path)
        frame["run_id"] = source_output.name
        frames.append(_normalize_summary_frame(frame))
    if not frames:
        return pd.DataFrame(columns=SENSITIVITY_COLUMNS)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined.sort_values(
        ["source_output", "sweep_parameter", "sweep_value", "estimator"],
        kind="mergesort",
    ).reset_index(drop=True)


def _normalize_summary_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in SENSITIVITY_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = np.nan
    return normalized[SENSITIVITY_COLUMNS]


def _category_frame(combined: pd.DataFrame, category: str) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame(columns=SENSITIVITY_COLUMNS)
    parameters = combined["sweep_parameter"].astype(str)
    if category == "alpha":
        mask = parameters.str.contains("alpha", case=False, regex=False)
    elif category == "noise":
        mask = parameters.eq("scenario.noise_std") | parameters.str.contains(
            "noise_std",
            regex=False,
        )
    elif category == "missing":
        mask = parameters.eq("scenario.missing_ratio") | parameters.str.contains(
            "missing_ratio",
            regex=False,
        )
    elif category == "bad_data":
        mask = parameters.str.contains("scenario.bad_data", regex=False) | parameters.str.contains(
            "bad_data",
            regex=False,
        )
    else:
        raise ValueError(f"unknown sensitivity category: {category}")
    return combined.loc[mask, SENSITIVITY_COLUMNS].reset_index(drop=True)


def _write_category_plot(frame: pd.DataFrame, category: str, figures_dir: Path) -> None:
    if frame.empty or "rmse_mean" not in frame.columns:
        return
    numeric = frame.copy()
    numeric["sweep_value"] = pd.to_numeric(numeric["sweep_value"], errors="coerce")
    numeric["rmse_mean"] = pd.to_numeric(numeric["rmse_mean"], errors="coerce")
    numeric = numeric.dropna(subset=["sweep_value", "rmse_mean"])
    if numeric.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    group_columns = ["sweep_parameter", "estimator"]
    for (parameter, estimator), estimator_frame in numeric.groupby(group_columns, dropna=False):
        sorted_frame = estimator_frame.sort_values("sweep_value")
        ax.plot(
            sorted_frame["sweep_value"],
            sorted_frame["rmse_mean"],
            marker="o",
            label=f"{estimator} ({parameter})",
        )
    ax.set_xlabel("Sweep value")
    ax.set_ylabel("Mean RMSE")
    ax.set_title(f"{category.replace('_', ' ').title()} sensitivity")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize="x-small")
    fig.tight_layout()
    fig.savefig(figures_dir / f"{category}_sensitivity_rmse.png", dpi=150)
    plt.close(fig)


def _summary_markdown(
    category_frames: dict[str, pd.DataFrame],
    missing_categories: list[str],
    recommended_path: Path,
) -> str:
    lines = [
        "# Sensitivity Summary",
        "",
        "This summary is built from existing output artifacts only. It does not run long "
        "experiments and does not fabricate missing sweeps.",
        "",
        "## Available Sensitivity Rows",
        "",
        "| Category | Rows | Sweep parameters | Source outputs |",
        "| --- | ---: | --- | ---: |",
    ]
    for category, frame in category_frames.items():
        parameters = _joined_unique(frame, "sweep_parameter")
        source_count = int(frame["source_output"].nunique()) if not frame.empty else 0
        lines.append(f"| {category} | {len(frame)} | {parameters or 'missing'} | {source_count} |")
    alpha_rows = len(category_frames["alpha"])
    lines.extend(
        [
            "",
            "## Alpha Sensitivity Status",
            "",
            f"- Alpha rows found: {alpha_rows}",
        ]
    )
    if alpha_rows == 0:
        lines.extend(
            [
                "- Alpha sensitivity still needs experiment execution.",
                "- Recommended alpha configs and commands are listed in "
                "`recommended_sensitivity_runs.md`.",
            ]
        )
    else:
        lines.append("- Alpha sensitivity rows are available in existing summary artifacts.")
    lines.extend(
        [
            "",
            "## Missing Evidence",
            "",
        ]
    )
    if missing_categories:
        for category in missing_categories:
            lines.append(f"- No existing {category} sweep summary was found.")
    else:
        lines.append("- All tracked sensitivity categories have at least one existing summary row.")
    lines.extend(
        [
            "",
            "## Recommended Runs",
            "",
            f"Recommended run suggestions were written to `{_display_path(recommended_path)}`.",
            "",
            "Use these suggestions as draft configs or experiment requests; they were not run "
            "by this script.",
        ]
    )
    return "\n".join(lines)


def _recommended_runs_text(
    category_frames: dict[str, pd.DataFrame],
    missing_categories: list[str],
) -> str:
    lines = [
        "# Recommended Sensitivity Runs",
        "",
        "The current summary does not fabricate missing sweep results. The following "
        "draft config names are recommended for paper-level sensitivity coverage:",
        "",
    ]
    for config_name in RECOMMENDED_CONFIGS:
        lines.append(f"- `configs/{config_name}`")
    lines.extend(
        [
            "",
            "## Suggested Coverage",
            "",
            "- `alpha_sensitivity_real_ieee14.yaml`: sweep ridge/qsvt alpha on IEEE14.",
            "- `alpha_sensitivity_real_ieee118.yaml`: sweep ridge/qsvt alpha on IEEE118.",
            "- `alpha_sensitivity_real_ieee300_reduced.yaml`: sweep ridge/qsvt alpha on IEEE300 "
            "with a reduced runtime budget.",
            "- `alpha_sensitivity_nonlinear_ac_ieee14.yaml`: lightweight nonlinear AC alpha sweep.",
            "- `noise_missing_bad_data_sensitivity_ieee14.yaml`: joint but lightweight "
            "noise, missing, bad-data ratio, and bad-data magnitude sweeps.",
            "- `noise_missing_bad_data_sensitivity_ieee118.yaml`: same stress factors on a "
            "larger but still tractable case.",
            "",
            "## Current Gaps",
            "",
        ]
    )
    if missing_categories:
        for category in missing_categories:
            lines.append(f"- Missing existing output summary for `{category}` sensitivity.")
    if category_frames.get("alpha", pd.DataFrame()).empty:
        lines.append("- Alpha rows found: 0.")
        lines.append("- Alpha sensitivity still needs experiment execution.")
    for category, frame in category_frames.items():
        if not frame.empty:
            lines.append(
                f"- Existing `{category}` summary rows found: {len(frame)} "
                f"from {frame['source_output'].nunique()} output folder(s)."
            )
    lines.extend(
        [
            "",
            "## Minimum Config Requirements",
            "",
            "- Include explicit `sweeps` entries for `scenario.noise_std`, "
            "`scenario.missing_ratio`, `scenario.bad_data.ratio`, "
            "`scenario.bad_data.magnitude`, and estimator `alpha` where supported.",
            "- For alpha sweeps in the current indexed estimator layout, use "
            "`parameter: estimators.1.alpha` and "
            "`linked_parameters: [estimators.3.alpha]` when the estimator order is "
            "pseudoinverse, ridge, truncated_svd, qsvt_regularized.",
            "- Use seeded trials and checkpointing.",
            "- Keep IEEE300 alpha runs separated from multi-factor stress sweeps to avoid "
            "long default runtimes.",
            "",
            "## Suggested Commands",
            "",
            "```bash",
            ".venv/bin/python -m robust_qsvt_se.experiments.run_benchmark \\",
            "  --config configs/alpha_sensitivity_real_ieee14.yaml",
            ".venv/bin/python -m robust_qsvt_se.experiments.run_benchmark \\",
            "  --config configs/alpha_sensitivity_real_ieee118.yaml",
            ".venv/bin/python -m robust_qsvt_se.experiments.run_benchmark \\",
            "  --config configs/alpha_sensitivity_real_ieee300_reduced.yaml --resume",
            ".venv/bin/python -m robust_qsvt_se.experiments.run_benchmark \\",
            "  --config configs/alpha_sensitivity_nonlinear_ac_ieee14.yaml",
            ".venv/bin/python scripts/build_sensitivity_summary.py",
            "```",
        ]
    )
    return "\n".join(lines)


def _joined_unique(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame:
        return ""
    values = sorted({str(value) for value in frame[column].dropna().unique()})
    return "; ".join(values)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "sensitivity_summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_sensitivity_summary(outputs_dir=args.outputs_dir, output_dir=args.output_dir)
    print(f"Sensitivity summary written to {result['output_dir']}")
    for category, frame in result["category_frames"].items():
        print(f"{category}: {len(frame)} rows")


if __name__ == "__main__":
    main()
