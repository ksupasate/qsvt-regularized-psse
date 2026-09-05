from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from robust_qsvt_se.cli.build_report import main as report_cli_main
from robust_qsvt_se.experiments.report_builder import (
    build_estimator_ranking,
    build_report,
    load_report_config,
)


def _write_config_resolved(path: Path, *, run_id: str, mode: str) -> None:
    with (path / "config_resolved.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            {
                "run_name": run_id,
                "output": {"run_id": run_id},
                "system": {"mode": mode},
                "scenario": {"name": "test_scenario"},
            },
            file,
        )


def _write_single_run(path: Path, *, run_id: str, mode: str = "synthetic_linearized") -> None:
    path.mkdir(parents=True)
    _write_config_resolved(path, run_id=run_id, mode=mode)
    pd.DataFrame(
        [
            {
                "estimator": "pseudoinverse",
                "rmse": 2.0,
                "weighted_residual": 0.2,
                "residual_norm": 0.2,
                "condition_number": 1000.0,
                "runtime_seconds": 0.01,
                "converged": True,
                "failed": False,
                "mode": mode,
            },
            {
                "estimator": "qsvt_regularized",
                "rmse": 1.0,
                "weighted_residual": 0.3,
                "residual_norm": 0.3,
                "condition_number": 1000.0,
                "runtime_seconds": 0.02,
                "converged": True,
                "failed": False,
                "mode": mode,
            },
        ]
    ).to_csv(path / "metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "degree": 4,
                "max_error": 0.2,
                "target_error": 0.1,
                "recommended_degree": None,
                "block_encoding_normalization": 1.0,
                "effective_condition_number": 1000.0,
                "proxy_query_count": 9,
                "resource_estimation_scope": "single_run",
                "estimator": "qsvt_regularized",
            },
            {
                "degree": 8,
                "max_error": 0.05,
                "target_error": 0.1,
                "recommended_degree": 8,
                "block_encoding_normalization": 1.0,
                "effective_condition_number": 1000.0,
                "proxy_query_count": 17,
                "resource_estimation_scope": "single_run",
                "estimator": "qsvt_regularized",
            },
        ]
    ).to_csv(path / "qsvt_resource_estimates.csv", index=False)


def _write_sweep_run(path: Path) -> None:
    path.mkdir(parents=True)
    _write_config_resolved(path, run_id="bad_data_sweep", mode="ac_power_flow_linearized")
    aggregate = pd.DataFrame(
        [
            {
                "trial_id": "ratio_0p1_seed1",
                "sweep_name": "bad_data_ratio_sweep",
                "sweep_parameter": "scenario.bad_data.ratio",
                "sweep_value": 0.1,
                "seed": 1,
                "estimator": "pseudoinverse",
                "rmse": 3.0,
                "weighted_residual": 5.0,
                "residual_norm": 5.0,
                "condition_number": 50.0,
                "runtime_seconds": 0.01,
                "failed": False,
                "bad_data_count": 2,
                "bad_data_ratio": 0.1,
                "mode": "ac_power_flow_linearized",
            },
            {
                "trial_id": "ratio_0p1_seed1",
                "sweep_name": "bad_data_ratio_sweep",
                "sweep_parameter": "scenario.bad_data.ratio",
                "sweep_value": 0.1,
                "seed": 1,
                "estimator": "huber_irls",
                "rmse": 1.5,
                "weighted_residual": 4.0,
                "residual_norm": 4.0,
                "condition_number": 50.0,
                "runtime_seconds": 0.03,
                "failed": False,
                "bad_data_count": 2,
                "bad_data_ratio": 0.1,
                "mode": "ac_power_flow_linearized",
            },
        ]
    )
    aggregate.to_csv(path / "aggregate_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "sweep_name": "bad_data_ratio_sweep",
                "sweep_parameter": "scenario.bad_data.ratio",
                "sweep_value": 0.1,
                "estimator": "pseudoinverse",
                "n_trials": 1,
                "failed_count": 0,
                "failure_rate": 0.0,
                "rmse_mean": 3.0,
                "rmse_std": 0.0,
                "weighted_residual_mean": 5.0,
                "weighted_residual_std": 0.0,
                "condition_number_mean": 50.0,
                "condition_number_std": 0.0,
            },
            {
                "sweep_name": "bad_data_ratio_sweep",
                "sweep_parameter": "scenario.bad_data.ratio",
                "sweep_value": 0.1,
                "estimator": "huber_irls",
                "n_trials": 1,
                "failed_count": 0,
                "failure_rate": 0.0,
                "rmse_mean": 1.5,
                "rmse_std": 0.0,
                "weighted_residual_mean": 4.0,
                "weighted_residual_std": 0.0,
                "condition_number_mean": 50.0,
                "condition_number_std": 0.0,
            },
        ]
    ).to_csv(path / "summary_metrics.csv", index=False)


def _write_phase_demo(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "qsvt_demo_plot.png").write_bytes(b"fake-png")
    with (path / "circuit_summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "degree": 5,
                "n_phase_angles": 6,
                "phase_synthesis_method": "scipy_least_squares_scalar_qsp_poc",
                "qiskit_available": False,
                "circuit_depth": None,
                "gate_count_total": 0,
                "target_scale": 1.0,
            },
            file,
        )
    pd.DataFrame(
        {
            "normalized_singular_value": [0.1, 0.5],
            "abs_error": [0.2, 0.1],
        }
    ).to_csv(path / "approximation_error.csv", index=False)
    pd.DataFrame(
        {
            "singular_value": [0.1, 0.5],
            "qsp_abs_error": [0.2, 0.1],
        }
    ).to_csv(path / "qsvt_demo_results.csv", index=False)


def _report_config(tmp_path: Path, runs: list[dict[str, str]]) -> dict:
    return {
        "report": {
            "report_id": "test_report",
            "output_dir": str(tmp_path / "report"),
            "compile_pdf": False,
            "estimator_order": ["pseudoinverse", "qsvt_regularized", "huber_irls"],
            "input_runs": runs,
        }
    }


def test_report_builder_writes_tables_figures_and_manifest(tmp_path: Path) -> None:
    single = tmp_path / "single"
    sweep = tmp_path / "sweep"
    phase_demo = tmp_path / "phase_demo"
    _write_single_run(single, run_id="single")
    _write_sweep_run(sweep)
    _write_phase_demo(phase_demo)
    config = _report_config(
        tmp_path,
        [
            {"path": str(single), "label": "Single & Smoke"},
            {"path": str(sweep), "label": "Bad-data sweep"},
        ],
    )
    config["report"]["phase_demo_inputs"] = [{"path": str(phase_demo), "label": "QSVT phase demo"}]

    report = build_report(config)
    output_dir = report["output_dir"]

    assert (output_dir / "report.tex").is_file()
    assert (output_dir / "tables" / "estimator_performance.tex").is_file()
    assert (output_dir / "tables" / "qsvt_resource_summary.tex").is_file()
    assert (output_dir / "tables" / "qsvt_phase_demo_summary.tex").is_file()
    assert (output_dir / "figures" / "rmse_by_estimator_and_benchmark.png").is_file()
    assert (output_dir / "figures" / "bad_data_rmse_sweep.png").is_file()
    assert list((output_dir / "figures").glob("qsvt_phase_demo_*.png"))
    assert (output_dir / "report_manifest.json").is_file()
    assert (output_dir / "run.log").is_file()
    assert report["manifest"]["pdf_compiled"] is False

    combined = pd.read_csv(output_dir / "combined_metrics.csv")
    ranking = pd.read_csv(output_dir / "estimator_ranking.csv")
    robust = pd.read_csv(output_dir / "robust_bad_data_comparison.csv")
    resource_summary = pd.read_csv(output_dir / "qsvt_resource_summary.csv")
    phase_summary = pd.read_csv(output_dir / "qsvt_phase_demo_summary.csv")

    assert set(combined["report_run_label"]) == {"Single & Smoke", "Bad-data sweep"}
    assert ranking["rmse_rank"].min() == 1.0
    assert set(robust["estimator"]) == {"pseudoinverse", "huber_irls"}
    assert set(resource_summary["degree"]) == {4, 8}
    assert phase_summary.loc[0, "n_phase_angles"] == 6
    assert "Single \\& Smoke" in (output_dir / "tables" / "estimator_performance.tex").read_text(
        encoding="utf-8"
    )


def test_report_builder_handles_missing_optional_resource_file(tmp_path: Path) -> None:
    single = tmp_path / "single_no_resource"
    _write_single_run(single, run_id="single_no_resource")
    (single / "qsvt_resource_estimates.csv").unlink()

    report = build_report(
        _report_config(tmp_path, [{"path": str(single), "label": "No resource run"}])
    )

    output_dir = report["output_dir"]
    assert (output_dir / "qsvt_resource_summary.csv").is_file()
    assert pd.read_csv(output_dir / "combined_metrics.csv").shape[0] == 2


def test_report_builder_missing_required_metrics_fails(tmp_path: Path) -> None:
    broken = tmp_path / "broken"
    broken.mkdir()
    _write_config_resolved(broken, run_id="broken", mode="synthetic_linearized")

    with pytest.raises(FileNotFoundError, match=r"missing metrics\.csv or aggregate_metrics\.csv"):
        build_report(_report_config(tmp_path, [{"path": str(broken), "label": "Broken"}]))


def test_estimator_ranking_orders_custom_estimator_order() -> None:
    summary = pd.DataFrame(
        [
            {"report_run_label": "run", "sweep_name": "single", "estimator": "z", "rmse_mean": 2.0},
            {"report_run_label": "run", "sweep_name": "single", "estimator": "a", "rmse_mean": 1.0},
        ]
    )

    ranking = build_estimator_ranking(summary, ["z", "a"])

    assert list(ranking["estimator"]) == ["z", "a"]
    assert ranking.set_index("estimator")["rmse_rank"].to_dict() == {
        "z": 2.0,
        "a": 1.0,
    }


def test_report_cli_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:  # type: ignore[no-untyped-def]
    single = tmp_path / "single"
    _write_single_run(single, run_id="single")
    config_path = tmp_path / "report.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            _report_config(tmp_path, [{"path": str(single), "label": "CLI smoke"}]),
            file,
        )

    loaded = load_report_config(config_path)
    assert loaded["report"]["report_id"] == "test_report"

    monkeypatch.setattr(sys, "argv", ["build_report", "--config", str(config_path)])
    report_cli_main()

    captured = capsys.readouterr()
    assert "Report complete" in captured.out
    assert (tmp_path / "report" / "report.tex").is_file()
