from __future__ import annotations

import platform
from copy import deepcopy
from logging import Logger
from pathlib import Path
from typing import Any

import pandas as pd

from robust_qsvt_se import __version__
from robust_qsvt_se.estimators.base import Estimator
from robust_qsvt_se.estimators.hhl_style_inverse_proxy import HHLStyleInverseProxyEstimator
from robust_qsvt_se.estimators.huber_irls import HuberIRLSEstimator
from robust_qsvt_se.estimators.lav import LAVEstimator
from robust_qsvt_se.estimators.normal_equation_wls import NormalEquationWLSEstimator
from robust_qsvt_se.estimators.pseudoinverse import PseudoinverseEstimator
from robust_qsvt_se.estimators.qsvt_spectral import QSVTSpectralEstimator
from robust_qsvt_se.estimators.qsvt_unregularized_inverse import (
    QSVTUnregularizedInverseEstimator,
)
from robust_qsvt_se.estimators.ridge import RidgeEstimator
from robust_qsvt_se.estimators.truncated_svd import TruncatedSVDEstimator
from robust_qsvt_se.experiments.iterative_ac import run_iterative_ac_experiment
from robust_qsvt_se.experiments.reporting import write_artifacts
from robust_qsvt_se.experiments.scenarios import build_system_from_config
from robust_qsvt_se.experiments.sweeps import run_sweeps
from robust_qsvt_se.utils.io import ensure_directory
from robust_qsvt_se.utils.logging import configure_run_logger
from robust_qsvt_se.utils.seed import make_rng


def run_experiment(config: dict[str, Any], *, resume: bool = False) -> dict[str, Any]:
    resolved_config = _with_runtime_metadata(config)
    output_dir = _output_dir(resolved_config)
    if (
        output_dir.exists()
        and any(output_dir.iterdir())
        and not resolved_config["output"].get("overwrite", False)
        and not resume
    ):
        raise FileExistsError(f"output directory already exists and overwrite=false: {output_dir}")
    ensure_directory(output_dir)
    logger = configure_run_logger(output_dir / "run.log")
    logger.info("Starting run %s", resolved_config["output"]["run_id"])

    if resolved_config["system"].get("mode") in {
        "ac_iterative_state_estimation",
        "nonlinear_ac_state_estimation",
    }:
        return run_iterative_ac_experiment(
            config=resolved_config,
            output_dir=output_dir,
            logger=logger,
            resume=resume,
        )

    if resolved_config.get("sweeps"):
        return run_sweeps(
            config=resolved_config,
            output_dir=output_dir,
            logger=logger,
            trial_runner=_run_trial,
            resume=resume,
        )

    system, results = _run_trial(resolved_config, logger)
    artifacts = write_artifacts(
        output_dir=output_dir,
        config=resolved_config,
        system=system,
        results=results,
        save_plots=bool(resolved_config["output"].get("save_plots", False)),
    )
    logger.info("Completed run %s", resolved_config["output"]["run_id"])
    metrics = pd.read_csv(artifacts["metrics"])
    return {
        "output_dir": output_dir,
        "artifacts": artifacts,
        "metrics": metrics,
        "results": results,
        "system": system,
    }


def _run_trial(
    config: dict[str, Any],
    logger: Logger,
) -> tuple[Any, list[Any]]:
    rng = make_rng(int(config["seed"]))
    system = build_system_from_config(config, rng)
    estimators = [_build_estimator(item) for item in config["estimators"]]
    results = []
    for estimator in estimators:
        logger.info("Running estimator %s", estimator.name)
        result = estimator.solve(system)
        results.append(result)
        logger.info(
            "Estimator %s finished failed=%s rmse=%s residual=%s",
            result.name,
            result.failed,
            result.rmse,
            result.residual_norm,
        )
    return system, results


def _build_estimator(config: dict[str, Any]) -> Estimator:
    name = config["name"]
    if name == "pseudoinverse":
        return PseudoinverseEstimator(rcond=float(config.get("rcond", 1e-12)))
    if name == "normal_equation_wls":
        return NormalEquationWLSEstimator(
            max_gain_condition_number=float(config.get("max_gain_condition_number", float("inf")))
        )
    if name == "ridge":
        return RidgeEstimator(alpha=float(config["alpha"]))
    if name == "truncated_svd":
        return TruncatedSVDEstimator(tau=float(config["tau"]))
    if name == "qsvt_regularized":
        return QSVTSpectralEstimator(alpha=float(config["alpha"]))
    if name == "qsvt_unregularized_inverse":
        return QSVTUnregularizedInverseEstimator(cutoff=float(config.get("cutoff", 1.0e-8)))
    if name == "hhl_style_inverse_proxy":
        return HHLStyleInverseProxyEstimator(
            cutoff=float(config.get("cutoff", 1.0e-8)),
            precision=float(config.get("precision", 1.0e-3)),
            instability_condition_threshold=float(
                config.get("instability_condition_threshold", 1.0e8)
            ),
            fail_on_instability=bool(config.get("fail_on_instability", False)),
        )
    if name == "huber_irls":
        return HuberIRLSEstimator(
            delta=float(config["delta"]),
            max_iterations=int(config.get("max_iterations", 50)),
            tolerance=float(config.get("tolerance", 1.0e-8)),
        )
    if name == "lav":
        return LAVEstimator()
    raise ValueError(f"unknown estimator name: {name}")


def _output_dir(config: dict[str, Any]) -> Path:
    output = config["output"]
    return Path(output["root"]) / output["run_id"]


def _with_runtime_metadata(config: dict[str, Any]) -> dict[str, Any]:
    resolved = deepcopy(config)
    resolved["output"].setdefault("run_id", f"{config['run_name']}_seed{config['seed']}")
    resolved["runtime"] = {
        "python_version": platform.python_version(),
        "package_version": __version__,
        "implementation_scope": (
            "Synthetic, DC, AC-linearized, nonlinear AC, and QSVT resource benchmarks"
        ),
    }
    return resolved
