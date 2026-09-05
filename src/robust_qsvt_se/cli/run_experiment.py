from __future__ import annotations

import argparse
from pathlib import Path

from robust_qsvt_se.experiments.runner import run_experiment
from robust_qsvt_se.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the robust QSVT spectral benchmark.")
    parser.add_argument("--config", required=True, help="Path to YAML experiment config.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a sweep run from trial_results.jsonl when present.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    run = run_experiment(config, resume=args.resume)
    print(f"Run complete: {run['output_dir']}")
    if "metrics" in run:
        metrics = run["metrics"]
        print(metrics[["estimator", "rmse", "residual_norm", "failed"]].to_string(index=False))
    else:
        summary = run["summary_metrics"]
        print(
            summary[
                [
                    "sweep_name",
                    "sweep_value",
                    "estimator",
                    "rmse_mean",
                    "failure_rate",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
