from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.residual_feasible_gate_validation import (  # noqa: E402
    run_residual_feasible_gate_validation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate-level validation of residual-feasible QSVT configurations"
    )
    parser.add_argument(
        "--input",
        default="outputs/qsvt_residual_feasible_config_search/residual_feasible_configs.csv",
    )
    parser.add_argument(
        "--fallback-input",
        default="outputs/qsvt_residual_feasible_config_search/all_config_results.csv",
    )
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--max-configs", type=int, default=3)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument("--max-manageable-degree", type=int, default=101)
    parser.add_argument(
        "--output-dir",
        default="outputs/qsvt_residual_feasible_gate_validation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_residual_feasible_gate_validation(
        {
            "input": args.input,
            "fallback_input": args.fallback_input,
            "fallback": not args.no_fallback,
            "max_configs": args.max_configs,
            "shots": args.shots,
            "seed": args.seed,
            "case_source": args.case_source,
            "submatrix_size": args.submatrix_size,
            "max_manageable_degree": args.max_manageable_degree,
            "output_dir": args.output_dir,
        }
    )
    completed = [row for row in run["rows"] if row["validation_status"] != "failed"]
    print(f"QSVT residual-feasible gate validation complete: {run['output_dir']}")
    print(
        f"configs={len(run['rows'])} successful={len(completed)} "
        f"used_fallback={run['used_fallback']}"
    )


if __name__ == "__main__":
    main()
