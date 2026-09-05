from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.best_config_gate_validation import (  # noqa: E402
    run_best_config_gate_validation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run best-config gate-level QSVT validation")
    parser.add_argument(
        "--input",
        default="outputs/qsvt_polynomial_action_decomposition/polynomial_action_residuals.csv",
    )
    parser.add_argument("--max-configs", type=int, default=5)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs/qsvt_best_config_gate_validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_best_config_gate_validation(
        {
            "input": args.input,
            "max_configs": args.max_configs,
            "shots": args.shots,
            "seed": args.seed,
            "case_source": args.case_source,
            "submatrix_size": args.submatrix_size,
            "output_dir": args.output_dir,
        }
    )
    completed = [row for row in run["rows"] if row["run_status"] == "completed"]
    print(f"QSVT best-config gate validation complete: {run['output_dir']}")
    print(f"selected={len(run['selected_configs'])} completed={len(completed)}")


if __name__ == "__main__":
    main()
