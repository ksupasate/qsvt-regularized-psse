from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.gate_validation_with_amplitude_recovery import (  # noqa: E402
    run_gate_validation_with_amplitude_recovery,
)

_SEARCH_DIR = "outputs/qsvt_residual_feasible_with_amplitude_recovery"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate-level QSVT validation with amplitude-based norm recovery"
    )
    parser.add_argument("--input", default=f"{_SEARCH_DIR}/residual_feasible_configs.csv")
    parser.add_argument(
        "--fallback-input", default=f"{_SEARCH_DIR}/diagnostic_feasible_configs.csv"
    )
    parser.add_argument(
        "--closest-input", default=f"{_SEARCH_DIR}/all_amplitude_recovery_configs.csv"
    )
    parser.add_argument("--max-configs", type=int, default=3)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument("--grover-powers", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    parser.add_argument(
        "--output-dir", default="outputs/qsvt_gate_validation_with_amplitude_recovery"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_gate_validation_with_amplitude_recovery(
        {
            "input": args.input,
            "fallback_input": args.fallback_input,
            "closest_input": args.closest_input,
            "max_configs": args.max_configs,
            "shots": args.shots,
            "seed": args.seed,
            "case_source": args.case_source,
            "submatrix_size": args.submatrix_size,
            "grover_powers": args.grover_powers,
            "output_dir": args.output_dir,
        }
    )
    completed = [row for row in run["rows"] if row["gate_run_status"] == "completed"]
    print(f"QSVT gate validation with amplitude recovery complete: {run['output_dir']}")
    print(f"configs={len(run['rows'])} completed={len(completed)} source={run['source']}")


if __name__ == "__main__":
    main()
