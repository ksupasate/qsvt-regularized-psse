from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.codesigned_gate_validation import (  # noqa: E402
    run_qsvt_codesigned_gate_validation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dense gate-level validation of co-designed QSVT-safe targets."
    )
    parser.add_argument(
        "--input",
        default="outputs/qsvt_residual_feasible_codesigned_search/"
        "deployable_residual_feasible_configs.csv",
    )
    parser.add_argument(
        "--fallback-input",
        default="outputs/qsvt_residual_feasible_codesigned_search/diagnostic_feasible_configs.csv",
    )
    parser.add_argument("--max-configs", type=int, default=3)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/qsvt_codesigned_gate_validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "input": args.input,
        "fallback_input": args.fallback_input,
        "max_configs": args.max_configs,
        "shots": args.shots,
        "seed": args.seed,
        "output_dir": args.output_dir,
    }
    run = run_qsvt_codesigned_gate_validation(config)
    rows = run["rows"]
    feasible = sum(1 for row in rows if bool(row.get("residual_feasible_after_gate")))
    print(f"Wrote co-designed gate validation to {run['output_dir']} (source={run['source']})")
    print(f"Validated: {len(rows)}; residual-feasible after gate: {feasible}")


if __name__ == "__main__":
    main()
