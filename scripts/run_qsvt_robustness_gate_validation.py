from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.robustness_gate_validation import (  # noqa: E402
    run_qsvt_robustness_gate_validation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dense gate-level validation of co-designed robustness candidates."
    )
    parser.add_argument(
        "--input",
        default="outputs/qsvt_codesigned_subproblem_robustness/"
        "robustness_gate_validation_candidates.csv",
    )
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--model", default="ac_linearized")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--max-configs", type=int, default=3)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/qsvt_robustness_gate_validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "input": args.input,
        "case": args.case,
        "model": args.model,
        "case_source": args.case_source,
        "max_configs": args.max_configs,
        "shots": args.shots,
        "seed": args.seed,
        "output_dir": args.output_dir,
    }
    run = run_qsvt_robustness_gate_validation(config)
    rows = run["rows"]
    feasible = sum(1 for row in rows if bool(row.get("residual_feasible_after_gate")))
    print(f"Wrote robustness gate validation to {run['output_dir']}")
    print(f"Validated: {len(rows)}; residual-feasible after gate: {feasible}")


if __name__ == "__main__":
    main()
