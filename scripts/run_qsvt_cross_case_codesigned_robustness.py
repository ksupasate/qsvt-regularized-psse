from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.cross_case_codesigned_robustness import (  # noqa: E402
    DEFAULT_ALPHAS,
    DEFAULT_CASES,
    DEFAULT_DEGREES,
    DEFAULT_SELECTION_MODES,
    DEFAULT_TARGET_FAMILIES,
    run_qsvt_cross_case_codesigned_robustness,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-case robustness of co-designed QSVT targets on IEEE14/30/57 subproblems."
    )
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--model", default="ac_linearized")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument("--selection-modes", nargs="+", default=list(DEFAULT_SELECTION_MODES))
    parser.add_argument("--target-families", nargs="+", default=list(DEFAULT_TARGET_FAMILIES))
    parser.add_argument("--degrees", type=int, nargs="+", default=list(DEFAULT_DEGREES))
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/qsvt_cross_case_codesigned_robustness")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "cases": args.cases,
        "model": args.model,
        "case_source": args.case_source,
        "submatrix_size": args.submatrix_size,
        "selection_modes": args.selection_modes,
        "target_families": args.target_families,
        "degrees": args.degrees,
        "alphas": args.alphas,
        "seed": args.seed,
        "output_dir": args.output_dir,
    }
    run = run_qsvt_cross_case_codesigned_robustness(config)
    rows = run["rows"]
    feasible = sum(1 for row in rows if bool(row.get("residual_feasible")))
    candidates = sum(1 for row in rows if bool(row.get("gate_validation_recommended")))
    print(f"Wrote cross-case robustness study to {run['output_dir']}")
    print(f"Configs: {len(rows)}; residual-feasible: {feasible}; gate candidates: {candidates}")


if __name__ == "__main__":
    main()
