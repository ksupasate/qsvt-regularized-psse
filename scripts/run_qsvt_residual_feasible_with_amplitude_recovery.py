from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.residual_feasible_with_amplitude_recovery import (  # noqa: E402
    DEFAULT_METHODS,
    DEFAULT_TARGET_DESIGNS,
    run_residual_feasible_with_amplitude_recovery,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Residual-feasible QSVT search using actual amplitude/norm recovery"
    )
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--model", default="ac_linearized")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    parser.add_argument("--degrees", type=int, nargs="+", default=[35, 51, 75])
    parser.add_argument("--target-designs", nargs="+", default=list(DEFAULT_TARGET_DESIGNS))
    parser.add_argument("--norm-recovery-methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--shots", type=int, nargs="+", default=[100, 1000, 10000])
    parser.add_argument("--grover-powers", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output-dir", default="outputs/qsvt_residual_feasible_with_amplitude_recovery"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_residual_feasible_with_amplitude_recovery(
        {
            "case": args.case,
            "model": args.model,
            "case_source": args.case_source,
            "submatrix_size": args.submatrix_size,
            "alphas": args.alphas,
            "degrees": args.degrees,
            "target_designs": args.target_designs,
            "norm_recovery_methods": args.norm_recovery_methods,
            "shots": args.shots,
            "grover_powers": args.grover_powers,
            "seed": args.seed,
            "output_dir": args.output_dir,
        }
    )
    rows = run["rows"]
    feasible = sum(1 for row in rows if row["residual_feasible"])
    diagnostic = sum(1 for row in rows if row["diagnostic_feasible"])
    print(f"QSVT residual-feasible search with amplitude recovery complete: {run['output_dir']}")
    print(f"configs={len(rows)} deployable_feasible={feasible} diagnostic_feasible={diagnostic}")


if __name__ == "__main__":
    main()
