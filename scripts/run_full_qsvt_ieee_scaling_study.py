from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robust_qsvt_se.qsvt.full_qsvt_ieee_pathway import run_ieee_scaling_study  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full QSVT IEEE scaling/resource study")
    parser.add_argument(
        "--cases", nargs="+", default=["ieee14", "ieee30", "ieee57", "ieee118", "ieee300"]
    )
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--matrix-source", default="weighted_jacobian")
    parser.add_argument(
        "--implemented-submatrix-sizes", nargs="+", type=int, default=[4, 8, 16, 32]
    )
    parser.add_argument("--alpha", type=float, default=1.0e-4)
    parser.add_argument("--degree", type=int, default=51)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/full_qsvt_ieee_scaling")
    args = parser.parse_args()
    run = run_ieee_scaling_study(
        {
            "cases": args.cases,
            "case_source": args.case_source,
            "matrix_source": args.matrix_source,
            "implemented_submatrix_sizes": args.implemented_submatrix_sizes,
            "alpha": args.alpha,
            "degree": args.degree,
            "seed": args.seed,
            "output_dir": args.output_dir,
        }
    )
    simulations = run["implemented_simulations"]
    executed = simulations[simulations["status"] == "ok"]
    print(f"Full QSVT IEEE scaling study complete: {run['output_dir']}")
    print(f"simulation_rows={len(simulations)} executed_explicit_dense={len(executed)}")


if __name__ == "__main__":
    main()
