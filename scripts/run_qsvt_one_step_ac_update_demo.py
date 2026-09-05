from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robust_qsvt_se.qsvt.full_qsvt_ieee_pathway import run_one_step_ac_update_demo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-step AC QSVT update subproblem demo")
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--matrix-source", default="weighted_jacobian")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0e-4)
    parser.add_argument("--degree", type=int, default=51)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/qsvt_one_step_ac_update_demo")
    args = parser.parse_args()
    run = run_one_step_ac_update_demo(
        {
            "case": args.case,
            "case_source": args.case_source,
            "matrix_source": args.matrix_source,
            "submatrix_size": args.submatrix_size,
            "alpha": args.alpha,
            "degree": args.degree,
            "seed": args.seed,
            "output_dir": args.output_dir,
        }
    )
    row = run["update_summary"].iloc[0]
    print(f"One-step AC QSVT update demo complete: {run['output_dir']}")
    state_error = row["phase_aligned_state_l2_error_vs_ridge"]
    print(f"submatrix_size={int(row['submatrix_size'])} state_error={state_error:.3e}")


if __name__ == "__main__":
    main()
