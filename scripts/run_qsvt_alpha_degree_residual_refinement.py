from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.alpha_degree_refinement import (  # noqa: E402
    run_alpha_degree_residual_refinement,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QSVT alpha/degree residual refinement")
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--model", default="ac_linearized")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0e-4])
    parser.add_argument("--degrees", type=int, nargs="+", default=[51])
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/qsvt_alpha_degree_residual_refinement")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_alpha_degree_residual_refinement(
        {
            "case": args.case,
            "model": args.model,
            "case_source": args.case_source,
            "submatrix_size": args.submatrix_size,
            "alphas": args.alphas,
            "degrees": args.degrees,
            "shots": args.shots,
            "seed": args.seed,
            "output_dir": args.output_dir,
        }
    )
    completed = [row for row in run["rows"] if row["run_status"] == "completed"]
    print(f"QSVT alpha/degree refinement complete: {run['output_dir']}")
    print(f"rows={len(run['rows'])} completed={len(completed)}")


if __name__ == "__main__":
    main()
