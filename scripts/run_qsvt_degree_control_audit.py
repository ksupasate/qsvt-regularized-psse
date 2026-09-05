from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.degree_control_audit import run_degree_control_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QSVT true-degree control audit")
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--model", default="ac_linearized")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0e-4])
    parser.add_argument("--degrees", type=int, nargs="+", default=[35, 51])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--grid-size", type=int, default=2048)
    parser.add_argument("--angle-solver", default="iterative")
    parser.add_argument("--phase-timeout-seconds", type=int, default=20)
    parser.add_argument("--output-dir", default="outputs/qsvt_degree_control_audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_degree_control_audit(
        {
            "case": args.case,
            "model": args.model,
            "case_source": args.case_source,
            "submatrix_size": args.submatrix_size,
            "alphas": args.alphas,
            "degrees": args.degrees,
            "seed": args.seed,
            "grid_size": args.grid_size,
            "angle_solver": args.angle_solver,
            "phase_timeout_seconds": args.phase_timeout_seconds,
            "output_dir": args.output_dir,
            "phase_cache_dir": str(Path(args.output_dir) / "phase_cache"),
        }
    )
    completed = [
        row for row in run["rows"] if row["backend_status"] in {"synthesized", "cache_hit"}
    ]
    print(f"QSVT degree-control audit complete: {run['output_dir']}")
    print(f"rows={len(run['rows'])} synthesized_or_cached={len(completed)}")


if __name__ == "__main__":
    main()
