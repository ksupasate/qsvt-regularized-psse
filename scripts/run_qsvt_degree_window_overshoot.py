from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.degree_window_overshoot import (  # noqa: E402
    DEFAULT_ALPHAS,
    DEFAULT_DEGREES,
    DEFAULT_TARGET_FAMILIES,
    run_qsvt_degree_window_overshoot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map the degree-vs-overshoot window for co-designed QSVT-safe targets."
    )
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--model", default="ac_linearized")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--degrees", type=int, nargs="+", default=list(DEFAULT_DEGREES))
    parser.add_argument("--target-families", nargs="+", default=list(DEFAULT_TARGET_FAMILIES))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/qsvt_degree_window_overshoot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "case": args.case,
        "model": args.model,
        "case_source": args.case_source,
        "submatrix_size": args.submatrix_size,
        "alphas": args.alphas,
        "degrees": args.degrees,
        "target_families": args.target_families,
        "seed": args.seed,
        "output_dir": args.output_dir,
    }
    run = run_qsvt_degree_window_overshoot(config)
    rows = run["rows"]
    feasible = sum(1 for row in rows if row.get("degree_window_class") == "residual_feasible")
    print(f"Wrote degree-window overshoot study to {run['output_dir']}")
    print(f"Total configs: {len(rows)}; residual-feasible: {feasible}")


if __name__ == "__main__":
    main()
