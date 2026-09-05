from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.initializer_ablation import (  # noqa: E402
    DEFAULT_ESTIMATORS,
    DEFAULT_INITIALIZERS,
    build_initializer_ablation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 7 (optional): nonlinear AC x0 initializer mini-ablation."
    )
    parser.add_argument("--cases", nargs="+", default=["ieee14", "ieee57"])
    parser.add_argument("--initializers", nargs="+", default=list(DEFAULT_INITIALIZERS))
    parser.add_argument("--estimators", nargs="+", default=list(DEFAULT_ESTIMATORS))
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/nonlinear_initializer_ablation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = build_initializer_ablation(
        {
            "cases": args.cases,
            "initializers": args.initializers,
            "estimators": args.estimators,
            "alpha": args.alpha,
            "seeds": args.seeds,
            "input_root": args.input_root,
            "output_dir": args.output_dir,
        }
    )
    print(f"Wrote initializer ablation to {run['output_dir']}")
    print(
        f"computed_rows={run['computed_rows']} qsvt==ridge={run['qsvt_ridge_equivalent']} "
        f"missing={len(run['missing_rows'])}"
    )


if __name__ == "__main__":
    main()
