from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.nonlinear_ac_alpha_stress import (  # noqa: E402
    DEFAULT_ALPHAS,
    DEFAULT_CASES,
    DEFAULT_ESTIMATORS,
    DEFAULT_SEEDS,
    DEFAULT_STRESS_TYPES,
    build_nonlinear_ac_alpha_stress,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4: focused nonlinear AC alpha / weak-area / compound stress."
    )
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--workflow", default="nonlinear_ac")
    parser.add_argument("--stress-types", nargs="+", default=list(DEFAULT_STRESS_TYPES))
    parser.add_argument("--estimators", nargs="+", default=list(DEFAULT_ESTIMATORS))
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/nonlinear_ac_alpha_weak_stress")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = build_nonlinear_ac_alpha_stress(
        {
            "cases": args.cases,
            "workflow": args.workflow,
            "stress_types": args.stress_types,
            "estimators": args.estimators,
            "alphas": args.alphas,
            "seeds": args.seeds,
            "case_source": args.case_source,
            "input_root": args.input_root,
            "output_dir": args.output_dir,
        }
    )
    print(f"Wrote nonlinear AC alpha/weak stress to {run['output_dir']}")
    print(
        f"rows={len(run['rows'])} converged={run['converged_rows']} "
        f"qsvt_ridge_equivalent={run['qsvt_ridge_equivalent']}"
    )


if __name__ == "__main__":
    main()
