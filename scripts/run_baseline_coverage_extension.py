from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.baseline_coverage_extension import (  # noqa: E402
    DEFAULT_CASES,
    DEFAULT_ESTIMATORS,
    DEFAULT_SEEDS,
    DEFAULT_STRESS_TYPES,
    build_baseline_coverage_extension,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5: minimal baseline coverage extension (LAV / WLS / HHL-style proxy)."
    )
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--workflow", default="ac_linearized")
    parser.add_argument("--stress-types", nargs="+", default=list(DEFAULT_STRESS_TYPES))
    parser.add_argument("--estimators", nargs="+", default=list(DEFAULT_ESTIMATORS))
    parser.add_argument("--alpha", type=float, default=1.0e-4)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/baseline_coverage_extension")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = build_baseline_coverage_extension(
        {
            "cases": args.cases,
            "workflow": args.workflow,
            "stress_types": args.stress_types,
            "estimators": args.estimators,
            "alpha": args.alpha,
            "seeds": args.seeds,
            "case_source": args.case_source,
            "input_root": args.input_root,
            "output_dir": args.output_dir,
        }
    )
    print(f"Wrote baseline coverage extension to {run['output_dir']}")
    print(
        f"rows={len(run['rows'])} robust_beats_ridge_bad_data={run['robust_beats_ridge_bad_data']}"
    )
    for estimator, info in run["coverage"].items():
        fail = info["failure_rate"]
        print(f"  {estimator}: {info['coverage']} cases={info['cases']} fail={fail:.2f}")


if __name__ == "__main__":
    main()
