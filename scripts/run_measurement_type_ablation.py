from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.measurement_type_ablation import (  # noqa: E402
    DEFAULT_CASES,
    DEFAULT_ESTIMATORS,
    DEFAULT_SEEDS,
    DEFAULT_SUBSETS,
    build_measurement_type_ablation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: per-measurement-type ablation on controlled IEEE benchmarks."
    )
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--workflow", default="ac_linearized")
    parser.add_argument("--measurement-subsets", nargs="+", default=list(DEFAULT_SUBSETS))
    parser.add_argument("--estimators", nargs="+", default=list(DEFAULT_ESTIMATORS))
    parser.add_argument("--alpha", type=float, default=1.0e-4)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/measurement_type_ablation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = build_measurement_type_ablation(
        {
            "cases": args.cases,
            "workflow": args.workflow,
            "measurement_subsets": args.measurement_subsets,
            "estimators": args.estimators,
            "alpha": args.alpha,
            "seeds": args.seeds,
            "case_source": args.case_source,
            "input_root": args.input_root,
            "output_dir": args.output_dir,
        }
    )
    print(f"Wrote measurement-type ablation to {run['output_dir']}")
    print(
        f"rows={len(run['rows'])} computed={run['computed_rows']} "
        f"rank_deficient={run['rank_deficient_rows']} missing={len(run['missing'])}"
    )
    print(f"findings={run['findings']}")


if __name__ == "__main__":
    main()
