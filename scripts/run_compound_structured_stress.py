from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.compound_structured_stress import (  # noqa: E402
    DEFAULT_CASES,
    DEFAULT_ESTIMATORS,
    DEFAULT_SEEDS,
    DEFAULT_STRESS_TYPES,
    build_compound_structured_stress,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3: compound / weak-area / spatial structured stress."
    )
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--workflow", default="ac_linearized")
    parser.add_argument("--stress-types", nargs="+", default=list(DEFAULT_STRESS_TYPES))
    parser.add_argument("--estimators", nargs="+", default=list(DEFAULT_ESTIMATORS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/structured_compound_stress")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = build_compound_structured_stress(
        {
            "cases": args.cases,
            "workflow": args.workflow,
            "stress_types": args.stress_types,
            "estimators": args.estimators,
            "seeds": args.seeds,
            "case_source": args.case_source,
            "input_root": args.input_root,
            "output_dir": args.output_dir,
        }
    )
    print(f"Wrote compound structured stress to {run['output_dir']}")
    print(
        f"rows={len(run['rows'])} computed={run['computed_rows']} "
        f"controlled_assumption={run['controlled_assumption_rows']} "
        f"available_types={len(run['available_types'])}"
    )
    print(f"findings={run['findings']}")


if __name__ == "__main__":
    main()
