from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.jacobian_validation import (  # noqa: E402
    DEFAULT_CASES,
    DEFAULT_EPSILON,
    build_jacobian_validation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Implementation verification: finite-difference AC/DC Jacobian validation "
        "and weighted-Jacobian consistency audit."
    )
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/jacobian_validation")
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = build_jacobian_validation(
        {
            "input_root": args.input_root,
            "output_dir": args.output_dir,
            "cases": args.cases,
            "case_source": args.case_source,
            "epsilon": args.epsilon,
        }
    )
    print(f"Wrote Jacobian validation to {run['output_dir']}")
    print(
        f"ac_status={run['ac_status']} ac_max_rel={run['ac_max_relative_error']:.3e} "
        f"ac_cases={run['ac_cases']}"
    )
    print(
        f"dc_status={run['dc_status']} dc_max_rel={run['dc_max_relative_error']:.3e} "
        f"dc_cases={run['dc_cases']}"
    )
    print(f"weighted_audit_rows={len(run['weighted'])}")


if __name__ == "__main__":
    main()
