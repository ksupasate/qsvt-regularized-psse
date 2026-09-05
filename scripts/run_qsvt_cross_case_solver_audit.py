from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.cross_case_solver_audit import (  # noqa: E402
    CROSS_CASE_SELECTION_MODES,
    DEFAULT_CASES,
    run_qsvt_cross_case_solver_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-case readiness audit for the selected QSVT solver prototype."
    )
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--model", default="ac_linearized")
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--submatrix-size", type=int, default=4)
    parser.add_argument("--selection-modes", nargs="+", default=list(CROSS_CASE_SELECTION_MODES))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="outputs/qsvt_cross_case_solver_audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "input_root": args.input_root,
        "cases": args.cases,
        "model": args.model,
        "case_source": args.case_source,
        "submatrix_size": args.submatrix_size,
        "selection_modes": args.selection_modes,
        "seed": args.seed,
        "output_dir": args.output_dir,
    }
    run = run_qsvt_cross_case_solver_audit(config)
    candidates = run["candidates"]
    available = int(candidates["available"].astype(bool).sum()) if not candidates.empty else 0
    print(f"Wrote cross-case solver audit to {run['output_dir']}")
    print(f"Cases inspected: {len(candidates)}; available: {available}")


if __name__ == "__main__":
    main()
