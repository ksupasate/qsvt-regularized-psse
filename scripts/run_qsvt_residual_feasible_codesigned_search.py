from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.residual_feasible_codesigned_search import (  # noqa: E402
    run_qsvt_residual_feasible_codesigned_search,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Residual-feasible search over co-designed QSVT-safe targets."
    )
    parser.add_argument(
        "--input",
        default="outputs/qsvt_codesigned_bounded_target_study/codesigned_target_summary.csv",
    )
    parser.add_argument("--no-build-if-missing", action="store_true")
    parser.add_argument("--output-dir", default="outputs/qsvt_residual_feasible_codesigned_search")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "input": args.input,
        "build_if_missing": not args.no_build_if_missing,
        "output_dir": args.output_dir,
    }
    run = run_qsvt_residual_feasible_codesigned_search(config)
    rows = run["rows"]
    feasible = sum(1 for row in rows if bool(row.get("residual_feasible")))
    diagnostic = sum(1 for row in rows if bool(row.get("diagnostic_feasible")))
    print(f"Wrote residual-feasible co-designed search to {run['output_dir']}")
    print(f"Total: {len(rows)}; deployable-feasible: {feasible}; diagnostic-feasible: {diagnostic}")


if __name__ == "__main__":
    main()
