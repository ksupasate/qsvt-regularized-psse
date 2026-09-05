from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.sparse_oracle_assumption_ledger import (  # noqa: E402
    build_sparse_oracle_assumption_ledger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sparse-oracle assumption ledger")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["ieee14", "ieee30", "ieee57", "ieee118", "ieee300"],
    )
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--alpha", type=float, default=1.0e-4)
    parser.add_argument("--degree", type=int, default=51)
    parser.add_argument("--output-dir", default="outputs/sparse_oracle_assumption_ledger")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = build_sparse_oracle_assumption_ledger(
        {
            "cases": args.cases,
            "case_source": args.case_source,
            "alpha": args.alpha,
            "degree": args.degree,
            "output_dir": args.output_dir,
        }
    )
    print(f"Sparse-oracle assumption ledger complete: {run['output_dir']}")
    print(f"assumptions={len(run['ledger_rows'])}")


if __name__ == "__main__":
    main()
