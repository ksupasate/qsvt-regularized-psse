from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.measurement_row_metadata_audit import (  # noqa: E402
    DEFAULT_CASES,
    build_measurement_row_metadata_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Implementation verification: measurement row metadata and subset-mask audit."
    )
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/measurement_row_metadata_audit")
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--case-source", default="pypower")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = build_measurement_row_metadata_audit(
        {
            "input_root": args.input_root,
            "output_dir": args.output_dir,
            "cases": args.cases,
            "case_source": args.case_source,
        }
    )
    print(f"Wrote measurement row metadata audit to {run['output_dir']}")
    print(
        f"status={run['status']} metadata_rows={len(run['metadata_rows'])} "
        f"subset_rows={len(run['subset_rows'])} mask_failures={run['mask_failures']}"
    )


if __name__ == "__main__":
    main()
