from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robust_qsvt_se.qsvt.full_qsvt_ieee_pathway import build_full_engineering_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full QSVT IEEE engineering report")
    parser.add_argument("--output-dir", default="outputs/full_qsvt_ieee_engineering_report")
    parser.add_argument("--input-root", default="outputs")
    args = parser.parse_args()
    run = build_full_engineering_report(
        {
            "output_dir": args.output_dir,
            "input_root": args.input_root,
            "require_inputs": True,
        }
    )
    print(f"Full QSVT IEEE engineering report complete: {run['output_dir']}")


if __name__ == "__main__":
    main()
