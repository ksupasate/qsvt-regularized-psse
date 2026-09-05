from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.scalable_qsvt_report import build_scalable_qsvt_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/scalable_qsvt_ieee_report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = build_scalable_qsvt_report({"output_dir": args.output_dir})
    print(f"Scalable QSVT IEEE report complete: {run['output_dir']}")


if __name__ == "__main__":
    main()
