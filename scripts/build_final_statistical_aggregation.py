from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.final_statistical_aggregation import (  # noqa: E402
    build_final_statistical_aggregation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build final statistical summaries from existing manuscript artifacts."
    )
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument("--package-root", default="outputs/final_manuscript_package")
    parser.add_argument(
        "--output-dir",
        default="outputs/final_manuscript_package/statistical_summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = build_final_statistical_aggregation(
        {
            "input_root": args.input_root,
            "package_root": args.package_root,
            "output_dir": args.output_dir,
        }
    )
    print(f"Wrote final statistical aggregation to {run['output_dir']}")
    for key, rows in run["rows"].items():
        print(f"{key}: {len(rows)} rows")
    missing = [row for row in run["manifest_rows"] if str(row["status"]).startswith("missing")]
    print(f"missing_sources={len(missing)}")


if __name__ == "__main__":
    main()
