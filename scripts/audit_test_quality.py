from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.test_quality_audit import build_test_quality_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase E: audit test quality and identify smoke tests."
    )
    parser.add_argument("--test-root", default="tests")
    parser.add_argument(
        "--output-dir",
        default="outputs/final_manuscript_package/test_quality_audit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = build_test_quality_audit({"test_root": args.test_root, "output_dir": args.output_dir})
    print(f"Wrote test-quality audit to {run['output_dir']}")
    print(
        f"tests={len(run['inventory_rows'])} categories={run['category_counts']} "
        f"relies_on_smoke={run['relies_on_smoke']}"
    )


if __name__ == "__main__":
    main()
