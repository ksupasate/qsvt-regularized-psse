from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.test_quality_appendix import build_test_quality_appendix  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 9: build the test-quality appendix from the test-quality audit."
    )
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument(
        "--output-dir", default="outputs/final_manuscript_package/test_quality_appendix"
    )
    parser.add_argument("--package-root", default=None)
    parser.add_argument("--full-suite-status", default="run_separately_see_reproducibility")
    parser.add_argument("--ruff-status", default="run_separately_see_reproducibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "input_root": args.input_root,
        "output_dir": args.output_dir,
        "full_suite_status": args.full_suite_status,
        "ruff_status": args.ruff_status,
    }
    if args.package_root:
        config["package_root"] = args.package_root
    run = build_test_quality_appendix(config)
    print(f"Wrote test-quality appendix to {run['output_dir']}")
    print(f"total_tests={run['total']} relies_on_smoke={run['relies_on_smoke']}")


if __name__ == "__main__":
    main()
