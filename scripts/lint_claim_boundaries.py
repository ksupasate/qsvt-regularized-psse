from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.claim_lint import build_claim_lint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4 hardening: lint manuscript text and artifacts for overclaims."
    )
    parser.add_argument("--input-root", default="outputs/final_manuscript_package")
    parser.add_argument("--output-dir", default="outputs/final_manuscript_package/claim_lint")
    parser.add_argument("--manuscript", default=None, help="optional manuscript file to scan")
    parser.add_argument(
        "--extra-path", action="append", default=[], help="additional file or directory to scan"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = build_claim_lint(
        {
            "input_root": args.input_root,
            "output_dir": args.output_dir,
            "manuscript": args.manuscript,
            "extra_paths": args.extra_path,
        }
    )
    print(f"Wrote claim lint report to {run['output_dir']}")
    print(
        f"scanned_files={len(run['scanned_files'])} hits={len(run['rows'])} "
        f"high_risk={run['high_risk_count']}"
    )
    for finding in run["high_risk"][:10]:
        print(f"  HIGH {finding['file_path']}:{finding['line_number']} {finding['matched_phrase']}")


if __name__ == "__main__":
    main()
