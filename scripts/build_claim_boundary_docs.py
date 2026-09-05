from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.claim_boundary_writer import build_claim_boundary_docs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5: build the claim-boundary table and DO_NOT_CLAIM checklist."
    )
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/final_manuscript_package/claim_boundaries")
    parser.add_argument("--package-root", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {"input_root": args.input_root, "output_dir": args.output_dir}
    if args.package_root:
        config["package_root"] = args.package_root
    run = build_claim_boundary_docs(config)
    print(f"Wrote claim-boundary docs to {run['output_dir']}")
    print(f"counts={run['counts']} do_not_claim_items={len(run['do_not_claim_items'])}")


if __name__ == "__main__":
    main()
