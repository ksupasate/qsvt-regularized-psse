from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.classical_audit_recheck import (  # noqa: E402
    build_classical_audit_recheck,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase A: recheck classical spectral-filtering estimator/result coverage."
    )
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument(
        "--phase-package-dir",
        default="outputs/final_manuscript_package/phase2_classical_spectral_filtering",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/final_manuscript_package/phase2_classical_recheck",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = build_classical_audit_recheck(
        {
            "input_root": args.input_root,
            "phase_package_dir": args.phase_package_dir,
            "output_dir": args.output_dir,
        }
    )
    print(f"Wrote Phase 2 classical recheck to {run['output_dir']}")
    statuses = {r["estimator"]: r["coverage_status"] for r in run["coverage_rows"]}
    print(f"estimators={len(run['coverage_rows'])} statuses={statuses}")


if __name__ == "__main__":
    main()
