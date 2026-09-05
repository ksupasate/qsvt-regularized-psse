from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.evidence_claim_support_report import (  # noqa: E402
    build_evidence_claim_support_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build QSVT evidence claim-support report")
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/qsvt_evidence_claim_support_report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = build_evidence_claim_support_report(
        {"input_root": args.input_root, "output_dir": args.output_dir}
    )
    print(f"QSVT evidence claim-support report complete: {run['output_dir']}")
    print(f"evidence_layers={len(run['evidence_rows'])}")
    print(f"solver_readiness={run['solver_readiness']}")


if __name__ == "__main__":
    main()
