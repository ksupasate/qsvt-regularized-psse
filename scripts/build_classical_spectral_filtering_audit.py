from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.classical_spectral_filtering_audit import (  # noqa: E402
    build_classical_spectral_filtering_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2: audit and consolidate classical spectral-filtering outputs."
    )
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument(
        "--output-dir",
        default="outputs/final_manuscript_package/phase2_classical_spectral_filtering",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = build_classical_spectral_filtering_audit(
        {"input_root": args.input_root, "output_dir": args.output_dir}
    )
    print(f"Wrote Phase 2 classical spectral-filtering audit to {run['output_dir']}")
    print(f"main_results={len(run['main_rows'])} spectrum={len(run['spectrum_rows'])}")


if __name__ == "__main__":
    main()
