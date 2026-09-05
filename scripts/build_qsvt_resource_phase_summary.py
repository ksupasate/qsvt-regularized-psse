from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.qsvt_resource_phase_summary import (  # noqa: E402
    build_qsvt_resource_phase_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 6: consolidate QSVT resource, phase, gate, and readout evidence."
    )
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument(
        "--output-dir", default="outputs/final_manuscript_package/phase6_qsvt_resource_phase"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = build_qsvt_resource_phase_summary(
        {"input_root": args.input_root, "output_dir": args.output_dir}
    )
    print(f"Wrote Phase 6 QSVT resource/phase summary to {run['output_dir']}")
    print(
        f"target_window={len(run['target_rows'])} gate={len(run['gate_rows'])} "
        f"observable={len(run['observable_rows'])} resource={len(run['resource_rows'])}"
    )


if __name__ == "__main__":
    main()
