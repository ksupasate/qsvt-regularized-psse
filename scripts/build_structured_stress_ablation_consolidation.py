from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.structured_stress_ablation_consolidation import (  # noqa: E402
    build_structured_stress_ablation_consolidation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5: consolidate structured stress and measurement ablation evidence."
    )
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument("--config-root", default="configs")
    parser.add_argument(
        "--output-dir",
        default="outputs/final_manuscript_package/phase5_structured_stress_ablation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = build_structured_stress_ablation_consolidation(
        {
            "input_root": args.input_root,
            "config_root": args.config_root,
            "output_dir": args.output_dir,
        }
    )
    print(f"Wrote Phase 5 structured stress / measurement ablation to {run['output_dir']}")
    print(
        f"stress_rows={len(run['stress_rows'])} measurement_rows={len(run['measurement_rows'])} "
        f"diagnostic_rows={len(run['diagnostic_rows'])} "
        f"available_stress_types={run['available_stress_types']}"
    )


if __name__ == "__main__":
    main()
