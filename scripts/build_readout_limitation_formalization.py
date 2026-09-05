from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.readout_limitation_formalization import (  # noqa: E402
    build_readout_limitation_formalization,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 6: formalize the QSVT readout limitation from existing artifacts."
    )
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/readout_limitation_formalization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = build_readout_limitation_formalization(
        {"input_root": args.input_root, "output_dir": args.output_dir}
    )
    print(f"Wrote readout limitation formalization to {run['output_dir']}")
    print(
        f"observables={len(run['observable_rows'])} cost_rows={len(run['cost_rows'])} "
        f"topk_rows={len(run['topk_rows'])} "
        f"full_vector_assumption_recorded={run['full_vector_assumption_recorded']}"
    )


if __name__ == "__main__":
    main()
