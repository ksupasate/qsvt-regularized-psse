from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.qsvt.solver_prototype_consolidation_audit import (  # noqa: E402
    run_qsvt_solver_prototype_consolidation_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate the latest co-designed QSVT solver outputs into an audit."
    )
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument("--max-best-configs", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs/qsvt_solver_prototype_consolidation_audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "input_root": args.input_root,
        "max_best_configs": args.max_best_configs,
        "output_dir": args.output_dir,
    }
    run = run_qsvt_solver_prototype_consolidation_audit(config)
    print(f"Wrote solver-prototype consolidation audit to {run['output_dir']}")
    print(
        f"Best configs: {len(run['best_configs'])}; gate-validated: {len(run['gate_configs'])}; "
        f"observable rows: {len(run['observable_rows'])}"
    )


if __name__ == "__main__":
    main()
