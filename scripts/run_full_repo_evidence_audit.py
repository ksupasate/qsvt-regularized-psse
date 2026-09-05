from __future__ import annotations

# ruff: noqa: E402,I001

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.full_repo_evidence_audit import run_cli


if __name__ == "__main__":
    raise SystemExit(run_cli(sys.argv[1:]))
