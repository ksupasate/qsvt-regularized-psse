#!/usr/bin/env python3
"""Regenerate the normalized/propagated sparse value-oracle quantization error report."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.sparse_quantization_error_report import main  # noqa: E402

if __name__ == "__main__":
    main(sys.argv[1:])
