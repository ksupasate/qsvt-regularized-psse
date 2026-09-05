from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "outputs" / "final_falsification_and_submission"


def csv(name: str) -> pd.DataFrame:
    path = FINAL / name
    assert path.is_file(), f"missing final audit artifact: {path}"
    return pd.read_csv(path)


def text(name: str) -> str:
    path = FINAL / name
    assert path.is_file(), f"missing final audit artifact: {path}"
    return path.read_text(encoding="utf-8")
