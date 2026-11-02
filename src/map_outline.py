"""Build the land shape for the dashboard map from the FIRMS India detections.

Every 0.15 degree cell with a detection becomes one point, which draws India with no
map tiles, no internet and no neighbouring countries. It shows where detections fall,
so it is not an official boundary.

Usage: python src/map_outline.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_prep import OUTPUT_PATH

OUTLINE_PATH = Path(__file__).resolve().parents[1] / "data" / "india_outline.csv"
CELL_DEGREES = 0.15


def outline_cells(df: pd.DataFrame, size: float = CELL_DEGREES) -> pd.DataFrame:
    """Unique cell centres of the detections."""
    cells = pd.DataFrame({
        "lat": (np.floor(df["lat"] / size) + 0.5) * size,
        "lon": (np.floor(df["lon"] / size) + 0.5) * size,
    })
    return cells.round(3).drop_duplicates().sort_values(["lat", "lon"], ignore_index=True)


def main() -> None:
    outline_cells(pd.read_csv(OUTPUT_PATH, usecols=["lat", "lon"])).to_csv(OUTLINE_PATH, index=False)


if __name__ == "__main__":
    main()
