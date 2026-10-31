from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
EXTRACTED_DIR = RAW_DIR / "extracted"
OUTPUT_PATH = ROOT / "data" / "firms_india_raw.csv"
VEGETATION_FIRE_TYPE = 0

REQUIRED_COLUMNS = [
    "latitude",
    "longitude",
    "brightness",
    "scan",
    "track",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "version",
    "bright_t31",
    "frp",
    "daynight",
    "type",
]


def extract_archives(force: bool = False) -> Path:
    """Extract raw FIRMS ZIP files into data/raw/extracted/ if needed."""
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    for archive_path in sorted(RAW_DIR.glob("*.zip")):
        with zipfile.ZipFile(archive_path) as zf:
            for member in zf.namelist():
                if not member.lower().endswith(".csv"):
                    continue
                target = EXTRACTED_DIR / Path(member).name
                if target.exists() and not force:
                    continue
                target.write_bytes(zf.read(member))
    return EXTRACTED_DIR


def _read_fire_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "type" not in df.columns:
        df["type"] = pd.NA  # near-real-time files have no type, so they cannot be confirmed as vegetation fires
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} missing required columns: {sorted(missing)}")
    return df[REQUIRED_COLUMNS].copy()


def prepare_firms_dataset(output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    """Combine all monthly FIRMS CSVs, clean rows, and save the India subset."""
    extract_archives()
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(EXTRACTED_DIR.glob("*.csv")):
        if not csv_path.name.startswith("fire_"):
            continue
        frames.append(_read_fire_csv(csv_path))

    if not frames:
        raise FileNotFoundError(f"No FIRMS CSV files found under {EXTRACTED_DIR}")

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})

    # Keep only valid Indian box values and fire records with usable signal.
    # FIRMS type 0 is "presumed vegetation fire"; 1-3 are volcanoes, static land sources and offshore.
    df = df[
        df["lat"].between(8.0, 35.2)
        & df["lon"].between(68.4, 97.3)
        & df["type"].eq(VEGETATION_FIRE_TYPE)
        & df["frp"].notna()
        & df["brightness"].notna()
        & df["confidence"].notna()
    ].copy()

    df["acq_time"] = df["acq_time"].astype(str).str.replace(".0", "", regex=False).str.zfill(4)
    df["frp"] = pd.to_numeric(df["frp"], errors="coerce")
    df["brightness"] = pd.to_numeric(df["brightness"], errors="coerce")
    df["confidence"] = df["confidence"].astype(str).str.lower().str.strip()

    df = df[df["frp"] > 0].copy()
    df = df.drop_duplicates().reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    prepare_firms_dataset()
    print(f"Saved cleaned FIRMS dataset to {OUTPUT_PATH}")
