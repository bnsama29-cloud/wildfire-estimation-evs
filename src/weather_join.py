"""Join FIRMS fire points with Open-Meteo hourly weather.

Open-Meteo counts every 2 weeks of data per location as one call, so one 2-year
series costs about 52 calls and the free tier allows 5,000 calls an hour. Weather
is therefore fetched for 1-degree grid cells, busiest cells first, and each cell
is cached on disk so an interrupted run resumes without repeating requests.

Usage:
    python src/weather_join.py fetch --cells 180     # slow, paced, resumable
    python src/weather_join.py join                  # offline, uses the cache only
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from data_prep import OUTPUT_PATH

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache" / "weather"
FULL_JOIN_PATH = ROOT / "data" / "india_fires_weather_full.csv.gz"
SAMPLE_JOIN_PATH = ROOT / "data" / "india_fires_weather.csv"

API_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARIABLES = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]
START_DATE, END_DATE = "2024-01-01", "2025-12-31"
CELL_DEGREES = 1.0
RAIN_WINDOW_HOURS = 168
SECONDS_PER_CELL = 42  # about 4,450 weighted calls an hour, under the 5,000 limit
RATE_LIMIT_WAIT_SECONDS = 600
MAX_ATTEMPTS = 12
DASHBOARD_SAMPLE_ROWS = 30_000

logger = logging.getLogger("weather_join")


def cell_center(values, size: float = CELL_DEGREES) -> np.ndarray:
    """Centre of the grid cell that holds each coordinate."""
    return (np.floor(np.asarray(values, dtype=np.float64) / size) + 0.5) * size


def load_fires(path: str | Path = OUTPUT_PATH, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["lat", "lon", "acq_date", "acq_time", "frp", "confidence", "daynight"],
        dtype={"lat": "float32", "lon": "float32", "frp": "float32", "acq_date": "string", "acq_time": "string"},
    )
    df = df[df["acq_date"].between(start, end)].copy()
    df["time_utc"] = pd.to_datetime(
        df["acq_date"] + df["acq_time"].str.zfill(4), format="%Y-%m-%d%H%M", errors="coerce"
    )
    df = df.dropna(subset=["time_utc"])
    df["cell_lat"] = cell_center(df["lat"])
    df["cell_lon"] = cell_center(df["lon"])
    return df


def rank_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Grid cells ordered by detection count, busiest first."""
    return (
        df.groupby(["cell_lat", "cell_lon"]).size().rename("fires").reset_index()
        .sort_values("fires", ascending=False, ignore_index=True)
    )


def cache_path(cell_lat: float, cell_lon: float) -> Path:
    return CACHE_DIR / f"{cell_lat:.2f}_{cell_lon:.2f}.npz"


def rain_last_week(precipitation: np.ndarray, hours: int = RAIN_WINDOW_HOURS) -> np.ndarray:
    """Rain total over the trailing window, including the current hour. Missing hours count as dry."""
    totals = np.cumsum(np.nan_to_num(precipitation), dtype=np.float64)
    shifted = np.concatenate([np.zeros(hours), totals[:-hours]]) if len(totals) > hours else np.zeros(len(totals))
    return (totals - shifted).astype(np.float32)


def fetch_cell(cell_lat: float, cell_lon: float, session: requests.Session | None = None) -> Path:
    """Download one cell into the cache. A rate-limited request waits and retries."""
    destination = cache_path(cell_lat, cell_lon)
    if destination.exists():
        return destination
    session = session or requests.Session()
    params = {
        "latitude": cell_lat,
        "longitude": cell_lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "GMT",
        "wind_speed_unit": "kmh",
    }
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = session.get(API_URL, params=params, timeout=120)
        except requests.RequestException as exc:
            logger.warning("network error for %s,%s: %s", cell_lat, cell_lon, exc)
            time.sleep(30 * attempt)
            continue
        if response.status_code == 429:
            logger.warning("rate limited, waiting %ss", RATE_LIMIT_WAIT_SECONDS)
            time.sleep(RATE_LIMIT_WAIT_SECONDS)
            continue
        if response.status_code >= 500:
            time.sleep(30 * attempt)
            continue
        response.raise_for_status()
        hourly = response.json()["hourly"]
        arrays = {name: np.array(hourly[name], dtype=np.float32) for name in HOURLY_VARIABLES}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(".part.npz")
        np.savez_compressed(partial, **arrays)
        partial.replace(destination)
        return destination
    raise RuntimeError(f"gave up on cell {cell_lat},{cell_lon} after {MAX_ATTEMPTS} attempts")


def fetch_cells(cells: pd.DataFrame, limit: int) -> None:
    session = requests.Session()
    for number, row in enumerate(cells.head(limit).itertuples(index=False), start=1):
        if cache_path(row.cell_lat, row.cell_lon).exists():
            continue
        fetch_cell(row.cell_lat, row.cell_lon, session)
        logger.info("cached cell %d/%d (%.1f, %.1f, %d fires)", number, limit, row.cell_lat, row.cell_lon, row.fires)
        if number < limit:
            time.sleep(SECONDS_PER_CELL)


def attach_weather(group: pd.DataFrame, weather: dict[str, np.ndarray], start: str = START_DATE) -> pd.DataFrame:
    """Add the weather of the hour nearest each satellite pass to one cell's fire points."""
    hours = (group["time_utc"] - pd.Timestamp(start)) / pd.Timedelta(hours=1)
    index = np.clip(np.rint(hours.to_numpy(dtype=np.float64)).astype(np.int64), 0, len(weather["temperature_2m"]) - 1)
    out = group.copy()
    out["temp_c"] = weather["temperature_2m"][index]
    out["rh_percent"] = weather["relative_humidity_2m"][index]
    out["wind_kmh"] = weather["wind_speed_10m"][index]
    out["rain_7d_mm"] = rain_last_week(weather["precipitation"])[index]
    return out


def join_cached(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for (cell_lat, cell_lon), group in df.groupby(["cell_lat", "cell_lon"]):
        path = cache_path(cell_lat, cell_lon)
        if not path.exists():
            continue
        with np.load(path) as cached:
            weather = {name: cached[name] for name in HOURLY_VARIABLES}
        parts.append(attach_weather(group, weather))
    if not parts:
        raise FileNotFoundError(f"no cached weather under {CACHE_DIR}; run the fetch step first")
    joined = pd.concat(parts, ignore_index=True)
    joined = joined.dropna(subset=["temp_c", "rh_percent", "wind_kmh"])
    columns = ["lat", "lon", "acq_date", "acq_time", "frp", "confidence", "daynight",
               "temp_c", "rh_percent", "wind_kmh", "rain_7d_mm"]
    return joined[columns].round({"lat": 4, "lon": 4, "temp_c": 1, "rh_percent": 1, "wind_kmh": 1, "rain_7d_mm": 1})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["fetch", "join"])
    parser.add_argument("--cells", type=int, default=180, help="Number of busiest grid cells to fetch")
    args = parser.parse_args()

    fires = load_fires()
    if args.step == "fetch":
        cells = rank_cells(fires)
        covered = cells["fires"].head(args.cells).sum() / cells["fires"].sum()
        logger.info("fetching %d of %d cells, covering %.1f%% of detections", args.cells, len(cells), 100 * covered)
        fetch_cells(cells, args.cells)
        return

    joined = join_cached(fires)
    logger.info("joined %d of %d fire points", len(joined), len(fires))
    joined.to_csv(FULL_JOIN_PATH, index=False)
    joined.sample(min(DASHBOARD_SAMPLE_ROWS, len(joined)), random_state=42).to_csv(SAMPLE_JOIN_PATH, index=False)


if __name__ == "__main__":
    main()
