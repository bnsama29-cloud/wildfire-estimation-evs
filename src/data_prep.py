from __future__ import annotations

import argparse
import gzip
import json
import threading
import time
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests import Response, Session
from requests.exceptions import RequestException

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
EXTRACTED_DIR = RAW_DIR / "extracted"
OUTPUT_PATH = ROOT / "data" / "firms_india_raw.csv"
WEATHER_OUTPUT_PATH = ROOT / "data" / "india_fires_weather.csv"
CACHE_DIR = ROOT / "data" / "cache"
WEATHER_ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
)
WEATHER_COLUMNS = ("temp_c", "rh_percent", "wind_kmh")
REQUEST_INTERVAL_SECONDS = 0.25
API_MIN_REQUEST_INTERVAL_SECONDS = 4.0
HOURLY_LIMIT_RETRY_SECONDS = 3600.0
MAX_REQUEST_RETRIES = 5

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
        df["type"] = 0
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
    df = df.dropna(
        subset=[
            "lat",
            "lon",
            "acq_date",
            "acq_time",
            "frp",
            "confidence",
            "brightness",
            "satellite",
        ]
    )
    df = df[
        df["lat"].between(8.0, 35.1)
        & df["lon"].between(68.5, 97.2)
    ].copy()

    df["acq_time"] = (
        df["acq_time"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    )
    df["frp"] = pd.to_numeric(df["frp"], errors="coerce")
    df["brightness"] = pd.to_numeric(df["brightness"], errors="coerce")
    df["confidence"] = df["confidence"].astype(str).str.lower().str.strip()

    df = df[df["frp"] > 0].copy()
    df = df.drop_duplicates().reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def weather_cell(lat: float, lon: float) -> tuple[int, int]:
    """Return the integer indices for a nearest 0.25-degree weather cell."""
    return int(np.rint(float(lat) * 4)), int(np.rint(float(lon) * 4))


def weather_cache_path(
    cell: tuple[int, int], start_date: str, end_date: str, cache_dir: Path = CACHE_DIR
) -> Path:
    return cache_dir / (
        f"weather_{cell[0]}_{cell[1]}_{start_date}_{end_date}.json.gz"
    )


def _retry_delay(response: Response | None, attempt: int) -> float:
    if response is not None:
        if "hourly api request limit exceeded" in response.text.casefold():
            return HOURLY_LIMIT_RETRY_SECONDS
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                try:
                    retry_time = parsedate_to_datetime(retry_after)
                except (TypeError, ValueError, OverflowError):
                    retry_time = None
                if retry_time is not None:
                    if retry_time.tzinfo is None:
                        retry_time = retry_time.replace(tzinfo=timezone.utc)
                    return max(
                        0.0,
                        (retry_time - datetime.now(timezone.utc)).total_seconds(),
                    )
    return min(60.0, 1.0 * (2**attempt))


def _validate_weather_payload(payload: dict[str, Any], source: Path | None = None) -> None:
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        location = f" in {source}" if source else ""
        raise ValueError(f"Open-Meteo response has no hourly data{location}")
    required = ("time", *WEATHER_VARIABLES)
    missing = [column for column in required if column not in hourly]
    if missing:
        location = f" in {source}" if source else ""
        raise ValueError(f"Open-Meteo response is missing {missing}{location}")
    lengths = {len(hourly[column]) for column in required}
    if len(lengths) != 1:
        location = f" in {source}" if source else ""
        raise ValueError(f"Open-Meteo hourly arrays have different lengths{location}")


def fetch_weather_cell(
    cell: tuple[int, int],
    start_date: str,
    end_date: str,
    cache_dir: Path = CACHE_DIR,
    session: Session | None = None,
    sleep_seconds: float = REQUEST_INTERVAL_SECONDS,
    max_retries: int = MAX_REQUEST_RETRIES,
    request_gate: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Fetch/cache one complete hourly Open-Meteo series in UTC."""
    cache_path = weather_cache_path(cell, start_date, end_date, cache_dir)
    if cache_path.exists():
        with gzip.open(cache_path, "rt", encoding="utf-8") as cache_file:
            cached = json.load(cache_file)
        expected_request = {
            "latitude": cell[0] / 4,
            "longitude": cell[1] / 4,
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "UTC",
            "hourly": list(WEATHER_VARIABLES),
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
        }
        if cached.get("request") != expected_request:
            raise ValueError(f"Weather cache metadata does not match request: {cache_path}")
        payload = cached.get("response")
        if not isinstance(payload, dict):
            raise ValueError(f"Weather cache is missing its API response: {cache_path}")
        _validate_weather_payload(payload, cache_path)
        return payload

    owns_session = session is None
    http = session or requests.Session()
    request_params = {
        "latitude": cell[0] / 4,
        "longitude": cell[1] / 4,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(WEATHER_VARIABLES),
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
    }
    response: Response | None = None
    try:
        for attempt in range(max_retries + 1):
            if request_gate is not None:
                request_gate()
            elif sleep_seconds > 0:
                time.sleep(sleep_seconds)
            try:
                response = http.get(WEATHER_ENDPOINT, params=request_params, timeout=90)
            except RequestException:
                if attempt == max_retries:
                    raise
                time.sleep(_retry_delay(None, attempt))
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == max_retries:
                    response.raise_for_status()
                time.sleep(_retry_delay(response, attempt))
                continue
            response.raise_for_status()
            payload = response.json()
            _validate_weather_payload(payload)

            cached = {
                "request": {
                    "latitude": request_params["latitude"],
                    "longitude": request_params["longitude"],
                    "start_date": start_date,
                    "end_date": end_date,
                    "timezone": "UTC",
                    "hourly": list(WEATHER_VARIABLES),
                    "temperature_unit": "celsius",
                    "wind_speed_unit": "kmh",
                },
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "response": payload,
            }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            with gzip.open(temporary_path, "wt", encoding="utf-8") as cache_file:
                json.dump(cached, cache_file, separators=(",", ":"))
            temporary_path.replace(cache_path)
            return payload
    finally:
        if owns_session:
            http.close()
    raise RuntimeError(f"Could not fetch Open-Meteo data for weather cell {cell}")


def nearest_weather_indices(
    weather_times: pd.DatetimeIndex, target_times: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """Find each UTC target's nearest weather hour; ties resolve to the earlier hour."""
    if len(weather_times) == 0:
        raise ValueError("Weather series contains no hourly timestamps")
    weather_ns = weather_times.asi8
    targets = pd.DatetimeIndex(target_times).asi8
    positions = np.searchsorted(weather_ns, targets, side="left")
    right = np.minimum(positions, len(weather_ns) - 1)
    left = np.maximum(positions - 1, 0)
    choose_left = (positions == len(weather_ns)) | (
        (positions > 0)
        & ((targets - weather_ns[left]) <= (weather_ns[right] - targets))
    )
    indices = np.where(choose_left, left, right)
    valid = ~pd.isna(target_times).to_numpy()
    return indices, valid


def _weather_values_at(
    payload: dict[str, Any], acquisition_times: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hourly = payload["hourly"]
    weather_times = pd.DatetimeIndex(pd.to_datetime(hourly["time"], utc=True))
    indices, valid = nearest_weather_indices(weather_times, acquisition_times)
    output = []
    for column in WEATHER_VARIABLES:
        values = np.asarray(hourly[column], dtype=np.float32)
        matched = values[indices].copy()
        matched[~valid] = np.nan
        output.append(matched)
    return output[0], output[1], output[2]


def _scan_weather_cells(
    data_path: Path, max_points: int | None, chunk_size: int
) -> tuple[list[tuple[int, int]], str, str]:
    cell_parts: list[np.ndarray] = []
    min_date: pd.Timestamp | None = None
    max_date: pd.Timestamp | None = None
    for chunk in pd.read_csv(
        data_path,
        usecols=["lat", "lon", "acq_date"],
        dtype={"lat": "float32", "lon": "float32", "acq_date": "string"},
        chunksize=chunk_size,
        nrows=max_points,
    ):
        cell_values = np.column_stack(
            (
                np.rint(chunk["lat"].to_numpy() * 4),
                np.rint(chunk["lon"].to_numpy() * 4),
            )
        ).astype(np.int32)
        cell_parts.append(np.unique(cell_values, axis=0))
        dates = pd.to_datetime(chunk["acq_date"], errors="coerce").dropna()
        if not dates.empty:
            chunk_min, chunk_max = dates.min(), dates.max()
            min_date = chunk_min if min_date is None else min(min_date, chunk_min)
            max_date = chunk_max if max_date is None else max(max_date, chunk_max)
    if not cell_parts or min_date is None or max_date is None:
        raise ValueError(f"No valid FIRMS rows found in {data_path}")
    cells = np.unique(np.concatenate(cell_parts), axis=0)
    return (
        [tuple(map(int, cell)) for cell in cells],
        min_date.strftime("%Y-%m-%d"),
        max_date.strftime("%Y-%m-%d"),
    )


def _count_csv_rows(data_path: Path, chunk_size: int = 500_000) -> int:
    return sum(
        len(chunk)
        for chunk in pd.read_csv(
            data_path, usecols=["lat"], chunksize=chunk_size
        )
    )


def weather_dataset_complete(
    data_path: str | Path = OUTPUT_PATH,
    weather_path: str | Path = WEATHER_OUTPUT_PATH,
) -> bool:
    source = Path(data_path)
    joined = Path(weather_path)
    if not source.exists() or not joined.exists():
        return False
    columns = set(pd.read_csv(joined, nrows=0).columns)
    if not set(WEATHER_COLUMNS).issubset(columns):
        return False
    return _count_csv_rows(source) == _count_csv_rows(joined)


def _partition_firms_rows(
    data_path: Path,
    stage_dir: Path,
    max_points: int | None,
    chunk_size: int,
) -> int:
    stage_dir.mkdir(parents=True, exist_ok=True)
    row_count = 0
    for chunk in pd.read_csv(
        data_path, chunksize=chunk_size, nrows=max_points, low_memory=False
    ):
        cell_lat = np.rint(chunk["lat"].to_numpy(dtype=np.float64) * 4).astype(np.int32)
        cell_lon = np.rint(chunk["lon"].to_numpy(dtype=np.float64) * 4).astype(np.int32)
        chunk["_cell_lat"] = cell_lat
        chunk["_cell_lon"] = cell_lon
        for (lat_index, lon_index), group in chunk.groupby(
            ["_cell_lat", "_cell_lon"], sort=False
        ):
            partition_path = stage_dir / f"{lat_index}_{lon_index}.csv"
            group.drop(columns=["_cell_lat", "_cell_lon"]).to_csv(
                partition_path,
                mode="a",
                header=not partition_path.exists(),
                index=False,
            )
            row_count += len(group)
    return row_count


def prepare_weather_dataset(
    data_path: str | Path = OUTPUT_PATH,
    output_path: str | Path = WEATHER_OUTPUT_PATH,
    cache_dir: str | Path = CACHE_DIR,
    max_points: int | None = None,
    sleep_seconds: float = REQUEST_INTERVAL_SECONDS,
    chunk_size: int = 100_000,
    force: bool = False,
) -> Path:
    """Join FIRMS points with each cell's nearest cached UTC weather hour."""
    source = Path(data_path)
    destination = Path(output_path)
    cache_root = Path(cache_dir)
    if not source.exists():
        raise FileNotFoundError(f"Cleaned FIRMS dataset not found: {source}")
    if max_points is not None and max_points <= 0:
        raise ValueError("max_points must be greater than zero")
    if destination.exists() and not force:
        existing_columns = set(pd.read_csv(destination, nrows=0).columns)
        if not set(WEATHER_COLUMNS).issubset(existing_columns):
            raise ValueError(
                f"{destination} exists without the required weather columns; use force=True to rebuild"
            )
        source_rows = _count_csv_rows(source)
        expected_rows = min(max_points, source_rows) if max_points else source_rows
        actual_rows = _count_csv_rows(destination)
        if actual_rows == expected_rows:
            return destination
        raise ValueError(
            f"Existing weather join has {actual_rows:,} rows; expected {expected_rows:,}. "
            "Use force=True to rebuild it."
        )

    cells, start_date, end_date = _scan_weather_cells(source, max_points, chunk_size)
    print(
        f"Weather join: {len(cells):,} cells, {start_date} through {end_date}, "
        f"{'all rows' if max_points is None else f'first {max_points:,} rows'}"
    )
    stage_dir = cache_root / f"join_stage_{start_date}_{end_date}_{max_points or 'all'}"
    stage_manifest = stage_dir / "complete.json"
    if stage_manifest.exists():
        manifest = json.loads(stage_manifest.read_text(encoding="utf-8"))
        expected_manifest = {
            "source": str(source.resolve()),
            "source_size": source.stat().st_size,
            "source_mtime_ns": source.stat().st_mtime_ns,
            "rows": max_points,
            "start_date": start_date,
            "end_date": end_date,
        }
        if any(manifest.get(key) != value for key, value in expected_manifest.items()):
            raise ValueError(f"FIRMS partition manifest does not match this input: {stage_dir}")
        row_count = int(manifest["row_count"])
    else:
        if stage_dir.exists() and any(stage_dir.iterdir()):
            raise ValueError(
                f"Incomplete FIRMS partition staging data exists at {stage_dir}; "
                "move it aside before restarting the weather join"
            )
        row_count = _partition_firms_rows(source, stage_dir, max_points, chunk_size)
        manifest = {
            "source": str(source.resolve()),
            "source_size": source.stat().st_size,
            "source_mtime_ns": source.stat().st_mtime_ns,
            "rows": max_points,
            "start_date": start_date,
            "end_date": end_date,
            "row_count": row_count,
        }
        stage_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    session_local = threading.local()
    worker_sessions: list[Session] = []
    session_lock = threading.Lock()
    request_gate_lock = threading.Lock()
    last_request_started = 0.0

    def wait_for_request_slot() -> None:
        nonlocal last_request_started
        with request_gate_lock:
            interval = max(sleep_seconds, API_MIN_REQUEST_INTERVAL_SECONDS)
            delay = interval - (time.monotonic() - last_request_started)
            if delay > 0:
                time.sleep(delay)
            last_request_started = time.monotonic()

    def initialize_worker_session() -> None:
        worker_session = requests.Session()
        session_local.session = worker_session
        with session_lock:
            worker_sessions.append(worker_session)

    def cache_cell(cell: tuple[int, int]) -> None:
        fetch_weather_cell(
            cell,
            start_date,
            end_date,
            cache_dir=cache_root,
            session=session_local.session,
            request_gate=wait_for_request_slot,
        )

    try:
        with ThreadPoolExecutor(
            max_workers=4, initializer=initialize_worker_session
        ) as executor:
            futures = {executor.submit(cache_cell, cell): cell for cell in cells}
            for index, future in enumerate(as_completed(futures), start=1):
                future.result()
                if index % 100 == 0 or index == len(cells):
                    print(f"Cached weather for {index:,}/{len(cells):,} cells")
    finally:
        for worker_session in worker_sessions:
            worker_session.close()

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = destination.with_suffix(destination.suffix + ".tmp")
    if temporary_output.exists():
        temporary_output.unlink()
    written_rows = 0
    with requests.Session() as session:
        for index, cell in enumerate(cells, start=1):
            partition_path = stage_dir / f"{cell[0]}_{cell[1]}.csv"
            if not partition_path.exists():
                raise FileNotFoundError(
                    f"Missing FIRMS partition for weather cell {cell}: {partition_path}"
                )
            payload = fetch_weather_cell(
                cell,
                start_date,
                end_date,
                cache_dir=cache_root,
                session=session,
                sleep_seconds=0,
            )
            partition = pd.read_csv(partition_path, low_memory=False)
            acquisition_time = (
                partition["acq_date"].astype("string").str.strip()
                + " "
                + partition["acq_time"]
                .astype("string")
                .str.strip()
                .str.replace(r"\.0$", "", regex=True)
                .str.zfill(4)
            )
            acquisition_time = pd.to_datetime(
                acquisition_time, format="%Y-%m-%d %H%M", utc=True, errors="coerce"
            )
            temp, humidity, wind = _weather_values_at(payload, acquisition_time)
            partition["temp_c"] = temp
            partition["rh_percent"] = humidity
            partition["wind_kmh"] = wind
            partition.to_csv(
                temporary_output,
                mode="a",
                header=not temporary_output.exists(),
                index=False,
            )
            written_rows += len(partition)
            if index % 100 == 0 or index == len(cells):
                print(
                    f"Joined weather for {index:,}/{len(cells):,} cells "
                    f"({written_rows:,} rows)"
                )

    if written_rows != row_count:
        raise ValueError(
            f"Weather join wrote {written_rows:,} rows; expected {row_count:,}"
        )
    temporary_output.replace(destination)
    for cell in cells:
        (stage_dir / f"{cell[0]}_{cell[1]}.csv").unlink()
    stage_manifest.unlink()
    stage_dir.rmdir()
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare FIRMS data and optional weather join.")
    parser.add_argument("--weather", action="store_true", help="Build the Open-Meteo weather join")
    parser.add_argument("--max-points", type=int, help="Limit weather join to the first N FIRMS rows")
    parser.add_argument("--output", type=Path, default=WEATHER_OUTPUT_PATH, help="Weather CSV output path")
    parser.add_argument("--force-weather", action="store_true", help="Rebuild an existing weather CSV")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=REQUEST_INTERVAL_SECONDS,
        help="Extra delay between requests; the join has a 4-second global minimum (default: 0.25)",
    )
    args = parser.parse_args()
    if not OUTPUT_PATH.exists():
        prepare_firms_dataset()
        print(f"Saved cleaned FIRMS dataset to {OUTPUT_PATH}")
    if args.weather:
        result = prepare_weather_dataset(
            output_path=args.output,
            max_points=args.max_points,
            sleep_seconds=args.sleep_seconds,
            force=args.force_weather,
        )
        print(f"Saved weather-joined FIRMS dataset to {result}")
