from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


CONFIDENCE_ORDER = {"l": 0, "n": 1, "h": 2}
SATELLITE_FEATURE_COLUMNS = [
    "lat",
    "lon",
    "brightness",
    "scan",
    "track",
    "bright_t31",
    "month_sin",
    "month_cos",
    "time_sin",
    "time_cos",
    "confidence_score",
    "daynight",
    "satellite_N20",
    "satellite_SNPP",
]
WEATHER_FEATURE_COLUMNS = ["temp_c", "rh_percent", "wind_kmh"]
FEATURE_COLUMNS = SATELLITE_FEATURE_COLUMNS + WEATHER_FEATURE_COLUMNS
NO_WEATHER_FEATURE_COLUMNS = ["lat", "lon", "month"]


def confidence_score(value: str | float | int | None) -> int:
    if pd.isna(value):
        return 0
    key = str(value).strip().lower()
    return CONFIDENCE_ORDER.get(key, 0)


def risk_band(frp_mw: float) -> str:
    if frp_mw < 5:
        return "LOW"
    if frp_mw < 15:
        return "MEDIUM"
    return "HIGH"


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature-ready dataset with target and risk labels."""
    out = df.copy()
    out["confidence_score"] = out["confidence"].map(confidence_score)
    out["month"] = pd.to_datetime(out["acq_date"]).dt.month
    out["frp_numeric"] = pd.to_numeric(out["frp"], errors="coerce")
    out["target_log1p_frp"] = np.log1p(out["frp_numeric"].fillna(0.0))
    out["risk_band"] = out["frp_numeric"].map(risk_band)
    return out


def build_prediction_features(
    fire: Mapping[str, object],
    temp_c: float,
    rh_percent: float,
    wind_kmh: float,
) -> pd.DataFrame:
    """Build the combined satellite/weather feature row used by the trained model."""
    date = pd.to_datetime(fire["acq_date"], errors="coerce")
    if pd.isna(date):
        raise ValueError("Selected fire point has an invalid acquisition date")

    acquisition_time = str(fire["acq_time"]).strip()
    acquisition_time = acquisition_time.removesuffix(".0").zfill(4)
    if len(acquisition_time) != 4 or not acquisition_time.isdigit():
        raise ValueError("Selected fire point has an invalid acquisition time")
    hour, minute = int(acquisition_time[:2]), int(acquisition_time[2:])
    if hour > 23 or minute > 59:
        raise ValueError("Selected fire point has an invalid acquisition time")

    confidence = str(fire["confidence"]).strip().lower()
    satellite = str(fire["satellite"]).strip().upper()
    daynight = str(fire["daynight"]).strip().upper()
    if confidence not in CONFIDENCE_ORDER:
        raise ValueError(f"Unknown VIIRS confidence code: {confidence!r}")
    if satellite not in {"N20", "SNPP"}:
        raise ValueError(f"Unknown VIIRS satellite code: {satellite!r}")
    if daynight not in {"D", "N"}:
        raise ValueError(f"Unknown day/night code: {daynight!r}")

    month_angle = 2.0 * np.pi * (date.month - 1) / 12.0
    time_angle = 2.0 * np.pi * (hour + minute / 60.0) / 24.0
    values = {
        "lat": float(fire["lat"]),
        "lon": float(fire["lon"]),
        "brightness": float(fire["brightness"]),
        "scan": float(fire["scan"]),
        "track": float(fire["track"]),
        "bright_t31": float(fire["bright_t31"]),
        "month_sin": float(np.sin(month_angle)),
        "month_cos": float(np.cos(month_angle)),
        "time_sin": float(np.sin(time_angle)),
        "time_cos": float(np.cos(time_angle)),
        "confidence_score": float(CONFIDENCE_ORDER[confidence]),
        "daynight": float(daynight == "D"),
        "satellite_N20": float(satellite == "N20"),
        "satellite_SNPP": float(satellite == "SNPP"),
        "temp_c": float(temp_c),
        "rh_percent": float(rh_percent),
        "wind_kmh": float(wind_kmh),
    }
    if not np.isfinite(np.asarray(list(values.values()), dtype=np.float64)).all():
        raise ValueError("Model input features must all be finite numbers")
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)
