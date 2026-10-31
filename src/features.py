from __future__ import annotations

import numpy as np
import pandas as pd


CONFIDENCE_ORDER = {"l": 0, "n": 1, "h": 2}
WEATHER_FEATURES = [
    "temp_c",
    "rh_percent",
    "wind_kmh",
    "rain_7d_mm",
    "vpd_kpa",
    "month_sin",
    "month_cos",
    "is_day",
    "lat",
    "lon",
]


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


def vapour_pressure_deficit_kpa(temp_c, rh_percent):
    """Tetens estimate of how strongly the air dries out fuel, computed from temperature and humidity."""
    temp = np.asarray(temp_c, dtype=np.float64)
    saturation = 0.6108 * np.exp(17.27 * temp / (temp + 237.3))
    return saturation * (1.0 - np.asarray(rh_percent, dtype=np.float64) / 100.0)


def weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Model inputs. All of them are known before a fire is observed, so sliders and sensors can supply them."""
    angle = 2.0 * np.pi * (df["month"].to_numpy(dtype=np.float64) - 1.0) / 12.0
    columns = {
        "temp_c": df["temp_c"].to_numpy(dtype=np.float32),
        "rh_percent": df["rh_percent"].to_numpy(dtype=np.float32),
        "wind_kmh": df["wind_kmh"].to_numpy(dtype=np.float32),
        "rain_7d_mm": df["rain_7d_mm"].to_numpy(dtype=np.float32),
        "vpd_kpa": vapour_pressure_deficit_kpa(df["temp_c"], df["rh_percent"]).astype(np.float32),
        "month_sin": np.sin(angle).astype(np.float32),
        "month_cos": np.cos(angle).astype(np.float32),
        "is_day": df["is_day"].to_numpy(dtype=np.float32),
        "lat": df["lat"].to_numpy(dtype=np.float32),
        "lon": df["lon"].to_numpy(dtype=np.float32),
    }
    return pd.DataFrame(columns, index=df.index)[WEATHER_FEATURES]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature-ready dataset with target and risk labels."""
    out = df.copy()
    out["confidence_score"] = out["confidence"].map(confidence_score)
    out["month"] = pd.to_datetime(out["acq_date"]).dt.month
    out["frp_numeric"] = pd.to_numeric(out["frp"], errors="coerce")
    out["target_log1p_frp"] = np.log1p(out["frp_numeric"].fillna(0.0))
    out["risk_band"] = out["frp_numeric"].map(risk_band)
    return out
