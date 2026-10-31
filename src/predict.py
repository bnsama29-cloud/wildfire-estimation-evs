"""Turn slider or sensor inputs into a predicted FRP and a risk band."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from features import weather_features


@dataclass(frozen=True)
class Conditions:
    temp_c: float
    rh_percent: float
    wind_kmh: float
    rain_7d_mm: float
    month: int
    is_day: bool
    lat: float
    lon: float


def choose_conditions(sliders: Conditions, reading) -> tuple[Conditions, bool]:
    """Live sensor temperature and humidity replace the sliders. Returns the inputs and whether they are live."""
    if reading is None:
        return sliders, False
    live = Conditions(reading.temp_c, reading.rh_percent, sliders.wind_kmh, sliders.rain_7d_mm,
                      sliders.month, sliders.is_day, sliders.lat, sliders.lon)
    return live, True


def band_for(frp_mw: float, thresholds: tuple[float, float]) -> str:
    """Bands compare against predicted values on the training year, so LOW/MEDIUM/HIGH are relative risk."""
    if frp_mw < thresholds[0]:
        return "LOW"
    if frp_mw < thresholds[1]:
        return "MEDIUM"
    return "HIGH"


def predict_frp(bundle: dict, conditions: Conditions) -> tuple[float, str]:
    row = pd.DataFrame([{**asdict(conditions), "is_day": int(conditions.is_day)}])
    frp_mw = float(np.expm1(bundle["model"].predict(weather_features(row))[0]))
    return frp_mw, band_for(frp_mw, tuple(bundle["band_thresholds_mw"]))
