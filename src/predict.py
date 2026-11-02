"""Turn slider or sensor inputs into a predicted FRP and a risk band."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from features import fire_weather_index, occurrence_features, weather_features


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
    fires_last_7d: float = 0.0


def choose_conditions(sliders: Conditions, reading, historical: bool = False) -> tuple[Conditions, bool]:
    """Live sensor temperature and humidity replace the sliders. Returns the inputs and whether they are live.

    A historical fire keeps its own recorded weather, so a connected sensor does not change it.
    """
    if reading is None or historical:
        return sliders, False
    return replace(sliders, temp_c=reading.temp_c, rh_percent=reading.rh_percent), True


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


def predict_tomorrow(bundle: dict, conditions: Conditions) -> tuple[float, float]:
    """Chance of any fire and of a large fire tomorrow in the 1-degree cell around these conditions."""
    row = occurrence_features(pd.DataFrame([asdict(conditions)]))
    fire = float(bundle["models"]["fire"].predict_proba(row)[0, 1])
    large = float(bundle["models"]["large_fire"].predict_proba(row)[0, 1])
    return fire, min(fire, large)  # two separate models, but a large fire is still a fire


def fire_weather(bundle: dict, conditions: Conditions) -> tuple[float, str]:
    """A published formula, not the ML model, so it reacts directly to temperature, humidity and wind."""
    index = float(fire_weather_index(conditions.temp_c, conditions.rh_percent, conditions.wind_kmh))
    return index, band_for(index, tuple(bundle["fire_weather_thresholds"]))
