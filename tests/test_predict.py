import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import WEATHER_FEATURES, vapour_pressure_deficit_kpa  # noqa: E402
from predict import Conditions, band_for, choose_conditions, predict_frp  # noqa: E402
from serial_source import SensorReading  # noqa: E402

SLIDERS = Conditions(30.0, 50.0, 12.0, 3.0, 4, True, 21.8, 86.3)


def test_band_boundaries_are_inclusive_on_the_upper_band():
    thresholds = (5.0, 9.0)
    assert band_for(4.999, thresholds) == "LOW"
    assert band_for(5.0, thresholds) == "MEDIUM"
    assert band_for(8.999, thresholds) == "MEDIUM"
    assert band_for(9.0, thresholds) == "HIGH"


def test_no_reading_keeps_the_sliders():
    assert choose_conditions(SLIDERS, None) == (SLIDERS, False)


def test_live_reading_replaces_only_temperature_and_humidity():
    conditions, live = choose_conditions(SLIDERS, SensorReading(36.4, 24.0, 900))
    assert live
    assert (conditions.temp_c, conditions.rh_percent) == (36.4, 24.0)
    assert (conditions.wind_kmh, conditions.rain_7d_mm, conditions.lat) == (12.0, 3.0, 21.8)


def test_vpd_is_zero_at_saturation_and_about_4_2_kpa_when_hot_and_half_dry():
    assert vapour_pressure_deficit_kpa(30.0, 100.0) == pytest.approx(0.0)
    assert vapour_pressure_deficit_kpa(30.0, 50.0) == pytest.approx(2.12, abs=0.01)


class FixedModel:
    """Returns log1p(10 MW) and records the columns it was given."""

    def predict(self, frame):
        self.columns = list(frame.columns)
        self.row = frame.iloc[0]
        return np.array([np.log1p(10.0)])


def test_predict_frp_undoes_the_log_and_sends_columns_in_training_order():
    model = FixedModel()
    frp, band = predict_frp({"model": model, "band_thresholds_mw": [5.0, 9.0]}, SLIDERS)
    assert frp == pytest.approx(10.0)
    assert band == "HIGH"
    assert model.columns == WEATHER_FEATURES
    assert model.row["is_day"] == 1.0
