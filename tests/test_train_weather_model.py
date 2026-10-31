import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_weather_model import band_lift, load_joined, split_by_year  # noqa: E402


def test_load_joined_derives_month_year_and_day_flag(tmp_path):
    path = tmp_path / "joined.csv"
    pd.DataFrame({
        "lat": [21.0, 22.0], "lon": [79.0, 80.0], "acq_date": ["2024-03-15", "2025-11-02"],
        "acq_time": ["0745", "2010"], "frp": [4.0, 0.0], "confidence": ["n", "h"], "daynight": ["D", "N"],
        "temp_c": [30.0, 20.0], "rh_percent": [40.0, 60.0], "wind_kmh": [10.0, 5.0], "rain_7d_mm": [0.0, 1.0],
    }).to_csv(path, index=False)
    df = load_joined(path)
    assert len(df) == 1  # zero FRP removed
    assert (df["month"].iloc[0], df["year"].iloc[0], df["is_day"].iloc[0]) == (3, 2024, 1)


def test_split_by_year_keeps_test_year_out_of_training():
    df = pd.DataFrame({"year": [2024, 2024, 2025]})
    train, test = split_by_year(df)
    assert (len(train), len(test)) == (2, 1)
    assert set(train["year"]) == {2024}


def test_band_lift_reports_share_of_large_fires_per_band():
    frp = np.array([1.0, 20.0, 2.0, 30.0])
    predicted = np.array([1.0, 1.0, 6.0, 10.0])
    lift = band_lift(frp, predicted, (5.0, 9.0))
    assert lift["LOW"] == {"fires": 2, "share_large": 0.5}
    assert lift["MEDIUM"] == {"fires": 1, "share_large": 0.0}
    assert lift["HIGH"] == {"fires": 1, "share_large": 1.0}
