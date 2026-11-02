import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_occurrence_model import cell_days, climatology, split_weeks  # noqa: E402

START, END = "2024-01-01", "2024-01-20"
HOURS = 24 * 20


def weather():
    return {
        "temperature_2m": np.arange(HOURS, dtype=np.float32),
        "relative_humidity_2m": np.full(HOURS, 40.0, dtype=np.float32),
        "wind_speed_10m": np.full(HOURS, 10.0, dtype=np.float32),
        "precipitation": np.zeros(HOURS, dtype=np.float32),
    }


def days(fires):
    frame = pd.DataFrame(fires, columns=["acq_date", "frp"])
    return cell_days(frame, weather(), 21.5, 86.5, START, END).set_index("date")


def test_label_comes_from_the_next_day_and_large_needs_15_mw():
    out = days([("2024-01-11", 4.0), ("2024-01-12", 15.0)])
    assert out.loc["2024-01-10", "fire_tomorrow"] == 1
    assert out.loc["2024-01-10", "large_fire_tomorrow"] == 0
    assert out.loc["2024-01-11", "large_fire_tomorrow"] == 1
    assert out.loc["2024-01-12", "fire_tomorrow"] == 0


def test_history_excludes_today_and_counts_the_previous_seven_days():
    out = days([("2024-01-10", 4.0), ("2024-01-10", 5.0)])
    assert out.loc["2024-01-10", "fires_last_7d"] == 0
    assert out.loc["2024-01-11", "fires_last_7d"] == 2
    assert out.loc["2024-01-17", "fires_last_7d"] == 2
    assert out.loc["2024-01-18", "fires_last_7d"] == 0


def test_weather_is_read_at_the_issue_hour_of_that_day():
    assert days([]).loc["2024-01-10", "temp_c"] == 9 * 24 + 8


def test_rows_need_a_full_week_of_history_and_a_next_day():
    out = days([])
    assert (out.index.min(), out.index.max()) == (pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-19"))


def test_split_is_about_80_20_and_keeps_every_week_on_one_side():
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    fit, validation = split_weeks(pd.DataFrame({"date": dates}))
    assert 0.18 < len(validation) / len(dates) < 0.22
    assert not set((fit["date"] - dates[0]).dt.days // 7) & set((validation["date"] - dates[0]).dt.days // 7)


def test_climatology_uses_the_cell_month_rate_and_falls_back_to_the_overall_rate():
    train = pd.DataFrame({"lat": [1.0, 1.0, 2.0], "lon": [1.0, 1.0, 2.0], "month": [4, 4, 4], "y": [1, 0, 1]})
    other = pd.DataFrame({"lat": [1.0, 9.0], "lon": [1.0, 9.0], "month": [4, 4]})
    assert climatology(train, "y", other).tolist() == [0.5, 2 / 3]
