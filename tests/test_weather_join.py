import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import weather_join  # noqa: E402
from weather_join import attach_weather, cell_center, rain_last_week, rank_cells  # noqa: E402


def test_cell_center_uses_one_degree_cells():
    assert cell_center([21.97, 78.01, -0.2]).tolist() == [21.5, 78.5, -0.5]


def test_rank_cells_orders_busiest_first():
    df = pd.DataFrame({"cell_lat": [1.5, 2.5, 2.5], "cell_lon": [3.5, 4.5, 4.5]})
    ranked = rank_cells(df)
    assert ranked["cell_lat"].tolist() == [2.5, 1.5]
    assert ranked["fires"].tolist() == [2, 1]


def test_rain_last_week_is_a_trailing_sum_and_treats_missing_as_dry():
    rain = np.array([1.0, np.nan, 2.0, 4.0], dtype=np.float32)
    assert rain_last_week(rain, hours=2).tolist() == [1.0, 1.0, 2.0, 6.0]


def test_attach_weather_picks_nearest_hour():
    weather = {
        "temperature_2m": np.array([10.0, 20.0, 30.0], dtype=np.float32),
        "relative_humidity_2m": np.array([50.0, 40.0, 30.0], dtype=np.float32),
        "wind_speed_10m": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "precipitation": np.array([0.0, 0.0, 5.0], dtype=np.float32),
    }
    group = pd.DataFrame({"time_utc": pd.to_datetime(["2024-01-01 00:20", "2024-01-01 01:40"])})
    out = attach_weather(group, weather, start="2024-01-01")
    assert out["temp_c"].tolist() == [10.0, 30.0]
    assert out["rh_percent"].tolist() == [50.0, 30.0]
    assert out["rain_7d_mm"].tolist() == [0.0, 5.0]


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code, self._payload = status_code, payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        assert self.status_code == 200


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0

    def get(self, *args, **kwargs):
        self.calls += 1
        return self.responses.pop(0)


def test_fetch_cell_retries_after_rate_limit_then_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(weather_join, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(weather_join.time, "sleep", lambda _: None)
    hourly = {name: [1.0, None] for name in weather_join.HOURLY_VARIABLES}
    session = FakeSession([FakeResponse(429), FakeResponse(200, {"hourly": hourly})])

    path = weather_join.fetch_cell(21.5, 78.5, session)
    assert session.calls == 2
    with np.load(path) as cached:
        assert cached["temperature_2m"][0] == 1.0
        assert np.isnan(cached["temperature_2m"][1])

    weather_join.fetch_cell(21.5, 78.5, session)  # cached, so no further request
    assert session.calls == 2


def test_fetch_cell_gives_up_after_repeated_rate_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(weather_join, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(weather_join.time, "sleep", lambda _: None)
    session = FakeSession([FakeResponse(429)] * weather_join.MAX_ATTEMPTS)
    with pytest.raises(RuntimeError):
        weather_join.fetch_cell(21.5, 78.5, session)
