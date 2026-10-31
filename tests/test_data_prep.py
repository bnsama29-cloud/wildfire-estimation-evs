import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_prep import (  # noqa: E402
    _retry_delay,
    fetch_weather_cell,
    nearest_weather_indices,
    prepare_weather_dataset,
    weather_cell,
    weather_dataset_complete,
)


class FakeResponse:
    status_code = 200
    headers = {}

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        return FakeResponse(self.payload)


def _weather_payload():
    return {
        "hourly": {
            "time": ["2023-01-01T00:00", "2023-01-01T01:00"],
            "temperature_2m": [20.0, 21.0],
            "relative_humidity_2m": [60.0, 55.0],
            "wind_speed_10m": [4.0, 6.0],
        }
    }


def test_weather_cell_rounds_to_nearest_quarter_degree():
    assert weather_cell(21.972, 78.981) == (88, 316)


def test_weather_rate_limit_waits_for_hourly_reset():
    response = type(
        "RateLimitedResponse",
        (),
        {
            "text": '{"error":true,"reason":"Hourly API request limit exceeded"}',
            "headers": {},
        },
    )()

    assert _retry_delay(response, 0) == 3600


def test_nearest_weather_hour_resolves_ties_to_earlier_and_flags_nat():
    weather_times = pd.DatetimeIndex(
        ["2023-01-01T00:00Z", "2023-01-01T01:00Z"]
    )
    targets = pd.Series(
        pd.to_datetime(
            ["2023-01-01T00:30Z", "2023-01-01T00:40Z", None], utc=True
        )
    )

    indices, valid = nearest_weather_indices(weather_times, targets)

    assert indices.tolist() == [0, 1, 0]
    assert valid.tolist() == [True, True, False]


def test_weather_cell_cache_prevents_repeat_api_request(tmp_path):
    session = FakeSession(_weather_payload())

    first = fetch_weather_cell(
        (88, 316),
        "2023-01-01",
        "2023-01-01",
        cache_dir=tmp_path,
        session=session,
        sleep_seconds=0,
    )
    second = fetch_weather_cell(
        (88, 316),
        "2023-01-01",
        "2023-01-01",
        cache_dir=tmp_path,
        session=session,
        sleep_seconds=0,
    )

    assert session.calls == 1
    assert first == second
    cache_path = next(tmp_path.glob("*.json.gz"))
    with gzip.open(cache_path, "rt", encoding="utf-8") as cache_file:
        cache = json.load(cache_file)
    assert cache["request"]["timezone"] == "UTC"
    assert cache["request"]["hourly"] == [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
    ]
    assert "retrieved_at_utc" in cache


def test_weather_join_matches_nearest_hour_and_writes_all_rows(tmp_path, monkeypatch):
    source = tmp_path / "fires.csv"
    destination = tmp_path / "joined.csv"
    pd.DataFrame(
        [
            {"lat": 21.0, "lon": 79.0, "acq_date": "2023-01-01", "acq_time": "0030", "frp": 1.0},
            {"lat": 21.1, "lon": 79.0, "acq_date": "2023-01-01", "acq_time": "0040", "frp": 2.0},
            {"lat": 22.0, "lon": 80.0, "acq_date": "2023-01-01", "acq_time": "0000", "frp": 3.0},
        ]
    ).to_csv(source, index=False)
    payload = _weather_payload()
    monkeypatch.setattr(
        "data_prep.fetch_weather_cell",
        lambda *args, **kwargs: payload,
    )

    result = prepare_weather_dataset(
        data_path=source,
        output_path=destination,
        cache_dir=tmp_path / "cache",
        sleep_seconds=0,
        chunk_size=2,
    )
    joined = pd.read_csv(result)

    assert len(joined) == 3
    first_cell_rows = joined.loc[joined["lat"] < 22]
    assert np.allclose(first_cell_rows["temp_c"].sort_values(), [20.0, 21.0])
    assert np.allclose(first_cell_rows["rh_percent"].sort_values(), [55.0, 60.0])
    assert joined[["temp_c", "rh_percent", "wind_kmh"]].notna().all().all()


def test_incomplete_weather_file_is_not_accepted_as_full_dataset(tmp_path):
    source = tmp_path / "fires.csv"
    joined = tmp_path / "joined.csv"
    pd.DataFrame({"lat": [21.0, 22.0, 23.0]}).to_csv(source, index=False)
    pd.DataFrame(
        {
            "lat": [21.0, 22.0],
            "temp_c": [20.0, 21.0],
            "rh_percent": [60.0, 55.0],
            "wind_kmh": [4.0, 6.0],
        }
    ).to_csv(joined, index=False)

    assert not weather_dataset_complete(source, joined)

    pd.DataFrame(
        {
            "lat": [21.0, 22.0, 23.0],
            "temp_c": [20.0, 21.0, 22.0],
            "rh_percent": [60.0, 55.0, 50.0],
            "wind_kmh": [4.0, 6.0, 8.0],
        }
    ).to_csv(joined, index=False)
    assert weather_dataset_complete(source, joined)
