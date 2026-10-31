import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app import _fire_map, _prediction_row, load_dashboard_data  # noqa: E402
from features import FEATURE_COLUMNS, NO_WEATHER_FEATURE_COLUMNS  # noqa: E402


def _fire_row():
    return pd.Series(
        {
            "lat": 21.0,
            "lon": 79.0,
            "brightness": 330.0,
            "scan": 0.5,
            "track": 0.6,
            "bright_t31": 290.0,
            "acq_date": "2023-01-01",
            "acq_time": "0030",
            "confidence": "n",
            "satellite": "N20",
            "daynight": "D",
        }
    )


def test_dashboard_prediction_row_supports_weather_and_fallback_models():
    fire = _fire_row()

    weather_row = _prediction_row(fire, 26.0, 40.0, 8.0, FEATURE_COLUMNS)
    fallback_row = _prediction_row(
        fire, 26.0, 40.0, 8.0, NO_WEATHER_FEATURE_COLUMNS
    )

    assert list(weather_row.columns) == FEATURE_COLUMNS
    assert weather_row.loc[0, "temp_c"] == 26.0
    assert list(fallback_row.columns) == NO_WEATHER_FEATURE_COLUMNS
    assert fallback_row.loc[0, "month"] == 1


def test_dashboard_map_has_threshold_colored_fire_bands():
    data = pd.DataFrame(
        [
            {"lat": 21.0, "lon": 79.0, "acq_date": "2023-01-01", "frp": 2.0, "confidence": "n", "risk_band": "LOW"},
            {"lat": 22.0, "lon": 80.0, "acq_date": "2023-01-02", "frp": 10.0, "confidence": "h", "risk_band": "MEDIUM"},
            {"lat": 23.0, "lon": 81.0, "acq_date": "2023-01-03", "frp": 20.0, "confidence": "l", "risk_band": "HIGH"},
        ]
    )

    figure = _fire_map(data)

    assert [trace.name for trace in figure.data] == ["LOW", "MEDIUM", "HIGH"]


def test_dashboard_loader_samples_local_weather_csv(tmp_path):
    source = tmp_path / "joined.csv"
    pd.DataFrame(
        [
            {
                **_fire_row().to_dict(),
                "frp": float(index + 1),
                "temp_c": 25.0,
                "rh_percent": 45.0,
                "wind_kmh": 6.0,
            }
            for index in range(5)
        ]
    ).to_csv(source, index=False)

    sample = load_dashboard_data(str(source), sample_size=3)

    assert len(sample) == 3
    assert set(sample["risk_band"]).issubset({"LOW", "MEDIUM"})
    assert sample[["temp_c", "rh_percent", "wind_kmh"]].notna().all().all()
