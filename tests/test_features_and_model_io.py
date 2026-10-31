import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import (  # noqa: E402
    FEATURE_COLUMNS,
    build_prediction_features,
    risk_band,
)
from model_io import load_model, save_model  # noqa: E402


def test_risk_band_thresholds():
    assert risk_band(4.99) == "LOW"
    assert risk_band(5.0) == "MEDIUM"
    assert risk_band(14.99) == "MEDIUM"
    assert risk_band(15.0) == "HIGH"


def test_prediction_features_match_training_order():
    fire = {
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

    features = build_prediction_features(fire, 26.0, 40.0, 8.0)

    assert list(features.columns) == FEATURE_COLUMNS
    assert features.loc[0, "time_sin"] > 0
    assert features.loc[0, "satellite_N20"] == 1
    assert features.loc[0, "temp_c"] == 26.0
    assert np.isfinite(features.to_numpy()).all()


def test_joblib_model_round_trip(tmp_path):
    features = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    target = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    model = DummyRegressor(strategy="mean").fit(features, target)
    model_path = tmp_path / "model.joblib"

    save_model(model, model_path)
    loaded = load_model(model_path)

    assert np.array_equal(loaded.predict(features), model.predict(features))
