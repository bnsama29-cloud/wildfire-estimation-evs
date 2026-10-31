import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_model import FEATURE_COLUMNS, load_training_data  # noqa: E402


def test_load_training_data_builds_cyclic_time_and_sensor_features(tmp_path):
    rows = [
        {
            "lat": 21.0,
            "lon": 79.0,
            "brightness": 330.0,
            "scan": 0.5,
            "track": 0.6,
            "acq_date": "2023-01-01",
            "acq_time": "0000",
            "confidence": "n",
            "satellite": "N20",
            "bright_t31": 290.0,
            "daynight": "D",
            "frp": 1.0,
        },
        {
            "lat": 22.0,
            "lon": 80.0,
            "brightness": 340.0,
            "scan": 0.7,
            "track": 0.8,
            "acq_date": "2023-07-01",
            "acq_time": "1230",
            "confidence": "h",
            "satellite": "SNPP",
            "bright_t31": 295.0,
            "daynight": "N",
            "frp": 3.0,
        },
        {
            "lat": 23.0,
            "lon": 81.0,
            "brightness": 350.0,
            "scan": 0.7,
            "track": 0.8,
            "acq_date": "2023-07-01",
            "acq_time": "2460",
            "confidence": "h",
            "satellite": "SNPP",
            "bright_t31": 295.0,
            "daynight": "N",
            "frp": 3.0,
        },
    ]
    csv_path = tmp_path / "fires.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    X, y = load_training_data(csv_path)

    assert X.shape == (2, len(FEATURE_COLUMNS))
    assert np.all(np.isfinite(X))
    assert np.allclose(y, np.log1p([1.0, 3.0]))
    assert X[0, FEATURE_COLUMNS.index("month_sin")] == 0.0
    assert X[0, FEATURE_COLUMNS.index("month_cos")] == 1.0
    assert X[0, FEATURE_COLUMNS.index("time_sin")] == 0.0
    assert X[0, FEATURE_COLUMNS.index("satellite_N20")] == 1.0
    assert X[1, FEATURE_COLUMNS.index("satellite_SNPP")] == 1.0
