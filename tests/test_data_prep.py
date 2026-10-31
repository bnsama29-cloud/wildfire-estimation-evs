import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import data_prep  # noqa: E402


def row(lat, frp, **extra):
    base = {
        "latitude": lat, "longitude": 79.0, "brightness": 330.0, "scan": 0.5, "track": 0.5,
        "acq_date": "2024-03-01", "acq_time": 730, "satellite": "N20", "instrument": "VIIRS",
        "confidence": "n", "version": 2, "bright_t31": 290.0, "frp": frp, "daynight": "D",
    }
    return base | extra


def test_only_vegetation_fires_inside_india_are_kept(tmp_path, monkeypatch):
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    monkeypatch.setattr(data_prep, "RAW_DIR", tmp_path)
    monkeypatch.setattr(data_prep, "EXTRACTED_DIR", extracted)
    pd.DataFrame([
        row(21.0, 5.0, type=0),   # kept
        row(21.1, 5.0, type=2),   # static land source, dropped
        row(21.2, 0.0, type=0),   # no FRP, dropped
        row(50.0, 5.0, type=0),   # outside India, dropped
    ]).to_csv(extracted / "fire_archive_x.csv", index=False)
    pd.DataFrame([row(21.3, 5.0)]).to_csv(extracted / "fire_nrt_x.csv", index=False)  # no type column, dropped

    out = data_prep.prepare_firms_dataset(tmp_path / "out.csv")

    assert out["lat"].tolist() == [21.0]
    assert out["acq_time"].tolist() == ["0730"]
