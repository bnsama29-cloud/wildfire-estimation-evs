"""Runs the real dashboard script headlessly."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "app.py"
READY = (ROOT / "models" / "wildfire_weather_model.joblib").exists() and (ROOT / "data" / "india_fires_weather.csv").exists()


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    return AppTest.from_file(str(APP), default_timeout=60).run()


def metric(app, label):
    return next(m.value for m in app.metric if m.label == label)


@pytest.mark.skipif(not READY, reason="model and data not built")
def test_app_runs_without_hardware(app):
    assert not app.exception
    assert app.title[0].value == "Early Wildfire Detection and Analysis using ML"
    assert metric(app, "Predicted FRP").endswith("MW")
    assert metric(app, "Any fire").endswith("%")
    assert metric(app, "Large fire").endswith("%")
    assert any(text.value.startswith("Risk band:") for text in [*app.success, *app.warning, *app.error])


@pytest.mark.skipif(not READY, reason="model and data not built")
def test_sliders_change_the_prediction_inputs_without_error(app):
    app.slider[0].set_value(48)
    app.slider[1].set_value(8)
    before = metric(app, "Fire weather index")
    app.run()
    assert not app.exception
    assert int(metric(app, "Fire weather index")) > int(before)


@pytest.mark.skipif(not READY, reason="model and data not built")
def test_stopped_simulator_falls_back_to_sliders(app):
    app.radio[0].set_value("Wokwi").run()
    assert not app.exception
    assert metric(app, "Predicted FRP").endswith("MW")
    assert any("Waiting for data" in w.value for w in app.warning)


@pytest.mark.skipif(not READY, reason="model and data not built")
def test_live_source_locks_the_temperature_and_humidity_sliders(app):
    assert not app.slider[0].disabled
    app.radio[0].set_value("Wokwi").run()
    assert app.slider[0].disabled and app.slider[1].disabled
    assert not app.slider[2].disabled


@pytest.mark.skipif(not READY, reason="model and data not built")
def test_historical_point_shows_observed_value(app):
    app.radio[1].set_value("Historical fire").run()
    assert not app.exception
    assert any("Satellite observed" in c.value for c in app.caption)
