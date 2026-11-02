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


def slider(app, label):
    return next(s for s in app.slider if s.label == label)


@pytest.mark.skipif(not READY, reason="model and data not built")
def test_app_runs_without_hardware(app):
    assert not app.exception
    assert app.title[0].value == "Early Wildfire Detection and Analysis using ML"
    assert metric(app, "Predicted FRP").endswith("MW")
    assert metric(app, "Any fire").endswith("%")
    assert metric(app, "Large fire").endswith("%")
    assert any(text.value.startswith("Risk band:") for text in [*app.success, *app.warning, *app.error])


@pytest.mark.skipif(not READY, reason="model and data not built")
def test_collar_sliders_change_the_prediction_inputs_without_error(app):
    before = metric(app, "Fire weather index")
    slider(app, "Temperature sensor (°C)").set_value(48.0)
    slider(app, "Humidity sensor (%)").set_value(8.0)
    app.run()
    assert not app.exception
    assert int(metric(app, "Fire weather index")) > int(before)


@pytest.mark.skipif(not READY, reason="model and data not built")
def test_smoke_slider_raises_the_alert_without_a_board(app):
    assert not any("Smoke alert" in e.value for e in app.error)
    slider(app, "Smoke sensor (raw)").set_value(2500)
    app.run()
    assert not app.exception
    assert any("Smoke alert" in e.value for e in app.error)


@pytest.mark.skipif(not READY, reason="model and data not built")
def test_selecting_another_collar_changes_the_location_and_keeps_working(app):
    app.selectbox[0].set_value("Clouded leopard L3").run()
    assert not app.exception
    assert any("Clouded leopard L3" in s.value for s in app.subheader)


@pytest.mark.skipif(not READY, reason="model and data not built")
def test_stopped_simulator_falls_back_to_sliders(app):
    app.radio[0].set_value("Wokwi").run()
    assert not app.exception
    assert metric(app, "Predicted FRP").endswith("MW")
    assert any("Waiting for data" in w.value for w in app.warning)


@pytest.mark.skipif(not READY, reason="model and data not built")
def test_historical_point_shows_observed_value(app):
    app.radio[1].set_value("Historical fire").run()
    assert not app.exception
    assert any("Satellite observed" in c.value for c in app.caption)
