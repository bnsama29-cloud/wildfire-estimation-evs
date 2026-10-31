"""Streamlit dashboard. Run with: streamlit run src/app.py

Runs with no sensor attached and no internet once the model and CSV are in the repo.
The same panel reads a real ESP32 (COM port) or the Wokwi simulator (RFC2217 URL).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

from features import risk_band
from model_io import load_model
from predict import Conditions, choose_conditions, predict_frp
from serial_source import SMOKE_ALERT_RAW, WOKWI_URL, SerialSource, available_ports
from weather_join import SAMPLE_JOIN_PATH

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "wildfire_weather_model.joblib"
SITES = {
    "Similipal, Odisha": (21.85, 86.35),
    "Bandipur, Karnataka": (11.66, 76.63),
    "Corbett, Uttarakhand": (29.53, 78.77),
    "Gadchiroli, Maharashtra": (19.80, 80.00),
    "Mamit, Mizoram": (23.93, 92.48),
}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
BAND_COLOURS = {"LOW": [245, 197, 66, 150], "MEDIUM": [240, 130, 40, 170], "HIGH": [214, 40, 40, 200]}
SOURCE_NONE, SOURCE_SERIAL, SOURCE_WOKWI = "None (sliders only)", "ESP32 on USB", "Wokwi simulator"


@st.cache_resource
def load_bundle():
    return load_model(MODEL_PATH) if MODEL_PATH.exists() else None


@st.cache_data
def load_fires() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE_JOIN_PATH, dtype={"acq_date": "string", "acq_time": "string"})
    df["month"] = df["acq_date"].str[5:7].astype(int)
    df["band"] = df["frp"].map(risk_band)
    return df


@st.cache_resource
def get_source(port: str) -> SerialSource:
    return SerialSource(port)


def fire_map(df: pd.DataFrame, basemap: bool) -> pdk.Deck:
    points = df[["lat", "lon", "frp", "band", "acq_date"]].copy()
    points["colour"] = points["band"].map(BAND_COLOURS)
    layer = pdk.Layer(
        "ScatterplotLayer", points, get_position="[lon, lat]", get_fill_color="colour",
        get_radius=9000, radius_min_pixels=1, radius_max_pixels=6, pickable=True,
    )
    view = pdk.ViewState(latitude=22.5, longitude=82.5, zoom=3.5)
    tooltip = {"text": "{acq_date}\nFRP {frp} MW ({band})"}
    return pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip, map_style="light" if basemap else None)


def sidebar_controls() -> tuple[str | None, bool]:
    st.sidebar.header("Live sensor")
    kind = st.sidebar.radio("Source", [SOURCE_NONE, SOURCE_SERIAL, SOURCE_WOKWI])
    port = None
    if kind == SOURCE_SERIAL:
        ports = available_ports()
        port = st.sidebar.selectbox("Port", ports) if ports else None
        if not ports:
            st.sidebar.warning("No serial ports found. Check the USB cable.")
    elif kind == SOURCE_WOKWI:
        port = WOKWI_URL
        st.sidebar.caption("Start the simulation in VS Code and keep its tab visible.")
    st.sidebar.header("Map")
    return port, st.sidebar.checkbox("Show base map (needs internet)", value=False)


def input_panel(fires: pd.DataFrame) -> tuple[Conditions, float | None]:
    """Returns the slider conditions, plus the observed FRP when a historical point is selected."""
    st.subheader("Input conditions")
    mode = st.radio("Inputs", ["Set conditions", "Historical fire point"], horizontal=True, label_visibility="collapsed")
    if mode == "Historical fire point":
        rows = fires.head(200).reset_index(drop=True)
        labels = [f"{r.acq_date}  ({r.lat:.2f}, {r.lon:.2f})  {r.frp:.1f} MW" for r in rows.itertuples()]
        r = rows.iloc[st.selectbox("Fire point", range(len(rows)), format_func=lambda i: labels[i])]
        st.caption(f"{r.temp_c:.1f} °C, {r.rh_percent:.0f} % RH, {r.wind_kmh:.0f} km/h, {r.rain_7d_mm:.0f} mm rain in 7 days")
        return Conditions(r.temp_c, r.rh_percent, r.wind_kmh, r.rain_7d_mm, r.month,
                          r.daynight == "D", r.lat, r.lon), float(r.frp)

    temp = st.slider("Temperature (°C)", 0, 50, 36)
    rh = st.slider("Relative humidity (%)", 5, 100, 25)
    wind = st.slider("Wind speed (km/h)", 0, 60, 15)
    rain = st.slider("Rain in last 7 days (mm)", 0, 100, 0)
    site = st.selectbox("Location", list(SITES))
    month = st.select_slider("Month", MONTHS, value="Apr")
    is_day = st.toggle("Daytime", value=True)
    lat, lon = SITES[site]
    return Conditions(temp, rh, wind, rain, MONTHS.index(month) + 1, is_day, lat, lon), None


def show_sensor(reading, port: str | None) -> None:
    st.subheader("Live sensor (ESP32)")
    if port is None:
        st.caption("Not connected. Using the sliders.")
    elif reading is None:
        st.warning("Waiting for data. Using the sliders.")
    else:
        temp, rh, smoke = st.columns(3)
        temp.metric("Temp", f"{reading.temp_c:.1f} °C")
        rh.metric("Humidity", f"{reading.rh_percent:.0f} %")
        smoke.metric("Smoke (raw)", reading.smoke_raw, help=f"Relative ADC value, not ppm. Alert at {SMOKE_ALERT_RAW}.")
        if reading.smoke_alert:
            st.error("Smoke alert: reading is above the threshold.")
        else:
            st.success("Connected. No smoke alert.")


def prediction_panel(bundle: dict, sliders: Conditions, port: str | None, observed: float | None) -> None:
    reading = get_source(port).latest() if port else None
    conditions, live = choose_conditions(sliders, reading)
    st.subheader("Estimated fire intensity")
    if live:
        st.caption("Using live temperature and humidity.")
    if bundle is None:
        st.warning("Model not found. Run src/train_weather_model.py.")
    else:
        frp, band = predict_frp(bundle, conditions)
        st.metric("Predicted FRP if a fire burns here", f"{frp:.1f} MW")
        {"LOW": st.success, "MEDIUM": st.warning, "HIGH": st.error}[band](f"Risk band: {band}")
        if observed is not None:
            st.caption(f"Satellite observed {observed:.1f} MW for this fire.")
    show_sensor(reading, port)


def main() -> None:
    st.set_page_config(page_title="Wildfire Detection and Analysis", layout="wide")
    st.title("Early Wildfire Detection and Analysis using ML")
    st.caption("Educational prototype. Indian satellite fire points + weather. Not an official forecast.")

    port, basemap = sidebar_controls()
    bundle = load_bundle()
    fires = load_fires()

    left, centre, right = st.columns([1, 2, 1.1])
    with left:
        sliders, observed = input_panel(fires)
    with centre:
        st.subheader("Satellite fire detections (colour = FRP)")
        month = st.selectbox("Month shown", ["All"] + MONTHS)
        shown = fires if month == "All" else fires[fires["month"] == MONTHS.index(month) + 1]
        st.pydeck_chart(fire_map(shown, basemap), height=430)
        st.caption("Yellow below 5 MW, orange 5-15 MW, red 15 MW and above. FRP is an intensity proxy, not burned area.")
    with right:
        st.fragment(prediction_panel, run_every="2s" if port else None)(bundle, sliders, port, observed)

    table, results = st.columns([3, 2])
    with table:
        st.subheader("Sample fire points with weather")
        columns = ["acq_date", "lat", "lon", "frp", "band", "temp_c", "rh_percent", "wind_kmh", "rain_7d_mm"]
        st.dataframe(shown[columns].head(200), hide_index=True, height=260)
    with results:
        if bundle is not None:
            metrics = bundle["metrics"]
            st.subheader("Model check on unseen 2025 fires")
            model, baseline = metrics["test"], metrics["test_baseline_median"]
            error, r2 = st.columns(2)
            error.metric("Typical error", f"{model['mae_mw']:.2f} MW",
                         delta=f"{model['mae_mw'] - baseline['mae_mw']:.2f} vs median guess", delta_color="inverse")
            r2.metric("R² on log FRP", f"{model['r2_log1p_frp']:.2f}")
            importance = pd.Series(metrics["feature_importance"], name="importance")
            st.bar_chart(importance, horizontal=True, height=190)


main()
