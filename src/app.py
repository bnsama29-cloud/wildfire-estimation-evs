"""Streamlit dashboard. Run with: streamlit run src/app.py

Runs with no sensor attached and no internet once the models and CSVs are in the repo.
The same panel reads a real ESP32 (COM port) or the Wokwi simulator (RFC2217 URL).
"""
from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

from features import risk_band
from map_outline import OUTLINE_PATH
from model_io import load_model
from predict import Conditions, choose_conditions, fire_weather, predict_frp, predict_tomorrow
from serial_source import SMOKE_ALERT_RAW, WOKWI_URL, SerialSource, available_ports
from weather_join import SAMPLE_JOIN_PATH

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "wildfire_weather_model.joblib"
OCCURRENCE_PATH = MODEL_PATH.with_name("wildfire_occurrence_model.joblib")
THEME_PATH = Path(__file__).with_name("theme.css")
TYPICAL_RECENT_FIRES = 20
SITES = {
    "Similipal, Odisha": (21.85, 86.35),
    "Bandipur, Karnataka": (11.66, 76.63),
    "Corbett, Uttarakhand": (29.53, 78.77),
    "Gadchiroli, Maharashtra": (19.80, 80.00),
    "Mamit, Mizoram": (23.93, 92.48),
}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
FRP_COLOURS = {"LOW": [255, 196, 61, 170], "MEDIUM": [255, 128, 40, 200], "HIGH": [255, 45, 45, 235]}
LAND_COLOUR = [34, 46, 68, 255]
# Streamlit swaps an empty style for an online base map, so this hands it a blank local one.
BLANK_MAP_STYLE = ('data:application/json,{"version":8,"sources":{},"layers":[{"id":"sea","type":"background",'
                   '"paint":{"background-color":"%230b1220"}}]}')
SOURCE_NONE, SOURCE_SERIAL, SOURCE_WOKWI = "Sliders", "ESP32 on USB", "Wokwi"
FEATURE_NAMES = {
    "fires_last_7d": "Recent fires", "month_cos": "Season", "month_sin": "Season (phase)", "lat": "Latitude",
    "lon": "Longitude", "rh_percent": "Humidity", "temp_c": "Temperature", "wind_kmh": "Wind",
    "rain_7d_mm": "Rain, 7 days", "vpd_kpa": "Air dryness", "fwi": "Fire weather index",
}


@st.cache_resource
def load_bundle():
    return load_model(MODEL_PATH) if MODEL_PATH.exists() else None


@st.cache_resource
def load_occurrence():
    return load_model(OCCURRENCE_PATH) if OCCURRENCE_PATH.exists() else None


@st.cache_data
def load_fires() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE_JOIN_PATH, dtype={"acq_date": "string", "acq_time": "string"})
    df["month"] = df["acq_date"].str[5:7].astype(int)
    df["frp_class"] = df["frp"].map(risk_band)
    return df


@st.cache_data
def load_outline() -> pd.DataFrame:
    return pd.read_csv(OUTLINE_PATH)


@st.cache_resource
def get_source(port: str) -> SerialSource:
    return SerialSource(port)


def fire_map(df: pd.DataFrame, outline: pd.DataFrame) -> pdk.Deck:
    """India only: land is drawn from where satellites detected fires, so no map tiles or neighbours appear."""
    points = df[["lat", "lon", "frp", "frp_class", "acq_date"]].copy()
    points["colour"] = points["frp_class"].map(FRP_COLOURS)
    land = pdk.Layer("ScatterplotLayer", outline, get_position="[lon, lat]", get_fill_color=LAND_COLOUR,
                     get_radius=11000, radius_min_pixels=2, pickable=False)
    fires = pdk.Layer("ScatterplotLayer", points, get_position="[lon, lat]", get_fill_color="colour",
                      get_radius=7000, radius_min_pixels=1, radius_max_pixels=5, pickable=True)
    view = pdk.ViewState(latitude=22.0, longitude=82.8, zoom=3.7, min_zoom=3, max_zoom=9)
    tooltip = {"text": "{acq_date}\nFRP {frp} MW"}
    return pdk.Deck(layers=[land, fires], initial_view_state=view, tooltip=tooltip, map_style=BLANK_MAP_STYLE)


def source_control() -> str | None:
    kind = st.radio("Live source", [SOURCE_NONE, SOURCE_SERIAL, SOURCE_WOKWI], horizontal=True)
    if kind == SOURCE_WOKWI:
        return WOKWI_URL
    if kind == SOURCE_SERIAL:
        ports = available_ports()
        if ports:
            return st.selectbox("Port", ports)
        st.warning("No serial ports found. Check the USB cable.")
    return None


def input_panel(fires: pd.DataFrame, live: bool) -> tuple[Conditions, float | None]:
    """Returns the slider conditions, plus the observed FRP when a historical point is selected."""
    st.subheader("Input conditions")
    mode = st.radio("Inputs", ["Set conditions", "Historical fire"], horizontal=True, label_visibility="collapsed")
    if mode == "Historical fire":
        rows = fires.head(200).reset_index(drop=True)
        labels = [f"{r.acq_date}  ({r.lat:.2f}, {r.lon:.2f})  {r.frp:.1f} MW" for r in rows.itertuples()]
        r = rows.iloc[st.selectbox("Fire point", range(len(rows)), format_func=lambda i: labels[i])]
        st.caption(f"{r.temp_c:.1f} °C, {r.rh_percent:.0f} % RH, {r.wind_kmh:.0f} km/h, {r.rain_7d_mm:.0f} mm rain in 7 days")
        return Conditions(r.temp_c, r.rh_percent, r.wind_kmh, r.rain_7d_mm, r.month,
                          r.daynight == "D", r.lat, r.lon, TYPICAL_RECENT_FIRES), float(r.frp)

    temp = st.slider("Temperature (°C)", 0, 50, 36, disabled=live)
    rh = st.slider("Relative humidity (%)", 5, 100, 25, disabled=live)
    if live:
        st.caption("The live sensor sets temperature and humidity.")
    wind = st.slider("Wind speed (km/h)", 0, 60, 15)
    rain = st.slider("Rain in last 7 days (mm)", 0, 100, 0)
    site = st.selectbox("Location", list(SITES))
    month = st.select_slider("Month", MONTHS, value="Apr")
    recent = st.slider("Satellite fires nearby, last 7 days", 0, 300, TYPICAL_RECENT_FIRES)
    is_day = st.toggle("Daytime", value=True)
    lat, lon = SITES[site]
    return Conditions(temp, rh, wind, rain, MONTHS.index(month) + 1, is_day, lat, lon, recent), None


def show_sensor(reading, port: str | None) -> None:
    st.subheader("Live sensor (ESP32)")
    if port is None:
        st.caption("Not connected. Using the sliders.")
    elif reading is None:
        st.warning("Waiting for data. Using the sliders.")
    else:
        temp, rh, smoke = st.columns(3)
        temp.metric("Temp", f"{reading.temp_c:.1f}°C")
        rh.metric("Humidity", f"{reading.rh_percent:.0f}%")
        smoke.metric("Smoke", reading.smoke_raw, help=f"Raw sensor value, not ppm. Alert at {SMOKE_ALERT_RAW}.")
        if reading.smoke_alert:
            st.error("Smoke alert: reading is above the threshold.")
        else:
            st.success("Connected. No smoke alert.")


def prediction_panel(bundle: dict, occurrence: dict | None, sliders: Conditions, port: str | None,
                     observed: float | None) -> None:
    reading = get_source(port).latest() if port else None
    conditions, live = choose_conditions(sliders, reading, historical=observed is not None)
    if live:
        st.caption("Using live temperature and humidity.")
    if occurrence is not None:
        st.subheader("Chance of fire tomorrow")
        fire, large = predict_tomorrow(occurrence, conditions)
        any_fire, big_fire = st.columns(2)
        any_fire.metric("Any fire", f"{fire:.0%}")
        big_fire.metric("Large fire", f"{large:.0%}", help="A detection of 15 MW or more.")
        st.caption("Area of about 100 km. A fire here means a satellite detection.")
    st.subheader("Severity if a fire burns")
    if bundle is None:
        st.warning("Model not found. Run src/train_weather_model.py.")
    else:
        frp, band = predict_frp(bundle, conditions)
        index, weather_band = fire_weather(bundle, conditions)
        severity, weather = st.columns(2)
        severity.metric("Predicted FRP", f"{frp:.1f} MW")
        weather.metric("Fire weather index", f"{index:.0f}", help="Fosberg index, 0-100, from temperature, humidity and wind.")
        {"LOW": st.success, "MEDIUM": st.warning, "HIGH": st.error}[band](f"Risk band: {band}")
        share = bundle["metrics"]["test_band_lift"][band]["share_large"]
        st.caption(f"{share:.0%} of 2025 fires in this band were large. "
                   f"Weather alone rates {weather_band} against 2024 fire days.")
        if observed is not None:
            st.caption(f"Satellite observed {observed:.1f} MW for this fire.")
    show_sensor(reading, port)


def importance_chart(importance: dict) -> alt.Chart:
    top = pd.Series(importance).head(5).rename(FEATURE_NAMES).rename_axis("input").reset_index(name="share")
    return alt.Chart(top, height=120).mark_bar(color="#ff7a3d", cornerRadiusEnd=3).encode(
        x=alt.X("share:Q", title=None, axis=alt.Axis(format="%", grid=False)),
        y=alt.Y("input:N", title=None, sort=None),
    )


def results_panel(fires: pd.DataFrame, bundle: dict | None, occurrence: dict | None) -> None:
    if occurrence is not None:
        st.subheader("Model check: tomorrow's fire")
        large = occurrence["metrics"]["large_fire"]["validation_80_20"]
        fire = occurrence["metrics"]["fire"]["validation_80_20"]
        first, second = st.columns(2)
        first.metric("Any fire ROC-AUC", f"{fire['model']['roc_auc']:.2f}", help="80/20 split by whole weeks. 1.0 is perfect.")
        second.metric("Large fire PR-AUC", f"{large['model']['pr_auc']:.2f}",
                      help=f"The cell and month average scores {large['baseline_cell_month_rate']['pr_auc']:.2f}.")
    if bundle is not None:
        st.subheader("Model check: severity on 2025")
        model, baseline = bundle["metrics"]["test"], bundle["metrics"]["test_baseline_median"]
        error, r2 = st.columns(2)
        error.metric("Typical error", f"{model['mae_mw']:.2f} MW", help=f"A median guess gives {baseline['mae_mw']:.2f} MW.")
        r2.metric("R² on log FRP", f"{model['r2_log1p_frp']:.2f}")
    if occurrence is not None:
        st.caption("What drives the large fire prediction")
        st.altair_chart(importance_chart(occurrence["metrics"]["large_fire"]["feature_importance"]), width="stretch")
    st.subheader("Sample fire points with weather")
    columns = {"acq_date": "Date", "frp": "FRP", "frp_class": "FRP class", "temp_c": "°C", "rh_percent": "RH %",
               "wind_kmh": "Wind", "rain_7d_mm": "Rain"}
    st.dataframe(fires[list(columns)].head(200).rename(columns=columns), hide_index=True, height=230)


def main() -> None:
    st.set_page_config(page_title="Wildfire Detection and Analysis", layout="wide", initial_sidebar_state="collapsed")
    st.html(f"<style>{THEME_PATH.read_text(encoding='utf-8')}</style>")
    with st.container(key="header"):
        st.title("Early Wildfire Detection and Analysis using ML")
        st.caption("Educational prototype. Indian satellite fire points + weather. Not an official forecast.")

    bundle, occurrence = load_bundle(), load_occurrence()
    fires = load_fires()

    left, centre, right = st.columns([1.5, 1.45, 1], gap="small")
    with left, st.container(key="left_panel"):
        port = source_control()
        inputs, results = st.columns([1, 1.05], gap="small")
        with inputs:
            sliders, observed = input_panel(fires, port is not None)
        with results:
            st.fragment(prediction_panel, run_every="1s" if port else None)(bundle, occurrence, sliders, port, observed)
    with centre, st.container(key="map_panel"):
        st.subheader("Satellite fire detections in India")
        month = st.selectbox("Month shown", ["All"] + MONTHS, label_visibility="collapsed")
        shown = fires if month == "All" else fires[fires["month"] == MONTHS.index(month) + 1]
        st.pydeck_chart(fire_map(shown, load_outline()), height="stretch")
        st.caption("FRP class: yellow below 5 MW, orange 5-15 MW, red 15 MW and above. FRP is an intensity proxy, not burned area.")
    with right, st.container(key="right_panel"):
        results_panel(shown, bundle, occurrence)


main()
