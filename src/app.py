from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_prep import OUTPUT_PATH, WEATHER_COLUMNS, WEATHER_OUTPUT_PATH
from features import (
    FEATURE_COLUMNS,
    NO_WEATHER_FEATURE_COLUMNS,
    build_prediction_features,
    risk_band,
)
from model_io import load_model
from serial_source import SerialSource, available_ports
from train_model import DEFAULT_METRICS_PATH, DEFAULT_MODEL_PATH

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SIZE = 50_000
SELECTABLE_HISTORY_SIZE = 5_000
DASHBOARD_COLUMNS = [
    "lat",
    "lon",
    "brightness",
    "scan",
    "track",
    "acq_date",
    "acq_time",
    "confidence",
    "satellite",
    "bright_t31",
    "daynight",
    "frp",
    *WEATHER_COLUMNS,
]


@st.cache_data(show_spinner="Loading the local FIRMS dataset...")
def load_dashboard_data(
    path: str = str(WEATHER_OUTPUT_PATH),
    sample_size: int = SAMPLE_SIZE,
    allow_no_weather_fallback: bool = False,
) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Weather-joined FIRMS data is not available at {source}. "
            "Run `python src/data_prep.py --weather` first."
        )
    columns = set(pd.read_csv(source, nrows=0).columns)
    required_columns = set(DASHBOARD_COLUMNS)
    if allow_no_weather_fallback:
        required_columns.difference_update(WEATHER_COLUMNS)
    missing = required_columns - columns
    if missing:
        raise ValueError(f"Dashboard data is missing columns: {sorted(missing)}")
    read_columns = [column for column in DASHBOARD_COLUMNS if column in columns]
    rng = np.random.default_rng(42)
    priorities = np.empty(0, dtype=np.float64)
    sample = pd.DataFrame(columns=DASHBOARD_COLUMNS)
    for chunk in pd.read_csv(
        source,
        usecols=read_columns,
        chunksize=250_000,
        low_memory=False,
    ):
        chunk["frp"] = pd.to_numeric(chunk["frp"], errors="coerce")
        chunk = chunk.loc[chunk["frp"].gt(0)].dropna(
            subset=[
                "lat",
                "lon",
                "brightness",
                "scan",
                "track",
                "acq_date",
                "acq_time",
                "confidence",
                "satellite",
                "bright_t31",
                "daynight",
                "frp",
            ]
        )
        if chunk.empty:
            continue
        for column in WEATHER_COLUMNS:
            if column not in chunk:
                chunk[column] = np.nan
        combined_priorities = np.concatenate((priorities, rng.random(len(chunk))))
        combined_sample = pd.concat((sample, chunk), ignore_index=True)
        keep = min(sample_size, len(combined_priorities))
        if len(combined_priorities) > keep:
            selected = np.argpartition(combined_priorities, -keep)[-keep:]
            priorities = combined_priorities[selected]
            sample = combined_sample.iloc[selected].reset_index(drop=True)
        else:
            priorities = combined_priorities
            sample = combined_sample
    if sample.empty:
        raise ValueError(f"No valid fire detections were found in {source}")
    sample = sample[DASHBOARD_COLUMNS]
    sample["risk_band"] = sample["frp"].map(risk_band)
    return sample


@st.cache_resource(show_spinner=False)
def load_local_model(model_path: str, metrics_path: str):
    model_file = Path(model_path)
    metrics_file = Path(metrics_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Trained model is not available at {model_file}. "
            "Run `python src/train_model.py` first."
        )
    if not metrics_file.exists():
        raise FileNotFoundError(f"Training metrics are not available at {metrics_file}")
    with metrics_file.open(encoding="utf-8") as metrics_stream:
        metrics = json.load(metrics_stream)
    return load_model(model_file), metrics


def _finite_or(value: object, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if np.isfinite(result) else fallback


def _prediction_row(
    fire: pd.Series,
    temperature: float,
    humidity: float,
    wind: float,
    feature_columns: list[str],
) -> pd.DataFrame:
    if feature_columns == NO_WEATHER_FEATURE_COLUMNS:
        return pd.DataFrame(
            [
                {
                    "lat": float(fire["lat"]),
                    "lon": float(fire["lon"]),
                    "month": float(pd.to_datetime(fire["acq_date"]).month),
                }
            ],
            columns=NO_WEATHER_FEATURE_COLUMNS,
        )
    row = build_prediction_features(fire, temperature, humidity, wind)
    if feature_columns == FEATURE_COLUMNS:
        return row
    missing = set(feature_columns) - set(row.columns)
    if missing:
        raise ValueError(f"Model features are not supported by the dashboard: {sorted(missing)}")
    return row[feature_columns]


def _fire_map(data: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    colors = {"LOW": "#2a9d8f", "MEDIUM": "#e9c46a", "HIGH": "#e76f51"}
    for band in ("LOW", "MEDIUM", "HIGH"):
        points = data.loc[data["risk_band"] == band]
        if points.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=points["lon"],
                y=points["lat"],
                mode="markers",
                name=band,
                marker={
                    "color": colors[band],
                    "size": 6,
                    "opacity": 0.65,
                },
                customdata=np.column_stack(
                    (
                        points["acq_date"].astype(str).to_numpy(dtype=object),
                        points["frp"].to_numpy(dtype=float),
                        points["confidence"].astype(str).to_numpy(dtype=object),
                    )
                ),
                hovertemplate=(
                    "Lon %{x:.3f}<br>Lat %{y:.3f}<br>Date %{customdata[0]}"
                    "<br>FRP %{customdata[1]:.2f} MW"
                    "<br>Confidence %{customdata[2]}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title="Historical FIRMS fire detections across India",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        xaxis={"range": [68, 98], "constrain": "domain"},
        yaxis={"range": [6, 37], "scaleanchor": "x", "scaleratio": 1},
        legend_title="FRP band",
        height=520,
        margin={"l": 10, "r": 10, "t": 55, "b": 10},
    )
    return figure


def _feature_importance(model, feature_columns: list[str]) -> go.Figure:
    importance = np.asarray(getattr(model, "feature_importances_", []), dtype=float)
    if len(importance) != len(feature_columns):
        raise ValueError("Saved model feature importance does not match training metrics")
    order = np.argsort(importance)[-12:]
    figure = go.Figure(
        go.Bar(
            x=importance[order],
            y=np.asarray(feature_columns)[order],
            orientation="h",
            marker_color="#d95f02",
        )
    )
    figure.update_layout(
        title="XGBoost feature importance",
        xaxis_title="Importance",
        yaxis_title="",
        height=380,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return figure


def _sensor_reading(port: str, refresh: bool):
    if port == "No sensor":
        return None
    current = st.session_state.get("sensor_source")
    if current is None or current.port != port:
        if current is not None:
            current.close()
        current = SerialSource(port)
        st.session_state["sensor_source"] = current
    return current.latest() if refresh else None


def main() -> None:
    st.set_page_config(
        page_title="Early Wildfire Detection and Analysis using ML",
        layout="wide",
    )
    st.title("Early Wildfire Detection and Analysis using ML")
    st.caption(
        "Educational prototype. Indian satellite fire points + weather. "
        "Not an official forecast."
    )
    st.warning(
        "This prototype estimates FRP for an already detected fire point. "
        "It does not forecast fire occurrence or provide an operational warning."
    )
    try:
        data = load_dashboard_data()
        model, metrics = load_local_model(
            str(DEFAULT_MODEL_PATH), str(DEFAULT_METRICS_PATH)
        )
        feature_mode = metrics.get("feature_mode", "weather_and_satellite")
        data_path = WEATHER_OUTPUT_PATH
        if feature_mode == "fallback_no_weather" and not data_path.exists():
            data_path = OUTPUT_PATH
        data = load_dashboard_data(
            str(data_path),
            allow_no_weather_fallback=feature_mode == "fallback_no_weather",
        )
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as exc:
        st.error(str(exc))
        st.stop()

    feature_columns = list(metrics["feature_columns"])
    if feature_mode == "fallback_no_weather":
        st.warning("Model status: fallback: no weather (latitude, longitude, and month only).")
    else:
        st.success(
            f"Weather-informed FRP model; {metrics['weather_coverage']:.1%} of training rows "
            "have all three weather inputs."
        )

    sidebar = st.sidebar
    sidebar.header("Prediction inputs")
    selectable_count = min(SELECTABLE_HISTORY_SIZE, len(data))
    selectable_data = data.iloc[:selectable_count]
    labels = [
        f"{row.acq_date} | {row.lat:.3f}, {row.lon:.3f} | {row.frp:.2f} MW"
        for row in selectable_data.itertuples()
    ]
    selected_index = sidebar.selectbox(
        "Historical fire point",
        options=range(selectable_count),
        format_func=lambda index: labels[index],
    )
    fire = data.iloc[selected_index]
    temperature_default = _finite_or(fire["temp_c"], 26.0)
    humidity_default = _finite_or(fire["rh_percent"], 50.0)
    wind_default = _finite_or(fire["wind_kmh"], 10.0)
    if not np.isfinite(
        np.asarray(
            [fire["temp_c"], fire["rh_percent"], fire["wind_kmh"]], dtype=np.float64
        )
    ).all():
        sidebar.info(
            "This point has incomplete weather. Missing slider values start at "
            "26 °C, 50% humidity, and 10 km/h; adjust them as needed."
        )
    temperature_slider = sidebar.slider(
        "Temperature (°C)", -10.0, 55.0, float(np.clip(temperature_default, -10, 55)), 0.5
    )
    humidity_slider = sidebar.slider(
        "Relative humidity (%)",
        0.0,
        100.0,
        float(np.clip(humidity_default, 0, 100)),
        1.0,
    )
    wind = sidebar.slider(
        "Wind speed (km/h)", 0.0, 150.0, float(np.clip(wind_default, 0, 150)), 1.0
    )

    ports = available_ports()
    port = sidebar.selectbox("ESP32 serial port", ["No sensor", *ports])
    refresh = sidebar.button("Refresh live sensor reading")
    reading = _sensor_reading(port, refresh)
    temperature = temperature_slider
    humidity = humidity_slider
    if port != "No sensor":
        if reading is None:
            sidebar.info("No fresh serial reading; using the sliders.")
        else:
            temperature = reading.temp_c
            humidity = reading.rh_percent
            sidebar.metric("Live temperature", f"{reading.temp_c:.1f} °C")
            sidebar.metric("Live humidity", f"{reading.rh_percent:.1f}%")
            threshold = sidebar.slider(
                "Smoke alert threshold (raw ADC)",
                min_value=0,
                max_value=4095,
                value=2000,
                help="Calibrate for your own sensor and environment; this is not ppm.",
            )
            sidebar.metric("Live smoke index", f"{reading.smoke_raw}/4095")
            if reading.smoke_raw >= threshold:
                sidebar.warning("Smoke index is above the selected demo threshold.")
            else:
                sidebar.success("Smoke index is below the selected demo threshold.")
    if "sensor_source" in st.session_state and port == "No sensor":
        st.session_state["sensor_source"].close()
        del st.session_state["sensor_source"]

    try:
        prediction_row = _prediction_row(
            fire, temperature, humidity, wind, feature_columns
        )
        predicted_frp = float(
            np.expm1(model.predict(prediction_row.to_numpy(dtype=np.float32))[0])
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        st.error(f"Could not create a valid model input: {exc}")
        st.stop()

    left, center, right = st.columns([1.0, 1.8, 1.15])
    with left:
        st.subheader("Estimated intensity")
        st.metric("Predicted FRP", f"{predicted_frp:.2f} MW")
        st.metric("Risk band", risk_band(predicted_frp))
        st.caption(
            "FRP is a satellite intensity proxy, not burned area or a fire-occurrence probability."
        )
        st.subheader("Evaluation")
        st.metric("Random-test MAE", f"{metrics['mae_mw']:.2f} MW")
        st.metric("Random-test R²", f"{metrics['r2_log1p_frp']:.3f}")
    with center:
        st.subheader("Fire detections")
        map_sample = data.sample(
            n=min(8_000, len(data)), random_state=42
        )
        st.plotly_chart(
            _fire_map(map_sample), use_container_width=True, config={"displayModeBar": False}
        )
    with right:
        st.plotly_chart(
            _feature_importance(model, feature_columns),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.subheader("Sample fire points and matched weather")
    table_columns = [
        "lat",
        "lon",
        "acq_date",
        "acq_time",
        "frp",
        "confidence",
        "temp_c",
        "rh_percent",
        "wind_kmh",
        "risk_band",
    ]
    st.dataframe(
        data[table_columns].head(1000),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "The app loads only the local model and CSV. Weather is downloaded during data preparation; "
        "cached data supports offline use."
    )


if __name__ == "__main__":
    main()
