"""Train the weather-driven FRP model used by the dashboard.

Trains on 2024 and tests on 2025, so the score is not inflated by near-duplicate
detections. Models are compared on held-out 2024 dates; 2025 is only read for the
chosen model and the input comparison.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor

from features import WEATHER_FEATURES, weather_features
from model_io import save_metrics, save_model
from predict import band_for
from weather_join import FULL_JOIN_PATH

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "wildfire_weather_model.joblib"
METRICS_PATH = ROOT / "models" / "wildfire_weather_metrics.json"
BAND_QUANTILES = (0.5, 0.85)
LARGE_FIRE_MW = 15.0
RANDOM_STATE = 42
# Direction each input may push intensity: hotter, windier, drier air up; humid, rainy weather down.
MONOTONE = {"temp_c": 1, "rh_percent": -1, "wind_kmh": 1, "rain_7d_mm": -1, "vpd_kpa": 1}
WEATHER_ONLY = ["temp_c", "rh_percent", "wind_kmh", "rain_7d_mm", "vpd_kpa"]
PLACE_SEASON_ONLY = ["month_sin", "month_cos", "is_day", "lat", "lon"]

logger = logging.getLogger("train_weather_model")


def load_joined(path: str | Path = FULL_JOIN_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"acq_date": "string", "acq_time": "string"})
    df = df[df["frp"] > 0].copy()
    df["month"] = pd.to_numeric(df["acq_date"].str[5:7])
    df["year"] = pd.to_numeric(df["acq_date"].str[:4])
    df["is_day"] = df["daynight"].eq("D").astype(int)
    return df


def split_by_year(df: pd.DataFrame, train_year: int = 2024, test_year: int = 2025):
    return df[df["year"] == train_year], df[df["year"] == test_year]


def xgb(features: list[str], monotone: bool) -> XGBRegressor:
    constraints = tuple(MONOTONE.get(name, 0) for name in features) if monotone else None
    return XGBRegressor(
        objective="reg:squarederror", tree_method="hist", n_estimators=300, max_depth=7,
        learning_rate=0.08, subsample=0.8, colsample_bytree=0.9, min_child_weight=20,
        monotone_constraints=constraints, random_state=RANDOM_STATE, n_jobs=-1,
    )


def candidates() -> dict:
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=60, max_depth=14, min_samples_leaf=50, max_samples=0.2,
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "XGBoost": xgb(WEATHER_FEATURES, monotone=False),
        "XGBoost (monotonic weather)": xgb(WEATHER_FEATURES, monotone=True),
    }


def score(y_log: np.ndarray, pred_log: np.ndarray) -> dict:
    return {
        "mae_log1p_frp": float(mean_absolute_error(y_log, pred_log)),
        "r2_log1p_frp": float(r2_score(y_log, pred_log)),
        "mae_mw": float(mean_absolute_error(np.expm1(y_log), np.expm1(pred_log))),
    }


def band_lift(frp: np.ndarray, pred_mw: np.ndarray, thresholds: tuple[float, float]) -> dict:
    """Share of fires in each predicted band that really were large. This is what makes a band meaningful."""
    bands = np.array([band_for(value, thresholds) for value in pred_mw])
    lift = {}
    for band in ("LOW", "MEDIUM", "HIGH"):
        members = bands == band
        share = float((frp[members] >= LARGE_FIRE_MW).mean()) if members.any() else 0.0
        lift[band] = {"fires": int(members.sum()), "share_large": share}
    return lift


def train(df: pd.DataFrame) -> tuple[dict, dict]:
    train_df, test_df = split_by_year(df)
    X_all, y_all = weather_features(train_df), np.log1p(train_df["frp"].to_numpy(dtype=np.float64))
    X_test, y_test = weather_features(test_df), np.log1p(test_df["frp"].to_numpy(dtype=np.float64))

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    fit_idx, val_idx = next(splitter.split(X_all, groups=train_df["acq_date"]))
    validation = {}
    for name, model in candidates().items():
        model.fit(X_all.iloc[fit_idx], y_all[fit_idx])
        validation[name] = score(y_all[val_idx], model.predict(X_all.iloc[val_idx]))
        logger.info("%s validation %s", name, validation[name])
    best = min(validation, key=lambda name: validation[name]["mae_log1p_frp"])

    model = candidates()[best].fit(X_all, y_all)
    train_pred_mw = np.expm1(model.predict(X_all))
    thresholds = tuple(float(q) for q in np.quantile(train_pred_mw, BAND_QUANTILES))
    pred_log = model.predict(X_test)

    inputs_compared = {}
    for label, columns in (("weather only", WEATHER_ONLY), ("place and season only", PLACE_SEASON_ONLY)):
        partial = xgb(columns, monotone=False).fit(X_all[columns], y_all)
        inputs_compared[label] = score(y_test, partial.predict(X_test[columns]))

    importance = sorted(zip(WEATHER_FEATURES, model.feature_importances_), key=lambda item: -item[1])
    metrics = {
        "split": "train 2024, test 2025; model chosen on held-out 2024 dates",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "chosen_model": best,
        "validation": validation,
        "test": score(y_test, pred_log),
        "test_baseline_median": score(y_test, np.full_like(y_test, np.median(y_all))),
        "test_inputs_compared": inputs_compared,
        "band_thresholds_mw": list(thresholds),
        "test_band_lift": band_lift(test_df["frp"].to_numpy(), np.expm1(pred_log), thresholds),
        "large_fire_mw": LARGE_FIRE_MW,
        "feature_importance": {name: float(value) for name, value in importance},
    }
    bundle = {"model": model, "band_thresholds_mw": list(thresholds), "features": WEATHER_FEATURES, "metrics": metrics}
    return bundle, metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(FULL_JOIN_PATH))
    args = parser.parse_args()
    bundle, metrics = train(load_joined(args.data))
    save_model(bundle, MODEL_PATH)
    save_metrics(metrics, METRICS_PATH)
    logger.info("test %s, baseline %s", metrics["test"], metrics["test_baseline_median"])


if __name__ == "__main__":
    main()
