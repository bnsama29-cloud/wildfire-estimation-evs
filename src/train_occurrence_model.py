"""Train the next-day fire prediction used by the dashboard.

Each row is one 1-degree grid cell on one day. Inputs are the conditions at 08:00 UTC
(13:30 in India, a snapshot like a sensor reading). Targets are FIRMS detections on the
following day: any fire, and a large fire (15 MW or more).

The 80/20 split holds out whole weeks, because neighbouring days share weather and fire
episodes and a random row split would score far too high. Training on 2024 and testing on
2025 is reported as a harder check. Models are fixed before either score is read.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from features import OCCURRENCE_FEATURES, occurrence_features
from model_io import save_metrics, save_model
from weather_join import END_DATE, HOURLY_VARIABLES, START_DATE, cache_path, load_fires, rain_last_week

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "wildfire_occurrence_model.joblib"
METRICS_PATH = ROOT / "models" / "wildfire_occurrence_metrics.json"
ISSUE_HOUR_UTC = 8
LARGE_FIRE_MW = 15.0
VALIDATION_SHARE = 0.2
RANDOM_STATE = 42
TARGETS = {"fire": "fire_tomorrow", "large_fire": "large_fire_tomorrow"}
HISTORY_FEATURE = "fires_last_7d"  # detections in the cell over the 7 days before today

logger = logging.getLogger("train_occurrence_model")


def cell_days(fires: pd.DataFrame, weather: dict[str, np.ndarray], cell_lat: float, cell_lon: float,
              start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """One row per day for one cell. Rows without a full week of history or a next day are dropped."""
    dates = pd.date_range(start, end, freq="D")
    days = pd.to_datetime(fires["acq_date"])
    counts = fires.groupby(days).size().reindex(dates, fill_value=0).to_numpy()
    largest = fires.groupby(days)["frp"].max().reindex(dates, fill_value=0.0).to_numpy()
    hour = ISSUE_HOUR_UTC + 24 * np.arange(len(dates))
    out = pd.DataFrame({
        "date": dates,
        "lat": cell_lat,
        "lon": cell_lon,
        "month": dates.month,
        "temp_c": weather["temperature_2m"][hour],
        "rh_percent": weather["relative_humidity_2m"][hour],
        "wind_kmh": weather["wind_speed_10m"][hour],
        "rain_7d_mm": rain_last_week(weather["precipitation"])[hour],
    })
    tomorrow = pd.Series(counts).shift(-1)
    out["fire_tomorrow"] = (tomorrow > 0).astype(int)
    out["large_fire_tomorrow"] = (pd.Series(largest).shift(-1) >= LARGE_FIRE_MW).astype(int)
    # Today's detections are incomplete at the issue hour, so history stops at yesterday.
    out[HISTORY_FEATURE] = pd.Series(counts, dtype=np.float64).shift(1).rolling(7).sum()
    out["fire_yesterday"] = (pd.Series(counts).shift(1) > 0).astype(int)
    return out[tomorrow.notna() & out[HISTORY_FEATURE].notna()].reset_index(drop=True)


def build_dataset(fires: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for (cell_lat, cell_lon), group in fires.groupby(["cell_lat", "cell_lon"]):
        path = cache_path(cell_lat, cell_lon)
        if not path.exists():
            continue
        with np.load(path) as cached:
            weather = {name: cached[name] for name in HOURLY_VARIABLES}
        parts.append(cell_days(group, weather, float(cell_lat), float(cell_lon)))
    return pd.concat(parts, ignore_index=True).dropna(subset=["temp_c", "rh_percent", "wind_kmh"])


def week_groups(dates: pd.Series) -> pd.Series:
    """Weeks numbered from the first date, so one week in two calendar years stays one group."""
    return (dates - dates.min()).dt.days // 7


def split_weeks(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_SHARE, random_state=RANDOM_STATE)
    fit_idx, val_idx = next(splitter.split(df, groups=week_groups(df["date"])))
    return df.iloc[fit_idx], df.iloc[val_idx]


def classifier() -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic", tree_method="hist", n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.9, min_child_weight=20, random_state=RANDOM_STATE, n_jobs=-1,
    )


def score(y: np.ndarray, probability: np.ndarray) -> dict:
    flagged = probability >= 0.5
    return {
        "pr_auc": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "precision_at_50": float(precision_score(y, flagged, zero_division=0)),
        "recall_at_50": float(recall_score(y, flagged, zero_division=0)),
    }


def climatology(train: pd.DataFrame, target: str, other: pd.DataFrame) -> np.ndarray:
    """Baseline: how often this target happened in this cell and month during training."""
    rates = train.groupby(["lat", "lon", "month"])[target].mean().rename("rate").reset_index()
    merged = other[["lat", "lon", "month"]].merge(rates, on=["lat", "lon", "month"], how="left")
    return merged["rate"].fillna(train[target].mean()).to_numpy()


def evaluate(train: pd.DataFrame, held_out: pd.DataFrame, target: str) -> dict:
    y = held_out[target].to_numpy()
    y_train = train[target].to_numpy()
    X_train, X_held = occurrence_features(train), occurrence_features(held_out)
    model = classifier().fit(X_train, y_train)
    probability = model.predict_proba(X_held)[:, 1]

    no_history = [name for name in OCCURRENCE_FEATURES if name != HISTORY_FEATURE]
    no_history_model = classifier().fit(X_train[no_history], y_train)

    weather_only = ["temp_c", "rh_percent", "wind_kmh", "rain_7d_mm", "vpd_kpa", "fwi"]
    weather_model = classifier().fit(X_train[weather_only], y_train)

    bins = pd.cut(probability, np.linspace(0, 1, 6), include_lowest=True)
    reliability = pd.DataFrame({"predicted": probability, "observed": y}).groupby(bins, observed=True)
    return {
        "rows": int(len(held_out)),
        "base_rate": float(y.mean()),
        "model": score(y, probability),
        "baseline_always_base_rate": score(y, np.full(len(y), y_train.mean())),
        "baseline_cell_month_rate": score(y, climatology(train, target, held_out)),
        "baseline_same_as_yesterday": score(y, held_out["fire_yesterday"].to_numpy(dtype=np.float64)),
        "weather_only": score(y, weather_model.predict_proba(X_held[weather_only])[:, 1]),
        "without_recent_fire_history": score(y, no_history_model.predict_proba(X_held[no_history])[:, 1]),
        "reliability": [
            {"predicted": float(row.predicted), "observed": float(row.observed), "rows": int(row.rows)}
            for row in reliability.agg(predicted=("predicted", "mean"), observed=("observed", "mean"),
                                       rows=("observed", "size")).itertuples()
        ],
    }


def train(df: pd.DataFrame) -> tuple[dict, dict]:
    fit, validation = split_weeks(df)
    year_train, year_test = df[df["date"].dt.year == 2024], df[df["date"].dt.year == 2025]
    metrics = {
        "split": "80/20 by whole weeks; harder check trains on 2024 and tests on 2025",
        "train_rows": int(len(fit)),
        "cells": int(df[["lat", "lon"]].drop_duplicates().shape[0]),
        "issue_hour_utc": ISSUE_HOUR_UTC,
        "large_fire_mw": LARGE_FIRE_MW,
    }
    models = {}
    for name, target in TARGETS.items():
        metrics[name] = {
            "validation_80_20": evaluate(fit, validation, target),
            "train_2024_test_2025": evaluate(year_train, year_test, target),
        }
        models[name] = classifier().fit(occurrence_features(fit), fit[target].to_numpy())
        logger.info("%s %s", name, metrics[name]["validation_80_20"]["model"])
        metrics[name]["feature_importance"] = dict(sorted(
            zip(OCCURRENCE_FEATURES, map(float, models[name].feature_importances_)), key=lambda item: -item[1]
        ))
    return {"models": models, "features": OCCURRENCE_FEATURES, "metrics": metrics}, metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args()
    dataset = build_dataset(load_fires())
    logger.info("%d cell-days", len(dataset))
    bundle, metrics = train(dataset)
    save_model(bundle, MODEL_PATH)
    save_metrics(metrics, METRICS_PATH)


if __name__ == "__main__":
    main()
