from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from data_prep import OUTPUT_PATH, prepare_firms_dataset
from model_io import save_metrics, save_model

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "wildfire_model.joblib"
DATA_COLUMNS = [
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
FEATURE_COLUMNS = [
    "lat",
    "lon",
    "brightness",
    "scan",
    "track",
    "bright_t31",
    "month_sin",
    "month_cos",
    "time_sin",
    "time_cos",
    "confidence_score",
    "daynight",
    "satellite_N20",
    "satellite_SNPP",
]


def load_training_data(data_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(
        data_path,
        usecols=DATA_COLUMNS,
        dtype={
            "lat": "float32",
            "lon": "float32",
            "brightness": "float32",
            "scan": "float32",
            "track": "float32",
            "acq_date": "string",
            "acq_time": "string",
            "confidence": "category",
            "satellite": "category",
            "bright_t31": "float32",
            "daynight": "category",
            "frp": "float32",
        },
    )
    df = df.dropna(subset=DATA_COLUMNS)
    df = df[df["frp"] > 0].copy()

    dates = pd.to_datetime(df["acq_date"], errors="coerce")
    month = dates.dt.month
    acq_hhmm = pd.to_numeric(
        df["acq_time"].str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(4),
        errors="coerce",
    )
    hour = acq_hhmm // 100
    minute = acq_hhmm % 100
    time_valid = hour.between(0, 23) & minute.between(0, 59)
    fractional_hour = hour + minute / 60.0
    confidence = df["confidence"].astype("string").str.lower().map({"l": 0, "n": 1, "h": 2})
    satellite = df["satellite"].astype("string").map({"N20": 0, "SNPP": 1})
    daynight = df["daynight"].astype("string").str.upper().map({"D": 1, "N": 0})
    valid = dates.notna() & time_valid & confidence.notna() & satellite.notna() & daynight.notna()
    df = df.loc[valid]
    month = month.loc[valid]
    fractional_hour = fractional_hour.loc[valid]
    confidence = confidence.loc[valid]
    satellite = satellite.loc[valid]
    daynight = daynight.loc[valid]

    month_angle = 2.0 * np.pi * (month.to_numpy(dtype=np.float32) - 1.0) / 12.0
    time_angle = 2.0 * np.pi * fractional_hour.to_numpy(dtype=np.float32) / 24.0

    X = np.column_stack(
        [
            df["lat"].to_numpy(dtype=np.float32),
            df["lon"].to_numpy(dtype=np.float32),
            df["brightness"].to_numpy(dtype=np.float32),
            df["scan"].to_numpy(dtype=np.float32),
            df["track"].to_numpy(dtype=np.float32),
            df["bright_t31"].to_numpy(dtype=np.float32),
            np.sin(month_angle).astype(np.float32),
            np.cos(month_angle).astype(np.float32),
            np.sin(time_angle).astype(np.float32),
            np.cos(time_angle).astype(np.float32),
            confidence.to_numpy(dtype=np.float32),
            daynight.to_numpy(dtype=np.float32),
            satellite.eq(0).to_numpy(dtype=np.float32),
            satellite.eq(1).to_numpy(dtype=np.float32),
        ]
    )
    y = np.log1p(df["frp"].to_numpy(dtype=np.float32))
    return X, y


def train_model(
    data_path: str | Path = OUTPUT_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    max_samples: int | None = None,
    device: str = "cpu",
):
    X, y = load_training_data(data_path)
    if max_samples is not None and max_samples > 0 and len(y) > max_samples:
        rng = np.random.default_rng(42)
        selected = rng.choice(len(y), size=max_samples, replace=False)
        X, y = X[selected], y[selected]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    regressor = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        device=device,
        n_estimators=400,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    regressor.fit(X_train, y_train, verbose=False)

    pred_log = regressor.predict(X_test)
    mae_log = mean_absolute_error(y_test, pred_log)
    r2_log = r2_score(y_test, pred_log)

    test_frp = np.expm1(y_test)
    pred_frp = np.expm1(pred_log)
    mae_mw = mean_absolute_error(test_frp, pred_frp)

    train_median_log = float(np.median(y_train))
    baseline_mae_log = mean_absolute_error(y_test, np.full_like(y_test, train_median_log))
    train_median_mw = float(np.median(np.expm1(y_train)))
    baseline_mae_mw = mean_absolute_error(test_frp, np.full_like(test_frp, train_median_mw))

    metrics = {
        "n_samples": int(len(y)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "split": "random 80/20, random_state=42",
        "mae_log1p_frp": float(mae_log),
        "r2_log1p_frp": float(r2_log),
        "mae_mw": float(mae_mw),
        "baseline_mae_log1p_frp": float(baseline_mae_log),
        "baseline_mae_mw": float(baseline_mae_mw),
        "model": "XGBRegressor",
        "training_device": device,
        "feature_columns": FEATURE_COLUMNS,
        "improvements": [
            "Added brightness-temperature, pixel-footprint, day/night, and cyclic seasonal/pass-time features"
        ],
    }

    save_model(regressor, model_path)
    save_metrics(metrics, Path(model_path).with_name(Path(model_path).stem + "_metrics.json"))
    return regressor, metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train wildfire FRP model from repo FIRMS fire data.")
    parser.add_argument("--data", type=str, default=str(OUTPUT_PATH), help="Path to cleaned FIRMS CSV")
    parser.add_argument("--output", type=str, default=str(DEFAULT_MODEL_PATH), help="Model output path")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional row cap for experiments; omit to train on all rows",
    )
    parser.add_argument("--device", type=str, default="cpu", help="XGBoost device, for example cpu or cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        prepare_firms_dataset(data_path)
    _, metrics = train_model(
        data_path=data_path,
        model_path=args.output,
        max_samples=args.max_samples,
        device=args.device,
    )
    print(f"Model trained: {args.output}")
    print(f"Rows: {metrics['n_samples']:,} ({metrics['train_size']:,} train / {metrics['test_size']:,} test)")
    print(f"MAE on log1p(FRP): {metrics['mae_log1p_frp']:.4f}")
    print(f"MAE in MW: {metrics['mae_mw']:.4f}")
    print(f"R2 on log1p(FRP): {metrics['r2_log1p_frp']:.4f}")
    print(f"Median baseline MAE in MW: {metrics['baseline_mae_mw']:.4f}")
