from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from xgboost import XGBRegressor
from xgboost.core import XGBoostError

from data_prep import (
    OUTPUT_PATH,
    WEATHER_COLUMNS,
    WEATHER_OUTPUT_PATH,
    prepare_firms_dataset,
    prepare_weather_dataset,
    weather_dataset_complete,
)
from features import (
    NO_WEATHER_FEATURE_COLUMNS,
    SATELLITE_FEATURE_COLUMNS,
)
from model_io import save_metrics, save_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "readme_model_training"
DEFAULT_MODEL_PATH = MODEL_DIR / "wildfire_model.joblib"
DEFAULT_SATELLITE_MODEL_PATH = MODEL_DIR / "wildfire_satellite_model.joblib"
DEFAULT_METRICS_PATH = MODEL_DIR / "wildfire_model_metrics.json"
EVIDENCE_DIR = ROOT / "evidence"

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
FEATURE_COLUMNS = SATELLITE_FEATURE_COLUMNS
COMBINED_FEATURE_COLUMNS = FEATURE_COLUMNS + list(WEATHER_COLUMNS)
CHUNK_SIZE = 250_000
CV_MAX_SAMPLES = 100_000


def _read_feature_layout(
    data_path: str | Path,
    allow_no_weather_fallback: bool,
    satellite_only: bool = False,
) -> tuple[list[str], str]:
    columns = set(pd.read_csv(data_path, nrows=0).columns)
    present_weather = columns.intersection(WEATHER_COLUMNS)
    if satellite_only:
        return SATELLITE_FEATURE_COLUMNS, "satellite_only"
    if len(present_weather) == len(WEATHER_COLUMNS):
        return COMBINED_FEATURE_COLUMNS, "weather_and_satellite"
    if present_weather:
        missing = sorted(set(WEATHER_COLUMNS) - present_weather)
        if allow_no_weather_fallback:
            return NO_WEATHER_FEATURE_COLUMNS, "fallback_no_weather"
        raise ValueError(f"Weather dataset is incomplete; missing columns: {missing}")
    if not allow_no_weather_fallback:
        raise ValueError(
            "Weather data is missing. Build data/india_fires_weather.csv first, "
            "or explicitly enable the no-weather fallback."
        )
    return NO_WEATHER_FEATURE_COLUMNS, "fallback_no_weather"


def load_training_dataset(
    data_path: str | Path,
    allow_no_weather_fallback: bool = False,
    satellite_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Load model matrices in chunks while retaining dates for leakage-aware splits."""
    feature_columns, feature_mode = _read_feature_layout(
        data_path, allow_no_weather_fallback, satellite_only
    )
    usecols = DATA_COLUMNS.copy()
    if feature_mode == "weather_and_satellite":
        usecols.extend(WEATHER_COLUMNS)
    dtype: dict[str, str] = {
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
    }
    if feature_mode == "weather_and_satellite":
        dtype.update({column: "float32" for column in WEATHER_COLUMNS})

    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    year_parts: list[np.ndarray] = []
    date_parts: list[np.ndarray] = []
    total_rows = 0
    valid_weather_rows = 0
    for frame in pd.read_csv(
        data_path, usecols=usecols, dtype=dtype, chunksize=CHUNK_SIZE
    ):
        frame = frame.dropna(subset=DATA_COLUMNS)
        frame["frp"] = pd.to_numeric(frame["frp"], errors="coerce")
        frame = frame.loc[frame["frp"] > 0].copy()
        if frame.empty:
            continue

        dates = pd.to_datetime(frame["acq_date"], errors="coerce")
        acq_hhmm = pd.to_numeric(
            frame["acq_time"]
            .astype("string")
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(4),
            errors="coerce",
        )
        hour = acq_hhmm // 100
        minute = acq_hhmm % 100
        time_valid = hour.between(0, 23) & minute.between(0, 59)
        confidence = (
            frame["confidence"].astype("string").str.lower().map({"l": 0, "n": 1, "h": 2})
        )
        satellite = frame["satellite"].astype("string").map({"N20": 0, "SNPP": 1})
        daynight = frame["daynight"].astype("string").str.upper().map({"D": 1, "N": 0})
        valid = (
            dates.notna()
            & time_valid
            & confidence.notna()
            & satellite.notna()
            & daynight.notna()
        )
        frame = frame.loc[valid]
        dates = dates.loc[valid]
        hour = hour.loc[valid]
        minute = minute.loc[valid]
        confidence = confidence.loc[valid]
        satellite = satellite.loc[valid]
        daynight = daynight.loc[valid]
        if frame.empty:
            continue

        month_angle = 2.0 * np.pi * (dates.dt.month.to_numpy() - 1.0) / 12.0
        fractional_hour = hour.to_numpy(dtype=np.float32) + minute.to_numpy(
            dtype=np.float32
        ) / 60.0
        time_angle = 2.0 * np.pi * fractional_hour / 24.0
        satellite_features = np.column_stack(
            [
                frame["lat"].to_numpy(dtype=np.float32),
                frame["lon"].to_numpy(dtype=np.float32),
                frame["brightness"].to_numpy(dtype=np.float32),
                frame["scan"].to_numpy(dtype=np.float32),
                frame["track"].to_numpy(dtype=np.float32),
                frame["bright_t31"].to_numpy(dtype=np.float32),
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
        if feature_mode == "weather_and_satellite":
            weather_features = frame[list(WEATHER_COLUMNS)].to_numpy(
                dtype=np.float32, na_value=np.nan
            )
            valid_weather_rows += int(np.isfinite(weather_features).all(axis=1).sum())
            features = np.column_stack([satellite_features, weather_features])
        elif feature_mode == "satellite_only":
            features = satellite_features
        elif feature_mode == "fallback_no_weather":
            features = np.column_stack(
                [
                    frame["lat"].to_numpy(dtype=np.float32),
                    frame["lon"].to_numpy(dtype=np.float32),
                    dates.dt.month.to_numpy(dtype=np.float32),
                ]
            )
        else:
            features = satellite_features

        feature_parts.append(features.astype(np.float32, copy=False))
        target_parts.append(np.log1p(frame["frp"].to_numpy(dtype=np.float32)))
        year_parts.append(dates.dt.year.to_numpy(dtype=np.uint16))
        date_parts.append(dates.to_numpy(dtype="datetime64[D]").astype(np.int32))
        total_rows += len(frame)

    if not feature_parts:
        raise ValueError(f"No valid training rows found in {data_path}")
    X = np.concatenate(feature_parts)
    y = np.concatenate(target_parts)
    years = np.concatenate(year_parts)
    date_groups = np.concatenate(date_parts)
    metadata: dict[str, object] = {
        "feature_columns": feature_columns,
        "feature_mode": feature_mode,
        "weather_coverage": (
            valid_weather_rows / total_rows
            if feature_mode == "weather_and_satellite"
            else 0.0
        ),
        "weather_missing_rows": (
            total_rows - valid_weather_rows
            if feature_mode == "weather_and_satellite"
            else total_rows
        ),
        "n_samples": total_rows,
    }
    return X, y, years, date_groups, metadata


def load_training_data(
    data_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible feature/target loader for satellite-only data."""
    X, y, _, _, _ = load_training_dataset(
        data_path, allow_no_weather_fallback=True, satellite_only=True
    )
    return X, y


def _new_regressor(n_estimators: int = 400, device: str = "cuda") -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        device=device,
        n_estimators=n_estimators,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )


def _fit_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    preferred_device: str = "cuda",
    n_estimators: int = 400,
) -> tuple[XGBRegressor, str]:
    model = _new_regressor(n_estimators=n_estimators, device=preferred_device)
    try:
        model.fit(X_train, y_train, verbose=False)
        if preferred_device != "cpu":
            model.set_params(device="cpu")
        return model, preferred_device
    except XGBoostError as exc:
        if preferred_device != "cuda" or not any(
            token in str(exc).lower() for token in ("cuda", "gpu", "no visible device")
        ):
            raise
        warnings.warn(
            f"CUDA training failed; retrying explicitly on CPU: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        model = _new_regressor(n_estimators=n_estimators, device="cpu")
        model.fit(X_train, y_train, verbose=False)
        return model, "cpu"


def _score(
    y_true_log: np.ndarray, prediction_log: np.ndarray
) -> dict[str, float]:
    y_true_mw = np.expm1(y_true_log)
    prediction_mw = np.expm1(prediction_log)
    return {
        "mae_log1p_frp": float(mean_absolute_error(y_true_log, prediction_log)),
        "r2_log1p_frp": float(r2_score(y_true_log, prediction_log)),
        "mae_mw": float(mean_absolute_error(y_true_mw, prediction_mw)),
    }


def _baseline_metrics(y_train: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    median_log = float(np.median(y_train))
    median_mw = float(np.median(np.expm1(y_train)))
    return {
        "mae_log1p_frp": float(
            mean_absolute_error(y_test, np.full_like(y_test, median_log))
        ),
        "mae_mw": float(
            mean_absolute_error(
                np.expm1(y_test), np.full_like(y_test, median_mw)
            )
        ),
        "training_median_log1p_frp": median_log,
        "training_median_mw": median_mw,
    }


def _evaluate_time_split(
    X: np.ndarray, y: np.ndarray, years: np.ndarray, feature_mode: str
) -> dict[str, object]:
    train_indices = np.flatnonzero(years <= 2025)
    test_indices = np.flatnonzero(years == 2026)
    if not len(train_indices) or not len(test_indices):
        return {
            "status": "unavailable",
            "reason": "Both 2023-2025 training rows and 2026 test rows are required.",
        }
    model, device = _fit_regressor(X[train_indices], y[train_indices])
    prediction = model.predict(X[test_indices])
    return {
        "status": "complete",
        "split": "train 2023-2025, test 2026",
        "train_size": int(len(train_indices)),
        "test_size": int(len(test_indices)),
        "training_device": device,
        **_score(y[test_indices], prediction),
        "baseline": _baseline_metrics(y[train_indices], y[test_indices]),
        "feature_mode": feature_mode,
    }


def _grouped_date_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    date_groups: np.ndarray,
    train_indices: np.ndarray,
    folds: int,
    max_samples: int = CV_MAX_SAMPLES,
) -> dict[str, object]:
    if folds < 2:
        return {"status": "disabled", "reason": "At least two folds are required."}
    rng = np.random.default_rng(42)
    selected = train_indices
    if len(selected) > max_samples:
        selected = np.sort(rng.choice(selected, size=max_samples, replace=False))
    groups = date_groups[selected]
    if len(np.unique(groups)) < folds:
        return {
            "status": "unavailable",
            "reason": "The sampled training rows contain fewer distinct dates than folds.",
        }

    fold_metrics: list[dict[str, float]] = []
    splitter = GroupKFold(n_splits=folds)
    for train_local, test_local in splitter.split(X[selected], y[selected], groups):
        model, _ = _fit_regressor(
            X[selected[train_local]],
            y[selected[train_local]],
            n_estimators=100,
        )
        fold_metrics.append(
            _score(y[selected[test_local]], model.predict(X[selected[test_local]]))
        )
    return {
        "status": "complete",
        "grouped_by": "acq_date",
        "n_splits": folds,
        "n_samples": int(len(selected)),
        "folds": fold_metrics,
        "mean_mae_log1p_frp": float(
            np.mean([fold["mae_log1p_frp"] for fold in fold_metrics])
        ),
        "mean_mae_mw": float(np.mean([fold["mae_mw"] for fold in fold_metrics])),
    }


def _save_prediction_plot(
    y_test: np.ndarray, prediction: np.ndarray, output_path: Path
) -> None:
    actual_mw = np.expm1(y_test)
    predicted_mw = np.maximum(0.0, np.expm1(prediction))
    if len(actual_mw) > 50_000:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(actual_mw), size=50_000, replace=False)
        actual_mw = actual_mw[indices]
        predicted_mw = predicted_mw[indices]
    _, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(actual_mw, predicted_mw, s=5, alpha=0.25, rasterized=True)
    max_value = float(max(actual_mw.max(), predicted_mw.max(), 1.0))
    axis.plot([0, max_value], [0, max_value], color="black", linestyle="--", linewidth=1)
    axis.set(
        title="Predicted vs actual FRP on the random test set",
        xlabel="Actual FRP (MW)",
        ylabel="Predicted FRP (MW)",
    )
    axis.grid(alpha=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _save_random_forest_importance_plot(
    X: np.ndarray,
    y: np.ndarray,
    feature_columns: list[str],
    output_path: Path,
) -> int:
    weather_indices = [
        feature_columns.index(column) for column in WEATHER_COLUMNS
    ]
    weather_features = X[:, weather_indices]
    valid = np.isfinite(weather_features).all(axis=1)
    sample_indices = np.flatnonzero(valid)
    if not len(sample_indices):
        raise ValueError(
            "The README's weather feature-importance figure requires complete "
            "temperature, humidity, and wind rows."
        )
    rng = np.random.default_rng(42)
    if len(sample_indices) > CV_MAX_SAMPLES:
        sample_indices = rng.choice(sample_indices, size=CV_MAX_SAMPLES, replace=False)
    forest = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    forest.fit(weather_features[sample_indices], y[sample_indices])
    _, axis = plt.subplots(figsize=(9, 6))
    order = np.argsort(forest.feature_importances_)
    axis.barh(
        np.asarray(WEATHER_COLUMNS)[order],
        forest.feature_importances_[order],
        color="#d95f02",
    )
    axis.set(
        title="Random Forest weather feature importance",
        xlabel="Feature importance",
    )
    axis.grid(axis="x", alpha=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return len(sample_indices)


def train_model(
    data_path: str | Path = WEATHER_OUTPUT_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    max_samples: int | None = None,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    satellite_model_path: str | Path = DEFAULT_SATELLITE_MODEL_PATH,
    allow_no_weather_fallback: bool = False,
    cross_validation_folds: int = 3,
    evidence_dir: str | Path = EVIDENCE_DIR,
):
    X, y, years, date_groups, metadata = load_training_dataset(
        data_path, allow_no_weather_fallback=allow_no_weather_fallback
    )
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be greater than zero")
        if len(y) > max_samples:
            rng = np.random.default_rng(42)
            selected = rng.choice(len(y), size=max_samples, replace=False)
            X, y = X[selected], y[selected]
            years, date_groups = years[selected], date_groups[selected]

    train_indices, test_indices = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=42
    )
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    model, device = _fit_regressor(X_train, y_train)
    prediction = model.predict(X_test)
    random_metrics = _score(y_test, prediction)
    random_baseline = _baseline_metrics(y_train, y_test)
    save_model(model, model_path)

    satellite_model = None
    satellite_metrics = None
    if metadata["feature_mode"] == "weather_and_satellite":
        satellite_indices = [
            COMBINED_FEATURE_COLUMNS.index(column)
            for column in SATELLITE_FEATURE_COLUMNS
        ]
        satellite_model, satellite_device = _fit_regressor(
            X_train[:, satellite_indices], y_train
        )
        satellite_prediction = satellite_model.predict(X_test[:, satellite_indices])
        satellite_metrics = {
            "training_device": satellite_device,
            **_score(y_test, satellite_prediction),
            "baseline": random_baseline,
        }
        save_model(satellite_model, satellite_model_path)

    metrics: dict[str, object] = {
        "n_samples": int(len(y)),
        "train_size": int(len(train_indices)),
        "test_size": int(len(test_indices)),
        "split": "random 80/20, random_state=42",
        **random_metrics,
        "baseline": random_baseline,
        "model": "XGBRegressor",
        "training_device": device,
        "feature_columns": metadata["feature_columns"],
        "feature_mode": metadata["feature_mode"],
        "weather_coverage": metadata["weather_coverage"],
        "weather_missing_rows": metadata["weather_missing_rows"],
        "improvements": [
            "Added hourly temperature, relative humidity, and wind from the nearest UTC Open-Meteo hour"
            if metadata["feature_mode"] == "weather_and_satellite"
            else "Explicit no-weather fallback using location and month only"
        ],
        "time_based_evaluation": _evaluate_time_split(X, y, years, str(metadata["feature_mode"])),
        "grouped_date_cross_validation": _grouped_date_cross_validation(
            X, y, date_groups, train_indices, cross_validation_folds
        ),
    }
    if satellite_metrics is not None:
        metrics["satellite_only_model"] = {
            "model_path": str(Path(satellite_model_path)),
            **satellite_metrics,
        }

    evidence_root = Path(evidence_dir)
    _save_prediction_plot(
        y_test, prediction, evidence_root / "fig7_pred_vs_actual.png"
    )
    if metadata["feature_mode"] == "weather_and_satellite":
        importance_rows = _save_random_forest_importance_plot(
            X,
            y,
            list(metadata["feature_columns"]),
            evidence_root / "fig8_importance.png",
        )
        metrics["weather_feature_importance"] = {
            "figure_model": "RandomForestRegressor",
            "features": list(WEATHER_COLUMNS),
            "n_samples": importance_rows,
        }
    save_metrics(metrics, metrics_path)
    return model, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the wildfire FRP model."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=WEATHER_OUTPUT_PATH,
        help="Path to the complete weather-joined FIRMS CSV",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument(
        "--satellite-output", type=Path, default=DEFAULT_SATELLITE_MODEL_PATH
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional row cap for experiments; omit to train on all rows",
    )
    parser.add_argument(
        "--allow-no-weather-fallback",
        action="store_true",
        help="Train the documented lat/lon/month fallback if weather data is unavailable",
    )
    parser.add_argument(
        "--cross-validation-folds", type=int, default=3, help="Grouped date CV folds"
    )
    parser.add_argument(
        "--weather-sleep-seconds",
        type=float,
        default=0.25,
        help="Delay between Open-Meteo requests when the weather CSV must be built",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not OUTPUT_PATH.exists():
        prepare_firms_dataset()
        print(f"Saved cleaned FIRMS dataset to {OUTPUT_PATH}")
    if args.allow_no_weather_fallback and args.data.resolve() == WEATHER_OUTPUT_PATH.resolve():
        args.data = OUTPUT_PATH
    elif args.data.resolve() == WEATHER_OUTPUT_PATH.resolve():
        if not weather_dataset_complete(OUTPUT_PATH, args.data):
            prepare_weather_dataset(
                output_path=args.data,
                sleep_seconds=args.weather_sleep_seconds,
                force=args.data.exists(),
            )
    elif not args.data.exists():
        raise FileNotFoundError(f"Training data not found: {args.data}")
    _, result = train_model(
        data_path=args.data,
        model_path=args.output,
        metrics_path=args.metrics,
        satellite_model_path=args.satellite_output,
        max_samples=args.max_samples,
        allow_no_weather_fallback=args.allow_no_weather_fallback,
        cross_validation_folds=args.cross_validation_folds,
    )
    print(f"Model trained: {args.output}")
    print(
        f"Rows: {result['n_samples']:,} "
        f"({result['train_size']:,} train / {result['test_size']:,} test)"
    )
    print(f"MAE on log1p(FRP): {result['mae_log1p_frp']:.4f}")
    print(f"MAE in MW: {result['mae_mw']:.4f}")
    print(f"R2 on log1p(FRP): {result['r2_log1p_frp']:.4f}")
    print(f"Training device: {result['training_device']}")
    if "satellite_only_model" in result:
        print(
            "Satellite-only test MAE in MW: "
            f"{result['satellite_only_model']['mae_mw']:.4f}"
        )
