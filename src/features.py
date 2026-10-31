from __future__ import annotations

import numpy as np
import pandas as pd


CONFIDENCE_ORDER = {"l": 0, "n": 1, "h": 2}


def confidence_score(value: str | float | int | None) -> int:
    if pd.isna(value):
        return 0
    key = str(value).strip().lower()
    return CONFIDENCE_ORDER.get(key, 0)


def risk_band(frp_mw: float) -> str:
    if frp_mw < 5:
        return "LOW"
    if frp_mw < 15:
        return "MEDIUM"
    return "HIGH"


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature-ready dataset with target and risk labels."""
    out = df.copy()
    out["confidence_score"] = out["confidence"].map(confidence_score)
    out["month"] = pd.to_datetime(out["acq_date"]).dt.month
    out["frp_numeric"] = pd.to_numeric(out["frp"], errors="coerce")
    out["target_log1p_frp"] = np.log1p(out["frp_numeric"].fillna(0.0))
    out["risk_band"] = out["frp_numeric"].map(risk_band)
    return out
