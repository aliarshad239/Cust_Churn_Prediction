from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def _coerce_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    if "TotalCharges" not in df.columns:
        return df
    cleaned = df["TotalCharges"].astype(str).str.strip().replace({"": np.nan})
    df["TotalCharges"] = pd.to_numeric(cleaned, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if df["TotalCharges"].isna().any():
        median_val = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median_val)
    return df


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    if "services_count" in df.columns:
        df = df.drop(columns=["services_count"])
    if "tenure_bucket" in df.columns:
        df = df.drop(columns=["tenure_bucket"])
    if "est_lifetime_value" in df.columns:
        df = df.drop(columns=["est_lifetime_value"])

    service_cols = [col for col in SERVICE_COLUMNS if col in df.columns]
    if service_cols:
        df["services_count"] = (df[service_cols] == "Yes").sum(axis=1)
    else:
        df["services_count"] = 0

    if "tenure" in df.columns:
        bins = [0, 12, 24, 48, 60, np.inf]
        labels = ["0-12", "12-24", "24-48", "48-60", "60+"]
        df["tenure_bucket"] = pd.cut(df["tenure"], bins=bins, labels=labels, right=False)
    else:
        df["tenure_bucket"] = pd.NA

    if {"MonthlyCharges", "tenure"}.issubset(df.columns):
        df["est_lifetime_value"] = df["MonthlyCharges"] * df["tenure"]
    else:
        df["est_lifetime_value"] = np.nan

    return df


def get_feature_columns(cfg) -> list[str]:
    base = (
        cfg.features["numeric"]
        + cfg.features["binary_yes_no"]
        + cfg.features["categorical"]
    )
    engineered = cfg.features.get("engineered", [])
    return base + engineered


def prepare_dataset(df: pd.DataFrame, cfg, for_inference: bool = False) -> pd.DataFrame:
    df = df.copy()

    expected_cols = (
        cfg.features["numeric"]
        + cfg.features["binary_yes_no"]
        + cfg.features["categorical"]
    )
    for col in expected_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df = _coerce_total_charges(df)
    df = _add_engineered_features(df)

    for col in cfg.features.get("drop", []):
        if col in df.columns:
            df = df.drop(columns=[col])

    target_col = cfg.data["target"]
    if for_inference and target_col in df.columns:
        df = df.drop(columns=[target_col])

    final_cols: Iterable[str] = get_feature_columns(cfg)
    if not for_inference and target_col in df.columns:
        final_cols = list(final_cols) + [target_col]

    return df[list(final_cols)]
