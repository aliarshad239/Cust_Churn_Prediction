from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler



def _yes_no_to_int(data):
    return (data == "Yes").astype(int)


def _sanitize_numeric(data):
    arr = np.asarray(data, dtype="float64")
    arr[~np.isfinite(arr)] = np.nan
    return np.clip(arr, -1e6, 1e6)


def build_preprocessor(cfg) -> ColumnTransformer:
    numeric_features = cfg.features["numeric"] + [
        "services_count",
        "est_lifetime_value",
    ]
    categorical_features = cfg.features["categorical"] + ["tenure_bucket"]
    binary_features = cfg.features["binary_yes_no"]

    numeric_pipeline = Pipeline(
        steps=[
            ("sanitize", FunctionTransformer(_sanitize_numeric, validate=False)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("to_int", FunctionTransformer(_yes_no_to_int, validate=False)),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("bin", binary_pipeline, binary_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor


def build_model(cfg):
    model_type = cfg.train.get("model_type", "logistic").lower()
    random_state = cfg.random_state

    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception:
            model_type = "logistic"
        else:
            return XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            )

    if model_type == "logistic":
        return LogisticRegression(
            solver="liblinear",
            penalty="l2",
            C=1.0,
            max_iter=5000,
            class_weight="balanced",
            random_state=random_state,
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def build_pipeline(cfg) -> Pipeline:
    preprocessor = build_preprocessor(cfg)
    model = build_model(cfg)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
