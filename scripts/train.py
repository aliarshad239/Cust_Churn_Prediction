import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import prepare_dataset
from src.pipeline import build_pipeline
from src.utils import load_config

DEBUG = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train churn model.")
    parser.add_argument(
        "--config",
        default="config/params.yaml",
        help="Path to params.yaml",
    )
    return parser.parse_args()


def debug_numeric_report(df: pd.DataFrame, cfg) -> None:
    numeric_cols = cfg.features["numeric"] + ["services_count", "est_lifetime_value"]
    print("🔎 DEBUG numeric diagnostics")
    for col in numeric_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        nan_count = int(s.isna().sum())
        inf_count = int(np.isinf(s).sum())
        max_abs = float(s.abs().max(skipna=True))
        print(f"  {col}: nan={nan_count} inf={inf_count} max_abs={max_abs}")
        if nan_count > 0 or inf_count > 0 or (max_abs > 1e6):
            print(df[[col]].head(3))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    print("🔹 Stage: load")
    raw_path = cfg.data["raw_path"]
    df_raw = pd.read_csv(raw_path)
    print(f"Loaded raw data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    print("🔹 Stage: preprocess")
    df = prepare_dataset(df_raw, cfg, for_inference=False)

    if DEBUG:
        debug_numeric_report(df, cfg)

    processed_path = Path(cfg.data["processed_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Saved processed dataset to {processed_path}")

    target_col = cfg.data["target"]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print("🔹 Stage: split")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.train["test_size"],
        random_state=cfg.random_state,
        stratify=y,
    )

    print("🔹 Stage: train")
    pipeline = build_pipeline(cfg)
    Xt = pipeline.named_steps["preprocess"].fit_transform(X_train)
    if hasattr(Xt, "toarray"):
        Xt_arr = Xt.toarray()
    else:
        Xt_arr = np.asarray(Xt)
    nan_count = int(np.isnan(Xt_arr).sum())
    inf_count = int(np.isinf(Xt_arr).sum())
    max_abs = float(np.nanmax(np.abs(Xt_arr)))
    print(
        f"Transformed matrix diagnostics: nan={nan_count} inf={inf_count} max_abs={max_abs}"
    )
    if nan_count > 0 or inf_count > 0:
        raise ValueError("NaN/inf detected after preprocessing; aborting training.")
    pipeline.fit(X_train, y_train)
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_") and hasattr(model, "intercept_"):
        if not (np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all()):
            raise ValueError("Model coefficients contain NaN/inf after training.")

    print("🔹 Stage: eval")
    classes = list(pipeline.named_steps["model"].classes_)
    pos_label = cfg.train["positive_label"]
    if pos_label not in classes:
        raise ValueError(f"Positive label '{pos_label}' not found in classes: {classes}")
    pos_index = classes.index(pos_label)

    y_proba = pipeline.predict_proba(X_test)[:, pos_index]
    y_true = (y_test == pos_label).astype(int)
    threshold = cfg.train["threshold"]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "positive_label": pos_label,
        "test_size": float(cfg.train["test_size"]),
    }

    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    reports_dir = Path("reports")
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = reports_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    cm_display = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["No", "Yes"], cmap="Blues"
    )
    cm_path = figures_dir / "confusion_matrix.png"
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=160)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    RocCurveDisplay.from_predictions(y_true, y_proba)
    roc_path = figures_dir / "roc_curve.png"
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=160)
    plt.close()
    print(f"Saved ROC curve to {roc_path}")

    print("🔹 Stage: save model")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "churn_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
