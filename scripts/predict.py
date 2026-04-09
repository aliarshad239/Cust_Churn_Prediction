import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import prepare_dataset
from src.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict churn for a single record.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a JSON file or CSV file containing a single row.",
    )
    parser.add_argument(
        "--config",
        default="config/params.yaml",
        help="Path to params.yaml",
    )
    parser.add_argument(
        "--model",
        default="models/churn_model.joblib",
        help="Path to trained model artifact.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override classification threshold (default uses config).",
    )
    return parser.parse_args()


def load_single_record(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        with open(path, "r") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            payload = payload[0]
        return pd.DataFrame([payload])

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if df.shape[0] != 1:
            df = df.head(1)
        return df

    raise ValueError("Input must be a .json or .csv file.")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_path = Path(args.model)

    print("🔹 Stage: load model")
    model = joblib.load(model_path)

    print("🔹 Stage: load input")
    input_path = Path(args.input)
    df_raw = load_single_record(input_path)

    print("🔹 Stage: preprocess")
    df = prepare_dataset(df_raw, cfg, for_inference=True)

    print("🔹 Stage: predict")
    classes = list(model.named_steps["model"].classes_)
    pos_label = cfg.train["positive_label"]
    pos_index = classes.index(pos_label)
    proba = float(model.predict_proba(df)[:, pos_index][0])

    threshold = args.threshold if args.threshold is not None else cfg.train["threshold"]
    prediction = "Yes" if proba >= threshold else "No"

    print(f"Churn probability: {proba:.4f}")
    print(f"Predicted churn: {prediction} (threshold={threshold})")


if __name__ == "__main__":
    main()
