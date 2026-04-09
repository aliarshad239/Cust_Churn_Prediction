from __future__ import annotations

import os
import sys
from pathlib import Path

import boto3
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.features import prepare_dataset
from src.utils import load_config
from app.schemas import CustomerFeatures

app = FastAPI(title="Churn Inference API", version="2.0.0")

MODEL = None
CFG = None


def _download_from_s3(bucket: str, key: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, str(dest))


def _load_artifacts() -> None:
    global MODEL, CFG

    bucket = os.getenv("S3_BUCKET")
    model_key = os.getenv("MODEL_KEY")
    config_key = os.getenv("CONFIG_KEY")

    if bucket and model_key and config_key:
        model_path = Path("/tmp") / "churn_model.joblib"
        config_path = Path("/tmp") / "params.yaml"
        _download_from_s3(bucket, model_key, model_path)
        _download_from_s3(bucket, config_key, config_path)
    else:
        model_path = ROOT / "models" / "churn_model.joblib"
        config_path = ROOT / "config" / "params.yaml"

    CFG = load_config(str(config_path))
    MODEL = joblib.load(model_path)


@app.on_event("startup")
def startup_event() -> None:
    _load_artifacts()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: CustomerFeatures) -> dict:
    if MODEL is None or CFG is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    df = pd.DataFrame([payload.model_dump()])
    features = prepare_dataset(df, CFG, for_inference=True)

    classes = list(MODEL.named_steps["model"].classes_)
    pos_label = CFG.train["positive_label"]
    if pos_label not in classes:
        raise HTTPException(status_code=500, detail="Positive label not found.")
    pos_index = classes.index(pos_label)

    proba = float(MODEL.predict_proba(features)[:, pos_index][0])
    threshold = float(CFG.train["threshold"])
    prediction = "Yes" if proba >= threshold else "No"

    return {
        "churn_probability": proba,
        "prediction": prediction,
        "threshold": threshold,
    }
