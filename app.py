from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
CLASSIFIER_PATH = BASE_DIR / "simple_xgb_classifier.joblib"
SCALER_PATH = BASE_DIR / "deployment_11_feature_standard_scaler.joblib"
FEATURES_PATH = BASE_DIR / "deploy_features.json"
CLASS_METADATA_PATH = BASE_DIR / "metadata.json"
RF_DIR = BASE_DIR / "RF"
RF_MODEL_PATH = RF_DIR / "rf_regressor_11_features.joblib"
RF_METADATA_PATH = RF_DIR / "metadata.json"

classifier = joblib.load(CLASSIFIER_PATH)
scaler = joblib.load(SCALER_PATH)
rf_regressor = joblib.load(RF_MODEL_PATH)

FEATURES = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
CLASS_METADATA = json.loads(CLASS_METADATA_PATH.read_text(encoding="utf-8"))
RF_METADATA = json.loads(RF_METADATA_PATH.read_text(encoding="utf-8"))
CLASSES = CLASS_METADATA["classes"]
CLASS_DEFINITION = CLASS_METADATA.get("class_definition", {})

app = FastAPI(
    title="AQI Multi-Model Prediction API",
    description="Predict next-hour AQI level and next-hour AQI value from 11 raw environmental inputs.",
    version="2.0.0",
)


class AQIInput(BaseModel):
    PM25: float = Field(..., alias="PM2.5")
    PM10: float
    CO: float
    NO2: float
    SO2: float
    O3: float
    pressure: float
    humidity: float
    temperature: float
    wind_direction: float
    wind_speed: float

    class Config:
        populate_by_name = True



def build_raw_input(data: AQIInput) -> dict[str, float]:
    return {
        "PM2.5": float(data.PM25),
        "PM10": float(data.PM10),
        "CO": float(data.CO),
        "NO2": float(data.NO2),
        "SO2": float(data.SO2),
        "O3": float(data.O3),
        "pressure": float(data.pressure),
        "humidity": float(data.humidity),
        "temperature": float(data.temperature),
        "wind_direction": float(data.wind_direction),
        "wind_speed": float(data.wind_speed),
    }



def transform_input(raw_input: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = pd.DataFrame([raw_input], columns=FEATURES)
    scaled = scaler.transform(raw_df)
    scaled_df = pd.DataFrame(scaled, columns=FEATURES)
    return raw_df, scaled_df



def classify_from_scaled(scaled_df: pd.DataFrame) -> dict[str, object]:
    pred_id = int(classifier.predict(scaled_df)[0])
    pred_label = CLASSES[str(pred_id)]
    probabilities = classifier.predict_proba(scaled_df)[0]
    return {
        "prediction_label": pred_label,
        "prediction_class_id": pred_id,
        "probabilities": {
            CLASSES[str(i)]: float(probabilities[i]) for i in range(len(probabilities))
        },
        "class_definition": CLASS_DEFINITION.get(pred_label, ""),
    }



def regress_from_scaled(scaled_df: pd.DataFrame) -> dict[str, object]:
    value = float(rf_regressor.predict(scaled_df)[0])
    return {
        "prediction_value": value,
        "target": RF_METADATA.get("target", "AQI_next_raw"),
        "model": RF_METADATA.get("model_name", "rf_regressor_11_features"),
    }


@app.get("/")
def root():
    return {
        "message": "AQI multi-model prediction API is running.",
        "endpoints": {
            "combined": "/predict",
            "classification": "/predict/classification",
            "regression": "/predict/regression",
        },
        "features": FEATURES,
        "classification_model": CLASS_METADATA.get("model_name"),
        "regression_model": RF_METADATA.get("model_name"),
        "preprocessing": CLASS_METADATA.get("preprocessing"),
    }


@app.post("/predict")
def predict_all(data: AQIInput):
    raw_input = build_raw_input(data)
    _, scaled_df = transform_input(raw_input)
    return {
        "input_raw": raw_input,
        "input_standardized": {
            feature: float(scaled_df.iloc[0][feature]) for feature in FEATURES
        },
        "classification": classify_from_scaled(scaled_df),
        "regression": regress_from_scaled(scaled_df),
    }


@app.post("/predict/classification")
def predict_classification(data: AQIInput):
    raw_input = build_raw_input(data)
    _, scaled_df = transform_input(raw_input)
    return classify_from_scaled(scaled_df)


@app.post("/predict/regression")
def predict_regression(data: AQIInput):
    raw_input = build_raw_input(data)
    _, scaled_df = transform_input(raw_input)
    return regress_from_scaled(scaled_df)
