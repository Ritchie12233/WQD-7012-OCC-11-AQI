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
UNSUPERVISED_METADATA_PATH = BASE_DIR / "unsupervised_metadata.json"
KMEANS_MODEL_PATH = BASE_DIR / "kmeans_final.joblib"
ISOLATION_MODEL_PATH = BASE_DIR / "isolation_forest.joblib"

classifier = joblib.load(CLASSIFIER_PATH)
scaler = joblib.load(SCALER_PATH)
rf_regressor = joblib.load(RF_MODEL_PATH)
kmeans_model = joblib.load(KMEANS_MODEL_PATH)
isolation_model = joblib.load(ISOLATION_MODEL_PATH)

FEATURES = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
CLASS_METADATA = json.loads(CLASS_METADATA_PATH.read_text(encoding="utf-8"))
RF_METADATA = json.loads(RF_METADATA_PATH.read_text(encoding="utf-8"))
UNSUPERVISED_METADATA = json.loads(UNSUPERVISED_METADATA_PATH.read_text(encoding="utf-8"))
CLASSES = CLASS_METADATA["classes"]
CLASS_DEFINITION = CLASS_METADATA.get("class_definition", {})
KMEANS_FEATURES = UNSUPERVISED_METADATA["kmeans"]["features"]
ISOLATION_FEATURES = UNSUPERVISED_METADATA["isolation_forest"]["features"]
CLUSTER_PROFILES = UNSUPERVISED_METADATA["kmeans"]["cluster_profiles"]
RF_TOP_FEATURE = (
    FEATURES[int(max(range(len(FEATURES)), key=lambda index: rf_regressor.feature_importances_[index]))]
    if hasattr(rf_regressor, "feature_importances_")
    else "N/A"
)

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


def normalize_model_input(raw_input: dict[str, float]) -> dict[str, float]:
    model_input = {feature: float(raw_input[feature]) for feature in FEATURES}
    if model_input.get("pressure", 0.0) > 200:
        model_input["pressure"] = model_input["pressure"] / 10.0
    return model_input



def transform_input(raw_input: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_input = normalize_model_input(raw_input)
    model_df = pd.DataFrame([model_input], columns=FEATURES)
    scaled = scaler.transform(model_df)
    scaled_df = pd.DataFrame(scaled, columns=FEATURES)
    return model_df, scaled_df



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
            "unsupervised": "/predict/unsupervised",
        },
        "features": FEATURES,
        "classification_model": CLASS_METADATA.get("model_name"),
        "regression_model": RF_METADATA.get("model_name"),
        "unsupervised_artifact_mode": UNSUPERVISED_METADATA.get("artifact_mode"),
        "kmeans_features": KMEANS_FEATURES,
        "isolation_forest_features": ISOLATION_FEATURES,
        "preprocessing": CLASS_METADATA.get("preprocessing"),
    }


def calculate_sub_aqi(value: float, breakpoints: list[tuple[float, float, int, int]]) -> float:
    for bp_low, bp_high, aqi_low, aqi_high in breakpoints:
        if bp_low <= value <= bp_high:
            return ((aqi_high - aqi_low) / (bp_high - bp_low)) * (value - bp_low) + aqi_low
    bp_low, bp_high, aqi_low, aqi_high = breakpoints[-1]
    capped_value = min(value, bp_high)
    return ((aqi_high - aqi_low) / (bp_high - bp_low)) * (capped_value - bp_low) + aqi_low


def get_sub_aqi_scores(raw_input: dict[str, float]) -> dict[str, float]:
    pollutant_breakpoints = {
        "PM2.5": [(0.0, 12.0, 0, 50), (12.0, 35.4, 51, 100), (35.4, 55.4, 101, 150), (55.4, 150.4, 151, 200), (150.4, 250.4, 201, 300), (250.4, 500.4, 301, 500)],
        "PM10": [(0.0, 54.0, 0, 50), (54.0, 154.0, 51, 100), (154.0, 254.0, 101, 150), (254.0, 354.0, 151, 200), (354.0, 424.0, 201, 300), (424.0, 604.0, 301, 500)],
        "CO": [(0.0, 4.4, 0, 50), (4.4, 9.4, 51, 100), (9.4, 12.4, 101, 150), (12.4, 15.4, 151, 200), (15.4, 30.4, 201, 300), (30.4, 50.4, 301, 500)],
        "NO2": [(0.0, 53.0, 0, 50), (53.0, 100.0, 51, 100), (100.0, 360.0, 101, 150), (360.0, 649.0, 151, 200), (649.0, 1249.0, 201, 300), (1249.0, 2049.0, 301, 500)],
        "SO2": [(0.0, 35.0, 0, 50), (35.0, 75.0, 51, 100), (75.0, 185.0, 101, 150), (185.0, 304.0, 151, 200), (304.0, 604.0, 201, 300), (604.0, 1004.0, 301, 500)],
        "O3": [(0.0, 54.0, 0, 50), (54.0, 70.0, 51, 100), (70.0, 85.0, 101, 150), (85.0, 105.0, 151, 200), (105.0, 200.0, 201, 300)],
    }
    return {
        pollutant: calculate_sub_aqi(float(raw_input.get(pollutant, 0.0)), breakpoints)
        for pollutant, breakpoints in pollutant_breakpoints.items()
    }


def infer_kmeans_profile(scaled_df: pd.DataFrame, scaled_values: dict[str, float]) -> dict[str, object]:
    kmeans_input = scaled_df[KMEANS_FEATURES]
    cluster_id = int(kmeans_model.predict(kmeans_input)[0])
    cluster_distances = kmeans_model.transform(kmeans_input)[0]
    profile = CLUSTER_PROFILES.get(str(cluster_id), {
        "label": "Unlabelled Cluster",
        "pattern": "Model-derived cluster",
        "description": "This cluster was produced by the trained K-Means artifact.",
    })
    return {
        "cluster_id": cluster_id,
        "label": profile["label"],
        "pattern": profile["pattern"],
        "description": profile["description"],
        "distance_to_centroid": float(cluster_distances[cluster_id]),
        "cluster_distances": {
            str(index): float(distance) for index, distance in enumerate(cluster_distances)
        },
        "features_used": KMEANS_FEATURES,
        "model_file": UNSUPERVISED_METADATA["kmeans"].get("model_file", KMEANS_MODEL_PATH.name),
        "standardized_input": {feature: float(scaled_values[feature]) for feature in KMEANS_FEATURES},
    }


def infer_isolation_status(scaled_df: pd.DataFrame, scaled_values: dict[str, float]) -> dict[str, object]:
    isolation_input = scaled_df[ISOLATION_FEATURES]
    isolation_label = int(isolation_model.predict(isolation_input)[0])
    decision_score = float(isolation_model.decision_function(isolation_input)[0])
    score_sample = float(isolation_model.score_samples(isolation_input)[0])
    anomaly_score = min(max(50.0 - (decision_score * 250.0), 0.0), 100.0)

    if isolation_label == -1:
        label = "High anomaly risk"
    elif anomaly_score >= 45:
        label = "Watch zone"
    else:
        label = "Normal range"

    return {
        "label": label,
        "isolation_label": isolation_label,
        "anomaly_score": float(anomaly_score),
        "decision_score": decision_score,
        "score_sample": score_sample,
        "features_used": ISOLATION_FEATURES,
        "model_file": UNSUPERVISED_METADATA["isolation_forest"].get("model_file", ISOLATION_MODEL_PATH.name),
        "standardized_input": {feature: float(scaled_values[feature]) for feature in ISOLATION_FEATURES},
        "contamination": UNSUPERVISED_METADATA["isolation_forest"]["contamination"],
    }


def infer_unsupervised(raw_input: dict[str, float], scaled_df: pd.DataFrame) -> dict[str, object]:
    scaled_values = {feature: float(scaled_df.iloc[0][feature]) for feature in FEATURES}
    sub_scores = get_sub_aqi_scores(raw_input)
    driver = max(sub_scores, key=sub_scores.get)
    return {
        "kmeans": infer_kmeans_profile(scaled_df, scaled_values),
        "isolation_forest": infer_isolation_status(scaled_df, scaled_values),
        "driver_view": {
            "top_driver": driver,
            "current_aqi_estimate": float(sub_scores[driver]),
            "sub_aqi_scores": {feature: float(score) for feature, score in sub_scores.items()},
            "rf_top_feature": RF_TOP_FEATURE,
            "features_used": UNSUPERVISED_METADATA["driver_view"]["features"],
        },
        "data_contract": {
            "supervised_features": FEATURES,
            "kmeans_features": KMEANS_FEATURES,
            "isolation_forest_features": ISOLATION_FEATURES,
            "pressure_rule": "Pressure values above 200 are divided by 10 before standardization.",
        },
        "deployment_note": UNSUPERVISED_METADATA["deployment_note"],
    }


@app.post("/predict")
def predict_all(data: AQIInput):
    raw_input = build_raw_input(data)
    model_df, scaled_df = transform_input(raw_input)
    return {
        "input_raw": raw_input,
        "input_model_units": {
            feature: float(model_df.iloc[0][feature]) for feature in FEATURES
        },
        "input_standardized": {
            feature: float(scaled_df.iloc[0][feature]) for feature in FEATURES
        },
        "classification": classify_from_scaled(scaled_df),
        "regression": regress_from_scaled(scaled_df),
        "unsupervised": infer_unsupervised(raw_input, scaled_df),
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


@app.post("/predict/unsupervised")
def predict_unsupervised(data: AQIInput):
    raw_input = build_raw_input(data)
    _, scaled_df = transform_input(raw_input)
    return infer_unsupervised(raw_input, scaled_df)
