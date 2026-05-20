from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path(
    "/Users/user/Desktop/UM_S1_2025_2026_S2/"
    "WQD7012_Applied_Machine_Learning/99_Review_Summary/"
    "Airware-Haikou/2_filled_data"
)

FEATURES_PATH = BASE_DIR / "deploy_features.json"
SCALER_PATH = BASE_DIR / "deployment_11_feature_standard_scaler.joblib"
KMEANS_MODEL_PATH = BASE_DIR / "kmeans_final.joblib"
ISOLATION_MODEL_PATH = BASE_DIR / "isolation_forest.joblib"
UNSUPERVISED_METADATA_PATH = BASE_DIR / "unsupervised_metadata.json"
CLUSTER_CENTERS_PATH = BASE_DIR / "kmeans_cluster_centers.csv"
TRAINING_SUMMARY_PATH = BASE_DIR / "unsupervised_training_summary.json"

KMEANS_FEATURES = [
    "PM2.5",
    "PM10",
    "CO",
    "NO2",
    "SO2",
    "O3",
    "pressure",
    "humidity",
    "temperature",
    "wind_speed",
]
ISOLATION_FEATURES = ["PM2.5", "PM10", "CO", "NO2", "SO2", "O3"]


def sorted_csv_files(data_dir: Path) -> list[Path]:
    files = sorted(
        data_dir.glob("*.csv"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return files


def load_training_frame(csv_files: list[Path], features: list[str]) -> pd.DataFrame:
    frames = []
    for csv_file in csv_files:
        frame = pd.read_csv(csv_file, usecols=features)
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data = data.apply(pd.to_numeric, errors="coerce")

    # Frontend users may enter pressure in hPa, while the model was trained on kPa-like values.
    # The raw filled station data is already in the same pressure scale as the deployment scaler.
    data.loc[data["pressure"] > 200, "pressure"] = data.loc[data["pressure"] > 200, "pressure"] / 10.0
    return data.dropna(subset=features)


def assign_cluster_profiles(cluster_centers: pd.DataFrame) -> dict[str, dict[str, str]]:
    pollutant_mean = cluster_centers[ISOLATION_FEATURES].mean(axis=1)
    available = set(cluster_centers.index)

    low_pressure_id = int(cluster_centers.loc[list(available), "pressure"].idxmin())
    available.remove(low_pressure_id)

    severe_score = pollutant_mean + (0.25 * cluster_centers["humidity"]) + (0.25 * cluster_centers["O3"])
    severe_id = int(severe_score.loc[list(available)].idxmax())
    available.remove(severe_id)

    clean_score = pollutant_mean + (0.30 * cluster_centers["humidity"])
    clean_id = int(clean_score.loc[list(available)].idxmin())
    available.remove(clean_id)

    baseline_id = int(next(iter(available)))

    return {
        str(clean_id): {
            "label": "Clean & Dry",
            "pattern": "Good weather scenario",
            "description": "Lower pollutant load with dry-air conditions.",
        },
        str(baseline_id): {
            "label": "Average / Baseline",
            "pattern": "Regular background scenario",
            "description": "Most variables remain close to the dataset baseline.",
        },
        str(low_pressure_id): {
            "label": "Extreme Low Pressure",
            "pattern": "Extreme low-pressure fluctuation scenario",
            "description": "Pressure is strongly below the standard training baseline while pollutants are not necessarily high.",
        },
        str(severe_id): {
            "label": "Severe Pollution / Stagnant",
            "pattern": "Heavy pollution accumulation scenario",
            "description": "Pollutant load, ozone, or humidity indicates a high-risk accumulation mode.",
        },
    }


def train_unsupervised_artifacts(data_dir: Path) -> dict[str, object]:
    features = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    missing = [feature for feature in KMEANS_FEATURES + ISOLATION_FEATURES if feature not in features]
    if missing:
        raise ValueError(f"Missing deployment features: {missing}")

    csv_files = sorted_csv_files(data_dir)
    raw_data = load_training_frame(csv_files, features)

    scaler = joblib.load(SCALER_PATH)
    scaled_data = pd.DataFrame(scaler.transform(raw_data[features]), columns=features)

    kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans_model.fit(scaled_data[KMEANS_FEATURES])

    isolation_model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )
    isolation_model.fit(scaled_data[ISOLATION_FEATURES])

    cluster_centers = pd.DataFrame(kmeans_model.cluster_centers_, columns=KMEANS_FEATURES)
    cluster_centers.to_csv(CLUSTER_CENTERS_PATH, index_label="cluster_id")

    cluster_profiles = assign_cluster_profiles(cluster_centers)

    metadata = {
        "artifact_mode": "trained_joblib",
        "kmeans": {
            "method": "K-Means clustering",
            "model_file": KMEANS_MODEL_PATH.name,
            "n_clusters": 4,
            "random_state": 42,
            "n_init": 10,
            "feature_scale": "Standardized with deployment_11_feature_standard_scaler.joblib after pressure normalization.",
            "features": KMEANS_FEATURES,
            "cluster_profiles": cluster_profiles,
        },
        "isolation_forest": {
            "method": "Isolation Forest",
            "model_file": ISOLATION_MODEL_PATH.name,
            "n_estimators": 100,
            "contamination": 0.05,
            "random_state": 42,
            "feature_scale": "Standardized with deployment_11_feature_standard_scaler.joblib after pressure normalization.",
            "features": ISOLATION_FEATURES,
        },
        "driver_view": {
            "method": "Current sub-AQI driver ranking",
            "features": ISOLATION_FEATURES,
            "note": "Driver View ranks the current raw pollutant inputs by sub-AQI contribution, while supervised and unsupervised model inputs are standardized internally.",
        },
        "training_data": {
            "source_dir": str(data_dir),
            "csv_count": len(csv_files),
            "row_count_after_dropna": int(len(raw_data)),
            "feature_count": len(features),
        },
        "deployment_note": "Runtime unsupervised outputs are generated by trained K-Means and Isolation Forest joblib artifacts, using the same scaler and feature order as the deployed supervised models.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    joblib.dump(kmeans_model, KMEANS_MODEL_PATH)
    joblib.dump(isolation_model, ISOLATION_MODEL_PATH)
    UNSUPERVISED_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    TRAINING_SUMMARY_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train deployment K-Means and Isolation Forest artifacts.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    metadata = train_unsupervised_artifacts(args.data_dir)
    print(json.dumps(metadata["training_data"], indent=2))
    print(f"Saved {KMEANS_MODEL_PATH}")
    print(f"Saved {ISOLATION_MODEL_PATH}")
    print(f"Saved {UNSUPERVISED_METADATA_PATH}")


if __name__ == "__main__":
    main()
