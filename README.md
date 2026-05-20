# AQI Prediction Demo

This deployment folder now contains two prediction tasks built from the same 11 raw input features:

- AQI level classification with XGBoost
- Next-hour AQI value regression with Random Forest
- K-Means pollution pattern clustering loaded from `kmeans_final.joblib`
- Isolation Forest anomaly detection loaded from `isolation_forest.joblib`
- Driver View based on current pollutant sub-AQI contribution

## Folder Structure

- `simple_xgb_classifier.joblib`: XGBoost classifier
- `deployment_11_feature_standard_scaler.joblib`: shared standardization scaler
- `metadata.json`: classification metadata
- `deploy_features.json`: feature order used in deployment
- `RF/rf_regressor_11_features.joblib`: Random Forest regressor
- `RF/metadata.json`: regression metadata
- `kmeans_final.joblib`: trained K-Means artifact
- `isolation_forest.joblib`: trained Isolation Forest artifact
- `kmeans_cluster_centers.csv`: standardized K-Means centroids
- `train_unsupervised_artifacts.py`: reproducible training script for the unsupervised artifacts
- `unsupervised_metadata.json`: K-Means, Isolation Forest, and Driver View metadata
- `app.py`: FastAPI deployment entry
- `streamlit_app.py`: Streamlit demo page

## Input Features

The two models use the same 11 raw inputs:

- PM2.5
- PM10
- CO
- NO2
- SO2
- O3
- pressure, accepted as kPa-scale values around 100 or hPa-scale values around 1000
- humidity
- temperature
- wind_direction
- wind_speed

## Outputs

### Classification

- `Low`: AQI_next <= 50
- `Moderate`: 50 < AQI_next <= 100
- `High`: AQI_next > 100

### Regression

- Predicts the next-hour AQI numeric value directly

### Unsupervised and Driver View

- K-Means uses the trained `kmeans_final.joblib` artifact with standardized PM2.5, PM10, CO, NO2, SO2, O3, pressure, humidity, temperature, and wind_speed.
- Isolation Forest uses the trained `isolation_forest.joblib` artifact with standardized PM2.5, PM10, CO, NO2, SO2, and O3.
- Driver View ranks raw pollutant inputs by current sub-AQI contribution.

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Streamlit:

```bash
bash start_web.sh
```

Open the page:

```text
http://localhost:8501
```

Start API:

```bash
bash start_api.sh
```

API docs:

```text
http://localhost:8000/docs
```

## API Endpoints

- `POST /predict`: returns classification + regression together
- `POST /predict/classification`: returns AQI level prediction
- `POST /predict/regression`: returns next-hour AQI value prediction
- `POST /predict/unsupervised`: returns K-Means, Isolation Forest, and Driver View outputs

## Deployment Note

Both models expect standardized input internally.
Raw frontend values are automatically transformed with `deployment_11_feature_standard_scaler.joblib` before prediction.
If pressure is entered in hPa-scale form, such as 1012.5, the app converts it to the model's kPa-scale form, such as 101.25, before standardization.
The deployment folder now includes trained K-Means and Isolation Forest joblib files.
They were trained from the 95 filled station CSV files in `2_filled_data`, after transforming raw inputs with the same 11-feature deployment scaler used by the supervised models.
