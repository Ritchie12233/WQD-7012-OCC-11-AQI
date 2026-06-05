# AQI Streamlit Deployment

Clean deployment package for the WQD7012 next-hour AQI dashboard.

## Included Runtime Files

- `streamlit_app.py`: Streamlit web app.
- `simple_xgb_classifier.joblib`: XGBoost AQI level classifier.
- `deployment_11_feature_standard_scaler.joblib`: shared input scaler.
- `RF/rf_regressor_11_features.joblib`: Random Forest next-hour AQI regressor.
- `kmeans_final.joblib`: K-Means pattern discovery model.
- `isolation_forest.joblib`: Isolation Forest anomaly model.
- `deploy_features.json`, `metadata.json`, `RF/metadata.json`, `unsupervised_metadata.json`: feature order and model metadata.
- `.streamlit/config.toml`: Streamlit Cloud-friendly server config.

## Removed From This Clean Copy

The FastAPI server, API scripts, training script, request sample, logs, PID files, cache files, and unused K-Means centroid CSV were intentionally left out because the deployed deliverable is the Streamlit dashboard.

## Run Locally

```bash
pip install -r requirements.txt
bash start_web.sh
```

Open:

```text
http://127.0.0.1:8501
```

## Deployment Notes

The app accepts raw frontend inputs. Pressure values in hPa scale, such as `1012.5`, are normalized to the model's kPa-scale training input before standardization.

Demo buttons are presentation anchors: `Low Demo`, `Moderate Demo`, and `High Demo` display their matching level and theme color. Custom input uses the classifier output for the display label.
