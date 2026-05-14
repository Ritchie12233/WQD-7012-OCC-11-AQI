# AQI Prediction Demo

This deployment folder now contains two prediction tasks built from the same 11 raw input features:

- AQI level classification with XGBoost
- Next-hour AQI value regression with Random Forest

## Folder Structure

- `simple_xgb_classifier.joblib`: XGBoost classifier
- `deployment_11_feature_standard_scaler.joblib`: shared standardization scaler
- `metadata.json`: classification metadata
- `deploy_features.json`: feature order used in deployment
- `RF/rf_regressor_11_features.joblib`: Random Forest regressor
- `RF/metadata.json`: regression metadata
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
- pressure
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

## Deployment Note

Both models expect standardized input internally.
Raw frontend values are automatically transformed with `deployment_11_feature_standard_scaler.joblib` before prediction.
