from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
CLASSIFIER_PATH = BASE_DIR / "simple_xgb_classifier.joblib"
SCALER_PATH = BASE_DIR / "deployment_11_feature_standard_scaler.joblib"
FEATURES_PATH = BASE_DIR / "deploy_features.json"
CLASS_METADATA_PATH = BASE_DIR / "metadata.json"
RF_DIR = BASE_DIR / "RF"
RF_MODEL_PATH = RF_DIR / "rf_regressor_11_features.joblib"
RF_METADATA_PATH = RF_DIR / "metadata.json"


@st.cache_resource
def load_artifacts():
    classifier = joblib.load(CLASSIFIER_PATH)
    scaler = joblib.load(SCALER_PATH)
    rf_regressor = joblib.load(RF_MODEL_PATH)
    features = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    class_metadata = json.loads(CLASS_METADATA_PATH.read_text(encoding="utf-8"))
    rf_metadata = json.loads(RF_METADATA_PATH.read_text(encoding="utf-8"))
    return classifier, scaler, rf_regressor, features, class_metadata, rf_metadata


classifier, scaler, rf_regressor, FEATURES, CLASS_METADATA, RF_METADATA = load_artifacts()
CLASSES = CLASS_METADATA["classes"]
CLASS_DEFINITION = CLASS_METADATA.get("class_definition", {})
RF_TOP_FEATURE = (
    FEATURES[int(max(range(len(FEATURES)), key=lambda index: rf_regressor.feature_importances_[index]))]
    if hasattr(rf_regressor, "feature_importances_")
    else "N/A"
)

DEFAULT_VALUES = {
    "PM2.5": 12.50,
    "PM10": 30.20,
    "CO": 0.60,
    "NO2": 18.00,
    "SO2": 3.20,
    "O3": 45.10,
    "pressure": 1012.50,
    "humidity": 68.00,
    "temperature": 28.50,
    "wind_direction": 180.00,
    "wind_speed": 2.40,
}

LOW_DEMO_VALUES = {
    "PM2.5": 6.00,
    "PM10": 18.00,
    "CO": 0.20,
    "NO2": 8.00,
    "SO2": 2.00,
    "O3": 28.00,
    "pressure": 1018.00,
    "humidity": 42.00,
    "temperature": 24.00,
    "wind_direction": 60.00,
    "wind_speed": 4.50,
}

MODERATE_DEMO_VALUES = {
    "PM2.5": 45.00,
    "PM10": 88.00,
    "CO": 1.20,
    "NO2": 35.00,
    "SO2": 10.00,
    "O3": 82.00,
    "pressure": 1008.00,
    "humidity": 58.00,
    "temperature": 29.00,
    "wind_direction": 145.00,
    "wind_speed": 2.10,
}

HIGH_DEMO_VALUES = {
    "PM2.5": 135.00,
    "PM10": 220.00,
    "CO": 4.60,
    "NO2": 120.00,
    "SO2": 68.00,
    "O3": 168.00,
    "pressure": 998.00,
    "humidity": 80.00,
    "temperature": 36.00,
    "wind_direction": 210.00,
    "wind_speed": 0.80,
}

DEMO_TO_LABEL = {
    "Low Demo": "Low",
    "Moderate Demo": "Moderate",
    "High Demo": "High",
}

DEMO_PRESETS = {
    "Low Demo": LOW_DEMO_VALUES,
    "Moderate Demo": MODERATE_DEMO_VALUES,
    "High Demo": HIGH_DEMO_VALUES,
}

THEMES = {
    "default": {"body_class": "theme-default", "result_class": "result-default", "accent": "Air Quality Preview"},
    "Low": {"body_class": "theme-low", "result_class": "result-low", "accent": "Clear Sky"},
    "Moderate": {"body_class": "theme-moderate", "result_class": "result-moderate", "accent": "Soft Haze"},
    "High": {"body_class": "theme-high", "result_class": "result-high", "accent": "Smog Warning"},
}

THEME_PALETTES = {
    "default": {
        "primary": "#7A9DB8",
        "secondary": "#A3BFD1",
        "soft": "#F4F7FA",
        "accent": "#6B8FA8",
        "decor": "#B8CFDE",
        "highlight": "#3D5D74",
        "ink": "#1E2F3A",
        "shadow": "61,93,116",
        "surface": "linear-gradient(135deg, rgba(244,247,250,0.68) 0%, rgba(163,191,209,0.44) 100%)",
        "surface_strong": "linear-gradient(135deg, rgba(122,157,184,0.64) 0%, rgba(163,191,209,0.58) 100%)",
        "page_bg": "radial-gradient(circle at 16% 14%, rgba(122,157,184,0.16) 0%, transparent 36%), radial-gradient(circle at 84% 16%, rgba(107,143,168,0.12) 0%, transparent 34%), radial-gradient(circle at 50% 92%, rgba(163,191,209,0.10) 0%, transparent 40%), linear-gradient(135deg, #F4F7FA 0%, #EBF1F5 48%, #F2F5F8 100%)",
    },
    "Low": {
        "primary": "#6B9E85",
        "secondary": "#93BAA6",
        "soft": "#F4F8F5",
        "accent": "#5A8D72",
        "decor": "#A8CCB8",
        "highlight": "#3D6B54",
        "ink": "#1F3529",
        "shadow": "61,107,84",
        "surface": "linear-gradient(135deg, rgba(244,248,245,0.68) 0%, rgba(147,186,166,0.44) 100%)",
        "surface_strong": "linear-gradient(135deg, rgba(107,158,133,0.64) 0%, rgba(147,186,166,0.58) 100%)",
        "page_bg": "radial-gradient(circle at 16% 14%, rgba(107,158,133,0.14) 0%, transparent 36%), radial-gradient(circle at 84% 16%, rgba(90,141,114,0.10) 0%, transparent 34%), radial-gradient(circle at 50% 92%, rgba(147,186,166,0.08) 0%, transparent 40%), linear-gradient(135deg, #F4F8F5 0%, #EAF2ED 48%, #F1F6F3 100%)",
    },
    "Moderate": {
        "primary": "#9EAAB5",
        "secondary": "#BCC4CD",
        "soft": "#F5F6F7",
        "accent": "#8B97A3",
        "decor": "#C8CFD6",
        "highlight": "#5A6875",
        "ink": "#2B3239",
        "shadow": "90,104,117",
        "surface": "linear-gradient(135deg, rgba(245,246,247,0.68) 0%, rgba(188,196,205,0.44) 100%)",
        "surface_strong": "linear-gradient(135deg, rgba(158,170,181,0.64) 0%, rgba(188,196,205,0.58) 100%)",
        "page_bg": "radial-gradient(circle at 16% 14%, rgba(158,170,181,0.14) 0%, transparent 36%), radial-gradient(circle at 84% 16%, rgba(139,151,163,0.10) 0%, transparent 34%), radial-gradient(circle at 50% 92%, rgba(188,196,205,0.08) 0%, transparent 40%), linear-gradient(135deg, #F5F6F7 0%, #EDEFF2 48%, #F2F4F5 100%)",
    },
    "High": {
        "primary": "#B8956E",
        "secondary": "#CCB098",
        "soft": "#F9F6F2",
        "accent": "#A07852",
        "decor": "#D4BFA8",
        "highlight": "#6B4D31",
        "ink": "#352A1F",
        "shadow": "107,77,49",
        "surface": "linear-gradient(135deg, rgba(249,246,242,0.68) 0%, rgba(204,176,152,0.44) 100%)",
        "surface_strong": "linear-gradient(135deg, rgba(184,149,110,0.64) 0%, rgba(204,176,152,0.58) 100%)",
        "page_bg": "radial-gradient(circle at 16% 14%, rgba(184,149,110,0.14) 0%, transparent 36%), radial-gradient(circle at 84% 16%, rgba(160,120,82,0.10) 0%, transparent 34%), radial-gradient(circle at 50% 92%, rgba(204,176,152,0.08) 0%, transparent 40%), linear-gradient(135deg, #F9F6F2 0%, #F2ECE4 48%, #F6F3ED 100%)",
    },
}


PROBABILITY_COLORS = {"Low": "#6B9E85", "Moderate": "#9EAAB5", "High": "#B8956E"}

GRADE_REFERENCE = [
    ("Grade 1", "Low", "AQI 0-50"),
    ("Grade 2", "Moderate", "AQI 51-100"),
    ("Grade 3", "High", "AQI 101+"),
]

FEATURE_UNITS = {
    "PM2.5": "µg/m³",
    "PM10": "µg/m³",
    "CO": "ppm",
    "NO2": "ppb",
    "SO2": "ppb",
    "O3": "ppb",
    "pressure": "hPa",
    "humidity": "%",
    "temperature": "°C",
    "wind_direction": "°",
    "wind_speed": "m/s",
}

LEVEL_ICONS = {
    "Low": "<svg viewBox='0 0 84 84' class='level-icon-svg' aria-hidden='true'><circle cx='42' cy='42' r='38' fill='rgba(255,255,255,0.22)' /><circle cx='25' cy='23' r='7' fill='#fff6b8'/><path d='M12 56 C22 44, 28 44, 39 56 Z' fill='#3ca66f'/><path d='M31 56 C42 38, 51 38, 63 56 Z' fill='#4ab37d'/><path d='M8 61 C24 56, 54 56, 75 62 L75 69 L8 69 Z' fill='#72d4f2'/></svg>",
    "Moderate": "<svg viewBox='0 0 84 84' class='level-icon-svg' aria-hidden='true'><circle cx='42' cy='42' r='38' fill='rgba(255,255,255,0.20)' /><rect x='18' y='49' width='28' height='7' rx='3.5' fill='#ffffff'/><rect x='46' y='50' width='10' height='5' rx='2.5' fill='#d8a36a'/><path d='M56 48 C62 44, 64 38, 60 34 C68 31, 67 24, 61 21' fill='none' stroke='#eef4f8' stroke-width='4' stroke-linecap='round'/><path d='M48 45 C53 40, 53 34, 48 30 C56 26, 56 19, 50 15' fill='none' stroke='#dce5eb' stroke-width='4' stroke-linecap='round' opacity='0.9'/></svg>",
    "High": "<svg viewBox='0 0 84 84' class='level-icon-svg' aria-hidden='true'><circle cx='42' cy='42' r='38' fill='rgba(255,255,255,0.16)' /><path d='M16 60 C25 44, 30 37, 42 24 C52 37, 58 44, 68 60 Z' fill='#7a5443'/><path d='M34 47 C37 42, 39 37, 42 31 C45 37, 47 42, 50 47 Z' fill='#311814'/><path d='M40 31 C44 24, 48 21, 50 15 C55 21, 56 29, 52 35 C58 34, 61 40, 58 45 C51 43, 44 39, 40 31 Z' fill='#e7ddd4' opacity='0.95'/><path d='M39 37 C41 34, 42 31, 43 28 C45 31, 47 33, 48 36 C47 39, 44 41, 41 42 C39 41, 38 39, 39 37 Z' fill='#ff925c'/></svg>",
}


def apply_preset(values: dict[str, float], name: str) -> None:
    for feature, value in values.items():
        st.session_state[f"feature_{feature}"] = round(float(value), 2)
    st.session_state.active_demo = name


def detect_input_mode(payload: dict[str, float]) -> str:
    for name, values in DEMO_PRESETS.items():
        if all(abs(float(payload[feature]) - float(values[feature])) < 0.005 for feature in FEATURES):
            return name
    return "Custom Input"


def transform_payload(payload: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = pd.DataFrame([payload])[FEATURES]
    scaled = scaler.transform(raw_df)
    scaled_df = pd.DataFrame(scaled, columns=FEATURES)
    return raw_df, scaled_df


def classify_and_regress(payload: dict[str, float]) -> dict[str, object]:
    _, scaled_df = transform_payload(payload)
    pred_id = int(classifier.predict(scaled_df)[0])
    pred_label = CLASSES[str(pred_id)]
    probabilities = classifier.predict_proba(scaled_df)[0]
    reg_value = float(rf_regressor.predict(scaled_df)[0])
    return {
        "prediction_label": pred_label,
        "prediction_class_id": pred_id,
        "probabilities": {CLASSES[str(i)]: float(probabilities[i]) for i in range(len(probabilities))},
        "predicted_aqi_value": reg_value,
        "class_definition": CLASS_DEFINITION.get(pred_label, ""),
    }


def render_level_icon(label: str) -> str:
    return LEVEL_ICONS.get(label, "")


def format_aqi_band(value: float) -> str:
    if value <= 50:
        return "Grade 1 · Low"
    if value <= 100:
        return "Grade 2 · Moderate"
    return "Grade 3 · High"


def label_from_aqi_value(value: float) -> str:
    if value <= 50:
        return "Low"
    if value <= 100:
        return "Moderate"
    return "High"


def calculate_sub_aqi(value: float, breakpoints: list[tuple[float, float, int, int]]) -> float:
    for bp_low, bp_high, aqi_low, aqi_high in breakpoints:
        if bp_low <= value <= bp_high:
            return ((aqi_high - aqi_low) / (bp_high - bp_low)) * (value - bp_low) + aqi_low
    bp_low, bp_high, aqi_low, aqi_high = breakpoints[-1]
    capped_value = min(value, bp_high)
    return ((aqi_high - aqi_low) / (bp_high - bp_low)) * (capped_value - bp_low) + aqi_low


def get_sub_aqi_scores(payload: dict[str, float]) -> dict[str, float]:
    pollutant_breakpoints = {
        "PM2.5": [(0.0, 12.0, 0, 50), (12.0, 35.4, 51, 100), (35.4, 55.4, 101, 150), (55.4, 150.4, 151, 200), (150.4, 250.4, 201, 300), (250.4, 500.4, 301, 500)],
        "PM10": [(0.0, 54.0, 0, 50), (54.0, 154.0, 51, 100), (154.0, 254.0, 101, 150), (254.0, 354.0, 151, 200), (354.0, 424.0, 201, 300), (424.0, 604.0, 301, 500)],
        "CO":    [(0.0, 4.4, 0, 50), (4.4, 9.4, 51, 100), (9.4, 12.4, 101, 150), (12.4, 15.4, 151, 200), (15.4, 30.4, 201, 300), (30.4, 50.4, 301, 500)],
        "NO2":   [(0.0, 53.0, 0, 50), (53.0, 100.0, 51, 100), (100.0, 360.0, 101, 150), (360.0, 649.0, 151, 200), (649.0, 1249.0, 201, 300), (1249.0, 2049.0, 301, 500)],
        "SO2":   [(0.0, 35.0, 0, 50), (35.0, 75.0, 51, 100), (75.0, 185.0, 101, 150), (185.0, 304.0, 151, 200), (304.0, 604.0, 201, 300), (604.0, 1004.0, 301, 500)],
        "O3":    [(0.0, 54.0, 0, 50), (54.0, 70.0, 51, 100), (70.0, 85.0, 101, 150), (85.0, 105.0, 151, 200), (105.0, 200.0, 201, 300)],
    }
    sub_scores = {
        pollutant: calculate_sub_aqi(float(payload.get(pollutant, 0.0)), breakpoints)
        for pollutant, breakpoints in pollutant_breakpoints.items()
    }
    return sub_scores


def estimate_current_aqi(payload: dict[str, float]) -> tuple[float, str]:
    sub_scores = get_sub_aqi_scores(payload)
    driver = max(sub_scores, key=sub_scores.get)
    return float(sub_scores[driver]), driver


def render_aqi_trend_card(current_aqi: float, predicted_aqi: float, driver: str, rf_driver: str, band: str) -> str:
    delta = predicted_aqi - current_aqi
    change_pct = abs(delta) / max(current_aqi, 1.0) * 100
    if delta > 5:
        trend_label = "Rising"
        trend_note = "Air quality may get worse next hour."
        trend_class = "trend-rising"
        arrow_path = "M42 12 L70 40 H55 V72 H29 V40 H14 Z"
    elif delta < -5:
        trend_label = "Falling"
        trend_note = "Air quality may improve next hour."
        trend_class = "trend-falling"
        arrow_path = "M42 72 L14 44 H29 V12 H55 V44 H70 Z"
    else:
        trend_label = "Stable"
        trend_note = "Air quality is expected to stay similar."
        trend_class = "trend-stable"
        arrow_path = "M12 34 H53 V22 L74 42 L53 62 V50 H12 Z"

    return dedent(
        f"""
        <div class='glass-card aqi-compare-card method-block'>
            <div class='aqi-compare-head'>
                <div>
                    <div class='method-label'>Method 2</div>
                    <div class='aqi-hero-kicker'>Random Forest Regression</div>
                    <div class='aqi-compare-title'>RF next-hour AQI value</div>
                </div>
                <div class='trend-pill {trend_class}'>{trend_label} · {change_pct:.1f}%</div>
            </div>
            <div class='aqi-compare-grid'>
                <div class='aqi-value-panel current-panel'>
                    <div class='aqi-panel-label'>Current AQI Estimate</div>
                    <div class='aqi-panel-value'>{current_aqi:.2f}</div>
                    <div class='aqi-panel-note'>Highest current sub-AQI<br><span class='driver-pill'>{driver}</span></div>
                </div>
                <div class='aqi-arrow {trend_class}'>
                    <div class='aqi-arrow-spacer'></div>
                    <svg viewBox='0 0 84 84' class='trend-arrow-svg' aria-hidden='true'>
                        <path d='{arrow_path}' />
                    </svg>
                    <div class='trend-arrow-pct'>{change_pct:.1f}%</div>
                </div>
                <div class='aqi-value-panel next-panel'>
                    <div class='aqi-panel-label'>RF Next-hour AQI</div>
                    <div class='aqi-panel-value'>{predicted_aqi:.2f}</div>
                    <div class='aqi-panel-note'>RF key feature<br><span class='driver-pill rf-pill'>{rf_driver}</span></div>
                    <div class='aqi-panel-band'>{band}</div>
                </div>
            </div>
            <div class='aqi-compare-note'>{trend_note}</div>
        </div>
        """
    ).strip()


def render_probability_mini(probabilities: dict[str, float], predicted_label: str) -> str:
    low = float(probabilities.get("Low", 0.0)) * 100
    moderate = float(probabilities.get("Moderate", 0.0)) * 100
    high = float(probabilities.get("High", 0.0)) * 100
    top_value = float(probabilities.get(predicted_label, 0.0)) * 100
    low_offset = 0
    moderate_offset = -low
    high_offset = -(low + moderate)
    return dedent(
        f"""
        <div class='donut-wrap'>
            <div class='prob-donut' aria-label='XGBoost class probability donut chart'>
                <svg viewBox='0 0 120 120' class='donut-svg' aria-hidden='true'>
                    <circle cx='60' cy='60' r='46' fill='none' stroke='rgba(255,255,255,0.16)' stroke-width='22' />
                    <circle class='donut-track' cx='60' cy='60' r='44' pathLength='100' />
                    <circle class='donut-segment donut-low' cx='60' cy='60' r='44' pathLength='100' stroke-dasharray='{max(low - 0.4, 0.05):.2f} {100 - low + 0.4:.2f}' stroke-dashoffset='{low_offset:.2f}' />
                    <circle class='donut-segment donut-moderate' cx='60' cy='60' r='44' pathLength='100' stroke-dasharray='{max(moderate - 0.4, 0.05):.2f} {100 - moderate + 0.4:.2f}' stroke-dashoffset='{moderate_offset:.2f}' />
                    <circle class='donut-segment donut-high' cx='60' cy='60' r='44' pathLength='100' stroke-dasharray='{max(high - 0.4, 0.05):.2f} {100 - high + 0.4:.2f}' stroke-dashoffset='{high_offset:.2f}' />
                </svg>
                <div class='donut-center'>
                    <div class='donut-value'>{top_value:.1f}%</div>
                </div>
            </div>
            <div class='donut-legend'>
                <div><span style='background:#7BC8A4;'></span>Low <strong>{low:.1f}%</strong></div>
                <div><span style='background:#C8B898;'></span>Moderate <strong>{moderate:.1f}%</strong></div>
                <div><span style='background:#D49880;'></span>High <strong>{high:.1f}%</strong></div>
            </div>
        </div>
        """
    ).strip()


def infer_unsupervised_preview(payload: dict[str, float], current_aqi: float, driver: str) -> dict[str, object]:
    particle_load = float(payload["PM2.5"]) + float(payload["PM10"])
    gas_load = float(payload["CO"]) + float(payload["NO2"]) + float(payload["SO2"])
    ozone_value = float(payload["O3"])
    max_load = max(particle_load, gas_load * 4, ozone_value, 1.0)
    particle_score = min(particle_load / max_load * 100, 100)
    gas_score = min((gas_load * 4) / max_load * 100, 100)
    ozone_score = min(ozone_value / max_load * 100, 100)
    anomaly_score = min(current_aqi / 180 * 100, 100)

    if current_aqi > 150:
        anomaly = "High anomaly risk"
        anomaly_note = "Extreme pollution-like input"
        anomaly_color = "#D49880"
        anomaly_level = "HIGH"
    elif current_aqi > 100:
        anomaly = "Watch zone"
        anomaly_note = "Elevated but not extreme"
        anomaly_color = "#E8A860"
        anomaly_level = "WATCH"
    else:
        anomaly = "Normal range"
        anomaly_note = "No strong anomaly signal"
        anomaly_color = "#7BC8A4"
        anomaly_level = "NORMAL"

    if particle_load >= max(gas_load * 8, ozone_value):
        cluster = "Cluster A"
        pattern = "Particle-heavy pattern"
    elif ozone_value >= max(particle_load, gas_load * 6):
        cluster = "Cluster B"
        pattern = "Ozone / photochemical pattern"
    elif gas_load > 45:
        cluster = "Cluster C"
        pattern = "Combustion gas pattern"
    else:
        cluster = "Cluster D"
        pattern = "Background air pattern"

    return {
        "cluster": cluster,
        "pattern": pattern,
        "anomaly": anomaly,
        "anomaly_note": anomaly_note,
        "anomaly_color": anomaly_color,
        "anomaly_level": anomaly_level,
        "driver": driver,
        "particle_score": particle_score,
        "gas_score": gas_score,
        "ozone_score": ozone_score,
        "anomaly_score": anomaly_score,
    }


def render_load_bar(name: str, value: float, color: str) -> str:
    return (
        f"<div class='unsup-viz-row'>"
        f"<div class='unsup-viz-head'><span>{name}</span><strong>{value:.0f}</strong></div>"
        f"<div class='unsup-track'><div class='unsup-fill' style='width:{value:.1f}%; background:{color};'></div></div>"
        f"</div>"
    )


def render_cluster_map(preview: dict[str, object]) -> str:
    particle = float(preview["particle_score"])
    gas = float(preview["gas_score"])
    ozone = float(preview["ozone_score"])
    active_index = max(range(3), key=[particle, gas, ozone].__getitem__)
    points = [
        ("Particles", 24, 68, particle, "#7BC8A4"),
        ("Gas mix", 74, 42, gas, "#C8B898"),
        ("Ozone", 46, 20, ozone, "#D49880"),
    ]
    bubbles = []
    labels = []
    for index, (name, cx, cy, score, color) in enumerate(points):
        radius = 10 + score * 0.13
        active = "cluster-active" if index == active_index else ""
        bubbles.append(
            f"<circle class='cluster-dot {active}' cx='{cx}' cy='{cy}' r='{radius:.1f}' fill='{color}' />"
        )
        labels.append(
            f"<div class='cluster-legend-item'><span style='background:{color};'></span>{name}<strong>{score:.0f}</strong></div>"
        )
    return (
        "<div class='cluster-map'>"
        "<svg viewBox='0 0 100 86' class='cluster-svg' aria-hidden='true'>"
        "<path d='M18 70 C35 18, 62 10, 82 44' fill='none' stroke='rgba(255,255,255,0.68)' stroke-width='3' stroke-linecap='round'/>"
        + "".join(bubbles)
        + "</svg>"
        f"<div class='cluster-legend'>{''.join(labels)}</div>"
        "</div>"
    )


def render_driver_bars(sub_scores: dict[str, float]) -> str:
    top_scores = sorted(sub_scores.items(), key=lambda item: item[1], reverse=True)[:4]
    max_score = max([score for _, score in top_scores] + [1.0])
    rows = []
    for index, (feature, score) in enumerate(top_scores):
        color = "var(--theme-highlight)" if index == 0 else "#8A8A8A"
        rows.append(
            f"<div class='driver-row {'top-driver' if index == 0 else ''}'>"
            f"<div class='driver-row-head'><span>{feature}</span><strong>{score:.1f}</strong></div>"
            f"<div class='driver-track'><div class='driver-fill' style='width:{score / max_score * 100:.1f}%; background:{color};'></div></div>"
            f"</div>"
        )
    return "".join(rows)


def render_unsupervised_section(payload: dict[str, float], current_aqi: float, driver: str, sub_scores: dict[str, float]) -> str:
    preview = infer_unsupervised_preview(payload, current_aqi, driver)
    cluster_map = render_cluster_map(preview)
    driver_bars = render_driver_bars(sub_scores)
    anomaly_score = float(preview["anomaly_score"])
    cluster_title = f'{preview["cluster"]} ({preview["pattern"]})'
    return dedent(
        f"""
        <div class='section-shell unsup-shell'>
            <div class='method-zone-head'>
                <div>
                    <div class='section-eyebrow'>Unsupervised Learning</div>
                    <div class='section-heading'>Pattern Discovery Dashboard</div>
                </div>
                <div class='method-zone-tag'>Method 3 + Method 4</div>
            </div>
            <div class='section-subtitle'>Looks for pollution patterns and unusual input profiles without using target labels.</div>
            <div class='unsup-dashboard'>
                <div class='unsup-card cluster-card'>
                    <div class='unsup-method'>Method 3 · K-Means Clustering</div>
                    <div class='unsup-main'>{cluster_title}</div>
                    <div class='unsup-label'>Pollution profile cluster</div>
                    <div class='unsup-visual'>{cluster_map}</div>
                </div>
                <div class='unsup-card anomaly-card'>
                    <div class='unsup-method'>Method 4 · Isolation Forest</div>
                    <div class='unsup-main'>{preview["anomaly"]}</div>
                    <div class='unsup-label'>{preview["anomaly_note"]}</div>
                    <div class='risk-gauge' style='--risk:{anomaly_score:.1f}%; --anomaly-color:{preview["anomaly_color"]};'>
                        <div class='risk-gauge-center'><span class='risk-gauge-pct'>{anomaly_score:.0f}%</span><span class='risk-gauge-lvl'>{preview["anomaly_level"]}</span></div>
                    </div>
                    <div class='risk-legend'>
                        <div class='risk-legend-item'><span style='background:#7BC8A4;'></span>0&ndash;33%</div>
                        <div class='risk-legend-item'><span style='background:#E8A860;'></span>33&ndash;66%</div>
                        <div class='risk-legend-item'><span style='background:#D49880;'></span>66&ndash;100%</div>
                    </div>
                </div>
                <div class='unsup-card explain-card'>
                    <div class='unsup-method'>Driver View</div>
                    <div class='unsup-main'>{preview["driver"]}</div>
                    <div class='unsup-label'>Highest current sub-AQI</div>
                    <div class='driver-stack'>{driver_bars}</div>
                </div>
            </div>
        </div>
        """
    ).strip()


def render_background_art() -> str:
    return """
    <div class='bg-art bg-art-primary'>
        <svg viewBox='0 0 900 900' class='bg-scene-svg' aria-hidden='true'>
            <defs>
                <linearGradient id='skyGlow' x1='0' y1='0' x2='1' y2='1'>
                    <stop offset='0%' stop-color='#dff7ff'/>
                    <stop offset='50%' stop-color='#bfeefe'/>
                    <stop offset='100%' stop-color='#eaf9ff'/>
                </linearGradient>
            </defs>
            <circle cx='450' cy='420' r='300' fill='url(#skyGlow)' opacity='0.95'/>
            <circle cx='450' cy='420' r='210' fill='none' stroke='#8fdcf3' stroke-width='18' opacity='0.34'/>
            <circle cx='450' cy='420' r='160' fill='none' stroke='#c8eff7' stroke-width='10' opacity='0.42'/>
            <rect x='388' y='244' width='124' height='288' rx='42' fill='#5dbfe8' opacity='0.68'/>
            <rect x='408' y='278' width='84' height='118' rx='24' fill='#ecfbff' opacity='0.92'/>
            <circle cx='450' cy='456' r='32' fill='#effcff' opacity='0.94'/>
            <circle cx='450' cy='456' r='16' fill='#8dddf2'/>
            <path d='M124 570 C218 508, 328 500, 436 540' fill='none' stroke='#66cae4' stroke-width='14' stroke-linecap='round' opacity='0.58'/>
            <path d='M164 620 C278 558, 386 554, 488 594' fill='none' stroke='#95dde9' stroke-width='10' stroke-linecap='round' opacity='0.52'/>
            <path d='M98 738 C230 660, 366 654, 534 712 C648 752, 734 754, 822 726 L822 828 L98 828 Z' fill='#dff9f4' opacity='0.78'/>
        </svg>
    </div>
    """


st.set_page_config(page_title="Next-hour AQI Prediction", page_icon="🌿", layout="wide")

if "active_demo" not in st.session_state:
    st.session_state.active_demo = "High Demo"

for feature, default_value in DEFAULT_VALUES.items():
    st.session_state.setdefault(f"feature_{feature}", round(float(default_value), 2))

current_payload = {feature: float(st.session_state[f"feature_{feature}"]) for feature in FEATURES}
st.session_state.active_demo = detect_input_mode(current_payload)
live_result = classify_and_regress(current_payload)
current_label = str(live_result["prediction_label"])
theme = THEMES.get(current_label, THEMES["default"])
palette = THEME_PALETTES.get(current_label, THEME_PALETTES["default"])

st.markdown(
    f"""
    <style>
    .stApp, .stApp * {{ font-family: 'Avenir Next', 'Trebuchet MS', 'Segoe UI', sans-serif; }}
    .stApp.theme-default {{ --theme-primary:#7A9DB8; --theme-secondary:#A3BFD1; --theme-soft:#F4F7FA; --theme-accent:#6B8FA8; --theme-ink:#1E2F3A; --theme-shadow:61,93,116; --theme-surface:rgba(244,247,250,0.68); --theme-surface-strong:rgba(122,157,184,0.64); --page-bg: radial-gradient(circle at 18% 18%, rgba(122,157,184,0.16) 0%, transparent 34%), radial-gradient(circle at 84% 20%, rgba(107,143,168,0.12) 0%, transparent 32%), linear-gradient(135deg, #F4F7FA 0%, #EBF1F5 46%, #F2F5F8 100%); background: var(--page-bg); }}
    .stApp.theme-low {{ --theme-primary:#6B9E85; --theme-secondary:#93BAA6; --theme-soft:#F4F8F5; --theme-accent:#5A8D72; --theme-ink:#1F3529; --theme-shadow:61,107,84; --theme-surface:rgba(244,248,245,0.68); --theme-surface-strong:rgba(107,158,133,0.64); --page-bg: radial-gradient(circle at 18% 18%, rgba(107,158,133,0.14) 0%, transparent 34%), radial-gradient(circle at 84% 20%, rgba(90,141,114,0.10) 0%, transparent 32%), linear-gradient(135deg, #F4F8F5 0%, #EAF2ED 46%, #F1F6F3 100%); background: var(--page-bg); }}
    .stApp.theme-moderate {{ --theme-primary:#9EAAB5; --theme-secondary:#BCC4CD; --theme-soft:#F5F6F7; --theme-accent:#8B97A3; --theme-ink:#2B3239; --theme-shadow:90,104,117; --theme-surface:rgba(245,246,247,0.68); --theme-surface-strong:rgba(158,170,181,0.64); --page-bg: radial-gradient(circle at 18% 18%, rgba(158,170,181,0.14) 0%, transparent 34%), radial-gradient(circle at 84% 20%, rgba(139,151,163,0.10) 0%, transparent 32%), linear-gradient(135deg, #F5F6F7 0%, #EDEFF2 48%, #F2F4F5 100%); background: var(--page-bg); }}
    .stApp.theme-high {{ --theme-primary:#B8956E; --theme-secondary:#CCB098; --theme-soft:#F9F6F2; --theme-accent:#A07852; --theme-ink:#352A1F; --theme-shadow:107,77,49; --theme-surface:rgba(249,246,242,0.68); --theme-surface-strong:rgba(184,149,110,0.64); --page-bg: radial-gradient(circle at 18% 18%, rgba(184,149,110,0.14) 0%, transparent 34%), radial-gradient(circle at 84% 20%, rgba(160,120,82,0.10) 0%, transparent 32%), linear-gradient(135deg, #F9F6F2 0%, #F2ECE4 48%, #F6F3ED 100%); background: var(--page-bg); }}
    .stApp, .stApp.theme-default, .stApp.theme-low, .stApp.theme-moderate, .stApp.theme-high {{ min-height:100vh; --theme-primary:{palette["primary"]}; --theme-secondary:{palette["secondary"]}; --theme-soft:{palette["soft"]}; --theme-accent:{palette["accent"]}; --theme-decor:{palette["decor"]}; --theme-highlight:{palette["highlight"]}; --theme-ink:{palette["ink"]}; --theme-shadow:{palette["shadow"]}; --theme-surface:{palette["surface"]}; --theme-surface-strong:{palette["surface_strong"]}; --page-bg:{palette["page_bg"]}; --glass-blur:22px; --glass-border:rgba(255,255,255,0.46); --glass-inner:inset 0 1px 0 rgba(255,255,255,0.34); --glass-shadow:0 18px 44px rgba(var(--theme-shadow),0.14); background:var(--page-bg); }}
    .stApp::before {{ content:''; position:fixed; inset:0; pointer-events:none; z-index:0; background: var(--page-bg); opacity:0.94; }}
    [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{ background: transparent !important; }}
    .block-container {{ max-width: 1220px; padding-top: 1.2rem; padding-bottom: 2rem; position: relative; z-index: 2; }}
    .bg-art {{ position: fixed; pointer-events: none; z-index: 0; opacity: 0.09; right: -2vw; top: 10vh; width: min(44vw, 610px); height: min(44vw, 610px); }}
    .bg-art .bg-scene-svg {{ width: 100%; height: 100%; }}
    .hero-strip, .glass-card, .result-card {{ background: var(--theme-surface); border: 1px solid var(--glass-border); box-shadow: var(--glass-shadow), var(--glass-inner); backdrop-filter: blur(var(--glass-blur)) saturate(145%); -webkit-backdrop-filter: blur(var(--glass-blur)) saturate(145%); border-radius: 24px; }}
    .method-block {{ position:relative; overflow:hidden; }}
    .method-label {{ display:inline-flex; align-items:center; width:max-content; border-radius:999px; padding:0.3rem 0.58rem; margin-bottom:0.38rem; background:rgba(255,255,255,0.28); border:1px solid rgba(255,255,255,0.34); box-shadow:inset 0 1px 0 rgba(255,255,255,0.22); font-size:0.68rem; font-weight:900; letter-spacing:0.1em; text-transform:uppercase; }}
    .hero-strip {{ padding: 0.95rem 1.15rem; margin-bottom: 0.75rem; min-height: 78px; display:flex; align-items:center; color:var(--theme-ink); background: linear-gradient(135deg, color-mix(in srgb, var(--theme-soft) 58%, var(--theme-primary)) 0%, color-mix(in srgb, var(--theme-primary) 32%, var(--theme-secondary)) 100%); border-left: 3px solid var(--theme-primary); }}
    .hero-title {{ font-size: 1.74rem; line-height: 1.12; font-weight: 800; letter-spacing: -0.02em; }}
    .top-control-row {{ display:grid; grid-template-columns: 1fr auto; align-items:center; gap:0.7rem; margin: 0 0 0.85rem 0; }}
    .demo-panel {{ padding:0.55rem; border-radius:20px; background:var(--theme-surface); border:1px solid rgba(255,255,255,0.42); box-shadow:0 12px 28px rgba(var(--theme-shadow),0.12); }}
    .settings-wrap {{ display:flex; justify-content:flex-end; }}
    .section-shell {{ margin: 0.95rem 0 1rem 0; padding: 1rem 1.1rem; border-radius: 24px; background: var(--theme-surface); border: 1px solid var(--glass-border); box-shadow: var(--glass-shadow), var(--glass-inner); backdrop-filter: blur(var(--glass-blur)) saturate(145%); -webkit-backdrop-filter: blur(var(--glass-blur)) saturate(145%); }}
    .section-eyebrow {{ font-size:0.72rem; text-transform:uppercase; letter-spacing:0.14em; color:color-mix(in srgb, var(--theme-ink) 82%, white); font-weight:900; }}
    .section-heading {{ margin-top:0.18rem; font-size:1.35rem; line-height:1.1; color:var(--theme-ink); font-weight:900; letter-spacing:-0.03em; }}
    .section-subtitle {{ margin-top:0.22rem; color:color-mix(in srgb, var(--theme-ink) 82%, white); font-size:0.86rem; font-weight:700; }}
    .method-zone-head {{ display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; }}
    .method-zone-tag {{ border-radius:999px; padding:0.48rem 0.75rem; background:rgba(255,255,255,0.48); border:1px solid rgba(255,255,255,0.58); color:var(--theme-ink); font-size:0.74rem; font-weight:950; letter-spacing:0.08em; text-transform:uppercase; white-space:nowrap; }}
    .glass-card {{ padding: 1rem 1.05rem; margin-bottom: 1rem; background: var(--theme-surface); }}
    .input-card {{ background: linear-gradient(145deg, color-mix(in srgb, var(--theme-primary) 22%, transparent) 0%, rgba(244, 251, 255, 0.56) 100%); }}
    .section-title {{ font-size: 1rem; font-weight: 800; color: var(--theme-ink); margin-bottom: 0.38rem; letter-spacing: -0.01em; }}
    .preset-strip {{ margin: 0.2rem 0 0.95rem 0; padding: 0.78rem 0.92rem; border-radius: 18px; background: linear-gradient(135deg, color-mix(in srgb, var(--theme-primary) 18%, transparent) 0%, color-mix(in srgb, var(--theme-secondary) 16%, transparent) 100%); border: 1px solid rgba(255,255,255,0.38); color: var(--theme-ink); display:flex; align-items:center; justify-content:space-between; gap: 1rem; }}
    .preset-kicker {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.7; font-weight: 700; }}
    .preset-name {{ margin-top: 0.18rem; font-size: 1rem; font-weight: 800; }}
    .preset-note {{ font-size: 0.83rem; color:#4d6778; text-align:right; }}
    .driver-pill {{ display:inline-block; margin-top:0.18rem; padding:0.16rem 0.48rem; border-radius:999px; background:linear-gradient(135deg, color-mix(in srgb, var(--theme-highlight) 18%, white), color-mix(in srgb, var(--theme-soft) 72%, white)); border:1px solid color-mix(in srgb, var(--theme-highlight) 34%, white); color:var(--theme-highlight); font-weight:900; }}
    .rf-pill {{ background:linear-gradient(135deg, color-mix(in srgb, var(--theme-decor) 30%, white), color-mix(in srgb, var(--theme-soft) 70%, white)); border-color:color-mix(in srgb, var(--theme-decor) 42%, white); color:var(--theme-highlight); }}
    .result-card {{ padding: 1rem 1.05rem; color: #f7fcff; border: 1px solid var(--glass-border); min-height: 354px; height:354px; box-shadow: 0 18px 44px rgba(var(--theme-shadow),0.16), var(--glass-inner); }}
    .result-default {{ background: linear-gradient(135deg, color-mix(in srgb, var(--theme-primary) 86%, #21465c) 0%, color-mix(in srgb, var(--theme-secondary) 84%, white) 100%); }}
    .result-low, .result-moderate, .result-high {{ background: var(--theme-surface-strong); }}
    .result-layout {{ display:grid; grid-template-columns: 1fr 1fr; gap: 0.9rem; align-items: stretch; height:100%; }}
    .result-label {{ font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.12em; opacity: 0.92; font-weight: 800; }}
    .result-title {{ margin-top: 0.22rem; font-size: 2.4rem; line-height: 1.02; font-weight: 900; letter-spacing: -0.03em; color:#ffffff; text-shadow:0 2px 20px rgba(var(--theme-shadow),0.28), 0 1px 3px rgba(0,0,0,0.10); }}
    .result-chip-row {{ display:flex; flex-direction:column; gap:0.72rem; margin-top:0.92rem; }}
    .result-chip {{ background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.18); border-radius: 16px; padding: 0.72rem 0.78rem; width: 100%; box-sizing: border-box; }}
    .result-chip-title {{ font-size: 0.71rem; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.82; }}
    .result-chip-value {{ margin-top: 0.16rem; font-size: 1.1rem; font-weight: 800; }}
    .icon-chip {{ display:flex; align-items:center; gap:0.58rem; width: 100%; }}
    .icon-chip-badge {{ width: 46px; height: 46px; border-radius: 14px; background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.22); display:flex; align-items:center; justify-content:center; flex: 0 0 auto; }}
    .level-icon-svg {{ width: 34px; height: 34px; display:block; }}
    .xgb-visual-panel {{ border-radius: 22px; background: rgba(255,255,255,0.13); border: 1px solid rgba(255,255,255,0.18); display:flex; flex-direction:column; align-items:center; justify-content:center; gap:1.05rem; padding:1rem; min-height:0; height:100%; }}
    .scene-hero {{ min-height: 0; border-radius: 18px; background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.14); display:flex; align-items:center; justify-content:center; }}
    .scene-hero .level-icon-svg {{ width: 62%; height: 62%; max-width:86px; max-height:86px; }}
    .mini-prob-stack {{ margin-top:0.62rem; display:flex; flex-direction:column; gap:0.38rem; }}
    .mini-prob-row {{ padding:0.42rem 0.52rem; border-radius:14px; background:rgba(255,255,255,0.13); border:1px solid rgba(255,255,255,0.15); }}
    .mini-prob-row.active {{ background:rgba(255,255,255,0.20); }}
    .mini-prob-head {{ display:flex; justify-content:space-between; gap:0.6rem; font-size:0.74rem; font-weight:800; }}
    .mini-prob-track {{ margin-top:0.38rem; height:8px; border-radius:999px; background:rgba(255,255,255,0.24); overflow:hidden; }}
    .mini-prob-fill {{ height:100%; border-radius:999px; }}
    .donut-wrap {{ width:100%; min-height:0; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:1rem; }}
    .prob-donut {{ width:min(200px, 88%); aspect-ratio:1; display:grid; place-items:center; position:relative; filter: drop-shadow(0 14px 28px rgba(36,91,122,0.14)); }}
    .donut-svg {{ position:absolute; inset:0; width:100%; height:100%; transform:rotate(-90deg); overflow:visible; }}
    .donut-track, .donut-segment {{ fill:none; stroke-width:20px; stroke-linecap:round; }}
    .donut-track {{ stroke:rgba(255,255,255,0.48); }}
    .donut-segment {{ transform-origin:60px 60px; }}
    .donut-low {{ stroke:#7BC8A4; }}
    .donut-moderate {{ stroke:#C8B898; }}
    .donut-high {{ stroke:#D49880; }}
    .donut-center {{ position:relative; z-index:2; width:56%; height:56%; border-radius:50%; display:grid; place-items:center; text-align:center; color:#ffffff; line-height:1; background:linear-gradient(145deg, color-mix(in srgb, var(--theme-primary) 78%, #1A1A2E), color-mix(in srgb, var(--theme-secondary) 68%, #1A1A2E)); border:1px solid rgba(255,255,255,0.55); }}
    .donut-value {{ font-size:1.55rem; font-weight:950; letter-spacing:-0.04em; text-shadow:0 2px 16px rgba(var(--theme-shadow),0.30), 0 1px 2px rgba(0,0,0,0.12); }}
    .donut-legend {{ width:100%; display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:0.42rem; color:#ffffff; font-size:0.72rem; font-weight:850; }}
    .donut-legend div {{ display:grid; grid-template-columns: 9px auto; grid-template-rows:auto auto; justify-content:center; align-items:center; column-gap:0.32rem; row-gap:0.08rem; text-align:center; padding:0.12rem 0.08rem; }}
    .donut-legend strong {{ grid-column:1 / -1; font-size:0.74rem; }}
    .donut-legend span {{ width:9px; height:9px; border-radius:999px; display:block; }}
    .aqi-hero-card {{ margin: 0.85rem 0 0.95rem 0; padding: 1rem 1.08rem; background: linear-gradient(135deg, rgba(47,143,176,0.22) 0%, rgba(83,188,176,0.18) 100%); }}
    .aqi-hero-kicker {{ font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 700; color:#426579; opacity:0.88; }}
    .aqi-hero-value {{ margin-top: 0.18rem; font-size: 3rem; line-height: 1; font-weight: 900; letter-spacing: -0.04em; color:var(--theme-highlight); text-shadow:0 2px 12px rgba(var(--theme-shadow),0.14); }}
    .aqi-hero-note {{ margin-top: 0.26rem; font-size: 0.88rem; color:color-mix(in srgb, var(--theme-ink) 78%, white); font-weight: 600; }}
    .aqi-compare-card {{ margin: 0; padding: 1rem 1.08rem; min-height:354px; height:354px; background: var(--theme-surface-strong); }}
    .aqi-compare-head {{ display:flex; align-items:flex-start; justify-content:space-between; gap:0.8rem; margin-bottom:0.8rem; }}
    .aqi-compare-title {{ margin-top:0.14rem; font-size:0.95rem; font-weight:800; color:var(--theme-ink); }}
    .trend-pill {{ border-radius:999px; padding:0.42rem 0.72rem; background:rgba(255,255,255,0.46); border:1px solid rgba(255,255,255,0.52); color:var(--theme-ink); font-size:0.75rem; font-weight:900; letter-spacing:0.04em; text-transform:uppercase; white-space:nowrap; }}
    .trend-rising {{ --trend-a:var(--theme-secondary); --trend-b:var(--theme-primary); --trend-text:var(--theme-ink); }}
    .trend-falling {{ --trend-a:var(--theme-secondary); --trend-b:var(--theme-primary); --trend-text:var(--theme-ink); }}
    .trend-stable {{ --trend-a:var(--theme-primary); --trend-b:var(--theme-secondary); --trend-text:var(--theme-ink); }}
    .trend-pill.trend-rising, .trend-pill.trend-falling, .trend-pill.trend-stable {{ background:linear-gradient(135deg, color-mix(in srgb, var(--trend-a) 24%, white) 0%, color-mix(in srgb, var(--trend-b) 30%, white) 100%); color:var(--trend-text); border-color:rgba(255,255,255,0.58); }}
    .aqi-compare-grid {{ display:grid; grid-template-columns: 1fr 88px 1fr; gap:0.62rem; align-items:stretch; }}
    .aqi-value-panel {{ border-radius:20px; padding:0.78rem 0.82rem; background:rgba(255,255,255,0.42); border:1px solid rgba(255,255,255,0.48); min-height:164px; display:flex; flex-direction:column; }}
    .current-panel {{ background:linear-gradient(145deg, rgba(255,255,255,0.68) 0%, rgba(255,255,255,0.34) 100%); }}
    .next-panel {{ background:linear-gradient(145deg, color-mix(in srgb, var(--theme-primary) 18%, var(--theme-soft)) 0%, color-mix(in srgb, var(--theme-secondary) 16%, var(--theme-soft)) 100%); }}
    .aqi-panel-label {{ min-height:2.15rem; display:flex; align-items:flex-end; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.1em; font-weight:800; color:color-mix(in srgb, var(--theme-ink) 78%, white); }}
    .aqi-panel-value {{ min-height:3rem; display:flex; align-items:center; margin-top:0.16rem; font-size:2.2rem; line-height:1; font-weight:900; letter-spacing:-0.04em; color:var(--theme-highlight); text-shadow:0 2px 10px rgba(var(--theme-shadow),0.12); }}
    .aqi-panel-note {{ margin-top:0.38rem; font-size:0.78rem; line-height:1.32; color:color-mix(in srgb, var(--theme-ink) 78%, white); font-weight:650; }}
    .aqi-panel-band {{ margin-top:auto; padding-top:0.28rem; font-size:0.76rem; color:color-mix(in srgb, var(--theme-ink) 80%, white); font-weight:850; }}
    .aqi-arrow {{ border-radius:20px; background:linear-gradient(145deg, color-mix(in srgb, var(--trend-a) 18%, white) 0%, color-mix(in srgb, var(--trend-b) 22%, white) 100%); border:1px solid rgba(255,255,255,0.54); display:flex; flex-direction:column; align-items:center; color:var(--trend-text); font-size:0.72rem; font-weight:900; letter-spacing:0.05em; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.22); }}
    .aqi-arrow-spacer {{ min-height:2.5rem; }}
    .trend-arrow-svg {{ width:42px; height:42px; display:block; filter: drop-shadow(0 8px 12px rgba(38,83,100,0.10)); }}
    .trend-arrow-svg path {{ fill:var(--theme-highlight); stroke:rgba(255,255,255,0.72); stroke-width:1.5px; }}
    .trend-arrow-pct {{ margin-top:0.1rem; font-size:0.74rem; color:var(--theme-highlight); }}
    .aqi-compare-note {{ margin-top:0.72rem; font-size:0.84rem; color:color-mix(in srgb, var(--theme-ink) 82%, white); font-weight:750; }}
    div[data-testid='stNumberInput'] {{ background: linear-gradient(145deg, rgba(255,255,255,0.58) 0%, rgba(228,245,252,0.76) 100%); border: 1px solid rgba(255,255,255,0.56); border-radius: 18px; padding: 0.55rem 0.75rem 0.3rem 0.75rem; box-shadow: 0 10px 24px rgba(86,126,154,0.07); margin-bottom: 0.56rem; backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px); }}
    div[data-testid='stNumberInput'] label p {{ color: #26485f !important; font-weight: 700 !important; font-size: 0.83rem !important; }}
    div[data-testid='stNumberInput'] input {{ background: rgba(255,255,255,0.62) !important; border-radius: 12px !important; color: #16384a !important; font-weight: 700 !important; }}
    div[data-testid='stButton'] > button {{ color: white; border: 1px solid rgba(255,255,255,0.30); border-radius: 14px; min-height: 2.75rem; font-weight: 700; box-shadow: 0 10px 24px rgba(var(--theme-shadow),0.14), inset 0 1px 0 rgba(255,255,255,0.26); }}
    [data-testid="stHorizontalBlock"] > div:nth-child(1) [data-testid="stButton"] > button {{ background: linear-gradient(135deg, #6B9E85 0%, #93BAA6 100%) !important; }}
    [data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stButton"] > button {{ background: linear-gradient(135deg, #9EAAB5 0%, #BCC4CD 100%) !important; }}
    [data-testid="stHorizontalBlock"] > div:nth-child(3) [data-testid="stButton"] > button {{ background: linear-gradient(135deg, #B8956E 0%, #CCB098 100%) !important; }}
    div[data-testid='stPopover'] > button {{ border-radius:999px !important; min-height:2.55rem !important; width:2.75rem !important; padding:0 !important; background:var(--theme-surface-strong) !important; color:var(--theme-ink) !important; border:1px solid rgba(255,255,255,0.54) !important; box-shadow:0 10px 24px rgba(var(--theme-shadow),0.16) !important; }}
    .auto-note {{ margin-top:0.7rem; padding:0.75rem 0.9rem; border-radius:16px; background:rgba(255,255,255,0.38); border:1px solid rgba(255,255,255,0.48); color:#4d6778; font-size:0.82rem; font-weight:700; }}
    .scale-shell {{ margin-top: 0.9rem; padding: 0.68rem 0.85rem; }}
    .grade-card-horizontal {{ display:flex; align-items:center; justify-content:space-between; gap:0.6rem; min-height: 114px; text-align:left; color:white; border-radius:18px; padding:0.72rem 0.72rem; opacity:0.92; border:1px solid rgba(255,255,255,0.28); box-shadow:0 12px 28px rgba(var(--theme-shadow),0.12); }}
    .grade-tone-1 {{ background:linear-gradient(145deg, color-mix(in srgb, #6B9E85 88%, white) 0%, color-mix(in srgb, #6B9E85 48%, #F4F8F5) 165%); }}
    .grade-tone-2 {{ background:linear-gradient(145deg, color-mix(in srgb, #BCC4CD 74%, #9EAAB5) 0%, color-mix(in srgb, #F5F6F7 54%, #BCC4CD) 165%); }}
    .grade-tone-3 {{ background:linear-gradient(145deg, color-mix(in srgb, #A07852 72%, #CCB098) 0%, color-mix(in srgb, #A07852 42%, #F9F6F2) 165%); }}
    .grade-copy {{ flex:1; }}
    .grade-icon-wrap {{ width: 74px; height: 74px; border-radius: 18px; display:flex; align-items:center; justify-content:center; background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.22); flex: 0 0 74px; }}
    .grade-icon-wrap .level-icon-svg {{ width: 56px; height: 56px; }}
    .unsup-shell {{ margin-top: 1rem; background: var(--theme-surface); }}
    .unsup-dashboard {{ display:grid; grid-template-columns: 1fr 1fr 1fr; gap:0.85rem; margin-top:0.95rem; align-items:stretch; }}
    .unsup-card {{ min-height:214px; border-radius:22px; padding:1rem; border:1px solid rgba(255,255,255,0.42); box-shadow: inset 0 0 0 1px rgba(255,255,255,0.16), 0 14px 28px rgba(var(--theme-shadow),0.12); }}
    .cluster-card {{ background:linear-gradient(145deg, color-mix(in srgb, var(--theme-primary) 22%, white) 0%, color-mix(in srgb, var(--theme-soft) 84%, white) 120%); }}
    .anomaly-card {{ background:linear-gradient(145deg, color-mix(in srgb, var(--theme-secondary) 20%, white) 0%, color-mix(in srgb, var(--theme-soft) 84%, white) 120%); }}
    .explain-card {{ background:linear-gradient(145deg, color-mix(in srgb, var(--theme-decor) 16%, white) 0%, color-mix(in srgb, var(--theme-soft) 86%, white) 120%); }}
    .unsup-method {{ font-size:0.72rem; text-transform:uppercase; letter-spacing:0.11em; font-weight:900; color:color-mix(in srgb, var(--theme-ink) 78%, white); }}
    .unsup-main {{ margin-top:0.42rem; color:var(--theme-ink); font-size:1.55rem; line-height:1.04; font-weight:950; letter-spacing:-0.035em; }}
    .unsup-label {{ margin-top:0.46rem; color:color-mix(in srgb, var(--theme-ink) 86%, white); font-size:0.92rem; font-weight:850; }}
    .unsup-note {{ margin-top:0.62rem; color:color-mix(in srgb, var(--theme-ink) 74%, white); font-size:0.78rem; line-height:1.35; font-weight:700; }}
    .unsup-visual {{ margin-top:0.9rem; display:flex; flex-direction:column; gap:0.6rem; }}
    .cluster-map {{ margin-top:0.82rem; display:grid; grid-template-columns: 1fr 1fr; gap:0.6rem; align-items:center; }}
    .cluster-svg {{ width:100%; min-height:112px; filter: drop-shadow(0 10px 18px rgba(71,119,143,0.10)); }}
    .cluster-dot {{ opacity:0.65; stroke:rgba(255,255,255,0.80); stroke-width:1.8; }}
    .cluster-dot.cluster-active {{ opacity:0.88; stroke-width:2.4; }}
    .cluster-legend {{ display:flex; flex-direction:column; gap:0.45rem; }}
    .cluster-legend-item {{ display:grid; grid-template-columns: 10px 1fr auto; gap:0.35rem; align-items:center; color:var(--theme-ink); font-size:0.72rem; font-weight:850; }}
    .cluster-legend-item span {{ width:10px; height:10px; border-radius:999px; display:block; }}
    .unsup-viz-head, .driver-row-head {{ display:flex; align-items:center; justify-content:space-between; gap:0.6rem; color:var(--theme-ink); font-size:0.76rem; font-weight:900; }}
    .unsup-track, .driver-track {{ margin-top:0.28rem; height:10px; border-radius:999px; background:rgba(255,255,255,0.48); overflow:hidden; box-shadow:inset 0 0 0 1px rgba(255,255,255,0.22); }}
    .unsup-fill, .driver-fill {{ height:100%; border-radius:999px; }}
    .risk-gauge {{ position:relative; width:156px; height:156px; margin:0.85rem auto 0; border-radius:50%; background:conic-gradient(var(--anomaly-color, #E8A860) var(--risk), rgba(255,255,255,0.18) 0); display:flex; align-items:center; justify-content:center; box-shadow:inset 0 0 0 1px rgba(255,255,255,0.42), 0 0 0 3px color-mix(in srgb, var(--theme-primary) 14%, rgba(255,255,255,0.30)); }}
    .risk-gauge::after {{ content:''; position:absolute; width:104px; height:104px; border-radius:50%; background:linear-gradient(145deg, rgba(255,255,255,0.58), rgba(248,252,255,0.42)); border:1px solid rgba(255,255,255,0.52); box-shadow:0 4px 16px rgba(0,0,0,0.06); }}
    .risk-gauge-center {{ position:relative; z-index:2; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:0.08rem; color:var(--theme-ink); line-height:1; }}
    .risk-gauge-pct {{ font-size:1.55rem; font-weight:950; letter-spacing:-0.03em; }}
    .risk-gauge-lvl {{ font-size:0.62rem; font-weight:900; text-transform:uppercase; letter-spacing:0.14em; opacity:0.62; }}
    .risk-legend {{ display:flex; justify-content:center; gap:0.62rem; margin-top:0.65rem; font-size:0.64rem; font-weight:750; color:var(--theme-ink); }}
    .risk-legend-item {{ display:flex; align-items:center; gap:0.28rem; }}
    .risk-legend-item span {{ width:8px; height:8px; border-radius:999px; display:block; }}
    .driver-stack {{ margin-top:0.82rem; display:flex; flex-direction:column; gap:0.55rem; }}
    .driver-row.top-driver {{ padding:0.46rem 0.52rem; border-radius:15px; background:color-mix(in srgb, var(--theme-highlight) 10%, white); border:1px solid color-mix(in srgb, var(--theme-highlight) 24%, white); }}
    @media (max-width: 980px) {{ .result-layout {{ grid-template-columns: 1fr; }} .aqi-compare-grid {{ grid-template-columns: 1fr; }} .unsup-dashboard {{ grid-template-columns: 1fr; }} .method-zone-head {{ flex-direction:column; }} .aqi-arrow {{ min-height:72px; }} .aqi-hero-value, .aqi-panel-value {{ font-size: 2.45rem; }} }}
    </style>
    <script>
    const app = window.parent.document.querySelector('.stApp');
    if (app) {{
        app.classList.remove('theme-default', 'theme-low', 'theme-moderate', 'theme-high');
        app.classList.add('{theme["body_class"]}');
    }}
    </script>
    """,
    unsafe_allow_html=True,
)

st.markdown(render_background_art(), unsafe_allow_html=True)
st.markdown("<div class='hero-strip'><div class='hero-title'>Next-hour AQI Prediction</div></div>", unsafe_allow_html=True)

demo_cols = st.columns([1, 1, 1, 0.18], gap="small")
with demo_cols[0]:
    if st.button("Low Demo", use_container_width=True):
        apply_preset(LOW_DEMO_VALUES, "Low Demo")
        st.rerun()
with demo_cols[1]:
    if st.button("Moderate Demo", use_container_width=True):
        apply_preset(MODERATE_DEMO_VALUES, "Moderate Demo")
        st.rerun()
with demo_cols[2]:
    if st.button("High Demo", use_container_width=True):
        apply_preset(HIGH_DEMO_VALUES, "High Demo")
        st.rerun()
with demo_cols[3]:
    with st.popover("⚙", use_container_width=True):
        st.markdown("<div class='glass-card input-card'><div class='section-title'>Input Panel</div></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='preset-strip'><div><div class='preset-kicker'>Current Demo</div><div class='preset-name'>{st.session_state.active_demo}</div></div><div class='preset-note'>Adjust values below.</div></div>",
            unsafe_allow_html=True,
        )

        for index, feature in enumerate(FEATURES):
            unit = FEATURE_UNITS.get(feature, "")
            label = f"{feature} ({unit})" if unit else feature
            st.number_input(label, step=0.01, format="%.2f", key=f"feature_{feature}")

        st.markdown("<div class='auto-note'>Auto-updating: results refresh when any input changes.</div>", unsafe_allow_html=True)

payload = {feature: float(st.session_state[f"feature_{feature}"]) for feature in FEATURES}
st.session_state.active_demo = detect_input_mode(payload)
result = classify_and_regress(payload)

pred_label = str(result["prediction_label"])
pred_id = str(result["prediction_class_id"])
probabilities = result["probabilities"]
reg_value = float(result["predicted_aqi_value"])
top_prob = max(probabilities.values())
display_label = pred_label
rf_label = label_from_aqi_value(reg_value)
result_theme = THEMES.get(display_label, THEMES["default"])
scene_text = result_theme["accent"]
aqi_band = format_aqi_band(reg_value)
current_aqi, current_driver = estimate_current_aqi(payload)
sub_aqi_scores = get_sub_aqi_scores(payload)
mode_value = st.session_state.active_demo
xgb_prob_html = render_probability_mini(probabilities, pred_label)

st.markdown(
    f"""
    <script>
    const liveApp = window.parent.document.querySelector('.stApp');
    if (liveApp) {{
        liveApp.classList.remove('theme-default', 'theme-low', 'theme-moderate', 'theme-high');
        liveApp.classList.add('{result_theme["body_class"]}');
    }}
    </script>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='section-shell'><div class='method-zone-head'><div><div class='section-eyebrow'>Supervised Learning</div><div class='section-heading'>Next-hour AQI Prediction Models</div></div><div class='method-zone-tag'>Method 1 + Method 2</div></div><div class='section-subtitle'>XGBoost is the main class prediction. Random Forest is the numeric AQI reference.</div></div>",
    unsafe_allow_html=True,
)

xgb_col, rf_col = st.columns([1, 1], gap="large")
with xgb_col:
    st.markdown(
        f"<div class='result-card {result_theme['result_class']} method-block'><div class='result-layout'><div><div class='method-label'>Method 1</div><div class='result-label'>XGBoost Classification</div><div class='result-title'>{display_label}</div><div class='result-chip-row'><div class='result-chip'><div class='result-chip-title'>Mode</div><div class='result-chip-value'>{mode_value}</div></div><div class='result-chip icon-chip'><div class='icon-chip-badge'>{render_level_icon(display_label)}</div><div><div class='result-chip-title'>Scene</div><div class='result-chip-value'>{scene_text}</div></div></div></div></div><div class='xgb-visual-panel'>{xgb_prob_html}</div></div></div>",
        unsafe_allow_html=True,
    )

with rf_col:
    st.markdown(render_aqi_trend_card(current_aqi, reg_value, current_driver, RF_TOP_FEATURE, aqi_band), unsafe_allow_html=True)

grade_label_for_scale = str(result.get("prediction_label", ""))
sub_aqi_scores = get_sub_aqi_scores(payload)

st.markdown(render_unsupervised_section(payload, current_aqi, current_driver, sub_aqi_scores), unsafe_allow_html=True)

st.markdown("<div class='glass-card scale-shell'><div class='section-title'>Next-hour AQI Grade Scale</div></div>", unsafe_allow_html=True)
grade_cols = st.columns(3, gap="small")
for index, (col, (grade, label, interval)) in enumerate(zip(grade_cols, GRADE_REFERENCE), start=1):
    with col:
        active_style = "box-shadow: inset 0 0 0 3px rgba(255,255,255,0.65);" if label == grade_label_for_scale else ""
        st.markdown(
            f"<div class='grade-card-horizontal grade-tone-{index}' style='{active_style}'><div class='grade-copy'><div style='font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; font-weight:700; opacity:0.92;'>{grade}</div><div style='margin-top:0.18rem; font-size:1rem; font-weight:800;'>{label}</div><div style='margin-top:0.12rem; font-size:0.8rem; opacity:0.95;'>{interval}</div></div><div class='grade-icon-wrap'>{render_level_icon(label)}</div></div>",
            unsafe_allow_html=True,
        )
