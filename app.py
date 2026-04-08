"""
ANAPCO ML Microservice
- Model 1: XGBoost Regressor — Monthly Cost Forecasting
- Model 2: XGBoost Classifier — Equipment/Site Risk Classification

Endpoints:
  POST /train              — Train both models from Spring Boot training data
  POST /predict/cost       — Predict next month total cost
  POST /predict/risk       — Predict risk class (LOW_RISK / MEDIUM_RISK / HIGH_RISK)
  POST /explain/cost       — SHAP explanation for cost prediction
  POST /explain/risk       — SHAP explanation for risk prediction
  GET  /health             — Health check
  GET  /model-info         — Model metadata
"""

import os
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import shap

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def to_float(val):
    """Safely convert any numpy/pandas scalar or 0-d array to a Python float."""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if hasattr(val, 'item'):
        return float(val.item())
    if hasattr(val, '__len__') and len(val) == 1:
        return float(val[0])
    # Fallback with debug
    try:
        return float(val)
    except Exception as e:
        logger.error(f"to_float failed for type {type(val)}: {e}")
        return 0.0

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

COST_MODEL_PATH = os.path.join(MODEL_DIR, "cost_forecast_model.joblib")
RISK_MODEL_PATH = os.path.join(MODEL_DIR, "risk_classification_model.joblib")
RISK_ENCODER_PATH = os.path.join(MODEL_DIR, "risk_label_encoder.joblib")
MODEL_META_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

# Feature columns used by both models (numeric + encoded categorical)
NUMERIC_FEATURES = [
    "total_cost_eur",
    "previous_month_total_cost_eur",
    "incident_cost_eur",
    "maintenance_cost_eur",
    "manual_expense_eur",
    "budget_eur",
    "budget_variance_pct",
    "incident_count",
    "critical_incident_count",
    "high_incident_count",
    "avg_incident_severity",
    "preventive_maintenance_count",
    "corrective_maintenance_count",
    "inspection_count",
    "corrective_preventive_ratio",
    "avg_mtbf",
    "avg_mttr",
    "oil_price_avg_usd",
    "gas_price_avg_eur_mwh",
    "electricity_price_avg_eur_mwh",
    "weather_risk_score_avg",
    "weather_alert_count",
    "equipment_count",
    "season",
]

CATEGORICAL_FEATURES = [
    "site_type",
    "dominant_equipment_category",
]

COST_TARGET = "next_month_total_cost_eur"
RISK_TARGET = "risk_class"

# Features that directly determine risk_class via the labeling rule:
#   HIGH_RISK  if budget_variance_pct>15 | critical_incident_count>=2 | corrective_preventive_ratio>3 | weather_risk_score_avg>=70
#   MEDIUM_RISK if budget_variance_pct>5  | critical_incident_count>=1 | corrective_preventive_ratio>1.5 | weather_risk_score_avg>=40
# Keeping these would cause near-perfect but meaningless accuracy (data leakage).
RISK_LEAKY_FEATURES = [
    "budget_variance_pct",
    "critical_incident_count",
    "corrective_preventive_ratio",
    "weather_risk_score_avg",
]

# Lag and rolling average configuration: (source_col, lag/window) → generated col name
LAG_SOURCES = [
    ("total_cost_eur", [1, 2, 3]),
    ("incident_count", [1, 2]),
    ("maintenance_cost_eur", [1]),
    ("incident_cost_eur", [1]),
    ("oil_price_avg_usd", [1]),
]

ROLLING_SOURCES = [
    ("total_cost_eur", [3, 6]),
    ("incident_count", [3, 6]),
    ("maintenance_cost_eur", [3]),
]


def engineer_temporal_features(df):
    """Add lag and rolling-average features per site, sorted chronologically.

    Expects columns: site_id, year, month + the source columns.
    Returns a new DataFrame with added lag_* and rolling_*m_* columns.
    """
    df = df.copy()

    # Ensure sort order: site → year → month
    for col in ["site_id", "year", "month"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["site_id", "year", "month"]).reset_index(drop=True)

    generated_cols = []

    # Lag features
    for src_col, lags in LAG_SOURCES:
        if src_col not in df.columns:
            df[src_col] = 0
        df[src_col] = pd.to_numeric(df[src_col], errors="coerce").fillna(0)
        for lag in lags:
            col_name = f"lag_{lag}_{src_col}"
            df[col_name] = df.groupby("site_id")[src_col].shift(lag)
            generated_cols.append(col_name)

    # Rolling averages
    for src_col, windows in ROLLING_SOURCES:
        if src_col not in df.columns:
            df[src_col] = 0
        df[src_col] = pd.to_numeric(df[src_col], errors="coerce").fillna(0)
        for w in windows:
            col_name = f"rolling_{w}m_{src_col}"
            df[col_name] = (
                df.groupby("site_id")[src_col]
                .transform(lambda s: s.rolling(window=w, min_periods=1).mean())
            )
            generated_cols.append(col_name)

    # Fill NaN from lags (first rows per site will have NaN)
    for col_name in generated_cols:
        df[col_name] = df[col_name].fillna(0)

    logger.info(f"Temporal features engineered: {generated_cols}")
    return df, generated_cols


def prepare_features(df, exclude_cols=None, extra_numeric=None):
    """Prepare feature matrix from raw dataframe.

    Args:
        df: Raw dataframe with all columns.
        exclude_cols: Optional list of numeric feature names to drop (e.g. leaky features).
        extra_numeric: Optional list of additional numeric columns (e.g. lag/rolling features).
    Returns:
        (feature_df, feature_col_list)
    """
    df = df.copy()
    exclude_cols = set(exclude_cols or [])
    extra_numeric = list(extra_numeric or [])

    numeric_used = [c for c in NUMERIC_FEATURES if c not in exclude_cols]
    all_numeric = numeric_used + [c for c in extra_numeric if c not in exclude_cols]

    # Fill nulls for numeric features
    for col in all_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    # One-hot encode categorical features
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("UNKNOWN")
        else:
            df[col] = "UNKNOWN"

    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, prefix=CATEGORICAL_FEATURES)

    # Return only feature columns (numeric + dummies)
    feature_cols = [c for c in df_encoded.columns if
                    c in all_numeric or
                    any(c.startswith(cat + "_") for cat in CATEGORICAL_FEATURES)]
    return df_encoded[feature_cols], feature_cols


def _validate_columns(df, required_col, dataset_label):
    """Raise ValueError if a required target column is missing or entirely empty."""
    if required_col not in df.columns:
        raise ValueError(
            f"{dataset_label}: missing required column '{required_col}'. "
            f"Available columns: {list(df.columns)}"
        )
    non_null = df[required_col].dropna()
    if non_null.empty or (non_null.astype(str).str.strip() == "").all():
        raise ValueError(
            f"{dataset_label}: column '{required_col}' exists but is entirely empty/null."
        )


# ── Training endpoint ──────────────────────────────────────

@app.route("/train", methods=["POST"])
def train():
    """
    Expects JSON body with two keys:
    {
      "cost_data": [ {feature_map}, ... ],    // rows with next_month_total_cost_eur filled
      "risk_data": [ {feature_map}, ... ]     // rows with risk_class filled
    }
    """
    try:
        data = request.get_json()
        results = {}

        # ── Model 1: Cost Forecast ──────────────────────────
        cost_data = data.get("cost_data", [])
        if len(cost_data) >= 10:
            df_cost = pd.DataFrame(cost_data)
            _validate_columns(df_cost, COST_TARGET, "cost_data")

            # Sprint 1: Engineer lag + rolling features
            df_cost, temporal_cols = engineer_temporal_features(df_cost)

            X_cost, cost_feature_cols = prepare_features(df_cost, extra_numeric=temporal_cols)
            y_cost = pd.to_numeric(df_cost[COST_TARGET], errors="coerce").fillna(0)

            # Sprint 1: Temporal split — train on first 80%, test on last 20% (chronological)
            split_idx = int(len(X_cost) * 0.8)
            X_train, X_test = X_cost.iloc[:split_idx], X_cost.iloc[split_idx:]
            y_train, y_test = y_cost.iloc[:split_idx], y_cost.iloc[split_idx:]

            cost_model = XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
            )
            cost_model.fit(X_train, y_train,
                           eval_set=[(X_test, y_test)],
                           verbose=False)

            y_pred = cost_model.predict(X_test)
            cost_metrics = {
                "mae": round(to_float(mean_absolute_error(y_test, y_pred)), 2),
                "rmse": round(to_float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
                "r2": round(to_float(r2_score(y_test, y_pred)), 4),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "split_type": "temporal",
                "temporal_features_added": temporal_cols,
                "total_features": len(cost_feature_cols),
            }

            # Save model + full feature metadata
            joblib.dump({
                "model": cost_model,
                "feature_cols": cost_feature_cols,
                "numeric_features": NUMERIC_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
                "temporal_features": temporal_cols,
                "excluded_features": [],
            }, COST_MODEL_PATH)
            results["cost_model"] = {"status": "trained", "metrics": cost_metrics, "feature_cols": cost_feature_cols}
            logger.info(f"Cost model trained (temporal split): R²={cost_metrics['r2']}, MAE={cost_metrics['mae']}, RMSE={cost_metrics['rmse']}")
        else:
            results["cost_model"] = {"status": "skipped", "reason": f"Not enough data ({len(cost_data)} rows, need >= 10)"}

        # ── Model 2: Risk Classification ────────────────────
        risk_data = data.get("risk_data", [])
        if len(risk_data) >= 10:
            df_risk = pd.DataFrame(risk_data)
            _validate_columns(df_risk, RISK_TARGET, "risk_data")

            # Sprint 1: Engineer lag + rolling features
            df_risk, risk_temporal_cols = engineer_temporal_features(df_risk)

            # Exclude features that directly determine the label (data leakage prevention)
            X_risk, risk_feature_cols = prepare_features(df_risk, exclude_cols=RISK_LEAKY_FEATURES, extra_numeric=risk_temporal_cols)
            logger.info(f"Risk model: excluded leaky features {RISK_LEAKY_FEATURES}")

            le = LabelEncoder()
            y_risk = le.fit_transform(df_risk[RISK_TARGET].astype(str))

            # Sprint 1: Temporal split
            split_idx = int(len(X_risk) * 0.8)
            X_train, X_test = X_risk.iloc[:split_idx], X_risk.iloc[split_idx:]
            y_train, y_test = y_risk[:split_idx], y_risk[split_idx:]

            risk_model = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric="mlogloss",
            )
            risk_model.fit(X_train, y_train,
                           eval_set=[(X_test, y_test)],
                           verbose=False)

            y_pred = risk_model.predict(X_test)
            risk_metrics = {
                "accuracy": round(to_float(accuracy_score(y_test, y_pred)), 4),
                "classification_report": classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "split_type": "temporal",
                "temporal_features_added": risk_temporal_cols,
                "total_features": len(risk_feature_cols),
                "classes": le.classes_.tolist(),
            }

            # Save model + full feature metadata
            joblib.dump({
                "model": risk_model,
                "feature_cols": risk_feature_cols,
                "numeric_features": [c for c in NUMERIC_FEATURES if c not in RISK_LEAKY_FEATURES],
                "categorical_features": CATEGORICAL_FEATURES,
                "temporal_features": risk_temporal_cols,
                "excluded_features": RISK_LEAKY_FEATURES,
            }, RISK_MODEL_PATH)
            joblib.dump(le, RISK_ENCODER_PATH)
            results["risk_model"] = {"status": "trained", "metrics": risk_metrics, "feature_cols": risk_feature_cols, "excluded_leaky": RISK_LEAKY_FEATURES}
            logger.info(f"Risk model trained (temporal split): accuracy={risk_metrics['accuracy']}")
        else:
            results["risk_model"] = {"status": "skipped", "reason": f"Not enough data ({len(risk_data)} rows, need >= 10)"}

        # Save metadata (A — persist all training context)
        meta = {
            "trained_at": datetime.now().isoformat(),
            "schema": {
                "numeric_features": NUMERIC_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
                "risk_leaky_features": RISK_LEAKY_FEATURES,
            },
            "cost_model": results.get("cost_model", {}),
            "risk_model": results.get("risk_model", {}),
        }
        with open(MODEL_META_PATH, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        return jsonify({"status": "OK", "results": results})

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return jsonify({"status": "ERROR", "error": str(e)}), 500


# ── Prediction endpoints ───────────────────────────────────

@app.route("/predict/cost", methods=["POST"])
def predict_cost():
    """
    Predict next month total cost.
    Expects JSON body:
      { "features": { feature_map } }                         — single prediction (lags=0)
      { "features": [ {map_t-5}, ..., {map_t} ] }             — with history for lags
      { "features": { feature_map }, "history": [ ... ] }      — explicit history
    """
    try:
        bundle = joblib.load(COST_MODEL_PATH)
        model = bundle["model"]
        trained_cols = bundle["feature_cols"]
        temporal_cols = bundle.get("temporal_features", [])

        data = request.get_json()
        features = data.get("features")
        history = data.get("history")

        if isinstance(features, dict):
            features = [features]

        # If explicit history provided, prepend it
        if history and isinstance(history, list):
            all_rows = history + features
        else:
            all_rows = features

        df = pd.DataFrame(all_rows)

        # Engineer temporal features if model was trained with them
        if temporal_cols:
            df, _ = engineer_temporal_features(df)

        X, _ = prepare_features(df, extra_numeric=temporal_cols)

        # Align columns with training
        for col in trained_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[trained_cols]

        # Predict only on the last N rows (the actual request, not history)
        predict_count = len(features)
        X_predict = X.iloc[-predict_count:]
        predictions = model.predict(X_predict)

        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "predicted_next_month_cost_eur": round(to_float(pred), 2),
                "site_id": features[i].get("site_id"),
                "year": features[i].get("year"),
                "month": features[i].get("month"),
            })

        return jsonify({"status": "OK", "predictions": results})

    except FileNotFoundError:
        return jsonify({"status": "ERROR", "error": "Cost model not trained yet. Call /train first."}), 404
    except Exception as e:
        logger.error(f"Cost prediction failed: {e}", exc_info=True)
        return jsonify({"status": "ERROR", "error": str(e)}), 500


@app.route("/predict/risk", methods=["POST"])
def predict_risk():
    """
    Predict risk class (LOW_RISK / MEDIUM_RISK / HIGH_RISK).
    Expects JSON body:
      { "features": { feature_map } }                         — single prediction (lags=0)
      { "features": [ {map_t-5}, ..., {map_t} ] }             — with history for lags
      { "features": { feature_map }, "history": [ ... ] }      — explicit history
    """
    try:
        bundle = joblib.load(RISK_MODEL_PATH)
        model = bundle["model"]
        trained_cols = bundle["feature_cols"]
        temporal_cols = bundle.get("temporal_features", [])
        excluded = bundle.get("excluded_features", [])
        le = joblib.load(RISK_ENCODER_PATH)

        data = request.get_json()
        features = data.get("features")
        history = data.get("history")

        if isinstance(features, dict):
            features = [features]

        if history and isinstance(history, list):
            all_rows = history + features
        else:
            all_rows = features

        df = pd.DataFrame(all_rows)

        if temporal_cols:
            df, _ = engineer_temporal_features(df)

        X, _ = prepare_features(df, exclude_cols=excluded, extra_numeric=temporal_cols)

        for col in trained_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[trained_cols]

        predict_count = len(features)
        X_predict = X.iloc[-predict_count:]
        pred_encoded = model.predict(X_predict)
        pred_proba = model.predict_proba(X_predict)
        pred_labels = le.inverse_transform(pred_encoded)

        results = []
        for i, label in enumerate(pred_labels):
            proba_dict = {le.classes_[j]: round(to_float(pred_proba[i][j]), 4) for j in range(len(le.classes_))}
            results.append({
                "risk_class": str(label),
                "probabilities": proba_dict,
                "site_id": features[i].get("site_id"),
                "year": features[i].get("year"),
                "month": features[i].get("month"),
            })

        return jsonify({"status": "OK", "predictions": results})

    except FileNotFoundError:
        return jsonify({"status": "ERROR", "error": "Risk model not trained yet. Call /train first."}), 404
    except Exception as e:
        logger.error(f"Risk prediction failed: {e}", exc_info=True)
        return jsonify({"status": "ERROR", "error": str(e)}), 500


# ── Sprint 2: SHAP Explainability endpoints ───────────────

@app.route("/explain/cost", methods=["POST"])
def explain_cost():
    """
    SHAP explanation for cost prediction.
    Expects JSON body: { "features": { feature_map } } or { "features": {...}, "history": [...] }
    Returns top contributing features with SHAP values.
    """
    try:
        bundle = joblib.load(COST_MODEL_PATH)
        model = bundle["model"]
        trained_cols = bundle["feature_cols"]
        temporal_cols = bundle.get("temporal_features", [])

        data = request.get_json()
        features = data.get("features")
        history = data.get("history")
        top_n = data.get("top_n", 10)

        if isinstance(features, dict):
            features = [features]

        if history and isinstance(history, list):
            all_rows = history + features
        else:
            all_rows = features

        df = pd.DataFrame(all_rows)
        if temporal_cols:
            df, _ = engineer_temporal_features(df)

        X, _ = prepare_features(df, extra_numeric=temporal_cols)
        for col in trained_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[trained_cols]

        X_explain = X.iloc[-len(features):]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)

        results = []
        for i in range(len(features)):
            sv = shap_values[i] if shap_values.ndim > 1 else shap_values
            feature_impact = sorted(
                [{"feature": col, "shap_value": round(to_float(sv[j]), 4), "feature_value": to_float(X_explain.iloc[i, j])}
                 for j, col in enumerate(trained_cols)],
                key=lambda x: abs(x["shap_value"]), reverse=True
            )[:top_n]

            prediction = round(to_float(model.predict(X_explain.iloc[[i]])[0]), 2)
            ev = explainer.expected_value
            base_val = to_float(ev[0]) if hasattr(ev, '__len__') else to_float(ev)
            results.append({
                "site_id": features[i].get("site_id"),
                "year": features[i].get("year"),
                "month": features[i].get("month"),
                "predicted_cost_eur": prediction,
                "base_value": round(base_val, 2),
                "top_features": feature_impact,
            })

        return jsonify({"status": "OK", "explanations": results})

    except FileNotFoundError:
        return jsonify({"status": "ERROR", "error": "Cost model not trained yet. Call /train first."}), 404
    except Exception as e:
        logger.error(f"Cost SHAP explanation failed: {e}", exc_info=True)
        return jsonify({"status": "ERROR", "error": str(e)}), 500


@app.route("/explain/risk", methods=["POST"])
def explain_risk():
    """
    SHAP explanation for risk prediction.
    Expects JSON body: { "features": { feature_map } } or { "features": {...}, "history": [...] }
    Returns top contributing features with SHAP values per class.
    """
    try:
        bundle = joblib.load(RISK_MODEL_PATH)
        model = bundle["model"]
        trained_cols = bundle["feature_cols"]
        temporal_cols = bundle.get("temporal_features", [])
        excluded = bundle.get("excluded_features", [])
        le = joblib.load(RISK_ENCODER_PATH)

        data = request.get_json()
        features = data.get("features")
        history = data.get("history")
        top_n = data.get("top_n", 10)

        if isinstance(features, dict):
            features = [features]

        if history and isinstance(history, list):
            all_rows = history + features
        else:
            all_rows = features

        df = pd.DataFrame(all_rows)
        if temporal_cols:
            df, _ = engineer_temporal_features(df)

        X, _ = prepare_features(df, exclude_cols=excluded, extra_numeric=temporal_cols)
        for col in trained_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[trained_cols]

        X_explain = X.iloc[-len(features):]

        # SHAP temporarily disabled due to numpy scalar conversion issues
        # explainer = shap.TreeExplainer(model)
        # shap_values = explainer.shap_values(X_explain)

        results = []
        for i in range(len(features)):
            pred_raw = model.predict(X_explain.iloc[[i]])[0]
            pred_encoded = to_float(pred_raw)
            pred_label = le.inverse_transform([int(pred_encoded)])[0]
            pred_proba = model.predict_proba(X_explain.iloc[[i]])[0]

            proba_dict = {le.classes_[j]: round(to_float(pred_proba[j]), 4) for j in range(len(le.classes_))}

            results.append({
                "site_id": features[i].get("site_id"),
                "year": features[i].get("year"),
                "month": features[i].get("month"),
                "predicted_risk_class": str(pred_label),
                "probabilities": proba_dict,
                "explanations_by_class": {"SHAP_temporarily_disabled": "numpy scalar conversion bug"},
            })

        return jsonify({"status": "OK", "explanations": results})

    except FileNotFoundError:
        return jsonify({"status": "ERROR", "error": "Risk model not trained yet. Call /train first."}), 404
    except Exception as e:
        logger.error(f"Risk SHAP explanation failed: {e}", exc_info=True)
        return jsonify({"status": "ERROR", "error": str(e)}), 500


# ── Info endpoints ─────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "UP",
        "cost_model_ready": os.path.exists(COST_MODEL_PATH),
        "risk_model_ready": os.path.exists(RISK_MODEL_PATH),
    })


@app.route("/model-info", methods=["GET"])
def model_info():
    if os.path.exists(MODEL_META_PATH):
        with open(MODEL_META_PATH) as f:
            meta = json.load(f)
        return jsonify(meta)
    return jsonify({"message": "No models trained yet"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
