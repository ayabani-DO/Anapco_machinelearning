"""
ANAPCO ML Microservice
- Model 1: XGBoost Regressor — Monthly Cost Forecasting
- Model 2: XGBoost Classifier — Equipment/Site Risk Classification

Endpoints:
  POST /train              — Train both models from Spring Boot training data
  POST /predict/cost       — Predict next month total cost
  POST /predict/risk       — Predict risk class (LOW_RISK / MEDIUM_RISK / HIGH_RISK)
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def prepare_features(df, exclude_cols=None):
    """Prepare feature matrix from raw dataframe.

    Args:
        df: Raw dataframe with all columns.
        exclude_cols: Optional list of numeric feature names to drop (e.g. leaky features).
    Returns:
        (feature_df, feature_col_list)
    """
    df = df.copy()
    exclude_cols = set(exclude_cols or [])

    numeric_used = [c for c in NUMERIC_FEATURES if c not in exclude_cols]

    # Fill nulls for numeric features
    for col in numeric_used:
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
                    c in numeric_used or
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
            X_cost, cost_feature_cols = prepare_features(df_cost)
            y_cost = pd.to_numeric(df_cost[COST_TARGET], errors="coerce").fillna(0)

            X_train, X_test, y_train, y_test = train_test_split(X_cost, y_cost, test_size=0.2, random_state=42)

            cost_model = XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
            cost_model.fit(X_train, y_train)

            y_pred = cost_model.predict(X_test)
            cost_metrics = {
                "mae": round(float(mean_absolute_error(y_test, y_pred)), 2),
                "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
                "r2": round(float(r2_score(y_test, y_pred)), 4),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            }

            # Save model + full feature metadata (A)
            joblib.dump({
                "model": cost_model,
                "feature_cols": cost_feature_cols,
                "numeric_features": NUMERIC_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
                "excluded_features": [],
            }, COST_MODEL_PATH)
            results["cost_model"] = {"status": "trained", "metrics": cost_metrics, "feature_cols": cost_feature_cols}
            logger.info(f"Cost model trained: {cost_metrics}")
        else:
            results["cost_model"] = {"status": "skipped", "reason": f"Not enough data ({len(cost_data)} rows, need >= 10)"}

        # ── Model 2: Risk Classification ────────────────────
        risk_data = data.get("risk_data", [])
        if len(risk_data) >= 10:
            df_risk = pd.DataFrame(risk_data)
            _validate_columns(df_risk, RISK_TARGET, "risk_data")

            # Exclude features that directly determine the label (B — data leakage prevention)
            X_risk, risk_feature_cols = prepare_features(df_risk, exclude_cols=RISK_LEAKY_FEATURES)
            logger.info(f"Risk model: excluded leaky features {RISK_LEAKY_FEATURES}")

            le = LabelEncoder()
            y_risk = le.fit_transform(df_risk[RISK_TARGET].astype(str))

            X_train, X_test, y_train, y_test = train_test_split(X_risk, y_risk, test_size=0.2, random_state=42, stratify=y_risk)

            risk_model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric="mlogloss",
            )
            risk_model.fit(X_train, y_train)

            y_pred = risk_model.predict(X_test)
            risk_metrics = {
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "classification_report": classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "classes": le.classes_.tolist(),
            }

            # Save model + full feature metadata (A)
            joblib.dump({
                "model": risk_model,
                "feature_cols": risk_feature_cols,
                "numeric_features": [c for c in NUMERIC_FEATURES if c not in RISK_LEAKY_FEATURES],
                "categorical_features": CATEGORICAL_FEATURES,
                "excluded_features": RISK_LEAKY_FEATURES,
            }, RISK_MODEL_PATH)
            joblib.dump(le, RISK_ENCODER_PATH)
            results["risk_model"] = {"status": "trained", "metrics": risk_metrics, "feature_cols": risk_feature_cols, "excluded_leaky": RISK_LEAKY_FEATURES}
            logger.info(f"Risk model trained: accuracy={risk_metrics['accuracy']}")
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
    Expects JSON body: { "features": { feature_map } } or { "features": [ {feature_map}, ... ] }
    """
    try:
        bundle = joblib.load(COST_MODEL_PATH)
        model = bundle["model"]
        trained_cols = bundle["feature_cols"]

        data = request.get_json()
        features = data.get("features")
        if isinstance(features, dict):
            features = [features]

        df = pd.DataFrame(features)
        X, _ = prepare_features(df)

        # Align columns with training
        for col in trained_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[trained_cols]

        predictions = model.predict(X)

        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "predicted_next_month_cost_eur": round(float(pred), 2),
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
    Expects JSON body: { "features": { feature_map } } or { "features": [ {feature_map}, ... ] }
    """
    try:
        bundle = joblib.load(RISK_MODEL_PATH)
        model = bundle["model"]
        trained_cols = bundle["feature_cols"]
        le = joblib.load(RISK_ENCODER_PATH)

        data = request.get_json()
        features = data.get("features")
        if isinstance(features, dict):
            features = [features]

        df = pd.DataFrame(features)
        excluded = bundle.get("excluded_features", [])
        X, _ = prepare_features(df, exclude_cols=excluded)

        for col in trained_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[trained_cols]

        pred_encoded = model.predict(X)
        pred_proba = model.predict_proba(X)
        pred_labels = le.inverse_transform(pred_encoded)

        results = []
        for i, label in enumerate(pred_labels):
            proba_dict = {le.classes_[j]: round(float(pred_proba[i][j]), 4) for j in range(len(le.classes_))}
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
