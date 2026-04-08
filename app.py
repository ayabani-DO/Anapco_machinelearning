"""
ANAPCO ML Microservice (Production API)

Architecture target:
Angular -> Spring Boot -> ML service + Groq

Primary stable endpoints (v1):
  GET  /api/v1/health
  POST /api/v1/prediction
  POST /api/v1/risk-classification
  POST /api/v1/rul-summary

Legacy compatibility endpoints remain available:
  POST /predict/cost
  POST /predict/risk
  GET  /health

Training and model-info endpoints are preserved for existing workflows:
  POST /train
  GET  /model-info
"""

import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple

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
    if hasattr(val, "item"):
        return float(val.item())
    if hasattr(val, "__len__") and len(val) == 1:
        return float(val[0])
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

RISK_LEAKY_FEATURES = [
    "budget_variance_pct",
    "critical_incident_count",
    "corrective_preventive_ratio",
    "weather_risk_score_avg",
]

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


# -----------------------------
# Response + validation helpers
# -----------------------------


def ok_response(data: Dict[str, Any], status_code: int = 200):
    return jsonify({"status": "OK", "data": data}), status_code


def error_response(code: str, message: str, details: Dict[str, Any] = None, status_code: int = 400):
    payload = {
        "status": "ERROR",
        "error": {
            "code": code,
            "message": message,
        },
    }
    if details:
        payload["error"]["details"] = details
    return jsonify(payload), status_code


def _json_body() -> Dict[str, Any]:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("Request body must be a valid JSON object.")
    return data


def _normalize_instances(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    DTO-like contract:
      {
        "instances": [ {feature_map}, ... ],
        "history": [ {feature_map}, ... ]  # optional global history
      }

    Backward compatible accepted input:
      {
        "features": { ... } | [ ... ],
        "history": [ ... ]
      }
    """
    instances = payload.get("instances")
    history = payload.get("history", [])

    if instances is None and "features" in payload:
        features = payload.get("features")
        if isinstance(features, dict):
            instances = [features]
        elif isinstance(features, list):
            instances = features

    if not isinstance(instances, list) or not instances:
        raise ValueError("'instances' must be a non-empty list of feature objects.")
    if not all(isinstance(item, dict) for item in instances):
        raise ValueError("Every element in 'instances' must be a JSON object.")

    if history is None:
        history = []
    if not isinstance(history, list):
        raise ValueError("'history' must be a list when provided.")
    if not all(isinstance(item, dict) for item in history):
        raise ValueError("Every element in 'history' must be a JSON object.")

    required_identity = ["site_id", "year", "month"]
    missing_fields = []
    for idx, row in enumerate(instances):
        for f in required_identity:
            if f not in row:
                missing_fields.append({"instance_index": idx, "field": f})

    if missing_fields:
        raise ValueError(f"Missing required identity fields in instances: {missing_fields}")

    return instances, history


def _load_bundle(model_path: str) -> Dict[str, Any]:
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file is not available. Train or deploy model artifacts first.")
    return joblib.load(model_path)


def engineer_temporal_features(df):
    """Add lag and rolling-average features per site, sorted chronologically."""
    df = df.copy()

    for col in ["site_id", "year", "month"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["site_id", "year", "month"]).reset_index(drop=True)

    generated_cols = []

    for src_col, lags in LAG_SOURCES:
        if src_col not in df.columns:
            df[src_col] = 0
        df[src_col] = pd.to_numeric(df[src_col], errors="coerce").fillna(0)
        for lag in lags:
            col_name = f"lag_{lag}_{src_col}"
            df[col_name] = df.groupby("site_id")[src_col].shift(lag)
            generated_cols.append(col_name)

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

    for col_name in generated_cols:
        df[col_name] = df[col_name].fillna(0)

    logger.info(f"Temporal features engineered: {generated_cols}")
    return df, generated_cols


def prepare_features(df, exclude_cols=None, extra_numeric=None):
    """Prepare feature matrix from raw dataframe."""
    df = df.copy()
    exclude_cols = set(exclude_cols or [])
    extra_numeric = list(extra_numeric or [])

    numeric_used = [c for c in NUMERIC_FEATURES if c not in exclude_cols]
    all_numeric = numeric_used + [c for c in extra_numeric if c not in exclude_cols]

    for col in all_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("UNKNOWN")
        else:
            df[col] = "UNKNOWN"

    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, prefix=CATEGORICAL_FEATURES)

    feature_cols = [
        c
        for c in df_encoded.columns
        if c in all_numeric or any(c.startswith(cat + "_") for cat in CATEGORICAL_FEATURES)
    ]
    return df_encoded[feature_cols], feature_cols


def _validate_columns(df, required_col, dataset_label):
    if required_col not in df.columns:
        raise ValueError(
            f"{dataset_label}: missing required column '{required_col}'. "
            f"Available columns: {list(df.columns)}"
        )
    non_null = df[required_col].dropna()
    if non_null.empty or (non_null.astype(str).str.strip() == "").all():
        raise ValueError(f"{dataset_label}: column '{required_col}' exists but is entirely empty/null.")


# ----------
# Training
# ----------

@app.route("/train", methods=["POST"])
def train():
    try:
        data = _json_body()
        results = {}

        cost_data = data.get("cost_data", [])
        if len(cost_data) >= 10:
            df_cost = pd.DataFrame(cost_data)
            _validate_columns(df_cost, COST_TARGET, "cost_data")

            df_cost, temporal_cols = engineer_temporal_features(df_cost)
            X_cost, cost_feature_cols = prepare_features(df_cost, extra_numeric=temporal_cols)
            y_cost = pd.to_numeric(df_cost[COST_TARGET], errors="coerce").fillna(0)

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
            cost_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

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

            joblib.dump(
                {
                    "model": cost_model,
                    "feature_cols": cost_feature_cols,
                    "numeric_features": NUMERIC_FEATURES,
                    "categorical_features": CATEGORICAL_FEATURES,
                    "temporal_features": temporal_cols,
                    "excluded_features": [],
                },
                COST_MODEL_PATH,
            )
            results["cost_model"] = {
                "status": "trained",
                "metrics": cost_metrics,
                "feature_cols": cost_feature_cols,
            }
        else:
            results["cost_model"] = {
                "status": "skipped",
                "reason": f"Not enough data ({len(cost_data)} rows, need >= 10)",
            }

        risk_data = data.get("risk_data", [])
        if len(risk_data) >= 10:
            df_risk = pd.DataFrame(risk_data)
            _validate_columns(df_risk, RISK_TARGET, "risk_data")

            df_risk, risk_temporal_cols = engineer_temporal_features(df_risk)
            X_risk, risk_feature_cols = prepare_features(
                df_risk, exclude_cols=RISK_LEAKY_FEATURES, extra_numeric=risk_temporal_cols
            )

            le = LabelEncoder()
            y_risk = le.fit_transform(df_risk[RISK_TARGET].astype(str))

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
            risk_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

            y_pred = risk_model.predict(X_test)
            risk_metrics = {
                "accuracy": round(to_float(accuracy_score(y_test, y_pred)), 4),
                "classification_report": classification_report(
                    y_test, y_pred, target_names=le.classes_, output_dict=True
                ),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "split_type": "temporal",
                "temporal_features_added": risk_temporal_cols,
                "total_features": len(risk_feature_cols),
                "classes": le.classes_.tolist(),
            }

            joblib.dump(
                {
                    "model": risk_model,
                    "feature_cols": risk_feature_cols,
                    "numeric_features": [c for c in NUMERIC_FEATURES if c not in RISK_LEAKY_FEATURES],
                    "categorical_features": CATEGORICAL_FEATURES,
                    "temporal_features": risk_temporal_cols,
                    "excluded_features": RISK_LEAKY_FEATURES,
                },
                RISK_MODEL_PATH,
            )
            joblib.dump(le, RISK_ENCODER_PATH)
            results["risk_model"] = {
                "status": "trained",
                "metrics": risk_metrics,
                "feature_cols": risk_feature_cols,
                "excluded_leaky": RISK_LEAKY_FEATURES,
            }
        else:
            results["risk_model"] = {
                "status": "skipped",
                "reason": f"Not enough data ({len(risk_data)} rows, need >= 10)",
            }

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

        return ok_response({"training": results})

    except ValueError as e:
        return error_response("VALIDATION_ERROR", str(e), status_code=400)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return error_response("INTERNAL_ERROR", str(e), status_code=500)


# --------------------
# Core inference logic
# --------------------


def _predict_cost_core(instances: List[Dict[str, Any]], history: List[Dict[str, Any]]):
    bundle = _load_bundle(COST_MODEL_PATH)
    model = bundle["model"]
    trained_cols = bundle["feature_cols"]
    temporal_cols = bundle.get("temporal_features", [])

    all_rows = history + instances
    df = pd.DataFrame(all_rows)

    if temporal_cols:
        df, _ = engineer_temporal_features(df)

    X, _ = prepare_features(df, extra_numeric=temporal_cols)

    for col in trained_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[trained_cols]

    predict_count = len(instances)
    X_predict = X.iloc[-predict_count:]
    predictions = model.predict(X_predict)

    output = []
    for i, pred in enumerate(predictions):
        output.append(
            {
                "site_id": instances[i].get("site_id"),
                "year": int(instances[i].get("year", 0)),
                "month": int(instances[i].get("month", 0)),
                "predicted_next_month_cost_eur": round(to_float(pred), 2),
            }
        )
    return output


def _predict_risk_core(instances: List[Dict[str, Any]], history: List[Dict[str, Any]]):
    bundle = _load_bundle(RISK_MODEL_PATH)
    model = bundle["model"]
    trained_cols = bundle["feature_cols"]
    temporal_cols = bundle.get("temporal_features", [])
    excluded = bundle.get("excluded_features", [])
    le = joblib.load(RISK_ENCODER_PATH)

    all_rows = history + instances
    df = pd.DataFrame(all_rows)

    if temporal_cols:
        df, _ = engineer_temporal_features(df)

    X, _ = prepare_features(df, exclude_cols=excluded, extra_numeric=temporal_cols)

    for col in trained_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[trained_cols]

    predict_count = len(instances)
    X_predict = X.iloc[-predict_count:]
    pred_encoded = model.predict(X_predict)
    pred_proba = model.predict_proba(X_predict)
    pred_labels = le.inverse_transform(pred_encoded)

    output = []
    for i, label in enumerate(pred_labels):
        proba_dict = {
            le.classes_[j]: round(to_float(pred_proba[i][j]), 4) for j in range(len(le.classes_))
        }
        output.append(
            {
                "site_id": instances[i].get("site_id"),
                "year": int(instances[i].get("year", 0)),
                "month": int(instances[i].get("month", 0)),
                "risk_class": str(label),
                "probabilities": proba_dict,
            }
        )
    return output


# --------------------
# Stable v1 endpoints
# --------------------

@app.route("/api/v1/health", methods=["GET"])
def health_v1():
    return ok_response(
        {
            "service": "anapco-ml-service",
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "models": {
                "cost_model_ready": os.path.exists(COST_MODEL_PATH),
                "risk_model_ready": os.path.exists(RISK_MODEL_PATH),
                "risk_encoder_ready": os.path.exists(RISK_ENCODER_PATH),
            },
        }
    )


@app.route("/api/v1/prediction", methods=["POST"])
def prediction_v1():
    try:
        payload = _json_body()
        instances, history = _normalize_instances(payload)
        predictions = _predict_cost_core(instances, history)
        return ok_response({"predictions": predictions})
    except FileNotFoundError as e:
        return error_response("MODEL_NOT_READY", str(e), status_code=404)
    except ValueError as e:
        return error_response("VALIDATION_ERROR", str(e), status_code=400)
    except Exception as e:
        logger.error(f"Cost prediction failed: {e}", exc_info=True)
        return error_response("INTERNAL_ERROR", str(e), status_code=500)


@app.route("/api/v1/risk-classification", methods=["POST"])
def risk_classification_v1():
    try:
        payload = _json_body()
        instances, history = _normalize_instances(payload)
        predictions = _predict_risk_core(instances, history)
        return ok_response({"predictions": predictions})
    except FileNotFoundError as e:
        return error_response("MODEL_NOT_READY", str(e), status_code=404)
    except ValueError as e:
        return error_response("VALIDATION_ERROR", str(e), status_code=400)
    except Exception as e:
        logger.error(f"Risk prediction failed: {e}", exc_info=True)
        return error_response("INTERNAL_ERROR", str(e), status_code=500)


@app.route("/api/v1/rul-summary", methods=["POST"])
def rul_summary_v1():
    """
    RUL summary endpoint contract is stable for Spring integration.
    Current repository does not include a dedicated RUL model artifact,
    so this endpoint returns a deterministic 'not_applicable' response.
    """
    try:
        payload = _json_body()
        instances, _ = _normalize_instances(payload)

        summaries = []
        for row in instances:
            summaries.append(
                {
                    "site_id": row.get("site_id"),
                    "year": int(row.get("year", 0)),
                    "month": int(row.get("month", 0)),
                    "rul_status": "not_applicable",
                    "message": "No RUL model deployed in current service artifacts.",
                }
            )
        return ok_response({"summaries": summaries})
    except ValueError as e:
        return error_response("VALIDATION_ERROR", str(e), status_code=400)
    except Exception as e:
        logger.error(f"RUL summary failed: {e}", exc_info=True)
        return error_response("INTERNAL_ERROR", str(e), status_code=500)


# -----------------------------
# Explainability (clean JSON)
# -----------------------------

@app.route("/api/v1/explain/cost", methods=["POST"])
def explain_cost_v1():
    """Clean JSON SHAP explanation for cost predictions."""
    try:
        payload = _json_body()
        instances, history = _normalize_instances(payload)
        top_n = int(payload.get("top_n", 10))

        bundle = _load_bundle(COST_MODEL_PATH)
        model = bundle["model"]
        trained_cols = bundle["feature_cols"]
        temporal_cols = bundle.get("temporal_features", [])

        all_rows = history + instances
        df = pd.DataFrame(all_rows)
        if temporal_cols:
            df, _ = engineer_temporal_features(df)

        X, _ = prepare_features(df, extra_numeric=temporal_cols)
        for col in trained_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[trained_cols]

        X_explain = X.iloc[-len(instances):]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)

        results = []
        for i in range(len(instances)):
            sv = shap_values[i] if shap_values.ndim > 1 else shap_values
            feature_impact = sorted(
                [
                    {
                        "feature": col,
                        "shap_value": round(to_float(sv[j]), 4),
                        "feature_value": to_float(X_explain.iloc[i, j]),
                    }
                    for j, col in enumerate(trained_cols)
                ],
                key=lambda x: abs(x["shap_value"]),
                reverse=True,
            )[:top_n]

            prediction = round(to_float(model.predict(X_explain.iloc[[i]])[0]), 2)
            ev = explainer.expected_value
            base_val = to_float(ev[0]) if hasattr(ev, "__len__") else to_float(ev)
            results.append(
                {
                    "site_id": instances[i].get("site_id"),
                    "year": int(instances[i].get("year", 0)),
                    "month": int(instances[i].get("month", 0)),
                    "predicted_next_month_cost_eur": prediction,
                    "base_value": round(base_val, 2),
                    "top_features": feature_impact,
                }
            )

        return ok_response({"explanations": results})
    except FileNotFoundError as e:
        return error_response("MODEL_NOT_READY", str(e), status_code=404)
    except ValueError as e:
        return error_response("VALIDATION_ERROR", str(e), status_code=400)
    except Exception as e:
        logger.error(f"Cost SHAP explanation failed: {e}", exc_info=True)
        return error_response("INTERNAL_ERROR", str(e), status_code=500)


# -------------------------------------
# Legacy compatibility endpoint aliases
# -------------------------------------

@app.route("/predict/cost", methods=["POST"])
def predict_cost_legacy():
    return prediction_v1()


@app.route("/predict/risk", methods=["POST"])
def predict_risk_legacy():
    return risk_classification_v1()


@app.route("/health", methods=["GET"])
def health_legacy():
    return health_v1()


@app.route("/model-info", methods=["GET"])
def model_info():
    if os.path.exists(MODEL_META_PATH):
        with open(MODEL_META_PATH) as f:
            meta = json.load(f)
        return ok_response({"model_info": meta})
    return ok_response({"model_info": None, "message": "No models trained yet"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
