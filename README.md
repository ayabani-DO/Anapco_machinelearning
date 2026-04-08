# ANAPCO Machine Learning Service

Production-ready Flask ML service for integration with Spring Boot.

**Target architecture:**
Angular -> Spring Boot -> ML service + Groq

## Existing ML logic (current repository)

The service currently contains:

1. **Cost forecasting model** (`XGBRegressor`)
   - Predicts `next_month_total_cost_eur`.
   - Uses numeric + one-hot categorical features.
   - Uses temporal feature engineering (lag + rolling windows by site and month).

2. **Risk classification model** (`XGBClassifier`)
   - Predicts risk class labels (e.g. `LOW_RISK`, `MEDIUM_RISK`, `HIGH_RISK`).
   - Applies leakage prevention by excluding directly rule-derived features from training.

3. **Preprocessing pipeline**
   - Missing numeric fields are coerced to `0`.
   - Categorical fields are one-hot encoded.
   - Inference aligns input columns to model-time feature columns.

4. **Model loading**
   - Models and metadata are loaded from `models/*.joblib` and `models/model_metadata.json`.

5. **Explainability flow**
   - SHAP-based cost explanation endpoint available at `/api/v1/explain/cost`.

---

## Stable API contracts (Spring Boot integration)

All responses are JSON-only and use a stable envelope:

- Success: `{ "status": "OK", "data": { ... } }`
- Error: `{ "status": "ERROR", "error": { "code": "...", "message": "...", "details": {...?} } }`

### 1) Health

`GET /api/v1/health`

**Response (200)**
```json
{
  "status": "OK",
  "data": {
    "service": "anapco-ml-service",
    "timestamp_utc": "2026-04-08T12:34:56.000000Z",
    "models": {
      "cost_model_ready": true,
      "risk_model_ready": true,
      "risk_encoder_ready": true
    }
  }
}
```

### 2) Prediction (Cost)

`POST /api/v1/prediction`

**Request**
```json
{
  "instances": [
    {
      "site_id": 101,
      "year": 2026,
      "month": 4,
      "total_cost_eur": 98000,
      "previous_month_total_cost_eur": 95000,
      "incident_cost_eur": 12000,
      "maintenance_cost_eur": 21000,
      "manual_expense_eur": 7000,
      "budget_eur": 110000,
      "budget_variance_pct": 3.5,
      "incident_count": 4,
      "critical_incident_count": 0,
      "high_incident_count": 1,
      "avg_incident_severity": 2.1,
      "preventive_maintenance_count": 10,
      "corrective_maintenance_count": 3,
      "inspection_count": 7,
      "corrective_preventive_ratio": 0.3,
      "avg_mtbf": 90,
      "avg_mttr": 8,
      "oil_price_avg_usd": 82,
      "gas_price_avg_eur_mwh": 44,
      "electricity_price_avg_eur_mwh": 107,
      "weather_risk_score_avg": 35,
      "weather_alert_count": 2,
      "equipment_count": 48,
      "season": 2,
      "site_type": "INDUSTRIAL",
      "dominant_equipment_category": "PUMP"
    }
  ],
  "history": []
}
```

**Response (200)**
```json
{
  "status": "OK",
  "data": {
    "predictions": [
      {
        "site_id": 101,
        "year": 2026,
        "month": 4,
        "predicted_next_month_cost_eur": 102345.67
      }
    ]
  }
}
```

### 3) Risk Classification

`POST /api/v1/risk-classification`

**Request**
```json
{
  "instances": [
    {
      "site_id": 101,
      "year": 2026,
      "month": 4,
      "total_cost_eur": 98000,
      "incident_count": 4,
      "maintenance_cost_eur": 21000,
      "site_type": "INDUSTRIAL",
      "dominant_equipment_category": "PUMP"
    }
  ]
}
```

**Response (200)**
```json
{
  "status": "OK",
  "data": {
    "predictions": [
      {
        "site_id": 101,
        "year": 2026,
        "month": 4,
        "risk_class": "MEDIUM_RISK",
        "probabilities": {
          "HIGH_RISK": 0.1542,
          "LOW_RISK": 0.2431,
          "MEDIUM_RISK": 0.6027
        }
      }
    ]
  }
}
```

### 4) RUL Summary

`POST /api/v1/rul-summary`

This endpoint is intentionally stable for backend wiring. In the current artifact set, no dedicated RUL model is deployed, so it returns a deterministic not-applicable result.

**Request**
```json
{
  "instances": [
    {
      "site_id": 101,
      "year": 2026,
      "month": 4
    }
  ]
}
```

**Response (200)**
```json
{
  "status": "OK",
  "data": {
    "summaries": [
      {
        "site_id": 101,
        "year": 2026,
        "month": 4,
        "rul_status": "not_applicable",
        "message": "No RUL model deployed in current service artifacts."
      }
    ]
  }
}
```

### Validation and error handling

Example validation error (HTTP 400):
```json
{
  "status": "ERROR",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "'instances' must be a non-empty list of feature objects."
  }
}
```

Example model-not-ready error (HTTP 404):
```json
{
  "status": "ERROR",
  "error": {
    "code": "MODEL_NOT_READY",
    "message": "Model file is not available. Train or deploy model artifacts first."
  }
}
```

---

## Backward compatibility

Legacy endpoints are preserved and mapped to v1 handlers:
- `GET /health` -> `/api/v1/health`
- `POST /predict/cost` -> `/api/v1/prediction`
- `POST /predict/risk` -> `/api/v1/risk-classification`

---

## Run

```bash
pip install -r requirements.txt
python app.py
```

Service listens on port `5001` by default.
