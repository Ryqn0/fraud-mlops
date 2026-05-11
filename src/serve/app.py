# src/serve/app.py
"""
Inference service: receives Pub/Sub push messages, computes features,
scores with LightGBM, writes predictions to fraud.predictions.

Run locally:
    MODEL_GCS_PATH=models/lgbm_fraud.pkl uvicorn src.serve.app:app --reload --port 8081
"""
from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from google.cloud import bigquery, storage as gcs
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("serve")

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID     = os.environ.get("PROJECT_ID", "fraud-mlops-portfolio")
GCS_BUCKET     = os.environ.get("GCS_BUCKET", "fraud-mlops-portfolio-fraud-artifacts")
MODEL_GCS_PATH = os.environ.get("MODEL_GCS_PATH", "models/lgbm_fraud.pkl")
MODEL_VERSION  = os.environ.get("MODEL_VERSION", "v1")
PRED_TABLE     = f"{PROJECT_ID}.fraud.predictions"
RAW_TABLE      = f"{PROJECT_ID}.fraud.txns_raw"
THRESHOLD      = float(os.environ.get("FRAUD_THRESHOLD", "0.5"))

# Feature order must match training exactly
FEATURE_COLS = [
    "amount",
    "type_is_transfer",
    "type_is_cashout",
    "balance_diff_orig",
    "account_drained",
    "tx_amount_sum_24h",
    "tx_count_24h",
]


# ── Model loading (at startup, once) ─────────────────────────────────────────
def load_model_from_gcs() -> lgb.LGBMClassifier:
    """Download model from GCS and load with joblib."""
    log.info("Loading model from gs://%s/%s ...", GCS_BUCKET, MODEL_GCS_PATH)
    storage_client = gcs.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(GCS_BUCKET)
    blob = bucket.blob(MODEL_GCS_PATH)

    # TODO 1: download the blob to a temporary file and load it with joblib.
    # Pattern:
    #   with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
    #       blob.download_to_filename(f.name)
    #       model = joblib.load(f.name)
    # Why NamedTemporaryFile? joblib.load needs a file path, not a file object.
    # Why delete=False? So the file still exists when joblib reads it after the
    # context manager closes. Clean it up manually after loading.
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        tmp_path = f.name
    
    blob.download_to_filename(tmp_path)
    model = joblib.load(tmp_path)
    os.remove(tmp_path)  # clean up the temp file
    return model


# Loaded once at startup — not per request
_model: lgb.LGBMClassifier | None = None
_bq: bigquery.Client | None = None


def get_model() -> lgb.LGBMClassifier:
    global _model
    if _model is None:
        _model = load_model_from_gcs()
        log.info("Model loaded.")
    return _model


def get_bq() -> bigquery.Client:
    global _bq
    if _bq is None:
        _bq = bigquery.Client(project=PROJECT_ID)
    return _bq


# ── Pydantic models (reuse envelope pattern from ingest service) ───────────────
class Transaction(BaseModel):
    event_id: str
    step: int = Field(ge=1, le=744)
    type: str
    amount: float = Field(ge=0)
    name_orig: str
    old_balance_org: float
    new_balance_org: float
    name_dest: str
    old_balance_dest: float
    new_balance_dest: float
    is_fraud: int


class PubSubMessage(BaseModel):
    data: str
    messageId: str
    attributes: dict[str, str] | None = None
    publishTime: str


class PubSubEnvelope(BaseModel):
    message: PubSubMessage
    subscription: str


# ── Feature computation ───────────────────────────────────────────────────────
def compute_static_features(tx: Transaction) -> dict:
    """
    Compute features derivable from the transaction payload alone.
    No BQ query needed.
    """
    # TODO 2: compute and return a dict with these keys:
    #   amount, type_is_transfer, type_is_cashout,
    #   balance_diff_orig, account_drained
    # These match exactly what we computed in offline.sql.
    # account_drained: old_balance_org > 0 AND new_balance_org == 0
    return {
        "amount": tx.amount,
        "type_is_transfer": int(tx.type == "TRANSFER"),
        "type_is_cashout": int(tx.type == "CASH_OUT"),
        "balance_diff_orig": tx.old_balance_org - tx.amount - tx.new_balance_org,
        "account_drained": int(tx.old_balance_org > 0 and tx.new_balance_org == 0),
    }


def compute_rolling_features(tx: Transaction) -> dict:
    """
    Query txns_raw for this account's last 24 steps to compute velocity features.
    This is the online equivalent of the RANGE BETWEEN window in offline.sql.
    """
    # TODO 3: run this parameterized BQ query and return the result as a dict.
    #
    # Query logic:
    #   SELECT
    #     COALESCE(SUM(amount), 0.0) AS tx_amount_sum_24h,
    #     COALESCE(COUNT(*),    0)   AS tx_count_24h
    #   FROM `{RAW_TABLE}`
    #   WHERE name_orig = @account_id
    #     AND step >= @min_step          -- current step - 24
    #     AND step <  @current_step      -- exclude current transaction
    #     AND type IN ('TRANSFER', 'CASH_OUT')
    #
    # Use parameterized queries to prevent SQL injection:
    #   job_config = bigquery.QueryJobConfig(
    #       query_parameters=[
    #           bigquery.ScalarQueryParameter("account_id", "STRING", tx.name_orig),
    #           bigquery.ScalarQueryParameter("min_step", "INT64", tx.step - 24),
    #           bigquery.ScalarQueryParameter("current_step", "INT64", tx.step),
    #       ]
    #   )
    #   row = list(get_bq().query(sql, job_config=job_config).result())[0]
    #   return {"tx_amount_sum_24h": float(row.tx_amount_sum_24h),
    #           "tx_count_24h": int(row.tx_count_24h)}
    #
    # Why parameterized? Write a one-line comment explaining SQL injection.

    # SQL Inkection is an attack technique where malicious users can manipulate the SQL query by injecting code through input parameters. Parameterized queries ensure that user input is treated as data, not executable code, thus preventing SQL injection attacks.
    sql = f"""
    SELECT
        COALESCE(SUM(amount), 0.0) AS tx_amount_sum_24h,
        COALESCE(COUNT(*),    0)   AS tx_count_24h
    FROM `{RAW_TABLE}`
    WHERE name_orig = @account_id
        AND step >= @min_step          
        AND step <  @current_step      
        AND type IN ('TRANSFER', 'CASH_OUT')
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("account_id", "STRING", tx.name_orig),
            bigquery.ScalarQueryParameter("min_step", "INT64", tx.step - 24),
            bigquery.ScalarQueryParameter("current_step", "INT64", tx.step),
        ]
    )
    row = list(get_bq().query(sql, job_config=job_config).result())[0]
    return {"tx_amount_sum_24h": float(row.tx_amount_sum_24h),
            "tx_count_24h": int(row.tx_count_24h)}


def build_feature_vector(tx: Transaction) -> list[float]:
    """
    Combine static and rolling features into a single list
    in the exact order FEATURE_COLS specifies.
    """
    # TODO 4: call compute_static_features and compute_rolling_features,
    # merge the two dicts, then return [merged[col] for col in FEATURE_COLS].
    # Order matters — LightGBM expects features in the same order as training.

    # Order matters because it will invert the features if the order is different from training, leading to incorrect predictions.

    static_feats = compute_static_features(tx)
    rolling_feats = compute_rolling_features(tx)
    merged = {**static_feats, **rolling_feats}
    return [merged[col] for col in FEATURE_COLS]


# ── Prediction writing ────────────────────────────────────────────────────────
def write_prediction(
    tx: Transaction,
    fraud_score: float,
    predicted_label: int,
) -> None:
    """Write prediction row to fraud.predictions."""
    # TODO 5: build the row dict matching the predictions schema exactly.
    # All fields: event_id, fraud_score, predicted_label, threshold_used,
    #             model_version, predicted_at, actual_label, feedback_at.
    # actual_label and feedback_at are NULL at prediction time — use None.
    # Insert with event_id as row_ids for idempotency (same pattern as ingest).
    row = {
        "event_id": tx.event_id,
        "fraud_score": fraud_score,
        "predicted_label": predicted_label,
        "threshold_used": THRESHOLD,
        "model_version": MODEL_VERSION,
        "predicted_at": datetime.now(timezone.utc).isoformat(),
        "actual_label": None,
        "feedback_at": None,
    }
    get_bq().insert_rows_json(PRED_TABLE, [row], row_ids=[tx.event_id])
    log.info("Prediction written to BigQuery: event_id=%s score=%.4f label=%d", tx.event_id, fraud_score, predicted_label)


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="fraud-mlops inference service")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/pubsub")
async def pubsub_push(request: Request) -> dict:
    """Score a transaction from a Pub/Sub push message."""

    # TODO 6: parse the envelope (same pattern as ingest service).
    # Decode base64 payload, validate against Transaction model.
    # Return 400 on malformed input.

    body = await request.body()
    try:
        envelope = PubSubEnvelope.model_validate_json(body)
        decoded_data = base64.b64decode(envelope.message.data).decode("utf-8")
        tx = Transaction.model_validate_json(decoded_data)
    except Exception as e:
        log.error("Malformed request: %s", e)
        raise HTTPException(status_code=400, detail="Malformed request")

    # TODO 7: call build_feature_vector(tx) to get features.

    features = build_feature_vector(tx)

    # TODO 8: score with the model.
    # model = get_model()
    # features = build_feature_vector(tx)
    # fraud_score = float(model.predict_proba([features])[0][1])
    # predicted_label = int(fraud_score >= THRESHOLD)
    # log.info("event=%s score=%.4f label=%d", tx.event_id, fraud_score, predicted_label)

    # [0][1] because predict_proba returns [[P(not fraud), P(fraud)]], we want the fraud probability for the single input row.

    model = get_model()
    fraud_score = float(model.predict_proba([features])[0][1])
    predicted_label = int(fraud_score >= THRESHOLD)
    log.info("event=%s score=%.4f label=%d", tx.event_id, fraud_score, predicted_label)

    # TODO 9: write prediction, return result.
    # write_prediction(tx, fraud_score, predicted_label)
    # return {"status": "ok", "event_id": tx.event_id,
    #         "fraud_score": fraud_score, "predicted_label": predicted_label}
    write_prediction(tx, fraud_score, predicted_label)
    return {"status": "ok", "event_id": tx.event_id, "fraud_score": fraud_score, "predicted_label": predicted_label}