# src/ingest/main.py
"""
Ingest service: receives Pub/Sub push messages, validates them,
streams them into BigQuery fraud.txns_raw.

Run locally:
    uvicorn src.ingest.main:app --reload --port 8080
"""
from __future__ import annotations

import base64
import json
import logging
import os
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from google.cloud import bigquery
from pydantic import BaseModel, Field, ValidationError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("ingest")

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT_ID = os.environ.get("PROJECT_ID", "fraud-mlops-portfolio")
DATASET    = os.environ.get("BQ_DATASET", "fraud")
TABLE      = os.environ.get("BQ_TABLE", "txns_raw")
TABLE_REF  = f"{PROJECT_ID}.{DATASET}.{TABLE}"


# ── Pydantic models ──────────────────────────────────────────────────────────
class Transaction(BaseModel):
    """Schema of the JSON payload inside the Pub/Sub message."""
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
    is_fraud: int = Field(ge=0, le=1)
    is_flagged_fraud: int = Field(ge=0, le=1)


class PubSubMessage(BaseModel):
    """Inner 'message' object from Pub/Sub push envelope."""
    data: str                                # base64-encoded payload
    messageId: str
    attributes: dict[str, str] | None = None
    publishTime: str


class PubSubEnvelope(BaseModel):
    """Outer envelope POSTed by Pub/Sub."""
    message: PubSubMessage
    subscription: str


# ── BigQuery client (lazy-initialised) ────────────────────────────────────────
_bq_client: bigquery.Client | None = None

def get_bq_client() -> bigquery.Client:
    global _bq_client
    if _bq_client is None:
        _bq_client = bigquery.Client(project=PROJECT_ID)
    return _bq_client


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="fraud-mlops ingest service")


@app.get("/health")
def health() -> dict[str, str]:
    """Health check for Cloud Run readiness probes."""
    return {"status": "ok"}


@app.post("/pubsub")
async def pubsub_push(request: Request) -> dict[str, str]:
    """Receive a Pub/Sub push message and write the payload to BigQuery."""

    # TODO 1: parse the request body as JSON and validate it against PubSubEnvelope.
    # Use `await request.json()` to get the body.
    # If validation fails, raise HTTPException(status_code=400, detail=...).
    # Why 400 and not 500? Connect this to the table of HTTP semantics above.
    # It says permanent failure, which means the client (Pub/Sub) is doing something wrong (e.g. sending invalid messages), so we should return 400 to indicate that the request is bad and should not be retried. If we returned 500, Pub/Sub would interpret that as a transient server error and would retry the message, which would likely lead to repeated failures.

    try:
        body_json = await request.json()
        envelope = PubSubEnvelope.model_validate(body_json)
    except (json.JSONDecodeError, ValidationError) as e:
        log.warning("Validation error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

    # TODO 2: decode the base64 payload.
    # Hint: base64.b64decode(envelope.message.data)
    #       Then .decode("utf-8") to get a string.

    decoded_string = base64.b64decode(envelope.message.data).decode("utf-8")

    # TODO 3: parse the decoded string as JSON and validate against Transaction.
    # Same rule as TODO 1: HTTPException(400) on validation failure.

    try:
        transaction = Transaction.model_validate_json(decoded_string)
    except ValidationError as e:
        log.warning("Validation error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

    # TODO 4: build the BigQuery row.
    # The row is a dict matching the txns_raw schema EXACTLY.
    # Add the two ingestion metadata columns that aren't in the Pub/Sub payload:
    #   - ingested_at: current UTC time as ISO format string
    #   - ingest_source: "pubsub"
    # Hint: datetime.now(timezone.utc).isoformat()

    row = transaction.model_dump()
    row["ingested_at"] = datetime.now(timezone.utc).isoformat()
    row["ingest_source"] = "pubsub"

    # TODO 5: streaming insert into BigQuery.
    # client = get_bq_client()
    # errors = client.insert_rows_json(
    #     TABLE_REF,
    #     [row],
    #     row_ids=[transaction.event_id],   # ← idempotency!
    # )
    # If errors is non-empty, log them and raise HTTPException(500, ...).
    # Why 500 and not 400? Same question as TODO 1, opposite answer.
    # 500 indicates a server error, which means something went wrong on our end while processing a valid request. Since the request is valid but we failed to process it (e.g. due to a BigQuery error), we should return 500 to indicate that the client can retry the request later.

    client = get_bq_client()
    errors = client.insert_rows_json(
        TABLE_REF,
        [row],
        row_ids=[transaction.event_id],   # ← idempotency!
    )

    if errors:
        log.error("BigQuery insert errors: %s", errors)
        raise HTTPException(status_code=500, detail=f"Failed to insert into BigQuery : {errors}")

    # TODO 6: return {"status": "ok", "event_id": transaction.event_id}
    
    return {"status": "ok", "event_id": transaction.event_id}


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Catch Pydantic validation errors and return 400 (don't retry)."""
    log.warning("Validation error: %s", exc)
    raise HTTPException(status_code=400, detail=str(exc))