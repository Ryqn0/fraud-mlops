-- src/features/schema.sql
-- BigQuery table schemas for fraud-mlops
-- Run via: bq query --use_legacy_sql=false < src/features/schema.sql

-- ── 1. Raw events (immutable log — never update, only append) ────────────────
-- One row per transaction as received from Pub/Sub.
-- Source of truth. Never modified after insert.

CREATE TABLE IF NOT EXISTS `fraud-mlops-portfolio.fraud.txns_raw` (
    -- From PaySim / Pub/Sub message
    event_id        STRING   OPTIONS(description="Unique Pub/Sub message ID"),
    step            INT64   OPTIONS(description="Hour of simulation (1–743)"),
    type            STRING   OPTIONS(description="Transaction type: TRANSFER, CASH_OUT, etc."),
    amount          FLOAT64   OPTIONS(description="Transaction amount"),
    name_orig       STRING   OPTIONS(description="Originating account ID"),
    old_balance_org FLOAT64   OPTIONS(description="Origin balance before transaction"),
    new_balance_org FLOAT64   OPTIONS(description="Origin balance after transaction"),
    name_dest       STRING   OPTIONS(description="Destination account ID"),
    old_balance_dest FLOAT64  OPTIONS(description="Destination balance before"),
    new_balance_dest FLOAT64  OPTIONS(description="Destination balance after"),
    is_fraud        INT64   OPTIONS(description="Ground truth label (arrives with delay)"),
    is_flagged_fraud INT64  OPTIONS(description="Legacy rule flag — not used in model"),

    -- Ingestion metadata
    ingested_at     TIMESTAMP   OPTIONS(description="Timestamp when row was written to BQ"),
    ingest_source   STRING   OPTIONS(description="'pubsub' or 'replay'")
)
OPTIONS(description="Immutable raw event log. Append-only.");

-- ── 2. Feature table (batch-computed, refreshed on schedule) ─────────────────
-- One row per transaction, after feature engineering.
-- Joined to txns_raw on event_id at training time.

CREATE TABLE IF NOT EXISTS `fraud-mlops-portfolio.fraud.features` (
    event_id            STRING   OPTIONS(description="FK to txns_raw.event_id"),
    step                INT64,
    amount              FLOAT64,
    type_is_transfer    BOOL   OPTIONS(description="1 if TRANSFER, 0 otherwise"),
    type_is_cashout     BOOL   OPTIONS(description="1 if CASH_OUT, 0 otherwise"),

    -- Balance features (from EDA section 5)
    balance_diff_orig   FLOAT64   OPTIONS(description="(old_balance_org - amount) - new_balance_org"),
    account_drained     BOOL   OPTIONS(description="True if old_balance_org > 0 AND new_balance_org == 0"),

    -- TODO: what two rolling/velocity features would you add here?
    -- Think: what patterns over the last N transactions per account
    -- would help distinguish fraud? Name them and give a description.
    -- (No code yet — just name and describe them as columns)
    tx_amount_sum_24h  FLOAT64  OPTIONS(description="Sum of amounts for nameOrig in past 24 steps"),
    tx_count_24h  INT64  OPTIONS(description="Number of transactions for nameOrig in past 24 steps"),

    feature_computed_at TIMESTAMP   OPTIONS(description="When this feature row was computed")
)
OPTIONS(description="Engineered features, batch-refreshed. Join to txns_raw for labels.");

-- ── 3. Predictions (model output + ground truth feedback loop) ───────────────
-- One row per scored transaction.
-- is_fraud starts NULL, filled in when ground truth arrives.

CREATE TABLE IF NOT EXISTS `fraud-mlops-portfolio.fraud.predictions` (
    event_id        STRING   OPTIONS(description="FK to txns_raw.event_id"),
    fraud_score     FLOAT64   OPTIONS(description="Model output probability [0, 1]"),
    predicted_label INT64   OPTIONS(description="1 if fraud_score > threshold, else 0"),
    threshold_used  FLOAT64   OPTIONS(description="Decision threshold at prediction time"),
    model_version   STRING   OPTIONS(description="Model version string, e.g. 'v1.2.0'"),
    predicted_at    TIMESTAMP   OPTIONS(description="When inference was run"),

    -- Ground truth (arrives later — fraud confirmed by investigation)
    actual_label    INT64   OPTIONS(description="True label. NULL until fraud investigation completes (delay: days to weeks)."),
    feedback_at     TIMESTAMP   OPTIONS(description="When ground truth was confirmed. NULL until then.")
)
OPTIONS(description="Model predictions + delayed ground truth. Used for drift monitoring.");