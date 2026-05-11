# src/features/lookup.py
"""
Runs the offline feature pipeline SQL against BigQuery.
Reads from fraud.txns_raw, writes to fraud.features.

Usage (one-off):
    python -m src.features.lookup

Later: triggered on a schedule by Cloud Scheduler → Cloud Run job.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from google.cloud import bigquery

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("feature-pipeline")

PROJECT_ID = os.environ.get("PROJECT_ID", "fraud-mlops-portfolio")
SQL_PATH = Path(__file__).parent / "offline.sql"


def run_feature_pipeline() -> None:
    client = bigquery.Client(project=PROJECT_ID)

    # TODO 1: read the SQL from SQL_PATH
    # Hint: SQL_PATH.read_text()
    sql = SQL_PATH.read_text()

    log.info("Running feature pipeline SQL...")
    t0 = time.time()

    # TODO 2: submit the query job and wait for completion
    # Hint: client.query(sql).result() — .result() blocks until the job finishes
    # Store the query job result so we can log stats
    query_job = client.query(sql)
    query_job.result()

    elapsed = time.time() - t0

    # TODO 3: log the following:
    #   - elapsed time in seconds (round to 1 decimal)
    #   - bytes processed (job.total_bytes_processed — convert to MB)
    #   - slot milliseconds (job.slot_millis — a measure of compute used)
    # Format: "Feature pipeline complete in X.Xs | Xmb processed | X slot-ms"
    log.info(
        f"Feature pipeline complete in {elapsed:.1f}s | {query_job.total_bytes_processed / (1024**2):.1f}mb processed | {query_job.slot_millis} slot-ms"
    )

    # TODO 4: query fraud.features to verify the row count and log it
    # Run: SELECT COUNT(*) as n FROM `fraud-mlops-portfolio.fraud.features`
    # Log the result

    count_job = client.query("SELECT COUNT(*) as n FROM `fraud-mlops-portfolio.fraud.features`").result()
    count = list(count_job)[0].n
    log.info(f"fraud.features row count: {count}")

    log.info("Done.")


if __name__ == "__main__":
    run_feature_pipeline()