# scripts/bulk_load.py
"""
One-off historical backfill: loads full PaySim CSV directly into
fraud.txns_raw, bypassing the streaming pipeline.

Usage:
    python scripts/bulk_load.py --csv data/paysim.csv
"""
from __future__ import annotations

import argparse
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("bulk-load")

PROJECT_ID = "fraud-mlops-portfolio"
TABLE_REF  = f"{PROJECT_ID}.fraud.txns_raw"
CHUNK_SIZE = 50_000   # rows per BQ insert batch


def load(csv_path: Path) -> None:
    log.info("Reading %s ...", csv_path)
    df = pd.read_csv(csv_path)
    log.info("Loaded %d rows.", len(df))

    # Rename CSV columns to match txns_raw schema
    df = df.rename(columns={
        "nameOrig":       "name_orig",
        "oldbalanceOrg":  "old_balance_org",
        "newbalanceOrig": "new_balance_org",
        "nameDest":       "name_dest",
        "oldbalanceDest": "old_balance_dest",
        "newbalanceDest": "new_balance_dest",
        "isFraud":        "is_fraud",
        "isFlaggedFraud": "is_flagged_fraud",
    })

    # Add metadata columns the ingest service would normally add
    now = datetime.now(timezone.utc).isoformat()
    df["event_id"]     = [uuid.uuid4().hex for _ in range(len(df))]
    df["ingested_at"]  = now
    df["ingest_source"] = "bulk_load"

    # Keep only columns that exist in the schema
    cols = [
        "event_id", "step", "type", "amount",
        "name_orig", "old_balance_org", "new_balance_org",
        "name_dest", "old_balance_dest", "new_balance_dest",
        "is_fraud", "is_flagged_fraud",
        "ingested_at", "ingest_source",
    ]
    df = df[cols]

    client = bigquery.Client(project=PROJECT_ID)

    # Load in chunks to avoid memory spikes and show progress
    total = 0
    for i in range(0, len(df), CHUNK_SIZE):
        chunk = df.iloc[i : i + CHUNK_SIZE]
        errors = client.insert_rows_from_dataframe(
            client.get_table(TABLE_REF), chunk
        )
        # errors is a list of lists; flatten and check
        flat_errors = [e for batch in errors for e in batch]
        if flat_errors:
            log.error("Errors in chunk %d: %s", i // CHUNK_SIZE, flat_errors[:3])
        total += len(chunk)
        log.info("Inserted %d / %d rows...", total, len(df))

    log.info("Bulk load complete. %d rows written to %s.", total, TABLE_REF)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    load(p.parse_args().csv)