# src/monitor/drift.py
"""
Drift monitoring pipeline.
Compares reference features (train period, step <= 600)
against current features (test period, step > 600).

Writes results to:
  - fraud.drift_reports (BigQuery)
  - custom.googleapis.com/fraud/* (Cloud Monitoring)

Usage:
    python -m src.monitor.drift
    python -m src.monitor.drift --inject-drift   # for demo: artificially drifts current data
"""
from __future__ import annotations

import argparse
import logging
import os
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from google.cloud import bigquery, monitoring_v3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("drift")

PROJECT_ID   = os.environ.get("PROJECT_ID", "fraud-mlops-portfolio")
FEATURES_TBL = f"{PROJECT_ID}.fraud.features"
DRIFT_TBL    = f"{PROJECT_ID}.fraud.drift_reports"
T_SPLIT      = 600

FEATURE_COLS = [
    "amount", "type_is_transfer", "type_is_cashout",
    "balance_diff_orig", "account_drained",
    "tx_amount_sum_24h", "tx_count_24h",
]


# ── Data loading ───────────────────────────────────────────────────────────────
def load_reference(client: bigquery.Client) -> pd.DataFrame:
    """
    Load reference dataset: features from the training period (step <= T_SPLIT).
    This is the baseline — what the model was trained on.

    TODO 1: write a SQL query that selects all FEATURE_COLS from FEATURES_TBL
            where step <= T_SPLIT.
            Limit to 50,000 rows to keep the report fast (use TABLESAMPLE or LIMIT).
    Return a DataFrame with only the FEATURE_COLS columns.
    """
    raise NotImplementedError


def load_current(client: bigquery.Client) -> pd.DataFrame:
    """
    Load current dataset: features from the test period (step > T_SPLIT).
    This simulates what the model sees in production today.

    TODO 2: same as load_reference but WHERE step > T_SPLIT.
    """
    raise NotImplementedError


def inject_drift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Artificially drift the current data for demo purposes.
    Multiplies amount and tx_amount_sum_24h by 10x,
    flips account_drained to simulate fraudsters changing behaviour.

    This proves your monitoring pipeline alerts correctly.
    Do NOT use in production — for demo only.
    """
    df = df.copy()
    df["amount"]            = df["amount"] * 10
    df["tx_amount_sum_24h"] = df["tx_amount_sum_24h"] * 10
    df["account_drained"]   = 1 - df["account_drained"]  # flip 0↔1
    log.info("⚠️  Synthetic drift injected into current data.")
    return df


# ── Evidently report ───────────────────────────────────────────────────────────
def run_evidently(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> dict:
    """
    Run Evidently DataDrift report and return the raw result dict.

    TODO 3: create a Report with DataDriftPreset(), run it on reference and current,
            return report.as_dict().
    Hint:
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        return report.as_dict()
    """
    raise NotImplementedError


def extract_metrics(report_dict: dict, report_id: str) -> list[dict]:
    """
    Parse Evidently's report dict into flat rows for BigQuery.
    Returns one dict per feature.

    Evidently's output structure (navigate carefully):
        report_dict["metrics"][0]["result"] → dataset-level results
        report_dict["metrics"][0]["result"]["drift_by_columns"] → per-feature

    TODO 4: parse the report dict to build a list of dicts, one per feature.
    Each dict must match the drift_reports schema:
        report_id, computed_at, reference_rows, current_rows,
        feature, drift_score, drift_detected, stat_test, dataset_drift

    Hints:
        result = report_dict["metrics"][0]["result"]
        dataset_drift = result["dataset_drift"]          # bool
        drift_by_cols = result["drift_by_columns"]       # dict keyed by feature name
        for feature, info in drift_by_cols.items():
            info["drift_score"]     # float
            info["drift_detected"]  # bool
            info["stattest_name"]   # string

    Log a summary: "X / Y features drifted. Dataset drift: True/False"
    """
    raise NotImplementedError


# ── BigQuery write ─────────────────────────────────────────────────────────────
def write_to_bq(client: bigquery.Client, rows: list[dict]) -> None:
    """
    Stream drift report rows into fraud.drift_reports.

    TODO 5: use client.insert_rows_json(DRIFT_TBL, rows).
    Convert computed_at to ISO string before inserting.
    Log how many rows were written.
    Check for errors and log them.
    """
    raise NotImplementedError


# ── Cloud Monitoring ───────────────────────────────────────────────────────────
def publish_to_monitoring(rows: list[dict]) -> None:
    """
    Publish drift metrics to Cloud Monitoring as custom time series.
    This enables alerting policies in GCP.

    Publishes two metric types:
      custom.googleapis.com/fraud/feature_drift_score  (one per feature)
      custom.googleapis.com/fraud/dataset_drift        (0 or 1, overall)

    TODO 6: for each row in rows, publish the drift_score with label feature=row["feature"].
            Also publish a single dataset_drift metric (0.0 or 1.0).

    Use this helper — don't implement from scratch:
    """
    mon = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{PROJECT_ID}"

    def _publish(metric_type: str, value: float, labels: dict = {}) -> None:
        series = monitoring_v3.TimeSeries()
        series.metric.type = f"custom.googleapis.com/fraud/{metric_type}"
        series.metric.labels.update(labels)
        series.resource.type = "global"
        series.resource.labels["project_id"] = PROJECT_ID

        now = datetime.now(timezone.utc).timestamp()
        interval = monitoring_v3.TimeInterval(
            end_time={"seconds": int(now), "nanos": int((now % 1) * 1e9)}
        )
        point = monitoring_v3.Point(
            interval=interval,
            value=monitoring_v3.TypedValue(double_value=float(value))
        )
        series.points = [point]
        mon.create_time_series(name=project_name, time_series=[series])

    # TODO 6a: for each row, call _publish("feature_drift_score", row["drift_score"],
    #           labels={"feature": row["feature"]})
    # TODO 6b: call _publish("dataset_drift", 1.0 if any row has dataset_drift else 0.0)
    # TODO 6c: log "Published X metrics to Cloud Monitoring"
    raise NotImplementedError


# ── Main ───────────────────────────────────────────────────────────────────────
def main(inject: bool = False) -> None:
    client = bigquery.Client(project=PROJECT_ID)
    report_id = uuid.uuid4().hex

    log.info("Loading reference data (step <= %d)...", T_SPLIT)
    reference = load_reference(client)

    log.info("Loading current data (step > %d)...", T_SPLIT)
    current = load_current(client)

    if inject:
        current = inject_drift(current)

    log.info("Reference: %d rows | Current: %d rows", len(reference), len(current))

    log.info("Running Evidently drift report...")
    report_dict = run_evidently(reference, current)

    rows = extract_metrics(report_dict, report_id)

    log.info("Writing drift report to BigQuery...")
    write_to_bq(client, rows)

    log.info("Publishing metrics to Cloud Monitoring...")
    publish_to_monitoring(rows)

    log.info("✅  Drift monitoring complete. report_id=%s", report_id)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inject-drift", action="store_true",
                   help="Artificially drift current data (for demo purposes)")
    args = p.parse_args()
    main(inject=args.inject_drift)