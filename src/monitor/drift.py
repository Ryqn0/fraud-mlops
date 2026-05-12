# src/monitor/drift.py
"""
Drift monitoring pipeline using scipy.stats.
Compares reference features (train period, step <= 600)
against current features (test period, step > 600).

Statistical tests:
  - KS test  : continuous features (amount, balance_diff_orig, tx_amount_sum_24h)
  - Chi-squared: binary features (type_is_transfer, type_is_cashout, account_drained)
  - KS test  : integer features (tx_count_24h)

Dataset drift = True when >= 50% of features drift.

Writes results to:
  - fraud.drift_reports (BigQuery)
  - custom.googleapis.com/fraud/* (Cloud Monitoring)

Usage:
    python -m src.monitor.drift
    python -m src.monitor.drift --inject-drift
"""
from __future__ import annotations

import argparse
import logging
import os
import uuid
from datetime import datetime, timezone

import pandas as pd
from scipy import stats
from google.cloud import bigquery, monitoring_v3
import numpy as np

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

# Features that are binary (0/1) — use chi-squared test
BINARY_FEATURES = {"type_is_transfer", "type_is_cashout", "account_drained"}
# Drift threshold: p-value below this → drift detected
P_VALUE_THRESHOLD = 0.05
# Dataset drift: True when this fraction of features drift
DATASET_DRIFT_THRESHOLD = 0.5


# ── Data loading ───────────────────────────────────────────────────────────────
def load_reference(client: bigquery.Client) -> pd.DataFrame:
    sql = f"""
    SELECT {', '.join(FEATURE_COLS)}
    FROM `{FEATURES_TBL}`
    WHERE step <= {T_SPLIT}
    LIMIT 50000
    """
    return client.query(sql).result().to_dataframe()


def load_current(client: bigquery.Client) -> pd.DataFrame:
    sql = f"""
    SELECT {', '.join(FEATURE_COLS)}
    FROM `{FEATURES_TBL}`
    WHERE step > {T_SPLIT}
    """
    return client.query(sql).result().to_dataframe()


def inject_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Artificially drift current data for demo. Do NOT use in production."""
    df = df.copy()
    df["amount"]            = df["amount"] * 10
    df["tx_amount_sum_24h"] = df["tx_amount_sum_24h"] * 10
    df["account_drained"]   = 1 - df["account_drained"]
    log.info("⚠️  Synthetic drift injected into current data.")
    return df


# ── Drift computation ─────────────────────────────────────────────────────────
def _ks_test(ref: pd.Series, curr: pd.Series) -> tuple[float, float]:
    """KS test for continuous/integer features. Returns (statistic, p_value)."""
    stat, p_value = stats.ks_2samp(ref.dropna(), curr.dropna())
    return float(stat), float(p_value)


def _chi2_test(ref: pd.Series, curr: pd.Series) -> tuple[float, float]:
    """Chi-squared test for binary features. Returns (statistic, p_value)."""
    ref_0  = int((ref == 0).sum())
    ref_1  = int((ref == 1).sum())
    curr_0 = int((curr == 0).sum())
    curr_1 = int((curr == 1).sum())

    # Need non-zero counts in all cells for chi-squared to be valid
    if ref_0 == 0 or ref_1 == 0 or curr_0 == 0 or curr_1 == 0:
        return 0.0, 1.0   # no test possible → no drift

    chi2, p_value, _, _ = stats.chi2_contingency(
        [[ref_0, ref_1], [curr_0, curr_1]]
    )
    return float(chi2), float(p_value)


def compute_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    report_id: str,
) -> list[dict]:
    """
    Compute per-feature drift using scipy.stats.
    Returns one dict per feature matching the drift_reports BQ schema.

    KS test for continuous features — statistic D ∈ [0,1], higher = more drift.
    Chi-squared for binary features — statistic = chi2, p-value decides drift.
    Dataset drift = True when >= 50% of features show drift.
    """
    now = datetime.now(timezone.utc)
    rows = []

    for col in FEATURE_COLS:
        if col in BINARY_FEATURES:
            stat, p_value = _chi2_test(reference[col], current[col])
            stat_test = "chi2"
            # For chi2, use p-value itself as the drift score (lower = more drift)
            drift_score = 1.0 - p_value
        else:
            stat, p_value = _ks_test(reference[col], current[col])
            stat_test = "ks"
            drift_score = stat   # KS statistic directly: 0 = no drift, 1 = max drift

        drift_detected = bool(p_value < P_VALUE_THRESHOLD)

        rows.append({
            "report_id":      report_id,
            "computed_at":    now,
            "reference_rows": len(reference),
            "current_rows":   len(current),
            "feature":        col,
            "drift_score":    drift_score,
            "drift_detected": drift_detected,
            "stat_test":      stat_test,
            "dataset_drift":  False,   # updated below after all features computed
        })

    # Compute dataset-level drift flag
    n_drifted = sum(r["drift_detected"] for r in rows)
    dataset_drift = (n_drifted / len(rows)) >= DATASET_DRIFT_THRESHOLD
    for row in rows:
        row["dataset_drift"] = dataset_drift

    log.info(
        "%d / %d features drifted (threshold: p < %.2f). Dataset drift: %s",
        n_drifted, len(rows), P_VALUE_THRESHOLD, dataset_drift,
    )
    return rows

def compute_psi(reference: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
    """Compute Population Stability Index (PSI) for a single feature."""
    # Create bins based on reference data quantiles
    bins = pd.qcut(reference, q=n_bins, duplicates="drop")
    ref_counts = bins.value_counts().sort_index()
    curr_counts = pd.cut(current, bins=bins.cat.categories).value_counts().sort_index()

    # Add small value to avoid division by zero
    ref_perc = (ref_counts + 1) / len(reference)
    curr_perc = (curr_counts + 1) / len(current)

    # Compute PSI
    psi = ((curr_perc - ref_perc) * np.log(curr_perc / ref_perc)).sum()
    return float(psi)


# ── BigQuery write ────────────────────────────────────────────────────────────
def write_to_bq(client: bigquery.Client, rows: list[dict]) -> None:
    bq_rows = []
    for row in rows:
        r = dict(row)
        r["computed_at"] = r["computed_at"].isoformat()
        bq_rows.append(r)

    errors = client.insert_rows_json(DRIFT_TBL, bq_rows)
    if errors:
        log.error("BQ insert errors: %s", errors)
    else:
        log.info("Inserted %d rows into %s", len(bq_rows), DRIFT_TBL)


# ── Cloud Monitoring ──────────────────────────────────────────────────────────
def publish_to_monitoring(rows: list[dict]) -> None:
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

    for row in rows:
        _publish(
            "feature_drift_score",
            row["drift_score"],
            labels={"feature": row["feature"]},
        )

    dataset_drift_value = 1.0 if any(r["dataset_drift"] for r in rows) else 0.0
    _publish("dataset_drift", dataset_drift_value)

    log.info(
        "Published %d feature drift scores + dataset_drift=%.0f to Cloud Monitoring",
        len(rows), dataset_drift_value,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
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

    rows = compute_drift(reference, current, report_id)

    log.info("Computed psi for 'amount' column is %.2f.", compute_psi(reference["amount"], current["amount"]))

    log.info("Writing drift report to BigQuery...")
    write_to_bq(client, rows)

    log.info("Publishing metrics to Cloud Monitoring...")
    publish_to_monitoring(rows)

    log.info("✅  Drift monitoring complete. report_id=%s", report_id)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inject-drift", action="store_true",
        help="Artificially drift current data (for demo purposes only)",
    )
    args = p.parse_args()
    main(inject=args.inject_drift)