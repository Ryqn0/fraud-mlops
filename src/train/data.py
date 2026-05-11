# src/train/data.py
"""
Loads features + labels from BigQuery and returns train/test splits.
Time-aware split: train on step <= T_SPLIT, test on step > T_SPLIT.
"""
from __future__ import annotations

import logging
import os

import pandas as pd
from google.cloud import bigquery

log = logging.getLogger("train.data")

PROJECT_ID = os.environ.get("PROJECT_ID", "fraud-mlops-portfolio")
T_SPLIT    = 600   # locked in EDA — steps 1-600 train, 601-743 test

# Features the model actually sees — no IDs, no timestamps
FEATURE_COLS = [
    "amount",
    "type_is_transfer",
    "type_is_cashout",
    "balance_diff_orig",
    "account_drained",
    "tx_amount_sum_24h",
    "tx_count_24h",
]
LABEL_COL = "is_fraud"


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Returns (X_train, y_train, X_test, y_test).

    Joins fraud.features (computed features) with fraud.txns_raw (labels).
    Splits by step <= T_SPLIT / step > T_SPLIT.
    """
    client = bigquery.Client(project=PROJECT_ID)

    # TODO 1: write the SQL query.
    # JOIN fraud.features to fraud.txns_raw ON event_id.
    # Select: all FEATURE_COLS from features, step from features, is_fraud from txns_raw.
    # Filter: event_id NOT LIKE 'test-%' AND is_fraud IS NOT NULL
    # Why the IS NOT NULL filter? Write a one-line comment connecting it
    # to label delay (concept from Ticket 4).

    # is_fraud IS NOT NULL filters out recently-ingested transactions where
    # the fraud investigation hasn't completed yet (label delay: days to weeks).
    # Training on NULL labels would silently corrupt the dataset.

    sql = """
    SELECT f.step, t.amount, f.type_is_transfer, f.type_is_cashout, f.balance_diff_orig, f.account_drained, f.tx_amount_sum_24h, f.tx_count_24h, t.is_fraud 
    FROM `fraud-mlops-portfolio.fraud.features` f
    JOIN `fraud-mlops-portfolio.fraud.txns_raw` t
    ON f.event_id = t.event_id
    WHERE f.event_id NOT LIKE 'test-%' AND t.is_fraud IS NOT NULL
    """

    log.info("Loading features + labels from BigQuery...")
    df = client.query(sql).result().to_dataframe()
    log.info("Loaded %d rows. Fraud rate: %.4f%%",
             len(df), df[LABEL_COL].mean() * 100)

    # TODO 2: apply the time-aware split.
    # train = rows where step <= T_SPLIT
    # test  = rows where step >  T_SPLIT
    # Drop the 'step' column after splitting — it's not a model feature,
    # it was only needed for the split.
    # Why must we never include 'step' as a model feature?
    # Write a one-line comment.

    # step is an artifact of the PaySim simulation, not a real-world signal.
    # In production, transactions don't have a 'step' — they have a timestamp.
    # Including step would cause the model to learn "step > 600 → higher fraud rate"
    # (the distribution shift we found in EDA), which is a spurious pattern
    # that won't generalize. The model must learn from transaction behaviour,
    # not from where it sits in the time series.

    train = df[df["step"] <= T_SPLIT].drop(columns=["step"])
    test  = df[df["step"] > T_SPLIT].drop(columns=["step"])

    X_train = train[FEATURE_COLS]
    y_train = train[LABEL_COL]
    X_test  = test[FEATURE_COLS]
    y_test  = test[LABEL_COL]

    # TODO 3: log the shape and fraud rate of both splits.
    # Format: "Train: X rows, Y fraud (Z%). Test: X rows, Y fraud (Z%)"

    log.info("Train: %d rows, %d fraud (%.4f%%)", len(train), train[LABEL_COL].sum(), train[LABEL_COL].mean() * 100)
    log.info("Test: %d rows, %d fraud (%.4f%%)", len(test), test[LABEL_COL].sum(), test[LABEL_COL].mean() * 100)

    return X_train, y_train, X_test, y_test