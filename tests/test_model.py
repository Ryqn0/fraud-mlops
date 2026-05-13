# tests/test_model.py
"""
Behavioral tests for the trained LightGBM model.
Loads the local model file — no GCP needed.
"""
import pytest
import joblib
from pathlib import Path

import pandas as pd

MODEL_PATH = Path("models/lgbm_fraud.pkl")
FEATURE_COLS = [
    "amount", "type_is_transfer", "type_is_cashout",
    "balance_diff_orig", "account_drained",
    "tx_amount_sum_24h", "tx_count_24h",
]


@pytest.fixture(scope="module")
def model():
    if not MODEL_PATH.exists():
        pytest.skip("Model file not found — run training first.")
    return joblib.load(MODEL_PATH)


def score(model, **feature_overrides) -> float:
    """Score a transaction given feature values."""
    defaults = dict(
        amount=1000.0, type_is_transfer=0, type_is_cashout=0,
        balance_diff_orig=0.0, account_drained=0,
        tx_amount_sum_24h=0.0, tx_count_24h=0,
    )
    defaults.update(feature_overrides)
    features = pd.DataFrame([defaults])[FEATURE_COLS]
    return float(model.predict_proba(features)[0][1])


class TestModelBehavior:

    def test_model_loads(self, model):
        assert model is not None

    def test_output_is_probability(self, model):
        s = score(model, type_is_transfer=1, account_drained=1, amount=50000)
        assert 0.0 <= s <= 1.0

    def test_drained_account_scores_higher(self, model):
        """Account drained → higher fraud score (our strongest feature)."""
        drained     = score(model, type_is_transfer=1, account_drained=1, amount=50000)
        not_drained = score(model, type_is_transfer=1, account_drained=0, amount=50000)
        assert drained > not_drained, \
            f"Drained account should score higher: {drained:.4f} vs {not_drained:.4f}"

    def test_transfer_with_fraud_signals_scores_higher(self, model):
        """Realistic fraud: TRANSFER + account drained >> benign PAYMENT."""
        fraud_transfer = score(
            model,
            type_is_transfer=1,
            account_drained=1,
            amount=50000,
            balance_diff_orig=50000,
        )
        benign_payment = score(
            model,
            type_is_transfer=0,
            type_is_cashout=0,
            account_drained=0,
            amount=500,
        )
        assert fraud_transfer > benign_payment, \
            f"Fraud TRANSFER should outscore benign PAYMENT: {fraud_transfer:.4f} vs {benign_payment:.4f}"

    def test_performance_regression(self, model):
        """Smoke test: high-risk transaction must score above 0.5."""
        high_risk = score(
            model,
            type_is_transfer=1,
            account_drained=1,
            amount=500000.0,
            balance_diff_orig=500000.0,
        )
        assert high_risk > 0.5, \
            f"High-risk transaction should score > 0.5, got {high_risk:.4f}"