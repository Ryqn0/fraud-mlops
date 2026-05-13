# tests/test_features.py
"""
Unit tests for feature computation.
These run without any GCP connection — pure Python.
"""
# import pytest
# from unittest.mock import MagicMock
import sys
import os

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── We need a Transaction-like object to test compute_static_features ─────────
# Import the function directly — it has no GCP dependencies
from src.serve.app import compute_static_features


class FakeTransaction:
    """Minimal stand-in for the Transaction Pydantic model."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def make_tx(**overrides):
    """Build a default legitimate transaction."""
    defaults = dict(
        event_id="test-001",
        step=1,
        type="TRANSFER",
        amount=1000.0,
        name_orig="C001",
        old_balance_org=5000.0,
        new_balance_org=4000.0,
        name_dest="C002",
        old_balance_dest=0.0,
        new_balance_dest=1000.0,
        is_fraud=0,
    )
    defaults.update(overrides)
    return FakeTransaction(**defaults)


class TestComputeStaticFeatures:

    def test_transfer_flag_set(self):
        tx = make_tx(type="TRANSFER")
        feats = compute_static_features(tx)
        assert feats["type_is_transfer"] == 1
        assert feats["type_is_cashout"] == 0

    def test_cashout_flag_set(self):
        tx = make_tx(type="CASH_OUT")
        feats = compute_static_features(tx)
        assert feats["type_is_transfer"] == 0
        assert feats["type_is_cashout"] == 1

    def test_other_type_both_flags_zero(self):
        tx = make_tx(type="PAYMENT")
        feats = compute_static_features(tx)
        assert feats["type_is_transfer"] == 0
        assert feats["type_is_cashout"] == 0

    def test_balance_diff_formula(self):
        # balance_diff_orig = (old_balance_org - amount) - new_balance_org
        tx = make_tx(old_balance_org=5000.0, amount=1000.0, new_balance_org=4000.0)
        feats = compute_static_features(tx)
        expected = (5000.0 - 1000.0) - 4000.0   # = 0.0 (perfect balance)
        assert abs(feats["balance_diff_orig"] - expected) < 1e-9

    def test_account_drained_when_emptied(self):
        # old > 0, new = 0 → drained
        tx = make_tx(old_balance_org=5000.0, new_balance_org=0.0)
        feats = compute_static_features(tx)
        assert feats["account_drained"] == 1

    def test_account_not_drained_when_partial(self):
        tx = make_tx(old_balance_org=5000.0, new_balance_org=2000.0)
        feats = compute_static_features(tx)
        assert feats["account_drained"] == 0

    def test_account_not_drained_when_already_empty(self):
        # old = 0 → not "drained" in our definition
        tx = make_tx(old_balance_org=0.0, new_balance_org=0.0)
        feats = compute_static_features(tx)
        assert feats["account_drained"] == 0

    def test_output_keys_match_feature_cols(self):
        """Static features must be a subset of FEATURE_COLS."""
        tx = make_tx()
        feats = compute_static_features(tx)
        static_keys = {"amount", "type_is_transfer", "type_is_cashout",
                       "balance_diff_orig", "account_drained"}
        assert set(feats.keys()) == static_keys

    def test_amount_passed_through(self):
        tx = make_tx(amount=12345.67)
        feats = compute_static_features(tx)
        assert feats["amount"] == 12345.67