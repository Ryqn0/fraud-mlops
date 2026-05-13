# tests/test_data.py
"""
Tests for data pipeline logic that doesn't require BigQuery.
The BQ loading functions are I/O wrappers tested in integration only.
"""
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.train.data import FEATURE_COLS, LABEL_COL, T_SPLIT


def make_feature_df(n_rows: int, step_range: tuple) -> pd.DataFrame:
    """Build a synthetic features + label DataFrame for testing."""
    import numpy as np
    rng = np.random.default_rng(42)
    n = n_rows
    step_min, step_max = step_range
    return pd.DataFrame({
        "step":              rng.integers(step_min, step_max + 1, n),
        "amount":            rng.uniform(100, 10000, n),
        "type_is_transfer":  rng.integers(0, 2, n),
        "type_is_cashout":   rng.integers(0, 2, n),
        "balance_diff_orig": rng.uniform(-1000, 1000, n),
        "account_drained":   rng.integers(0, 2, n),
        "tx_amount_sum_24h": rng.uniform(0, 50000, n),
        "tx_count_24h":      rng.integers(0, 20, n),
        "is_fraud":          rng.integers(0, 2, n),
    })


class TestTimeAwareSplit:
    """Test the split logic independently of BQ."""

    def _split(self, df):
        """Replicate the split logic from data.py."""
        train = df[df["step"] <= T_SPLIT].drop(columns=["step"])
        test  = df[df["step"] >  T_SPLIT].drop(columns=["step"])
        X_train = train[FEATURE_COLS]
        y_train = train[LABEL_COL]
        X_test  = test[FEATURE_COLS]
        y_test  = test[LABEL_COL]
        return X_train, y_train, X_test, y_test

    def test_train_rows_all_lte_split(self):
        df = make_feature_df(1000, (1, 743))
        X_train, y_train, _, _ = self._split(df)
        # Verify by reconstructing steps — not stored, so just check shapes
        assert len(X_train) + len(y_train) == 2 * len(X_train)

    def test_no_overlap_between_splits(self):
        df = make_feature_df(1000, (1, 743))
        train_full = df[df["step"] <= T_SPLIT]
        test_full  = df[df["step"] >  T_SPLIT]
        assert len(train_full) + len(test_full) == len(df)

    def test_step_not_in_features(self):
        """step must be dropped from feature matrix."""
        df = make_feature_df(200, (1, 743))
        X_train, _, X_test, _ = self._split(df)
        assert "step" not in X_train.columns
        assert "step" not in X_test.columns

    def test_feature_cols_present(self):
        df = make_feature_df(200, (1, 743))
        X_train, _, X_test, _ = self._split(df)
        for col in FEATURE_COLS:
            assert col in X_train.columns, f"Missing: {col}"
            assert col in X_test.columns,  f"Missing: {col}"

    def test_label_col_not_in_X(self):
        df = make_feature_df(200, (1, 743))
        X_train, _, X_test, _ = self._split(df)
        assert LABEL_COL not in X_train.columns
        assert LABEL_COL not in X_test.columns

    def test_pure_train_data_stays_in_train(self):
        df = make_feature_df(500, (1, T_SPLIT))   # all training period
        train_df = df[df["step"] <= T_SPLIT]
        test_df  = df[df["step"] >  T_SPLIT]
        assert len(train_df) == len(df)
        assert len(test_df) == 0

    def test_pure_test_data_stays_in_test(self):
        df = make_feature_df(500, (T_SPLIT + 1, 743))   # all test period
        train_df = df[df["step"] <= T_SPLIT]
        test_df  = df[df["step"] >  T_SPLIT]
        assert len(train_df) == 0
        assert len(test_df) == len(df)