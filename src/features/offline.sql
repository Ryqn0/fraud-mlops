-- src/features/offline.sql
-- Batch feature pipeline: reads txns_raw, computes features, writes to features table.
-- Run via: python -m src.features.lookup
-- Scheduled run: daily via Cloud Scheduler → Cloud Run job (Ticket 17)

CREATE OR REPLACE TABLE `fraud-mlops-portfolio.fraud.features`
OPTIONS(description="Engineered features, batch-refreshed. Join to txns_raw for labels.")
AS

WITH

-- ── Step 1: filter to scoreable transaction types ─────────────────────────────
-- TODO 1a: filter txns_raw to only TRANSFER and CASH_OUT transactions.
--          Also exclude test rows (event_id LIKE 'test-%').
--          Why do we filter here rather than at model training time?
--          Write your answer as a comment.

-- 1a: The first reason would be to save time by executing the subsequent feature engineering steps on a smaller dataset. The second reason is to be sure that we only have relevant sample for the training.

scoreable AS (
    SELECT *
    FROM `fraud-mlops-portfolio.fraud.txns_raw`
    WHERE type IN ('TRANSFER', 'CASH_OUT')
      AND event_id NOT LIKE 'test-%'
),

-- ── Step 2: static features (per-transaction, no window needed) ───────────────
static_features AS (
    SELECT
        event_id,
        step,
        amount,
        name_orig,

        -- TODO 2a: type flags
        -- Cast to INT64 not BOOL (we discussed why in the schema session)
        -- Hint: IF(condition, 1, 0)
        IF(type = 'TRANSFER', 1, 0) AS type_is_transfer,
        IF(type = 'CASH_OUT', 1, 0) AS type_is_cashout,

        -- TODO 2b: balance discrepancy feature
        -- Formula: (old_balance_org - amount) - new_balance_org
        (old_balance_org - amount) - new_balance_org AS balance_diff_org,

        -- TODO 2c: account_drained flag
        -- Condition: old_balance_org > 0 AND new_balance_org = 0
        -- Note: use = not == in SQL
        IF(old_balance_org > 0 AND new_balance_org = 0, 1, 0) AS account_drained

    FROM scoreable
),

-- ── Step 3: velocity features (rolling 24-step window per account) ────────────
-- New concept: window functions (OVER clause).
-- These compute aggregates over a sliding window of rows WITHOUT collapsing the table.
--
-- PARTITION BY name_orig: each account gets its own window
-- ORDER BY step: sort within the partition by time
-- RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING:
--   TODO 3a: write a comment explaining what this range means numerically.
--            What rows are included? Why 1 PRECEDING and not CURRENT ROW?

-- RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING means the window includes rows from 24 steps before up to 1 step before the current row, excluding the current row itself. We use 1 PRECEDING instead of CURRENT ROW because we want to compute features based only on past transactions, not including the current transaction which we are scoring.

velocity_features AS (
    SELECT
        event_id,

        -- Sum of amounts for this account in the 24 steps before this transaction
        -- TODO 3b: fill in the window function
        -- Use COALESCE(..., 0.0) to handle accounts with no prior transactions
        COALESCE(
            SUM(amount) OVER (PARTITION BY name_orig ORDER BY step RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING),
            0.0
        ) AS tx_amount_sum_24h,

        -- Count of transactions for this account in the 24 steps before this one
        -- TODO 3c: same window, COUNT(*) instead of SUM
        COALESCE(
            COUNT(*) OVER (PARTITION BY name_orig ORDER BY step RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING),
            0
        ) AS tx_count_24h

    FROM scoreable
)

-- ── Final join ────────────────────────────────────────────────────────────────
SELECT
    s.event_id,
    s.step,
    s.amount,
    s.type_is_transfer,
    s.type_is_cashout,
    s.balance_diff_org,
    s.account_drained,
    v.tx_amount_sum_24h,
    v.tx_count_24h,
    CURRENT_TIMESTAMP() AS feature_computed_at

FROM static_features s
JOIN velocity_features v USING (event_id)