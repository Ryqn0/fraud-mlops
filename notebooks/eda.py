# notebooks/eda.py
# Ticket 2: PaySim EDA + schema design
# Run from repo root: python notebooks/eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── 0. Load ───────────────────────────────────────────────────────────────────
CSV_PATH = "data/paysim.csv"
df = pd.read_csv(CSV_PATH)

print("Shape:", df.shape)
print("\nDtypes:\n", df.dtypes)
print("\nFirst 3 rows:\n", df.head(3))
print("\nMissing values:\n", df.isnull().sum())

# ── 1. Class distribution ─────────────────────────────────────────────────────
# TODO 1a: compute and print the fraud rate (fraction of rows where isFraud==1)
#           print it as a percentage with 4 decimal places
# TODO 1b: print absolute counts of fraud vs legitimate transactions
# TODO 1c: what does this number confirm about our choice of AUC-PR over accuracy?
#           write your answer as a comment

fraud_rate = df[df["isFraud"] == 1].shape[0] / df.shape[0]
print(f"\nFraud rate: {fraud_rate:.4%}")

fraud_count = df[df["isFraud"] == 1].shape[0]
legit_count = df[df["isFraud"] == 0].shape[0]
print(f"Fraud count: {fraud_count}")
print(f"Legitimate count: {legit_count}")

# The fraud rate is ver low 0.1291%, which confirms that accuracy would be a misleading metric. A model that always predicts "legitimate" would achieve over 99.87% accuracy, but would be useless for catching fraud. This is why we need to use AUC-PR, which focuses on the performance of the positive (fraud) class.

# ── 2. Transaction types ───────────────────────────────────────────────────────
# TODO 2a: for each transaction type, compute:
#           - total count
#           - fraud count
#           - fraud rate within that type
# Print as a sorted DataFrame
# TODO 2b: which types have ANY fraud at all?
#           write your answer as a comment — this will directly limit
#           which transactions we bother scoring in production


# Some transactions will have a frauds for example transfer might be one as either banks could get scammed or client could get scammed.

type_analysis = df.groupby("type").agg(
    total_count=("isFraud", "count"),
    fraud_count=("isFraud", "sum")
).reset_index()
type_analysis["fraud_rate"] = type_analysis["fraud_count"] / type_analysis["total_count"]
type_analysis = type_analysis.sort_values("fraud_rate", ascending=False)
print("\nType analysis:\n", type_analysis)

# It seems that TRANSFER and CASH_OUT transactions have fraud, while the other types (PAYMENT, DEBIT, CASH_IN) do not. This suggests that in production, we should focus our fraud detection efforts on TRANSFER and CASH_OUT transactions, as scoring the others would likely yield no benefit.

# ── 3. Temporal analysis ───────────────────────────────────────────────────────
# TODO 3a: plot fraud count per step (x=step, y=count of fraud transactions)
#           Use a bar chart or line chart. What pattern do you see?
# TODO 3b: plot legitimate transaction count per step on the same or adjacent axis
#           Is fraud uniformly distributed over time, or clustered?
# TODO 3c: compute how many unique steps exist. Does it match 744?
# TODO 3d: where would you place T_split=600 on this chart?
#           draw a vertical line at step 600

T_SPLIT = 600

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True) # figsize=(12, 6),
# TODO: fill in the plots

fraud_per_step = df[df["isFraud"] == 1].groupby("step").size()
legit_per_step = df[df["isFraud"] == 0].groupby("step").size()
# Plot fraud count
axes[0].bar(fraud_per_step.index, fraud_per_step.values, color="r")
axes[0].axvline(x=T_SPLIT, color="k", linestyle="--", label=f"T_split={T_SPLIT}")
axes[0].set_ylabel("Fraud count")

# The pattern i see is that fraud transactions are not uniformly distributed over time, but rather show some clustering. There are certain steps where the fraud count spikes, which could indicate periods of increased fraudulent activity or specific events that trigger more fraud.

# Plot legitimate count
axes[1].bar(legit_per_step.index, legit_per_step.values, color="g")
axes[1].axvline(x=T_SPLIT, color="k", linestyle="--", label=f"T_split={T_SPLIT}")
axes[1].set_ylabel("Legitimate count")
axes[1].legend()

# Fraud is not uniformly distributed over time, but rather shows some clustering. There are certain steps where the fraud count spikes, which could indicate periods of increased fraudulent activity or specific events that trigger more fraud.
plt.tight_layout()
plt.savefig("notebooks/temporal_distribution.png", dpi=150)
print("\nSaved temporal_distribution.png")
plt.show()

print(f"\nNumber of unique steps: {df['step'].nunique()}")

# 743 unique steps, not 744. Steps run 1-743 (one hour has no transactions).
# Not a zero-indexing issue — PaySim steps start at 1.

# ── 4. Amount analysis ────────────────────────────────────────────────────────
# TODO 4a: compute summary statistics (mean, median, p95, max) for `amount`
#           separately for fraud and legitimate transactions
# TODO 4b: are fraudulent amounts typically larger or smaller than legitimate?
#           write your answer as a comment with the numbers to back it up
# TODO 4c: what is the maximum fraud amount? the minimum?
# TODO: compute and print the stats

fraud_amounts = df[df["isFraud"] == 1]["amount"]
legit_amounts = df[df["isFraud"] == 0]["amount"]

fraud_stats = {
    "mean": fraud_amounts.mean(),
    "median": fraud_amounts.median(),
    "p95": fraud_amounts.quantile(0.95),
    "max": fraud_amounts.max(),
    "min": fraud_amounts.min()
}
legit_stats = {
    "mean": legit_amounts.mean(),
    "median": legit_amounts.median(),
    "p95": legit_amounts.quantile(0.95),
    "max": legit_amounts.max(),
    "min": legit_amounts.min()
}
print("\nFraud amount stats:", fraud_stats)
print("\nLegitimate amount stats:", legit_stats)

# It seems that the frauds stats are higher than the legitimate ones for example fraud amount means is 1467967.2991403872 while the legitimate amount mean is 178197.0417274075. The median for fraud is 441423.44 while for legitimate is 74684.72. The p95 for fraud is 8006429.04 while for legitimate is 515610.4229999998. The max for fraud is 10000000.0 while for legitimate is 92445516.64. The min for fraud is 0.0 while for legitimate is 0.01. This suggests that fraudulent transactions tend to involve larger amounts compared to legitimate transactions.

# ── 5. Balance discrepancy ────────────────────────────────────────────────────
# This is the most important analytical section for feature engineering later.
#
# For a legitimate transaction:
#   newbalanceOrig should ≈ oldbalanceOrg - amount
#
# For a fraudulent transaction, fraudsters often drain the account completely,
# creating a discrepancy between expected and actual balance.
#
# TODO 5a: create a new column `balance_diff_orig`:
#           balance_diff_orig = (oldbalanceOrg - amount) - newbalanceOrig
#           A value near 0 means the balance changed as expected.
#           A large value means something unexpected happened.
#
# TODO 5b: compute mean balance_diff_orig for fraud vs legitimate transactions
#           What do you find? Write the numbers as a comment.
#
# TODO 5c: what does this suggest about a feature we should engineer in Ticket 5?
#           Write your hypothesis as a comment — one sentence.

# balance_diff_orig captures the discrepancy between the expected balance after a transaction and the actual new balance. For legitimate transactions, we would expect this value to be close to zero, indicating that the balance changed as expected. For fraudulent transactions, we might see a large negative value, indicating that the account was drained more than expected.


df["balance_diff_orig"] = (df["oldbalanceOrg"] - df["amount"]) - df["newbalanceOrig"]
print("\nBalance discrepancy (fraud vs legit):")
print(df.groupby("isFraud")["balance_diff_orig"].mean())

# Better fraud signal: was the account completely drained?
df["account_drained"] = (
    (df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)
).astype(int)

print(df.groupby("isFraud")["account_drained"].mean())
# Hypothesis: account_drained rate will be much higher for fraud

# For legitimate transactions, the mean balance_diff_orig is -201338.558109, which suggests that the balance doesn't change as expected. For fraudulent transactions, the mean balance_diff_orig is -10692.325265, indicating that there is a large discrepancy between the expected and actual balance. This suggests that a feature capturing this balance discrepancy could be a strong indicator of fraud and should be engineered in Ticket 5.

# ── 6. Time-aware split preview ───────────────────────────────────────────────
T_SPLIT = 600

train = df[df["step"] <= T_SPLIT]
test  = df[df["step"] >  T_SPLIT]

# TODO 6a: print the shape of train and test
# TODO 6b: print the fraud rate in train and test separately
#           Are they similar? What would happen if they were very different?
# TODO 6c: print the number of unique nameOrig accounts that appear in BOTH
#           train and test — this is the account overlap we discussed.
#           Is it large or small? Does it concern you? Write a comment.

print("\n── Time-aware split ──")

# TODO: fill in

print("Train shape:", train.shape)
print("Test shape:", test.shape)

train_fraud_rate = train[train["isFraud"] == 1].shape[0] / train.shape[0]
test_fraud_rate = test[test["isFraud"] == 1].shape[0] / test.shape[0]
print(f"Train fraud rate: {train_fraud_rate:.4%}")
print(f"Test fraud rate: {test_fraud_rate:.4%}")

# The fraud rates in train is 0.1057% and in test is 1.5448%. They are not identical, but if they were very different, it could indicate a distribution shift between the training and testing data, which might lead to poor model performance on the test set.

print(f"Unique nameOrig in train: {train['nameOrig'].nunique()}")
print(f"Unique nameOrig in test: {test['nameOrig'].nunique()}")
train_accounts = set(train["nameOrig"].unique())
test_accounts = set(test["nameOrig"].unique())
overlap_accounts = train_accounts.intersection(test_accounts)
print(f"Unique nameOrig in both train and test: {len(overlap_accounts)}")

# There are 6250007 unique nameOrig accounts in train and 103571 in test, with an unique nameOrig in both train and test of 271. This is a very small overlap, which could be concerning because it means that the model will have to generalize to many new accounts in the test set that it has never seen during training. This could make the fraud detection task more challenging, as the model won't be able to learn account-specific patterns from the training data.


# ⚠️ CRITICAL FINDING: fraud rate is 15x higher in test than train.
# Root cause (from temporal chart): legitimate transaction volume collapses
# after step ~400 while fraud volume stays flat. This is distribution shift.
# Implication: model threshold must be calibrated on the test period distribution,
# not the training distribution. Monitoring must alert on fraud rate changes.


# ── 7. isFlaggedFraud sanity check ────────────────────────────────────────────
# TODO 7a: of the actual fraud cases (isFraud==1), how many does
#           isFlaggedFraud catch? Print as count and percentage.
# TODO 7b: what does this tell you about rule-based systems vs ML for fraud?
#           One sentence comment.

print("\n── isFlaggedFraud coverage ──")
# TODO: fill in
flagged_fraud_count = df[(df["isFraud"] == 1) & (df["isFlaggedFraud"] == 1)].shape[0]
total_fraud_count = df[df["isFraud"] == 1].shape[0]
flagged_fraud_rate = flagged_fraud_count / total_fraud_count
print(f"Flagged fraud count: {flagged_fraud_count}")
print(f"Total fraud count: {total_fraud_count}")
print(f"Flagged fraud rate: {flagged_fraud_rate:.4%}")

# The isFlaggedFraud column only catches 0.1948% of the actual fraud cases, which suggests that the rule-based system is extremely ineffective at identifying fraud in this dataset. This highlights the need for machine learning models, which can learn complex patterns and relationships in the data that simple rules cannot capture.

print("\n✅ EDA complete.")