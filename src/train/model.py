# src/train/model.py
"""
LightGBM fraud detection model.
Primary metric: AUC-PR (area under precision-recall curve).
Class imbalance handled via scale_pos_weight.
"""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,   # AUC-PR
    roc_auc_score,
    precision_recall_curve,
)

log = logging.getLogger("train.model")


# ── Hyperparameters ────────────────────────────────────────────────────────────
# These are reasonable defaults for a first run — not yet tuned.
# We'll iterate on these after seeing baseline results.
BASE_PARAMS: dict = {
    "objective":        "binary",       # binary classification
    "metric":           "average_precision",  # tracked during training
    "learning_rate":    0.05,
    "n_estimators":     500,
    "num_leaves":       31,             # controls model complexity
    "min_child_samples": 50,            # min samples per leaf — regularisation
    "subsample":        0.8,            # row subsampling per tree
    "colsample_bytree": 0.8,            # feature subsampling per tree
    "reg_alpha":        0.1,            # L1 regularisation
    "reg_lambda":       1.0,            # L2 regularisation
    "random_state":     42,
    "n_jobs":           -1,             # use all CPU cores
    "verbose":          -1,             # suppress LightGBM console spam
}


def compute_scale_pos_weight(y: pd.Series) -> float:
    """
    Compute the ratio of negative to positive examples.
    Passed to LightGBM as scale_pos_weight to handle class imbalance.

    TODO 1: implement this.
    Formula: n_negative / n_positive
    Log the result: "scale_pos_weight = X.Xf (X neg / X pos)"
    Return the float value.
    """
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    log.info("scale_pos_weight = %.4f (%.0f neg / %.0f pos)", scale_pos_weight, (y == 0).sum(), (y == 1).sum())
    return scale_pos_weight


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict | None = None,
) -> lgb.LGBMClassifier:
    """
    Train a LightGBM binary classifier.

    TODO 2: compute scale_pos_weight from y_train and add it to params.
    TODO 3: create lgb.LGBMClassifier(**params) and fit it on X_train, y_train.
    TODO 4: return the trained classifier.

    Hint: params = {**BASE_PARAMS, **(params or {})}
    This merges BASE_PARAMS with any overrides passed in.
    """
    scale_pos_weight = compute_scale_pos_weight(y_train)
    params = {**BASE_PARAMS, **(params or {}), "scale_pos_weight": scale_pos_weight}
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model


def evaluate(
    model: lgb.LGBMClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str,
) -> dict[str, float]:
    """
    Evaluate the model and return a dict of metrics.

    TODO 5: get fraud probability scores.
    Hint: model.predict_proba(X)[:, 1] returns P(fraud) for each row.

    TODO 6: compute and store these metrics:
        - auc_pr:  average_precision_score(y, scores)
        - auc_roc: roc_auc_score(y, scores)

    TODO 7: find the threshold that achieves >= 80% recall,
    then record the precision at that threshold.
    Hint: use precision_recall_curve(y, scores) which returns
          (precisions, recalls, thresholds) arrays.
          Find the index where recall >= 0.80, take the corresponding precision.
          Store as "precision_at_80_recall".

    We care about precision at 80% recall because catching 80% of fraud is a common business target.

    TODO 8: log all metrics clearly with split_name prefix.
    Format: "[train] auc_pr=0.XXX | auc_roc=0.XXX | prec@80rec=0.XXX"

    TODO 9: return the metrics dict.
    Keys must match the format: f"{split_name}_auc_pr", f"{split_name}_auc_roc", etc.
    This naming matters — MLflow uses these keys directly.
    """
    prob_scores = model.predict_proba(X)[:, 1]
    auc_pr = average_precision_score(y, prob_scores)
    auc_roc = roc_auc_score(y, prob_scores)
    precisions, recalls, thresholds = precision_recall_curve(y, prob_scores)
    indices = np.where(recalls >= 0.80)[0]
    if len(indices) == 0:
        precision_at_80_recall = 0.0   # model can't reach 80% recall at any threshold
    else:
        precision_at_80_recall = precisions[indices[-1]]  # ← [-1] not [0]        
    log.info("[%s] auc_pr=%.4f | auc_roc=%.4f | prec@80rec=%.4f", split_name, auc_pr, auc_roc, precision_at_80_recall)
    return {
        f"{split_name}_auc_pr": auc_pr,
        f"{split_name}_auc_roc": auc_roc,
        f"{split_name}_precision_at_80_recall": precision_at_80_recall
    }


def get_feature_importance(
    model: lgb.LGBMClassifier,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Returns a DataFrame of feature importances sorted descending.
    Uses LightGBM's 'gain' importance (total reduction in loss from splits on this feature).
    'gain' is more informative than 'split' (count of splits) for imbalanced problems.

    TODO 10: get importances via model.booster_.feature_importance(importance_type='gain')
    Build a DataFrame with columns ['feature', 'importance'], sorted by importance desc.
    Return it.
    """
    importances = model.booster_.feature_importance(importance_type='gain')
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)
    return importance_df