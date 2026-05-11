# src/train/train.py
"""
Training entrypoint.
Loads data → trains LightGBM → evaluates → logs to MLflow → saves model.

Usage:
    python -m src.train.train
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import mlflow
import mlflow.lightgbm

from src.train.data import load_data, FEATURE_COLS
from src.train.model import train_model, evaluate, get_feature_importance, BASE_PARAMS

import joblib
from google.cloud import storage as gcs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("train")

EXPERIMENT_NAME = "fraud-detection"
MODEL_DIR       = Path("models")
GCS_BUCKET  = os.environ.get("GCS_BUCKET", "fraud-mlops-portfolio-fraud-artifacts")
MODEL_GCS_PATH = "models/lgbm_fraud.pkl"
PROJECT_ID = os.environ.get("PROJECT_ID", "fraud-mlops-portfolio")

def run_training() -> None:
    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ── Load data ─────────────────────────────────────────────────────────────
    X_train, y_train, X_test, y_test = load_data()

    # ── Train + evaluate inside a single MLflow run ───────────────────────────
    with mlflow.start_run():

        # TODO 1: log BASE_PARAMS to MLflow.
        # Hint: mlflow.log_params(BASE_PARAMS)
        mlflow.log_params(BASE_PARAMS)

        # TODO 2: train the model.
        # Hint: model = train_model(X_train, y_train)
        model = train_model(X_train, y_train)

        # TODO 3: evaluate on both train AND test splits.
        # Log each metrics dict to MLflow.
        # Hint: mlflow.log_metrics(metrics_dict)
        # Why evaluate on train too? Write a one-line comment.

        # Evaluate on train to monitor for overfitting and ensure the model is learning.

        train_metrics = evaluate(model, X_train, y_train, split_name="train")
        test_metrics  = evaluate(model, X_test, y_test, split_name="test")
        mlflow.log_metrics(train_metrics)
        mlflow.log_metrics(test_metrics)

        # TODO 4: get feature importances, log as a MLflow artifact.
        # Steps:
        #   a. importance_df = get_feature_importance(model, FEATURE_COLS)
        #   b. print it to the console (log.info)
        #   c. save to "models/feature_importance.csv"
        #   d. mlflow.log_artifact("models/feature_importance.csv")

        importance_df = get_feature_importance(model, FEATURE_COLS)
        log.info("Feature importances:\n%s", importance_df)
        importance_df.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
        mlflow.log_artifact(str(MODEL_DIR / "feature_importance.csv"))  # correct

        # TODO 5: log the trained model itself.
        # Hint: mlflow.lightgbm.log_model(model, artifact_path="model")

        mlflow.lightgbm.log_model(model, name="model")

        # ── Save model to GCS for serving ────────────────────────────────────────────
        local_model_path = MODEL_DIR / "lgbm_fraud.pkl"
        joblib.dump(model, local_model_path)

        storage_client = gcs.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(GCS_BUCKET)
        blob = bucket.blob(MODEL_GCS_PATH)
        blob.upload_from_filename(str(local_model_path))
        log.info("Model saved to gs://%s/%s", GCS_BUCKET, MODEL_GCS_PATH)

        # TODO 6: get the MLflow run ID and log it.
        # Hint: mlflow.active_run().info.run_id
        # Log it with: log.info("MLflow run ID: %s", run_id)
        # This is how you find this run in the UI later.

        run_id = mlflow.active_run().info.run_id
        log.info("MLflow run ID: %s", run_id)

        log.info("Training complete.")


if __name__ == "__main__":
    MODEL_DIR.mkdir(exist_ok=True)
    run_training()