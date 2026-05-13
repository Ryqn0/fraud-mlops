#!/usr/bin/env bash
# infra/setup.sh
# Provisions all GCP resources for fraud-mlops.
# Run once from repo root: bash infra/setup.sh
# Safe to re-run — most commands are idempotent.
# Estimated cost to provision: $0.00 (billed on usage, not creation)


# After setup, restore application layer manually:
# 1. bq query ... < src/features/schema.sql   (recreate BQ tables)
# 2. gcloud run deploy ingest ...              (redeploy services)
# 3. python -m src.producer.replay ...         (repopulate data)
# setup.sh only restores infra skeleton — not data or deployments.

set -euo pipefail

# Ensure bq CLI finds Python on Windows/Git Bash
export CLOUDSDK_PYTHON=python

# TODO (understand, don't just copy):
#   -e  means: Exit when any command fails (non-zero exit code).
#   -u  means: Error in case of an undefined variable (e.g. typo) instead of silently continuing with an empty value.
#   -o pipefail means: If any command in a pipeline fails, the pipeline's exit code reflects that failure rather than the last command's exit code
# Look up "bash strict mode". One sentence per flag is enough.

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ID="fraud-mlops-portfolio"                              # TODO: your GCP project ID
REGION="europe-west1"                                  # TODO: your chosen region
BQ_DATASET="fraud"
GCS_BUCKET="${PROJECT_ID}-fraud-artifacts"    # globally unique by construction
PUBSUB_TOPIC="transactions"
PUBSUB_SUB_INGEST="transactions-ingest-sub"
PUBSUB_SUB_INFER="transactions-infer-sub"
AR_REPO="fraud-images"
SA_NAME="fraud-runner"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# ── Helper ────────────────────────────────────────────────────────────────────
log() { echo "▶  $*"; }

# ── 1. GCS bucket ─────────────────────────────────────────────────────────────
log "Creating GCS bucket: ${GCS_BUCKET}..."
# TODO: create the bucket in $REGION
# Hint: gcloud storage buckets create ...
# Flag you need: --location

gcloud storage buckets create gs://${GCS_BUCKET} --location=${REGION} --project=${PROJECT_ID} || echo "Bucket already exists, skipping creation."

# ── 2. BigQuery dataset ───────────────────────────────────────────────────────
log "Creating BigQuery dataset: ${BQ_DATASET}..."
# TODO: create the dataset
# Hint: bq mk ...
# Flag you need: --location (we discussed why this is irreversible)
# Flag you need: --dataset

bq --location=${REGION} mk --dataset --project_id=${PROJECT_ID} ${BQ_DATASET} || echo "Dataset already exists, skipping creation."

# ── 3. Pub/Sub topic ──────────────────────────────────────────────────────────
log "Creating Pub/Sub topic: ${PUBSUB_TOPIC}..."
# TODO: gcloud pubsub topics create ...
gcloud pubsub topics create ${PUBSUB_TOPIC} --project=${PROJECT_ID} || echo "Topic already exists, skipping creation."

# ── 4. Pub/Sub subscriptions ──────────────────────────────────────────────────
log "Creating subscriptions..."
# TODO: create $PUBSUB_SUB_INGEST as a pull subscription, ack-deadline 60s
gcloud pubsub subscriptions create ${PUBSUB_SUB_INGEST} --topic=${PUBSUB_TOPIC} --ack-deadline=60 --project=${PROJECT_ID} || echo "Subscription ${PUBSUB_SUB_INGEST} already exists, skipping creation."

# TODO: create $PUBSUB_SUB_INFER as a push subscription
#       leave --push-endpoint empty for now (set when Cloud Run is deployed)
#       hint: gcloud pubsub subscriptions create ... --push-endpoint=""

gcloud pubsub subscriptions create ${PUBSUB_SUB_INFER} --topic=${PUBSUB_TOPIC} --project=${PROJECT_ID} || echo "Subscription ${PUBSUB_SUB_INFER} already exists, skipping creation."

# ── 5. Artifact Registry repository ──────────────────────────────────────────
log "Creating Artifact Registry repo: ${AR_REPO}..."
# TODO: create a Docker-format repository
# Hint: gcloud artifacts repositories create ...
# Flags: --repository-format=docker, --location

gcloud artifacts repositories create ${AR_REPO} --repository-format=docker --location=${REGION} --project=${PROJECT_ID} || echo "Repository already exists, skipping creation."

# ── 6. Service account + IAM roles ───────────────────────────────────────────
log "Creating service account: ${SA_NAME}..."
# TODO: create the service account
# Hint: gcloud iam service-accounts create ...

gcloud iam service-accounts create ${SA_NAME} || echo "Service account already exists, skipping creation."

# Wait for the service account to propagate before granting roles.
# GCP IAM has eventual consistency — new SAs aren't immediately visible
# to the IAM policy service. Poll until describe succeeds.
log "Waiting for service account to propagate..."
for i in {1..30}; do
  if gcloud iam service-accounts describe ${SA_EMAIL} \
       --project=${PROJECT_ID} >/dev/null 2>&1; then
    log "Service account is visible after ${i}s."
    break
  fi
  sleep 1
done

log "Granting IAM roles..."
# TODO: grant each of the following roles to $SA_EMAIL on $PROJECT_ID
# Use a loop or separate lines — your choice
# Roles needed:
#   roles/bigquery.dataEditor     — read/write BQ tables
#   roles/bigquery.jobUser        — run BQ query jobs
#   roles/pubsub.subscriber       — pull/ack messages
#   roles/pubsub.publisher        — publish messages (producer)
#   roles/run.invoker             — trigger Cloud Run services
#   roles/storage.objectAdmin     — read/write GCS (model artifacts)
#   roles/aiplatform.user         — submit Vertex AI training jobs
# Hint: gcloud projects add-iam-policy-binding ...
# Flags: --member="serviceAccount:${SA_EMAIL}", --role

gcloud projects add-iam-policy-binding ${PROJECT_ID} --member="serviceAccount:${SA_EMAIL}" --role="roles/bigquery.dataEditor"
gcloud projects add-iam-policy-binding ${PROJECT_ID} --member="serviceAccount:${SA_EMAIL}" --role="roles/bigquery.jobUser"
gcloud projects add-iam-policy-binding ${PROJECT_ID} --member="serviceAccount:${SA_EMAIL}" --role="roles/pubsub.subscriber"
gcloud projects add-iam-policy-binding ${PROJECT_ID} --member="serviceAccount:${SA_EMAIL}" --role="roles/pubsub.publisher"
gcloud projects add-iam-policy-binding ${PROJECT_ID} --member="serviceAccount:${SA_EMAIL}" --role="roles/run.invoker"
gcloud projects add-iam-policy-binding ${PROJECT_ID} --member="serviceAccount:${SA_EMAIL}" --role="roles/storage.objectAdmin"
gcloud projects add-iam-policy-binding ${PROJECT_ID} --member="serviceAccount:${SA_EMAIL}" --role="roles/aiplatform.user"
gcloud projects add-iam-policy-binding ${PROJECT_ID} --member="serviceAccount:${SA_EMAIL}" --role="roles/artifactregistry.writer"
gcloud projects add-iam-policy-binding fraud-mlops-portfolio --member="serviceAccount:fraud-runner@fraud-mlops-portfolio.iam.gserviceaccount.com" --role="roles/run.developer"

log "✅  Setup complete."
log "Resources provisioned in: ${REGION}"
log "⚠️   Remember: run bash infra/teardown.sh before you stop for the day."