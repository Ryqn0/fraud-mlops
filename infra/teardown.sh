#!/usr/bin/env bash
# infra/teardown.sh
# Destroys ALL resources created by setup.sh.
# Run every time you stop working for the day.
# Cost when resources are deleted: $0.00/day on those resources.

set -euo pipefail

# Ensure bq CLI finds Python on Windows/Git Bash
export CLOUDSDK_PYTHON=python

# ── Configuration (must match setup.sh exactly) ───────────────────────────────
PROJECT_ID="fraud-mlops-portfolio"
REGION="europe-west1"
BQ_DATASET="fraud"
GCS_BUCKET="${PROJECT_ID}-fraud-artifacts"
PUBSUB_TOPIC="transactions"
PUBSUB_SUB_INGEST="transactions-ingest-sub"
PUBSUB_SUB_INFER="transactions-infer-sub"
AR_REPO="fraud-images"
SA_NAME="fraud-runner"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

log() { echo "🗑  $*"; }

# ── Destroy in reverse order of creation ──────────────────────────────────────
# Why reverse order? Think about it before reading on.
# Hint: some resources depend on others. Deleting a parent before a child
#       can either fail or leave orphaned resources that keep billing you.

# ── 1. Pub/Sub subscriptions (before topic) ───────────────────────────────────
log "Deleting Pub/Sub subscriptions..."
# TODO: delete both subscriptions
# || true — because if they don't exist, that's fine

# Reset push config before deletion to avoid dangling config warnings
gcloud pubsub subscriptions modify-push-config ${PUBSUB_SUB_INGEST} --push-endpoint="" --project=${PROJECT_ID} 2>/dev/null || true

gcloud pubsub subscriptions delete ${PUBSUB_SUB_INGEST} --project=${PROJECT_ID} || echo "Subscription ${PUBSUB_SUB_INGEST} does not exist, skipping deletion."
gcloud pubsub subscriptions delete ${PUBSUB_SUB_INFER} --project=${PROJECT_ID} || echo "Subscription ${PUBSUB_SUB_INFER} does not exist, skipping deletion."

# ── 2. Pub/Sub topic ──────────────────────────────────────────────────────────
log "Deleting Pub/Sub topic..."
# TODO: delete the topic

gcloud pubsub topics delete ${PUBSUB_TOPIC} --project=${PROJECT_ID} || echo "Topic does not exist, skipping deletion."

# ── 3. BigQuery dataset ───────────────────────────────────────────────────────
log "Deleting BigQuery dataset..."
# TODO: delete the dataset
# Warning flag you need: --recursive (why? what does it do?)

bq --location=${REGION} rm -r -f --project_id=${PROJECT_ID} ${BQ_DATASET} || echo "Dataset does not exist, skipping deletion."

# ── 4. GCS bucket ─────────────────────────────────────────────────────────────
log "Deleting GCS bucket..."
# TODO: delete the bucket
# Warning flag you need: --recursive (same question — why?)

gcloud storage rm -r gs://${GCS_BUCKET} --project=${PROJECT_ID} --quiet || echo "Bucket does not exist, skipping deletion."

# ── 5. Artifact Registry repository ──────────────────────────────────────────
log "Deleting Artifact Registry repo..."
# TODO: delete the repo

gcloud artifacts repositories delete ${AR_REPO} --location=${REGION} --project=${PROJECT_ID} --quiet || echo "Repository does not exist, skipping deletion."

# ── 5.5. Cloud Run services ───────────────────────────────────────────────────
log "Deleting Cloud Run services..."
gcloud run services delete ingest --region=${REGION} --project=${PROJECT_ID} --quiet || echo "Service 'ingest' does not exist, skipping."
gcloud run services delete serve --region=${REGION} --project=${PROJECT_ID} --quiet || echo "Service 'serve' does not exist, skipping."

# Add to infra/teardown.sh before service account deletion
log "Deleting alert policies..."
gcloud alpha monitoring policies list \
  --project=${PROJECT_ID} \
  --format="value(name)" | \
  xargs -I{} gcloud alpha monitoring policies delete {} \
  --project=${PROJECT_ID} --quiet 2>/dev/null || true

# ── Alert policies ────────────────────────────────────────────────────────────
log "Deleting alert policies..."
gcloud alpha monitoring policies list \
  --project=${PROJECT_ID} --format="value(name)" 2>/dev/null | \
  while read -r policy; do
    gcloud alpha monitoring policies delete "${policy}" \
      --project=${PROJECT_ID} --quiet 2>/dev/null || true
  done

# ── 6. Service account ────────────────────────────────────────────────────────
log "Deleting service account..."
# TODO: delete the service account
# Note: IAM bindings are automatically removed when the SA is deleted —
#       you don't need to explicitly revoke each role.

gcloud iam service-accounts delete ${SA_EMAIL} --project=${PROJECT_ID} --quiet || echo "Service account does not exist, skipping deletion."

log "✅  All resources destroyed."
log "Run bash infra/setup.sh to reprovision from scratch."