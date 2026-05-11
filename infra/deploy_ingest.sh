#!/usr/bin/env bash
# infra/deploy_ingest.sh
# Deploys the ingest Cloud Run service and wires up the Pub/Sub push subscription.
# Run after setup.sh whenever you resume the project.
#
# Prerequisites:
#   - setup.sh has been run (dataset, topic, subscriptions exist)
#   - schema.sql has been run (BQ tables exist) bq query --use_legacy_sql=false < src/features/schema.sql
#   - Docker image exists in Artifact Registry (if not, see step 0 below)

set -euo pipefail
export CLOUDSDK_PYTHON=python

PROJECT_ID="fraud-mlops-portfolio"
REGION="europe-west1"
SA_EMAIL="fraud-runner@${PROJECT_ID}.iam.gserviceaccount.com"
IMAGE="europe-west1-docker.pkg.dev/${PROJECT_ID}/fraud-images/ingest:v1"
PUBSUB_SUB_INGEST="transactions-ingest-sub"

log() { echo "▶  $*"; }

# ── Step 0: rebuild and push image (only needed if code changed) ──────────────
# Uncomment if you've modified src/ingest/main.py since last push:
# log "Building Docker image..."
# docker build -f src/ingest/Dockerfile -t ingest:v1 .
# gcloud auth configure-docker europe-west1-docker.pkg.dev --quiet
# docker tag ingest:v1 ${IMAGE}
# docker push ${IMAGE}

# ── Step 1: deploy Cloud Run service ─────────────────────────────────────────
log "Deploying ingest service to Cloud Run..."
gcloud run deploy ingest \
  --image=${IMAGE} \
  --region=${REGION} \
  --platform=managed \
  --service-account=${SA_EMAIL} \
  --no-allow-unauthenticated \
  --set-env-vars=PROJECT_ID=${PROJECT_ID},BQ_DATASET=fraud,BQ_TABLE=txns_raw \
  --memory=512Mi \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=3 \
  --port=8080 \
  --concurrency=80 \
  --quiet

# ── Step 2: get the deployed URL ─────────────────────────────────────────────
INGEST_URL=$(gcloud run services describe ingest \
  --region=${REGION} \
  --format='value(status.url)')
log "Ingest URL: ${INGEST_URL}"

# ── Step 3: smoke-test the health endpoint ────────────────────────────────────
log "Smoke-testing /health..."
HTTP_STATUS=$(curl --ssl-no-revoke -s -o /dev/null -w "%{http_code}" \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  ${INGEST_URL}/health)

if [ "${HTTP_STATUS}" != "200" ]; then
  echo "❌ Health check failed (HTTP ${HTTP_STATUS}). Aborting subscription wiring."
  exit 1
fi
log "Health check passed (HTTP 200)."

# ── Step 4: wire Pub/Sub push subscription to Cloud Run ──────────────────────
log "Configuring push subscription → ${INGEST_URL}/pubsub ..."
gcloud pubsub subscriptions modify-push-config ${PUBSUB_SUB_INGEST} \
  --push-endpoint=${INGEST_URL}/pubsub \
  --push-auth-service-account=${SA_EMAIL} \
  --project=${PROJECT_ID}

log "✅ Ingest pipeline restored."
log "   Service:      ${INGEST_URL}"
log "   Subscription: ${PUBSUB_SUB_INGEST} → push"
log ""
log "To send data through the pipeline:"
log "   python -m src.producer.replay \\"
log "     --csv data/paysim.csv \\"
log "     --project ${PROJECT_ID} \\"
log "     --topic transactions \\"
log "     --seconds-per-step 0.01 \\"
log "     --max-rows 10000"