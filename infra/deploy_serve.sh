#!/usr/bin/env bash
# infra/deploy_serve.sh
# Builds, pushes, and deploys the inference service to Cloud Run.
# Wires the transactions-infer-sub push subscription to the deployed URL.
#
# Usage:
#   bash infra/deploy_serve.sh            # deploy existing image
#   bash infra/deploy_serve.sh --rebuild  # rebuild image first

set -euo pipefail
export CLOUDSDK_PYTHON=python

PROJECT_ID="fraud-mlops-portfolio"
REGION="europe-west1"
SA_EMAIL="fraud-runner@${PROJECT_ID}.iam.gserviceaccount.com"
AR_REPO="europe-west1-docker.pkg.dev/${PROJECT_ID}/fraud-images"
IMAGE="${AR_REPO}/serve:v1"
PUBSUB_SUB_INFER="transactions-infer-sub"

log() { echo "▶  $*"; }

# ── Step 0 (optional): rebuild and push image ─────────────────────────────────
if [[ "${1:-}" == "--rebuild" ]]; then
    log "Building Docker image (linux/amd64)..."
    docker build --platform linux/amd64 \
        -f src/serve/Dockerfile \
        -t serve:v1 .

    log "Authenticating Docker to Artifact Registry..."
    gcloud auth configure-docker europe-west1-docker.pkg.dev --quiet

    log "Tagging and pushing image..."
    docker tag serve:v1 ${IMAGE}
    docker push ${IMAGE}
    log "Image pushed: ${IMAGE}"
fi

# ── Step 1: deploy Cloud Run service ─────────────────────────────────────────
log "Deploying inference service to Cloud Run..."
gcloud run deploy serve \
    --image=${IMAGE} \
    --region=${REGION} \
    --platform=managed \
    --service-account=${SA_EMAIL} \
    --no-allow-unauthenticated \
    --set-env-vars=PROJECT_ID=${PROJECT_ID},GCS_BUCKET=${PROJECT_ID}-fraud-artifacts,MODEL_GCS_PATH=models/lgbm_fraud.pkl,MODEL_VERSION=v1,FRAUD_THRESHOLD=0.5 \
    --memory=1Gi \
    --cpu=1 \
    --min-instances=0 \
    --max-instances=3 \
    --port=8080 \
    --concurrency=10 \
    --quiet

# ── Step 2: get the deployed URL ──────────────────────────────────────────────
SERVE_URL=$(gcloud run services describe serve \
    --region=${REGION} \
    --format='value(status.url)')
log "Serve URL: ${SERVE_URL}"

# ── Step 3: smoke-test the health endpoint ────────────────────────────────────
log "Smoke-testing /health..."
HTTP_STATUS=$(curl --ssl-no-revoke -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
    ${SERVE_URL}/health)

if [ "${HTTP_STATUS}" != "200" ]; then
    echo "❌ Health check failed (HTTP ${HTTP_STATUS}). Aborting."
    exit 1
fi
log "Health check passed (HTTP 200)."

# ── Step 4: wire Pub/Sub push subscription ────────────────────────────────────
log "Wiring ${PUBSUB_SUB_INFER} → ${SERVE_URL}/pubsub ..."
gcloud pubsub subscriptions modify-push-config ${PUBSUB_SUB_INFER} \
    --push-endpoint=${SERVE_URL}/pubsub \
    --push-auth-service-account=${SA_EMAIL} \
    --project=${PROJECT_ID}

log "✅  Inference service deployed and wired."
log ""
log "Usage:"
log "  Rebuild image:  bash infra/deploy_serve.sh --rebuild"
log "  Redeploy only:  bash infra/deploy_serve.sh"
log ""
log "To test end-to-end:"
log "  python -m src.producer.replay --csv data/paysim.csv \\"
log "    --project ${PROJECT_ID} --topic transactions \\"
log "    --seconds-per-step 0.01 --max-rows 500"