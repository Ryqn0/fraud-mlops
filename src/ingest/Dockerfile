# src/ingest/Dockerfile
# Production container for the fraud-mlops ingest service.

# ── Base image ────────────────────────────────────────────────────────────────
# python:3.11-slim is a minimal Debian image with Python 3.11.
# ~150MB vs ~1GB for the full python:3.11 — fewer libs, fewer CVEs.
FROM python:3.11-slim

# ── Build-time hygiene ────────────────────────────────────────────────────────
# Don't cache pip downloads (smaller image), don't buffer stdout (logs appear immediately).
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# ── Dependencies first (for layer caching) ────────────────────────────────────
# Install only what the ingest service needs — not the full pyproject.toml.
# Why: a smaller image cold-starts faster on Cloud Run. The model training and
# serving deps (lightgbm, sklearn, mlflow, evidently) have no place in this container.
RUN pip install \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.29.0" \
    "pydantic>=2.6.0" \
    "google-cloud-bigquery>=3.20.0" \
    "pyarrow>=15.0.0"

# ── Application code (changes most frequently — goes last) ────────────────────
COPY src/ingest/ /app/src/ingest/

# Make `src` importable as a package by adding an empty __init__.py
RUN touch /app/src/__init__.py /app/src/ingest/__init__.py

# ── Non-root user ─────────────────────────────────────────────────────────────
# Containers should never run as root. Cloud Run blocks root by policy anyway.
RUN useradd --create-home --uid 1000 appuser
USER appuser

# ── Runtime ───────────────────────────────────────────────────────────────────
# Cloud Run sets $PORT (default 8080). Bind uvicorn to it.
# Using JSON form to properly handle OS signals (SIGTERM/SIGINT).
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn src.ingest.main:app --host 0.0.0.0 --port ${PORT}"]