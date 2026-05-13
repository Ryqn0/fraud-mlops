# src/serve/Dockerfile
FROM python:3.11-slim

# Add OpenMP runtime required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Inference service deps — heavier than ingest (needs LightGBM + GCS)
RUN pip install \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.29.0" \
    "pydantic>=2.6.0" \
    "google-cloud-bigquery>=3.20.0" \
    "google-cloud-storage>=2.16.0" \
    "lightgbm>=4.3.0" \
    "scikit-learn>=1.4.0" \
    "joblib>=1.4.0" \
    "pyarrow>=15.0.0" \
    "db-dtypes>=1.2.0"

COPY src/serve/ /app/src/serve/
RUN touch /app/src/__init__.py /app/src/serve/__init__.py

RUN useradd --create-home --uid 1000 appuser
USER appuser

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "${PORT}"]