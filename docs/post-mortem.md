# Post-mortems

Production incidents encountered during development.
Each follows the format: timeline → root cause → resolution → prevention.

---

## PM-001 — Duplicate predictions from retry storm

**Date:** 2026-05-11
**Impact:** ~1,100 rows written for 500 unique transactions

**Timeline:**
- `serve` container deployed without scikit-learn → all requests returned HTTP 500
- Pub/Sub retried 500 messages for ~20 minutes with exponential backoff
- Service fixed, backlog drained, duplicates entered BQ past dedup window (~1 min)

**Root cause:** BQ streaming insert dedup (`insertId`) is best-effort within ~1 minute. Retries that arrived later created duplicate rows.

**Resolution:** `gcloud pubsub subscriptions seek --time=<now>` to drain backlog.

**Prevention:**
- Add scikit-learn to Dockerfile (done)
- Monitor `HTTP 5xx rate` on serving endpoint; alert if > 1% over 5 minutes
- Consider BigQuery Storage Write API for exactly-once semantics

---

## PM-002 — numpy 1.x segfault on Python 3.13 Windows

**Date:** 2026-05-12
**Impact:** Evidently drift monitoring non-functional

**Root cause:** Evidently 0.4.30 requires numpy < 2.0. numpy 1.26.4 built with MINGW-W64 on Python 3.13 Windows is explicitly experimental and crashes.

**Resolution:** Dropped Evidently dependency. Reimplemented KS test and chi-squared directly via `scipy.stats`. Equivalent statistical rigour, zero dependency conflicts.

**Prevention:** Pin major ML dependencies explicitly in `pyproject.toml`. Never use `>=X.Y` without an upper bound for libraries with unstable APIs.

---

## PM-003 — CMD variable expansion failure in CI-built image

**Date:** 2026-05-13
**Impact:** serve service failed to start after first CD deployment

**Root cause:** Dockerfile used shell-form `CMD exec uvicorn ... --port ${PORT}`. Variable expansion was unreliable when the image was built in GitHub Actions environment vs locally.

**Resolution:** Changed to explicit JSON form with sh -c:
```dockerfile
CMD ["sh", "-c", "exec uvicorn src.serve.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
```

**Prevention:** Always use JSON form CMD for production containers. Shell form behavior depends on the runtime environment's default shell.

---

## PM-004 — Artifact Registry permission gap in CD pipeline

**Date:** 2026-05-13
**Impact:** CD pipeline failed on first run; images not pushed

**Root cause:** `fraud-runner` SA was missing `roles/artifactregistry.writer`. Original `setup.sh` only granted roles for BQ, Pub/Sub, Storage, Cloud Run, and Vertex AI.

**Resolution:** Granted `roles/artifactregistry.writer` and added to `setup.sh`.

**Prevention:** Run CD pipeline on day 1 of a new project to surface permission gaps early. Don't wait until the system is complete.