┌─────────────────────────────────┐
                      │  Replay producer (local Python) │
                      │  reads PaySim CSV, paces it     │
                      │  to look like a real stream     │
                      └──────────────┬──────────────────┘
                                     │ publish
                                     ▼
                        ┌────────────────────────────┐
                        │  Pub/Sub topic: txns       │
                        └─────┬──────────────────┬───┘
                              │                  │
                  push sub    │                  │ pull sub
                              ▼                  ▼
              ┌──────────────────────┐   ┌─────────────────────────┐
              │ Cloud Run: inference │   │ Cloud Run: ingest/raw   │
              │ - load model         │   │ - validate schema       │
              │ - feature lookup     │   │ - stream-insert to BQ   │
              │ - score, log         │   └────────────┬────────────┘
              └──────┬───────────────┘                │
                     │ predictions + features         │ raw events
                     ▼                                ▼
              ┌─────────────────────────────────────────────┐
              │  BigQuery dataset: fraud                    │
              │  - txns_raw   (immutable event log)         │
              │  - features   (offline feature table)       │
              │  - predictions (model outputs + outcomes)   │
              └──────────────┬──────────────────────────────┘
                             │
              Cloud Scheduler ─► Cloud Run jobs (orchestration)
                             │
            ┌────────────────┼─────────────────┐
            ▼                ▼                 ▼
    ┌──────────────┐ ┌────────────────┐ ┌──────────────────┐
    │ Feature      │ │ Drift report   │ │ Retrain          │
    │ pipeline     │ │ (Evidently AI) │ │ (LightGBM)       │
    │ (BQ SQL)     │ │ → BQ + alert   │ │ → MLflow + GCS   │
    └──────────────┘ └────────┬───────┘ └────────┬─────────┘
                              │                  │
                              ▼                  ▼
                       Cloud Monitoring    Model registry
                       custom metric       (GCS + manifest)
                       + alert policy           │
                                                │
                                Cloud Build ◄───┘ (on registry update)
                                     │
                                     ▼
                          Redeploys inference Cloud Run


Replay producer  ──┐
                   │  publish
                   ▼
            ┌─────────────────┐
            │ Pub/Sub topic   │
            └────────┬────────┘
                     │ fan-out (1 message → 2 subscriptions)
        ┌────────────┴────────────┐
        ▼                         ▼
┌────────────────┐         ┌────────────────┐
│ ingest sub     │         │ infer sub      │
│ (push later)   │         │ (push later)   │
└────────┬───────┘         └────────────────┘
         │ HTTP POST              (Ticket 8)
         ▼
┌────────────────────────────────┐
│ Cloud Run: ingest service      │
│   POST /pubsub                 │
│   - parse Pub/Sub envelope     │
│   - decode base64 payload      │
│   - validate with Pydantic     │
│   - streaming insert → BQ      │
│   - 200 = ack, 500 = retry     │
│   GET /health                  │
└────────────┬───────────────────┘
             │ streaming insert
             ▼
   fraud.txns_raw (BigQuery)


# ⚠️ CRITICAL FINDING: fraud rate is 15x higher in test than train.
# Root cause (from temporal chart): legitimate transaction volume collapses
# after step ~400 while fraud volume stays flat. This is distribution shift.
# Implication: model threshold must be calibrated on the test period distribution,
# not the training distribution. Monitoring must alert on fraud rate changes.

## EDA Findings (PaySim, 6.36M transactions)

| Finding | Value | Implication |
|---|---|---|
| Fraud rate | 0.1291% | Use AUC-PR, not accuracy |
| Fraud transaction types | TRANSFER, CASH_OUT only | Only score these two types in production |
| TRANSFER fraud rate | 0.77% | Higher risk type |
| CASH_OUT fraud rate | 0.18% | Lower but significant |
| Fraud amount mean | $1.47M vs $178K legit | Amount is a strong feature |
| Fraud max | $10M (capped) | Simulator ceiling |
| account_drained rate | 97.6% fraud vs 23.8% legit | Strongest single feature |
| isFlaggedFraud recall | 0.19% (16/8213) | Rules are useless here |
| Train fraud rate | 0.1057% | — |
| Test fraud rate | 1.5448% | ⚠️ 15× distribution shift |
| Root cause of shift | Legit volume collapses after step 400 | Monitor fraud rate in production |
| Account overlap train/test | 271 accounts | Test is effectively unseen accounts |
| T_split | 600 (steps 1–600 train, 601–743 test) | Locked |