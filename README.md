"Streaming-insert deduplication via insertId is best-effort over a ~1-minute window. For longer-window deduplication, the daily batch pipeline runs SELECT event_id, ANY_VALUE(...) FROM txns_raw GROUP BY event_id before training. This handles late redeliveries and producer restarts that the streaming dedup cannot."

Windows curl gotcha. The system curl on Windows uses Microsoft's TLS stack (schannel) and is trying to check certificate revocation against Microsoft's servers — which sometimes fails in corporate/restricted networks.
Add the --ssl-no-revoke flag

> Note: AUC-PR of 0.998 reflects PaySim's synthetic, rule-based fraud patterns
> where engineered features (particularly account_drained) near-perfectly
> separate classes. Real-world fraud data would yield substantially lower
> metrics due to adversarial fraud patterns, label noise, and class overlap.
> The engineering pipeline, monitoring, and MLOps practices are the portfolio
> artefact — not the raw metric values.