"Streaming-insert deduplication via insertId is best-effort over a ~1-minute window. For longer-window deduplication, the daily batch pipeline runs SELECT event_id, ANY_VALUE(...) FROM txns_raw GROUP BY event_id before training. This handles late redeliveries and producer restarts that the streaming dedup cannot."

Windows curl gotcha. The system curl on Windows uses Microsoft's TLS stack (schannel) and is trying to check certificate revocation against Microsoft's servers — which sometimes fails in corporate/restricted networks.
Add the --ssl-no-revoke flag