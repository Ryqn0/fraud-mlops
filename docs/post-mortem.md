"BigQuery streaming buffer is immutable for ~90 minutes after insert. DELETE/UPDATE statements fail until flush. Test data is namespaced with prefix test-* and filtered at query time."

## Post-mortem #1 — duplicate predictions during sklearn outage

Root cause: serving container missing scikit-learn dependency caused 500 errors.
Pub/Sub retried ~125 messages beyond the BQ insertId dedup window (~1 min).
Result: 625 rows in predictions for 500 unique transactions.

Lesson: BQ streaming dedup is best-effort within ~1 min. Production fix would be 
a daily `DELETE FROM predictions WHERE event_id IN (SELECT event_id ... GROUP BY ... HAVING COUNT(*) > 1)` 
dedup job, or use BigQuery Storage Write API with exactly-once semantics.