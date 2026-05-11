# src/producer/replay.py
"""
Replay producer: reads PaySim CSV, publishes each row as a Pub/Sub message,
paces the stream to simulate real-time arrival.

Usage:
    python -m src.producer.replay \
        --csv data/paysim.csv \
        --project fraud-mlops-portfolio \
        --topic transactions \
        --seconds-per-step 0.05 \
        --max-rows 10000
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from concurrent.futures import Future
from pathlib import Path

import pandas as pd
from google.cloud import pubsub_v1

import urllib3
urllib3.disable_warnings(urllib3.exceptions.HTTPWarning)
# Optional: bump pool size so it actually caches
urllib3.util.connection.HAS_IPV6 = False  # minor speedup on some Windows setups

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("replay")

def _json_default(o):
    """Coerce numpy scalars to Python builtins for JSON."""
    if hasattr(o, "item"):  # numpy scalars all have .item()
        return o.item()
    raise TypeError(f"Not JSON serialisable: {type(o)}")


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay PaySim CSV to Pub/Sub.")
    p.add_argument("--csv", type=Path, required=True, help="Path to paysim CSV.")
    p.add_argument("--project", type=str, required=True, help="GCP project ID.")
    p.add_argument("--topic", type=str, required=True, help="Pub/Sub topic name.")
    p.add_argument(
        "--seconds-per-step",
        type=float,
        default=0.05,
        help="Wall-clock seconds per PaySim step. 0.05 = full dataset in ~37s of step-time pacing.",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Stop after publishing this many rows. Useful for testing.",
    )
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────
def load_paysim(csv_path: Path, max_rows: int | None) -> pd.DataFrame:
    """Load PaySim CSV, sorted by step so events arrive in time order."""
    # TODO 1: read the CSV with pandas. nrows=max_rows if provided.

    df = pd.read_csv(csv_path, nrows=max_rows)

    # TODO 2: sort by 'step' ascending. Why does order matter for a replay?
    #         Write a one-line comment explaining.
    # Order matter for replay because we want to simulate the real-time arrival of transactions as they occurred in the original dataset. If we don't sort by 'step', we might publish events out of order, which could lead to unrealistic scenarios and affect downstream processing that relies on the temporal sequence of events.
    
    df = df.sort_values("step", ascending=True).reset_index(drop=True)

    # TODO 3: log how many rows you loaded and the step range (min, max).

    log.info(f"Loaded {len(df)} rows from {csv_path}. Step range: {df['step'].min()} to {df['step'].max()}")

    # TODO 4: return the dataframe.

    return df


# ── Message construction ──────────────────────────────────────────────────────
def row_to_message(row: pd.Series) -> tuple[bytes, dict[str, str]]:
    """
    Convert one DataFrame row to a Pub/Sub message.

    Returns:
        (payload_bytes, attributes_dict)
        - payload_bytes: JSON-encoded transaction data
        - attributes_dict: metadata attached to the message (string-only values)
    """
    # TODO 5: build a dict mapping our schema column names to row values.
    #         Map PaySim columns to BQ column names:
    #           step → step
    #           type → type
    #           amount → amount
    #           nameOrig → name_orig
    #           oldbalanceOrg → old_balance_org
    #           newbalanceOrig → new_balance_org
    #           nameDest → name_dest
    #           oldbalanceDest → old_balance_dest
    #           newbalanceDest → new_balance_dest
    #           isFraud → is_fraud
    #           isFlaggedFraud → is_flagged_fraud

    dict_mapping = {
        "step": row["step"],
        "type": row["type"],
        "amount": row["amount"],
        "name_orig": row["nameOrig"],
        "old_balance_org": row["oldbalanceOrg"],
        "new_balance_org": row["newbalanceOrig"],
        "name_dest": row["nameDest"],
        "old_balance_dest": row["oldbalanceDest"],
        "new_balance_dest": row["newbalanceDest"],
        "is_fraud": row["isFraud"],
        "is_flagged_fraud": row["isFlaggedFraud"]
    }

    # TODO 6: generate a unique event_id (use uuid.uuid4().hex)
    #         and include it in the payload.
    
    event_id = uuid.uuid4().hex
    dict_mapping["event_id"] = event_id

    # TODO 7: serialise the dict to JSON bytes (UTF-8 encoded).
    
    payload_bytes = json.dumps(dict_mapping, default=_json_default).encode("utf-8")

    # TODO 8: build the attributes dict. Pub/Sub attributes must be string -> string.
    #         Include at least: {"event_id": <id>, "step": <str(step)>}
    #         Attributes are useful for filtering and routing without parsing the payload.
    
    attributes_dict = {
        "event_id": event_id,
        "step": str(row["step"])
    }

    # TODO 9: return (payload_bytes, attributes)

    return payload_bytes, attributes_dict


# ── Publish loop ──────────────────────────────────────────────────────────────
def publish_with_pacing(
    df: pd.DataFrame,
    publisher: pubsub_v1.PublisherClient,
    topic_path: str,
    seconds_per_step: float,
) -> None:
    """
    Publish rows in step order, pacing the stream so each step boundary
    advances seconds_per_step in wall-clock time.

    Pub/Sub's client batches automatically. We collect futures and resolve
    them in batches to surface errors.
    """
    futures: list[Future] = []
    current_step: int | None = None
    rows_published = 0

    for _, row in df.iterrows():
        # TODO 10: when the step changes, sleep seconds_per_step.
        #          The first row should not sleep — only on step transitions.
        #          Hint: track current_step, compare to row["step"].
        
        if current_step is not None and row["step"] != current_step:
            time.sleep(seconds_per_step)
        current_step = row["step"]

        # TODO 11: build the message via row_to_message().
        
        payload, attributes = row_to_message(row)

        # TODO 12: call publisher.publish(topic_path, payload, **attributes)
        #          This returns a Future; append it to `futures`.
        
        future = publisher.publish(topic_path, payload, **attributes)
        futures.append(future)

        # TODO 13: every 1000 published messages:
        #            - log progress
        #            - resolve all pending futures (call .result() on each)
        #            - clear the futures list
        #          Why? To surface errors and bound memory. What happens
        #          if you NEVER resolve futures? Write a one-line comment.
        # If you never resolve futures, you may run into memory issues as the list of futures grows indefinitely. Additionally, any exceptions that occur during publishing will not be raised until you call .result() on the future, so you would not be aware of any publish failures if you never resolve them.
        
        rows_published += 1
        if rows_published % 1000 == 0:
            log.info(f"Published {rows_published} messages. Resolving futures...")
            for f in futures:
                f.result()  # This will raise an exception if the publish failed.
            futures.clear()  # Clear the list to free memory.

    # TODO 14: after the loop, resolve any remaining futures.
    for f in futures:
        f.result()
    futures.clear()

    log.info("Done. Published %d messages.", rows_published)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    df = load_paysim(args.csv, args.max_rows)

    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(args.project, args.topic)
    log.info("Publishing to %s", topic_path)

    publish_with_pacing(df, publisher, topic_path, args.seconds_per_step)


if __name__ == "__main__":
    main()