import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from kafka import KafkaProducer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kafka producer for bank marketing events."
    )
    parser.add_argument(
        "--input",
        default="data/processed/bank.parquet",
        help="Input parquet path.",
    )
    parser.add_argument(
        "--broker",
        default="localhost:9092",
        help="Kafka bootstrap servers.",
    )
    parser.add_argument(
        "--topic",
        default="bank_raw",
        help="Kafka topic to publish to.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=50.0,
        help="Messages per second.",
    )
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="Repeat dataset indefinitely.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic shuffle.",
    )
    return parser.parse_args()


def to_python(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def load_records(path: str, seed):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input parquet not found: {path}")
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"No rows found in input dataset: {path}")
    records = df.to_dict(orient="records")
    if seed is not None:
        rng = np.random.RandomState(seed)
        rng.shuffle(records)
    return records


def build_message(record: dict) -> dict:
    payload = {key: to_python(value) for key, value in record.items()}
    if payload.get("y") is None:
        payload.pop("y", None)
    payload["event_time"] = datetime.now(timezone.utc).isoformat()
    return payload


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger("bank-producer")

    if args.rate <= 0:
        raise ValueError("--rate must be a positive number of messages per second.")

    records = load_records(args.input, args.seed)
    logger.info("Loaded %d records from %s", len(records), args.input)

    producer = KafkaProducer(
        bootstrap_servers=args.broker,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    )

    sent = 0
    start_time = time.monotonic()
    try:
        while True:
            for record in records:
                message = build_message(record)
                producer.send(args.topic, message)
                sent += 1

                next_send = start_time + (sent / args.rate)
                sleep_for = next_send - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)

                if sent % 1000 == 0:
                    logger.info("Sent %d messages to topic '%s'", sent, args.topic)
            if not args.repeat:
                break
    except KeyboardInterrupt:
        logger.info("Stopping producer after %d messages.", sent)
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()
