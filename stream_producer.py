from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from kafka import KafkaProducer

from config import NUMERIC_COLS


NUMERIC_SET = set(NUMERIC_COLS)


def iter_rows(path: str):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";", quotechar='"')
        for row in reader:
            yield row


def parse_row(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if k in NUMERIC_SET:
            try:
                out[k] = int(v)
            except (TypeError, ValueError):
                out[k] = None
        else:
            out[k] = v
    return out


def send_rows(
    producer: KafkaProducer,
    topic: str,
    path: str,
    rate: float,
    log_every: int,
    flush_every: int,
) -> int:
    interval = 1.0 / rate if rate and rate > 0 else 0.0
    next_time = time.time()
    sent = 0
    start = time.time()
    for row in iter_rows(path):
        if interval > 0:
            now = time.time()
            if now < next_time:
                time.sleep(next_time - now)
            next_time = max(next_time + interval, now)
        producer.send(topic, parse_row(row))
        sent += 1
        if sent == 1 or (log_every and sent % log_every == 0):
            elapsed = max(time.time() - start, 1e-6)
            rate_now = sent / elapsed
            print(f"sent={sent} rate={rate_now:.2f} msg/s elapsed={elapsed:.1f}s")
        if flush_every and sent % flush_every == 0:
            producer.flush()
    return sent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream CSV rows to Kafka as JSON")
    parser.add_argument("--input", default="data/bank-full.csv", help="Input CSV path")
    parser.add_argument("--bootstrap", default="localhost:9092", help="Kafka bootstrap servers")
    parser.add_argument("--topic", default="bank_raw", help="Kafka topic")
    parser.add_argument("--rate", type=float, default=50.0, help="Messages per second")
    parser.add_argument("--log-every", type=int, default=1000, help="Log progress every N messages")
    parser.add_argument("--flush-every", type=int, default=5000, help="Flush producer every N messages")
    parser.add_argument("--loop", action="store_true", help="Loop over file continuously")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    print(
        f"Starting producer: topic={args.topic} bootstrap={args.bootstrap} rate={args.rate} msg/s "
        f"log_every={args.log_every} flush_every={args.flush_every} loop={args.loop}"
    )

    sent_total = 0
    try:
        while True:
            sent = send_rows(
                producer,
                args.topic,
                args.input,
                args.rate,
                args.log_every,
                args.flush_every,
            )
            sent_total += sent
            print(f"Sent {sent} messages")
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("Stopping producer...")
    finally:
        producer.flush()
        producer.close()
        print(f"Total sent: {sent_total}")


if __name__ == "__main__":
    main()
