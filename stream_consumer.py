from __future__ import annotations

import argparse
import json

from kafka import KafkaConsumer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consume predictions from Kafka")
    parser.add_argument("--bootstrap", default="localhost:9092", help="Kafka bootstrap servers")
    parser.add_argument("--topic", default="bank_pred", help="Kafka topic")
    parser.add_argument("--group-id", default="bank_consumer", help="Kafka consumer group id")
    parser.add_argument(
        "--from-beginning",
        action="store_true",
        help="Read from earliest offsets (use with a new group-id to replay)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    offset_reset = "earliest" if args.from_beginning else "latest"
    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap,
        auto_offset_reset=offset_reset,
        enable_auto_commit=True,
        group_id=args.group_id,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    print(
        f"Listening on {args.topic} group_id={args.group_id} auto_offset_reset={offset_reset} ..."
    )
    try:
        for msg in consumer:
            val = msg.value
            ts = val.get("ingest_ts")
            pred = val.get("predicted_label")
            p1 = val.get("p1")
            age = val.get("age")
            job = val.get("job")
            print(f"ts={ts} pred={pred} p1={p1} age={age} job={job}")
    except KeyboardInterrupt:
        print("Stopping consumer...")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
