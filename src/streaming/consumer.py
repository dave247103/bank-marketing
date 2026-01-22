import argparse
import json
import logging

from kafka import KafkaConsumer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kafka consumer for scored bank marketing events."
    )
    parser.add_argument(
        "--broker",
        default="localhost:9092",
        help="Kafka bootstrap servers.",
    )
    parser.add_argument(
        "--topic",
        default="bank_scored",
        help="Kafka topic to consume.",
    )
    parser.add_argument(
        "--group",
        default="bank-scored-consumer",
        help="Consumer group id.",
    )
    parser.add_argument(
        "--from_beginning",
        action="store_true",
        help="Consume from earliest offset.",
    )
    return parser.parse_args()


def deserialize(payload: bytes):
    try:
        return json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError:
        return None


def normalize_label(value):
    if value is None:
        return None
    if isinstance(value, str):
        label = value.strip().lower()
        if label in {"yes", "no"}:
            return label
        if label in {"1", "0"}:
            return "yes" if label == "1" else "no"
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return "yes" if int(value) == 1 else "no"
    return None


def compute_metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )
    return total, precision, recall, f1


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger("bank-consumer")

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.broker,
        group_id=args.group,
        auto_offset_reset="earliest" if args.from_beginning else "latest",
        enable_auto_commit=True,
        value_deserializer=deserialize,
    )

    tp = fp = tn = fn = 0
    try:
        for message in consumer:
            payload = message.value
            if payload is None:
                logger.warning("Skipping invalid JSON message.")
                continue

            print(json.dumps(payload))

            y_label = normalize_label(payload.get("y"))
            pred_label = normalize_label(payload.get("prediction"))
            if pred_label is None:
                prob = payload.get("probability_yes")
                if prob is not None:
                    pred_label = "yes" if float(prob) >= 0.5 else "no"

            if y_label is None or pred_label is None:
                continue

            if y_label == "yes":
                if pred_label == "yes":
                    tp += 1
                else:
                    fn += 1
            elif y_label == "no":
                if pred_label == "yes":
                    fp += 1
                else:
                    tn += 1

            total, precision, recall, f1 = compute_metrics(tp, fp, tn, fn)
            print(
                "Metrics (n={total}): TP={tp} FP={fp} TN={tn} FN={fn} "
                "precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}".format(
                    total=total,
                    tp=tp,
                    fp=fp,
                    tn=tn,
                    fn=fn,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                )
            )
    except KeyboardInterrupt:
        logger.info("Stopping consumer.")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
