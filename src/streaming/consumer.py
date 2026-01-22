import argparse
import json
import logging
import math
import os
import statistics
from datetime import datetime, timezone

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
    parser.add_argument(
        "--latency_out",
        default="report/stream_latency.json",
        help="Optional JSON output path for latency summary.",
    )
    parser.add_argument(
        "--max_messages",
        type=int,
        default=None,
        help="Stop after N messages (optional).",
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


def parse_iso_timestamp(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def percentile(values, pct: float):
    if not values:
        return None
    if not 0.0 <= pct <= 1.0:
        raise ValueError("Percentile must be between 0 and 1.")
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    return float(
        sorted_vals[int(f)] + (sorted_vals[int(c)] - sorted_vals[int(f)]) * (k - f)
    )


def safe_mean(values):
    if not values:
        return None
    return float(sum(values) / len(values))


def safe_median(values):
    if not values:
        return None
    return float(statistics.median(values))


def summarize_latencies(latencies):
    if not latencies:
        return {
            "count": 0,
            "avg": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(latencies),
        "avg": safe_mean(latencies),
        "median": safe_median(latencies),
        "p95": percentile(latencies, 0.95),
        "min": float(min(latencies)),
        "max": float(max(latencies)),
    }


def format_ms(value):
    return "n/a" if value is None else f"{value:.2f}"


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger("bank-consumer")

    if args.max_messages is not None and args.max_messages <= 0:
        raise ValueError("--max_messages must be a positive integer.")

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.broker,
        group_id=args.group,
        auto_offset_reset="earliest" if args.from_beginning else "latest",
        enable_auto_commit=True,
        value_deserializer=deserialize,
    )

    tp = fp = tn = fn = 0
    message_count = 0
    latencies_ms = []
    latency_log_every = 100
    try:
        for message in consumer:
            message_count += 1
            payload = message.value
            if payload is None:
                logger.warning("Skipping invalid JSON message.")
                continue

            print(json.dumps(payload))

            event_time = parse_iso_timestamp(payload.get("event_time"))
            scored_at = parse_iso_timestamp(payload.get("scored_at"))
            if event_time and scored_at:
                latency_ms = (scored_at - event_time).total_seconds() * 1000.0
                latencies_ms.append(latency_ms)
                if len(latencies_ms) % latency_log_every == 0:
                    stats = summarize_latencies(latencies_ms)
                    logger.info(
                        "Latency stats (n=%d) avg=%s median=%s p95=%s min=%s max=%s ms",
                        stats["count"],
                        format_ms(stats["avg"]),
                        format_ms(stats["median"]),
                        format_ms(stats["p95"]),
                        format_ms(stats["min"]),
                        format_ms(stats["max"]),
                    )

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

            if args.max_messages is not None and message_count >= args.max_messages:
                logger.info("Reached max_messages=%d; stopping.", args.max_messages)
                break
    except KeyboardInterrupt:
        logger.info("Stopping consumer.")
    finally:
        if args.latency_out:
            summary = summarize_latencies(latencies_ms)
            summary["broker"] = args.broker
            summary["topic"] = args.topic
            summary["message_count"] = message_count
            summary["latencies_ms"] = latencies_ms
            try:
                out_dir = os.path.dirname(args.latency_out)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                with open(args.latency_out, "w", encoding="utf-8") as handle:
                    json.dump(summary, handle, indent=2, sort_keys=True)
                logger.info("Wrote latency summary to %s", args.latency_out)
            except OSError as exc:
                logger.warning("Failed to write latency summary: %s", exc)
        consumer.close()


if __name__ == "__main__":
    main()
