"""Summarize Spark Structured Streaming progress logs."""

import argparse
import json
import math
import os
import statistics


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for progress summarization."""
    parser = argparse.ArgumentParser(
        description="Summarize JSONL streaming progress metrics."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a JSONL progress log file.",
    )
    parser.add_argument(
        "--out_json",
        default="report/stream_summary.json",
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def percentile(values, pct: float):
    """Compute a percentile from a list of numeric values."""
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


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Progress log not found: {args.input}")

    processed_rows_per_second = []
    trigger_execution_ms = []
    offsets_behind = []

    with open(args.input, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            prps = payload.get("processedRowsPerSecond")
            if prps is not None:
                try:
                    processed_rows_per_second.append(float(prps))
                except (TypeError, ValueError):
                    pass

            duration_ms = payload.get("durationMs", {}) or {}
            trigger_ms = duration_ms.get("triggerExecution")
            if trigger_ms is not None:
                try:
                    trigger_execution_ms.append(float(trigger_ms))
                except (TypeError, ValueError):
                    pass

            for source in payload.get("sources", []) or []:
                behind = source.get("maxOffsetsBehindLatest")
                if behind is None:
                    behind = (source.get("metrics") or {}).get("maxOffsetsBehindLatest")
                if behind is None:
                    continue
                try:
                    offsets_behind.append(float(behind))
                except (TypeError, ValueError):
                    continue

    summary = {
        "input": args.input,
        "count": len(processed_rows_per_second),
        "processedRowsPerSecond": {
            "avg": safe_mean(processed_rows_per_second),
            "median": safe_median(processed_rows_per_second),
        },
        "triggerExecutionMs": {
            "p95": percentile(trigger_execution_ms, 0.95),
        },
        "maxOffsetsBehindLatest": max(offsets_behind) if offsets_behind else None,
    }

    print("Streaming progress summary:")
    print(
        "processedRowsPerSecond avg={avg} median={median}".format(
            avg=summary["processedRowsPerSecond"]["avg"],
            median=summary["processedRowsPerSecond"]["median"],
        )
    )
    print(
        "triggerExecutionMs p95={p95}".format(p95=summary["triggerExecutionMs"]["p95"])
    )
    print(
        "maxOffsetsBehindLatest={value}".format(value=summary["maxOffsetsBehindLatest"])
    )

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print(f"Wrote summary to: {args.out_json}")


if __name__ == "__main__":
    main()
