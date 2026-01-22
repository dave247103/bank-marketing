"""Generate report tables from metrics artifacts."""

import argparse
import json
import os
from glob import glob


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate markdown tables for the bank marketing report."
    )
    parser.add_argument(
        "--out_path",
        default="report/tables.md",
        help="Output markdown file path.",
    )
    parser.add_argument(
        "--models_path",
        default="report/metrics_models.json",
        help="Baseline model metrics JSON.",
    )
    parser.add_argument(
        "--tuning_glob",
        default="report/metrics_tuning_*.json",
        help="Glob for tuning metrics JSON files.",
    )
    parser.add_argument(
        "--latency_glob",
        default="report/stream_latency*.json",
        help="Glob for streaming latency JSON summaries.",
    )
    return parser.parse_args()


def load_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def fmt_float(value, digits=4):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def fmt_ms(value):
    return fmt_float(value, digits=2)


def fmt_int(value):
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "n/a"


def add_table(lines, title, headers, rows):
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    if rows:
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("| " + " | ".join(["n/a"] * len(headers)) + " |")
    lines.append("")


def model_comparison_table(models_path: str):
    data = load_json(models_path) or {}
    rows = []
    for entry in data.get("results", []) or []:
        metrics = entry.get("metrics", {}) or {}
        name = entry.get("model") or entry.get("model_path") or "model"
        rows.append(
            [
                str(name),
                fmt_float(metrics.get("auc")),
                fmt_float(metrics.get("f1")),
                fmt_float(metrics.get("accuracy")),
                fmt_float(metrics.get("precision")),
                fmt_float(metrics.get("recall")),
            ]
        )
    headers = ["Model", "AUC", "F1", "Accuracy", "Precision", "Recall"]
    return headers, rows


def format_params(params: dict) -> str:
    if not params:
        return "n/a"
    parts = []
    for key in sorted(params):
        parts.append(f"{key}={params[key]}")
    return ", ".join(parts)


def tuning_tables(tuning_glob: str):
    metric_rows = []
    param_rows = []
    for path in sorted(glob(tuning_glob)):
        data = load_json(path) or {}
        if not data:
            continue
        model = data.get("model") or os.path.splitext(os.path.basename(path))[0]
        baseline_metrics = (data.get("baseline") or {}).get("metrics") or {}
        tuned_metrics = (data.get("tuned") or {}).get("metrics") or {}
        if baseline_metrics:
            metric_rows.append(
                [
                    f"{model} baseline",
                    fmt_float(baseline_metrics.get("auc")),
                    fmt_float(baseline_metrics.get("f1")),
                    fmt_float(baseline_metrics.get("accuracy")),
                    fmt_float(baseline_metrics.get("precision")),
                    fmt_float(baseline_metrics.get("recall")),
                ]
            )
        if tuned_metrics:
            metric_rows.append(
                [
                    f"{model} tuned",
                    fmt_float(tuned_metrics.get("auc")),
                    fmt_float(tuned_metrics.get("f1")),
                    fmt_float(tuned_metrics.get("accuracy")),
                    fmt_float(tuned_metrics.get("precision")),
                    fmt_float(tuned_metrics.get("recall")),
                ]
            )
        best_params = (data.get("tuned") or {}).get("best_params") or {}
        if best_params:
            param_rows.append([str(model), format_params(best_params)])
    metric_headers = ["Model", "AUC", "F1", "Accuracy", "Precision", "Recall"]
    param_headers = ["Model", "Best params"]
    return metric_headers, metric_rows, param_headers, param_rows


def latency_run_label(path: str) -> str:
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    prefix = "stream_latency"
    if name == prefix:
        return "latest"
    if name.startswith(prefix + "_"):
        return name[len(prefix) + 1 :]
    return name


def streaming_latency_table(latency_glob: str):
    headers = [
        "Run",
        "Count",
        "Avg (ms)",
        "Median (ms)",
        "p95 (ms)",
        "Min (ms)",
        "Max (ms)",
    ]
    rows = []
    for path in sorted(glob(latency_glob)):
        data = load_json(path)
        if not data:
            continue
        rows.append(
            [
                latency_run_label(path),
                fmt_int(data.get("count")),
                fmt_ms(data.get("avg")),
                fmt_ms(data.get("median")),
                fmt_ms(data.get("p95")),
                fmt_ms(data.get("min")),
                fmt_ms(data.get("max")),
            ]
        )
    return headers, rows


def main() -> None:
    args = parse_args()

    lines = ["# Report Tables", ""]

    headers, rows = model_comparison_table(args.models_path)
    add_table(lines, "Model comparison", headers, rows)

    metric_headers, metric_rows, param_headers, param_rows = tuning_tables(
        args.tuning_glob
    )
    add_table(lines, "Tuning metrics", metric_headers, metric_rows)
    add_table(lines, "Tuning parameters", param_headers, param_rows)

    latency_headers, latency_rows = streaming_latency_table(args.latency_glob)
    add_table(lines, "Streaming latency", latency_headers, latency_rows)

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")
    print(f"Wrote tables to {args.out_path}")


if __name__ == "__main__":
    main()
