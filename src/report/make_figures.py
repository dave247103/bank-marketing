"""Generate report figures from model and streaming artifacts."""

import argparse
import json
import os
import re
from glob import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate figures for the bank marketing report."
    )
    parser.add_argument(
        "--out_dir",
        default="report/figures",
        help="Output directory for PNG figures.",
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


def percentile(values, pct: float):
    if not values:
        return None
    if not 0.0 <= pct <= 1.0:
        raise ValueError("Percentile must be between 0 and 1.")
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * pct
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def label_counts_from_parquet(path: str):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path, columns=["y"])
    except Exception:
        return None
    if "y" not in df.columns or df.empty:
        return None
    series = df["y"].dropna().astype(str).str.lower()
    counts = series.value_counts()
    return {
        "no": int(counts.get("no", 0)),
        "yes": int(counts.get("yes", 0)),
        "source": path,
    }


def label_counts_from_eda(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return None
    match_no = re.search(r"`no`:\s*([0-9,]+)", text)
    match_yes = re.search(r"`yes`:\s*([0-9,]+)", text)
    if not match_no or not match_yes:
        match_no = re.search(r"\bno:\s*([0-9,]+)", text)
        match_yes = re.search(r"\byes:\s*([0-9,]+)", text)
    if not match_no or not match_yes:
        return None
    return {
        "no": int(match_no.group(1).replace(",", "")),
        "yes": int(match_yes.group(1).replace(",", "")),
        "source": path,
    }


def load_label_counts():
    counts = label_counts_from_parquet("data/processed/bank.parquet")
    if counts:
        return counts
    return label_counts_from_eda("report/eda.md")


def plot_label_distribution(counts: dict, out_dir: str):
    labels = ["no", "yes"]
    values = [counts.get("no", 0), counts.get("yes", 0)]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(labels, values, color=["#4C78A8", "#F58518"])
    ax.set_title("Label distribution")
    ax.set_ylabel("Count")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    out_path = os.path.join(out_dir, "label_distribution.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def load_model_metrics():
    models = []
    models_path = "report/metrics_models.json"
    models_data = load_json(models_path)
    if models_data:
        for entry in models_data.get("results", []) or []:
            metrics = entry.get("metrics", {}) or {}
            name = entry.get("model") or entry.get("model_path") or "model"
            if metrics:
                models.append(
                    {
                        "name": str(name),
                        "auc": metrics.get("auc"),
                        "f1": metrics.get("f1"),
                    }
                )

    for path in sorted(glob("report/metrics_tuning_*.json")):
        tuning_data = load_json(path)
        if not tuning_data:
            continue
        tuned = tuning_data.get("tuned", {}) or {}
        metrics = tuned.get("metrics", {}) or {}
        if not metrics:
            continue
        model_name = tuning_data.get("model")
        name = (
            f"{model_name}_tuned"
            if model_name
            else os.path.splitext(os.path.basename(path))[0]
        )
        models.append(
            {
                "name": name,
                "auc": metrics.get("auc"),
                "f1": metrics.get("f1"),
            }
        )
    return [m for m in models if m.get("auc") is not None and m.get("f1") is not None]


def plot_model_comparison(models: list, out_dir: str):
    if not models:
        return
    names = [m["name"] for m in models]
    aucs = [m["auc"] for m in models]
    f1s = [m["f1"] for m in models]

    x = list(range(len(models)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(
        [idx - width / 2 for idx in x],
        aucs,
        width=width,
        label="AUC",
        color="#4C78A8",
    )
    ax.bar(
        [idx + width / 2 for idx in x],
        f1s,
        width=width,
        label="F1",
        color="#F58518",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison (AUC / F1)")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(out_dir, "model_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def resolve_feature_importance_path():
    preferred = "report/feature_importance_gbt.csv"
    if os.path.exists(preferred):
        return preferred
    paths = sorted(glob("report/feature_importance_*.csv"))
    return paths[0] if paths else None


def plot_feature_importance(path: str, out_dir: str):
    try:
        df = pd.read_csv(path)
    except Exception:
        return
    if "feature" not in df.columns or "importance" not in df.columns:
        return
    top = df.sort_values("importance", ascending=False).head(20)
    top = top.iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"], color="#54A24B")
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 feature importance")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "feature_importance_top20.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def load_stream_summary():
    return load_json("report/stream_summary.json")


def plot_latency_histogram(latency_path: str, stream_summary: dict, out_dir: str):
    latency_data = load_json(latency_path)
    if not latency_data:
        return
    latencies = latency_data.get("latencies_ms") or []
    if not latencies:
        return
    p95 = latency_data.get("p95")
    if p95 is None:
        p95 = percentile(latencies, 0.95)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(latencies, bins=30, color="#4C78A8", edgecolor="white")
    if p95 is not None:
        ax.axvline(
            p95,
            color="#E45756",
            linestyle="--",
            linewidth=1.5,
            label=f"p95 {p95:.1f} ms",
        )
        ax.legend()
    ax.set_title("Streaming latency")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")

    if stream_summary:
        prps = stream_summary.get("processedRowsPerSecond") or {}
        trigger = stream_summary.get("triggerExecutionMs") or {}
        lines = []
        if prps.get("avg") is not None:
            lines.append(f"rows/s avg {prps['avg']:.1f}")
        if prps.get("median") is not None:
            lines.append(f"rows/s median {prps['median']:.1f}")
        if trigger.get("p95") is not None:
            lines.append(f"trigger p95 {trigger['p95']:.1f} ms")
        if lines:
            ax.text(
                0.98,
                0.98,
                "\n".join(lines),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox={
                    "boxstyle": "round",
                    "facecolor": "white",
                    "alpha": 0.7,
                    "edgecolor": "#999999",
                },
            )

    fig.tight_layout()
    out_path = os.path.join(out_dir, "stream_latency_hist.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    counts = load_label_counts()
    if counts:
        plot_label_distribution(counts, args.out_dir)
    else:
        print("Label counts not found; skipping label distribution figure.")

    models = load_model_metrics()
    if models:
        plot_model_comparison(models, args.out_dir)
    else:
        print("Model metrics not found; skipping model comparison figure.")

    fi_path = resolve_feature_importance_path()
    if fi_path:
        plot_feature_importance(fi_path, args.out_dir)
    else:
        print("Feature importance CSV not found; skipping feature importance figure.")

    stream_summary = load_stream_summary()
    plot_latency_histogram("report/stream_latency.json", stream_summary, args.out_dir)


if __name__ == "__main__":
    main()
