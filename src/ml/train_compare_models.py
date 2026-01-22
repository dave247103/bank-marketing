"""Train and compare multiple Spark MLlib models on the bank dataset."""

import argparse
import json
import os
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SparkSession, functions as F

from features import build_preprocessing_stages, get_label_mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare LogisticRegression, RandomForest, and GBT models."
    )
    parser.add_argument(
        "--input",
        default="data/processed/bank.parquet",
        help="Input parquet path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split and models.",
    )
    parser.add_argument(
        "--report_out",
        default="report/metrics_models.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--model_dir",
        default="models",
        help="Directory to write PipelineModels.",
    )
    return parser.parse_args()


def compute_metrics(preds, pos_index: int):
    label_col = F.col("label").cast("int")
    pred_col = F.col("prediction").cast("int")

    agg_row = (
        preds.select(
            F.sum(
                F.when((label_col == pos_index) & (pred_col == pos_index), 1).otherwise(
                    0
                )
            ).alias("tp"),
            F.sum(
                F.when((label_col != pos_index) & (pred_col == pos_index), 1).otherwise(
                    0
                )
            ).alias("fp"),
            F.sum(
                F.when((label_col != pos_index) & (pred_col != pos_index), 1).otherwise(
                    0
                )
            ).alias("tn"),
            F.sum(
                F.when((label_col == pos_index) & (pred_col != pos_index), 1).otherwise(
                    0
                )
            ).alias("fn"),
        )
        .collect()[0]
        .asDict()
    )

    tp = int(agg_row["tp"])
    fp = int(agg_row["fp"])
    tn = int(agg_row["tn"])
    fn = int(agg_row["fn"])
    total = tp + fp + tn + fn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )

    counts = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return counts, metrics


def train_and_evaluate(model, model_name: str, train_df, test_df, seed: int):
    preprocess_stages = build_preprocessing_stages(label_col="y")
    pipeline = Pipeline(stages=preprocess_stages + [model])

    start_time = time.monotonic()
    pipeline_model = pipeline.fit(train_df)
    fit_seconds = time.monotonic() - start_time

    preds = pipeline_model.transform(test_df).cache()
    labels, pos_index = get_label_mapping(pipeline_model)

    counts, metrics = compute_metrics(preds, pos_index)
    auc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    ).evaluate(preds)

    preds.unpersist()

    output = {
        "model": model_name,
        "fit_seconds": fit_seconds,
        "metrics": {**metrics, "auc": auc},
        "confusion_matrix": counts,
        "label_mapping": {"labels": labels, "positive_label": "yes"},
    }
    return pipeline_model, output


def print_table(results):
    header = ("Model", "AUC", "F1", "Accuracy", "Precision", "Recall", "FitSeconds")
    widths = [12, 8, 8, 10, 10, 8, 12]
    fmt = "".join([f"{{:<{width}}}" for width in widths])
    print(fmt.format(*header))
    print(fmt.format(*["-" * (width - 1) for width in widths]))
    for row in results:
        metrics = row["metrics"]
        print(
            fmt.format(
                row["model"],
                f"{metrics['auc']:.4f}",
                f"{metrics['f1']:.4f}",
                f"{metrics['accuracy']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{row['fit_seconds']:.2f}",
            )
        )


def main() -> None:
    args = parse_args()
    spark = (
        SparkSession.builder.appName("bank-train-compare")
        .master("local[*]")
        .getOrCreate()
    )
    try:
        df = spark.read.parquet(args.input)
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=args.seed)

        models = {
            "lr": LogisticRegression(featuresCol="features", labelCol="label"),
            "rf": RandomForestClassifier(featuresCol="features", labelCol="label"),
            "gbt": GBTClassifier(featuresCol="features", labelCol="label"),
        }
        for model in models.values():
            if model.hasParam("seed"):
                model.setSeed(args.seed)

        results = []
        os.makedirs(args.model_dir, exist_ok=True)

        for name, model in models.items():
            pipeline_model, output = train_and_evaluate(
                model, name, train_df, test_df, args.seed
            )
            model_path = os.path.join(args.model_dir, f"pipeline_{name}")
            pipeline_model.write().overwrite().save(model_path)
            output["model_path"] = model_path
            results.append(output)
            print(f"Saved model to: {model_path}")

        sorted_results = sorted(
            results,
            key=lambda item: (item["metrics"]["auc"], item["metrics"]["f1"]),
            reverse=True,
        )
        print("\nModel comparison (sorted by AUC, then F1):")
        print_table(sorted_results)

        os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
        with open(args.report_out, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "seed": args.seed,
                    "train_rows": train_df.count(),
                    "test_rows": test_df.count(),
                    "results": results,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
        print(f"Wrote report to: {args.report_out}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
