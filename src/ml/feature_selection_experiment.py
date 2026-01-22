"""Compare full vs top-k feature pipelines for GBT or RF models."""

import argparse
import csv
import json
import os
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorSlicer
from pyspark.sql import SparkSession, functions as F

from feature_importance import extract_feature_names
from features import build_preprocessing_stages, get_label_mapping


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for feature selection experiment."""
    parser = argparse.ArgumentParser(
        description="Compare full feature vs top-k feature models (GBT/RF)."
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
        help="Random seed for train/test split and model training.",
    )
    parser.add_argument(
        "--model",
        choices=["gbt", "rf"],
        default="gbt",
        help="Model type to use (gbt|rf).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=30,
        help="Number of top features to keep.",
    )
    parser.add_argument(
        "--out_json",
        default="report/feature_selection_experiment.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--out_csv",
        default="report/feature_selection_topk.csv",
        help="Output CSV path for selected features.",
    )
    return parser.parse_args()


def compute_metrics(preds, pos_index: int):
    """Compute confusion counts and metrics for the positive index."""
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


def evaluate_model(pipeline_model, test_df):
    """Evaluate a PipelineModel on the test set."""
    preds = pipeline_model.transform(test_df).cache()
    labels, pos_index = get_label_mapping(pipeline_model)

    counts, metrics = compute_metrics(preds, pos_index)
    auc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    ).evaluate(preds)

    preds.unpersist()

    return {
        "label_mapping": {"labels": labels, "positive_label": "yes"},
        "confusion_matrix": counts,
        "metrics": {**metrics, "auc": auc},
    }


def build_classifier(model_type: str, features_col: str, seed: int):
    """Return a classifier instance configured for the requested model."""
    if model_type == "gbt":
        model = GBTClassifier(featuresCol=features_col, labelCol="label")
    elif model_type == "rf":
        model = RandomForestClassifier(featuresCol=features_col, labelCol="label")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    if model.hasParam("seed"):
        model.setSeed(seed)
    return model


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input parquet not found: {args.input}")
    if args.top_k <= 0:
        raise ValueError("--top_k must be a positive integer.")

    spark = (
        SparkSession.builder.appName("bank-feature-selection")
        .master("local[*]")
        .getOrCreate()
    )
    try:
        df = spark.read.parquet(args.input)
        if df.rdd.isEmpty():
            raise ValueError(f"No rows found in input dataset: {args.input}")

        train_df, test_df = df.randomSplit([0.8, 0.2], seed=args.seed)

        preprocess_stages = build_preprocessing_stages(label_col="y")
        baseline_classifier = build_classifier(args.model, "features", args.seed)
        baseline_pipeline = Pipeline(stages=preprocess_stages + [baseline_classifier])

        start_time = time.monotonic()
        baseline_model = baseline_pipeline.fit(train_df)
        baseline_fit_seconds = time.monotonic() - start_time

        baseline_report = evaluate_model(baseline_model, test_df)

        model_stage = baseline_model.stages[-1]
        if not hasattr(model_stage, "featureImportances"):
            raise ValueError("Model does not expose featureImportances.")

        importances = model_stage.featureImportances.toArray()
        total_features = len(importances)
        if args.top_k > total_features:
            raise ValueError(
                f"--top_k ({args.top_k}) exceeds feature count ({total_features})."
            )

        sample_df = train_df.limit(500)
        transformed = baseline_model.transform(sample_df)
        feature_names = extract_feature_names(transformed, total_features)
        if len(feature_names) != total_features:
            raise ValueError("Feature name count does not match importances length.")

        ranked = sorted(enumerate(importances), key=lambda pair: pair[1], reverse=True)[
            : args.top_k
        ]
        selected_indices = [idx for idx, _ in ranked]

        slicer = VectorSlicer(
            inputCol="features",
            outputCol="features_topk",
            indices=selected_indices,
        )
        reduced_preprocess = build_preprocessing_stages(label_col="y")
        reduced_classifier = build_classifier(args.model, "features_topk", args.seed)
        reduced_pipeline = Pipeline(
            stages=reduced_preprocess + [slicer, reduced_classifier]
        )

        start_time = time.monotonic()
        reduced_model = reduced_pipeline.fit(train_df)
        reduced_fit_seconds = time.monotonic() - start_time

        reduced_report = evaluate_model(reduced_model, test_df)

        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        with open(args.out_csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["rank", "feature", "importance"])
            for idx, (feature_idx, importance) in enumerate(ranked, start=1):
                writer.writerow([idx, feature_names[feature_idx], float(importance)])

        selected_features = []
        for rank, (feature_idx, importance) in enumerate(ranked, start=1):
            selected_features.append(
                {
                    "rank": rank,
                    "index": int(feature_idx),
                    "feature": feature_names[feature_idx],
                    "importance": float(importance),
                }
            )

        report = {
            "seed": args.seed,
            "model": args.model,
            "top_k": args.top_k,
            "train_rows": train_df.count(),
            "test_rows": test_df.count(),
            "timing": {
                "baseline_fit_seconds": baseline_fit_seconds,
                "reduced_fit_seconds": reduced_fit_seconds,
                "total_fit_seconds": baseline_fit_seconds + reduced_fit_seconds,
            },
            "baseline": baseline_report,
            "reduced": reduced_report,
            "selected_features": selected_features,
        }

        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)

        print(f"Wrote feature selection CSV to: {args.out_csv}")
        print(f"Wrote feature selection report to: {args.out_json}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
