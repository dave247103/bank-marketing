"""Tune Spark MLlib models with TrainValidationSplit on the bank dataset."""

import argparse
import json
import os
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import SparkSession, functions as F

from features import build_preprocessing_stages, get_label_mapping


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for model tuning."""
    parser = argparse.ArgumentParser(
        description="Tune GBT or RandomForest models with TrainValidationSplit."
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
        help="Random seed for splits and model training.",
    )
    parser.add_argument(
        "--model",
        choices=["gbt", "rf"],
        default="gbt",
        help="Model type to tune (gbt|rf).",
    )
    parser.add_argument(
        "--out_model",
        default="models/pipeline_tuned",
        help="Output path for saved best PipelineModel.",
    )
    parser.add_argument(
        "--report_out",
        default="report/metrics_tuning.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train ratio for both train/test split and TrainValidationSplit.",
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


def build_classifier(model_type: str):
    """Return a classifier instance and the tuning parameter names."""
    if model_type == "gbt":
        model = GBTClassifier(featuresCol="features", labelCol="label")
        param_names = ["maxDepth", "maxIter", "stepSize", "subsamplingRate"]
    elif model_type == "rf":
        model = RandomForestClassifier(featuresCol="features", labelCol="label")
        param_names = ["numTrees", "maxDepth", "featureSubsetStrategy"]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model, param_names


def build_param_grid(model, model_type: str):
    """Build a small grid of parameters for tuning."""
    if model_type == "gbt":
        return (
            ParamGridBuilder()
            .addGrid(model.maxDepth, [3, 5])
            .addGrid(model.maxIter, [20, 40])
            .addGrid(model.stepSize, [0.05, 0.1])
            .addGrid(model.subsamplingRate, [0.7, 1.0])
            .build()
        )
    if model_type == "rf":
        return (
            ParamGridBuilder()
            .addGrid(model.numTrees, [50, 100])
            .addGrid(model.maxDepth, [5, 10])
            .addGrid(model.featureSubsetStrategy, ["sqrt", "log2"])
            .build()
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def extract_best_params(model, param_names):
    """Extract chosen parameter values from a fitted model."""
    output = {}
    for name in param_names:
        if not model.hasParam(name):
            continue
        param = model.getParam(name)
        output[name] = model.getOrDefault(param)
    return output


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input parquet not found: {args.input}")
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train_ratio must be between 0 and 1.")

    spark = (
        SparkSession.builder.appName("bank-train-tune").master("local[*]").getOrCreate()
    )
    try:
        df = spark.read.parquet(args.input)
        train_df, test_df = df.randomSplit(
            [args.train_ratio, 1.0 - args.train_ratio], seed=args.seed
        )

        preprocess_stages = build_preprocessing_stages(label_col="y")
        classifier, param_names = build_classifier(args.model)
        if classifier.hasParam("seed"):
            classifier.setSeed(args.seed)

        baseline_pipeline = Pipeline(stages=preprocess_stages + [classifier])
        start_time = time.monotonic()
        baseline_model = baseline_pipeline.fit(train_df)
        baseline_fit_seconds = time.monotonic() - start_time

        baseline_report = evaluate_model(baseline_model, test_df)

        param_grid = build_param_grid(classifier, args.model)
        evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC",
        )
        tvs = TrainValidationSplit(
            estimator=baseline_pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            trainRatio=args.train_ratio,
            seed=args.seed,
        )

        start_time = time.monotonic()
        tvs_model = tvs.fit(train_df)
        tuned_fit_seconds = time.monotonic() - start_time

        tuned_model = tvs_model.bestModel
        tuned_report = evaluate_model(tuned_model, test_df)
        best_params = extract_best_params(tuned_model.stages[-1], param_names)

        tuned_model.write().overwrite().save(args.out_model)

        report = {
            "seed": args.seed,
            "model": args.model,
            "train_ratio": args.train_ratio,
            "train_rows": train_df.count(),
            "test_rows": test_df.count(),
            "baseline": {
                "fit_seconds": baseline_fit_seconds,
                **baseline_report,
            },
            "tuned": {
                "fit_seconds": tuned_fit_seconds,
                "best_params": best_params,
                **tuned_report,
            },
            "model_out": args.out_model,
        }

        os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
        with open(args.report_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)

        print(f"Saved tuned model to: {args.out_model}")
        print(f"Wrote tuning report to: {args.report_out}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
