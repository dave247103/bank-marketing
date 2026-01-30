from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    DecisionTreeClassificationModel,
    GBTClassificationModel,
    GBTClassifier,
    LogisticRegression,
    LogisticRegressionModel,
    RandomForestClassificationModel,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from config import DEFAULT_SEED
from preprocess import clean_df, get_feature_columns, get_spark, read_csv
from utils_report import save_fig, save_table


def build_preprocess_stages(cat_cols: List[str], num_cols: List[str]) -> List:
    stages: List = []
    ohe_cols: List[str] = []

    for col in cat_cols:
        idx = f"{col}_idx"
        ohe = f"{col}_ohe"
        stages.append(StringIndexer(inputCol=col, outputCol=idx, handleInvalid="keep"))
        stages.append(OneHotEncoder(inputCols=[idx], outputCols=[ohe], handleInvalid="keep"))
        ohe_cols.append(ohe)

    stages.append(VectorAssembler(inputCols=num_cols, outputCol="num_vec"))
    stages.append(StandardScaler(inputCol="num_vec", outputCol="num_scaled", withMean=False, withStd=True))
    stages.append(VectorAssembler(inputCols=ohe_cols + ["num_scaled"], outputCol="features"))
    return stages


def build_pipeline(model, cat_cols: List[str], num_cols: List[str]) -> Pipeline:
    stages = build_preprocess_stages(cat_cols, num_cols)
    stages.append(model)
    return Pipeline(stages=stages)


def evaluate_predictions(pred_df: DataFrame) -> Dict[str, float]:
    rdd = pred_df.select("prediction", "label").rdd.map(lambda r: (float(r[0]), float(r[1])))
    metrics = MulticlassMetrics(rdd)

    evaluator = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderROC"
    )
    auc = evaluator.evaluate(pred_df)

    return {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision(1.0),
        "recall": metrics.recall(1.0),
        "f1": metrics.fMeasure(1.0),
        "auc": auc,
    }


def confusion_matrix(pred_df: DataFrame) -> List[List[int]]:
    counts = pred_df.groupBy("label", "prediction").count().collect()
    matrix = [[0, 0], [0, 0]]
    for row in counts:
        label = int(row["label"])
        pred = int(row["prediction"])
        matrix[label][pred] = int(row["count"])
    return matrix


def plot_confusion_matrix(pred_df: DataFrame, report_dir: str) -> None:
    import matplotlib.pyplot as plt

    cm = confusion_matrix(pred_df)
    plt.figure(figsize=(4, 3))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks([0, 1], ["pred 0", "pred 1"])
    plt.yticks([0, 1], ["label 0", "label 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i][j], ha="center", va="center", color="black")
    save_fig(Path(report_dir) / "confusion_matrix.png")


def plot_feature_importance(pipeline_model, df: DataFrame, report_dir: str) -> bool:
    import pandas as pd
    import matplotlib.pyplot as plt

    model = pipeline_model.stages[-1]
    if isinstance(
        model,
        (
            RandomForestClassificationModel,
            GBTClassificationModel,
            DecisionTreeClassificationModel,
        ),
    ):
        importances = model.featureImportances.toArray()
        labels = [f"f{i}" for i in range(len(importances))]

        pdf = (
            pd.DataFrame({"feature": labels, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(20)
        )

        plt.figure(figsize=(7, 5))
        plt.barh(pdf["feature"], pdf["importance"], color="#54A24B")
        plt.gca().invert_yaxis()
        plt.title("Top Feature Importances")
        plt.xlabel("importance")
        save_fig(Path(report_dir) / "feature_importance.png")
        return True

    if isinstance(model, LogisticRegressionModel):
        return False

    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Spark ML models for bank marketing")
    parser.add_argument("--input", default="data/bank-full.csv", help="Input CSV path")
    parser.add_argument("--report-dir", default="report", help="Report output directory")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Artifacts output directory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--train-frac", type=float, default=0.8, help="Training fraction")
    parser.add_argument(
        "--pdays-features",
        choices=["both", "drop_pdays_clean", "drop_prev_contacted"],
        default="both",
        help="Choose pdays feature set",
    )

    def str2bool(v: str) -> bool:
        val = str(v).lower()
        if val in {"1", "true", "yes", "y"}:
            return True
        if val in {"0", "false", "no", "n"}:
            return False
        raise argparse.ArgumentTypeError(f"Invalid boolean: {v}")

    parser.add_argument(
        "--keep-duration",
        type=str2bool,
        default=False,
        help="Keep duration feature (default: false)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    Path(args.report_dir).mkdir(parents=True, exist_ok=True)
    Path(args.artifacts_dir).mkdir(parents=True, exist_ok=True)

    spark = get_spark("bank-train")
    spark.sparkContext.setLogLevel("WARN")

    df_raw = read_csv(spark, str(input_path))
    df = clean_df(df_raw, keep_duration=args.keep_duration, include_label=True)
    df = df.withColumn("row_id", F.monotonically_increasing_id())

    cat_cols, num_cols = get_feature_columns(args.keep_duration, args.pdays_features)

    fractions = {0: args.train_frac, 1: args.train_frac}
    train = df.sampleBy("label", fractions=fractions, seed=args.seed).cache()
    test = df.join(train.select("row_id"), on="row_id", how="left_anti").cache()

    if train.count() == 0 or test.count() == 0:
        raise RuntimeError("Train/test split produced empty set")

    models = [
        ("LogisticRegression", LogisticRegression(labelCol="label", featuresCol="features", maxIter=50)),
        (
            "RandomForest",
            RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, seed=args.seed),
        ),
        (
            "GBT",
            GBTClassifier(labelCol="label", featuresCol="features", maxIter=50, seed=args.seed),
        ),
    ]

    results = []

    for name, model in models:
        pipeline = build_pipeline(model, cat_cols, num_cols)
        fitted = pipeline.fit(train)
        preds = fitted.transform(test)
        metrics = evaluate_predictions(preds)
        results.append({"name": name, "tuned": False, "metrics": metrics, "model": fitted})
        print(f"Baseline {name}: f1={metrics['f1']:.4f} auc={metrics['auc']:.4f}")

    cv_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )

    tuned_specs = []
    lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=50)
    lr_grid = (
        ParamGridBuilder().addGrid(lr.regParam, [0.0, 0.1]).addGrid(lr.elasticNetParam, [0.0, 0.5]).build()
    )
    tuned_specs.append(("LogisticRegression", lr, lr_grid))

    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, seed=args.seed)
    rf_grid = ParamGridBuilder().addGrid(rf.maxDepth, [5, 10]).addGrid(rf.numTrees, [50, 100]).build()
    tuned_specs.append(("RandomForest", rf, rf_grid))

    for name, model, grid in tuned_specs:
        pipeline = build_pipeline(model, cat_cols, num_cols)
        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=grid,
            evaluator=cv_evaluator,
            numFolds=3,
            parallelism=2,
            seed=args.seed,
        )
        best = cv.fit(train).bestModel
        preds = best.transform(test)
        metrics = evaluate_predictions(preds)
        results.append({"name": name, "tuned": True, "metrics": metrics, "model": best})
        print(f"Tuned {name}: f1={metrics['f1']:.4f} auc={metrics['auc']:.4f}")

    def rank_key(res):
        return (res["metrics"]["f1"], res["metrics"]["auc"])

    best_result = max(results, key=rank_key)
    best_model = best_result["model"]

    preds_best = best_model.transform(test)
    plot_confusion_matrix(preds_best, args.report_dir)

    if not plot_feature_importance(best_model, train, args.report_dir):
        print("Feature importance skipped (final model is not tree-based).")

    metrics_rows = []
    for res in results:
        row = {
            "model": res["name"],
            "tuned": res["tuned"],
            **res["metrics"],
        }
        metrics_rows.append(row)

    import pandas as pd

    metrics_df = pd.DataFrame(metrics_rows)
    save_table(metrics_df, Path(args.report_dir) / "metrics_table.csv")

    model_path = Path(args.artifacts_dir) / "pipeline_model"
    best_model.write().overwrite().save(str(model_path))

    metadata = {
        "chosen_model": best_result["name"],
        "tuned": best_result["tuned"],
        "metrics": best_result["metrics"],
        "keep_duration": bool(args.keep_duration),
        "pdays_features": args.pdays_features,
        "train_frac": args.train_frac,
        "categorical_cols": cat_cols,
        "numeric_cols": num_cols,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(Path(args.artifacts_dir) / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"Best model: {best_result['name']} tuned={best_result['tuned']} f1={best_result['metrics']['f1']:.4f} auc={best_result['metrics']['auc']:.4f}"
    )
    print(f"Saved model to {model_path} and metrics to {args.report_dir}")

    spark.stop()


if __name__ == "__main__":
    main()
