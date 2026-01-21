import argparse

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession, functions as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline logistic regression.")
    parser.add_argument(
        "--input",
        default="data/processed/bank.parquet",
        help="Input parquet path.",
    )
    parser.add_argument(
        "--model_out",
        default="models/pipeline_lr",
        help="Output path for saved PipelineModel.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split and LR.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spark = (
        SparkSession.builder.appName("bank-train-lr")
        .master("local[*]")
        .getOrCreate()
    )
    try:
        df = spark.read.parquet(args.input)

        label_col = "y"
        numeric_cols = [
            "age",
            "balance",
            "day",
            "duration",
            "campaign",
            "pdays",
            "previous",
        ]
        categorical_cols = [
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "poutcome",
        ]

        label_indexer = StringIndexer(
            inputCol=label_col, outputCol="label", handleInvalid="error"
        )
        cat_indexers = [
            StringIndexer(
                inputCol=col_name,
                outputCol=f"{col_name}_idx",
                handleInvalid="keep",
            )
            for col_name in categorical_cols
        ]
        encoder = OneHotEncoder(
            inputCols=[f"{col_name}_idx" for col_name in categorical_cols],
            outputCols=[f"{col_name}_ohe" for col_name in categorical_cols],
            handleInvalid="keep",
        )
        assembler = VectorAssembler(
            inputCols=numeric_cols + [f"{col_name}_ohe" for col_name in categorical_cols],
            outputCol="features",
        )
        lr = LogisticRegression(featuresCol="features", labelCol="label")
        if lr.hasParam("seed"):
            lr.setSeed(args.seed)

        pipeline = Pipeline(
            stages=[label_indexer] + cat_indexers + [encoder, assembler, lr]
        )

        train_df, test_df = df.randomSplit([0.8, 0.2], seed=args.seed)
        model = pipeline.fit(train_df)

        preds = model.transform(test_df).cache()

        label_model = model.stages[0]
        label_values = list(label_model.labels)
        if "yes" not in label_values or "no" not in label_values:
            raise ValueError(f"Unexpected label mapping: {label_values}")
        pos_label = label_values.index("yes")
        neg_label = label_values.index("no")

        tp = preds.filter(
            (F.col("label") == pos_label) & (F.col("prediction") == pos_label)
        ).count()
        fp = preds.filter(
            (F.col("label") == neg_label) & (F.col("prediction") == pos_label)
        ).count()
        tn = preds.filter(
            (F.col("label") == neg_label) & (F.col("prediction") == neg_label)
        ).count()
        fn = preds.filter(
            (F.col("label") == pos_label) & (F.col("prediction") == neg_label)
        ).count()
        total = tp + fp + tn + fn

        accuracy = (tp + tn) / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        auc = BinaryClassificationEvaluator(
            labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
        ).evaluate(preds)

        print(f"Train rows: {train_df.count()}")
        print(f"Test rows: {test_df.count()}")
        print(f"Label mapping: {label_values} (positive='yes')")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print("Confusion matrix (positive class = 'yes'):")
        print(f"  TP: {tp}")
        print(f"  FP: {fp}")
        print(f"  TN: {tn}")
        print(f"  FN: {fn}")

        model.write().overwrite().save(args.model_out)
        print(f"Saved model to: {args.model_out}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
