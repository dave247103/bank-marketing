"""Export feature importances from a trained Spark MLlib pipeline model."""

import argparse
import csv
import os

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for feature importance export."""
    parser = argparse.ArgumentParser(
        description="Export feature importances from a GBT or RF PipelineModel."
    )
    parser.add_argument(
        "--input",
        default="data/processed/bank.parquet",
        help="Input parquet path.",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to saved PipelineModel (pipeline_gbt or pipeline_rf).",
    )
    parser.add_argument(
        "--out_csv",
        default="report/feature_importance.csv",
        help="Output CSV path for feature importances.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=30,
        help="Number of top features to export.",
    )
    return parser.parse_args()


def extract_feature_names(transformed_df, total_features: int):
    """Extract feature names from features column metadata (no AttributeGroup dependency)."""
    field = transformed_df.schema["features"]
    md = field.metadata or {}
    ml_attr = md.get("ml_attr") or {}
    attrs = ml_attr.get("attrs") or {}

    names = [None] * total_features

    # attrs groups are typically: numeric, binary, nominal
    for group in ("numeric", "binary", "nominal"):
        for a in attrs.get(group, []) or []:
            try:
                idx = int(a.get("idx"))
            except Exception:
                continue
            if 0 <= idx < total_features:
                names[idx] = a.get("name") or names[idx]

    return [n or f"feature_{i}" for i, n in enumerate(names)]


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input parquet not found: {args.input}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if args.top_k <= 0:
        raise ValueError("--top_k must be a positive integer.")

    spark = (
        SparkSession.builder.appName("bank-feature-importance")
        .master("local[*]")
        .getOrCreate()
    )
    try:
        df = spark.read.parquet(args.input)
        if df.rdd.isEmpty():
            raise ValueError(f"No rows found in input dataset: {args.input}")

        pipeline_model = PipelineModel.load(args.model_path)
        model_stage = pipeline_model.stages[-1]
        if not hasattr(model_stage, "featureImportances"):
            raise ValueError("Model does not expose featureImportances.")

        sample_df = df.limit(500)
        transformed = pipeline_model.transform(sample_df)

        importances = model_stage.featureImportances.toArray()
        feature_names = extract_feature_names(transformed, len(importances))
        if len(feature_names) != len(importances):
            raise ValueError("Feature name count does not match importances length.")

        ranked = sorted(enumerate(importances), key=lambda pair: pair[1], reverse=True)[
            : args.top_k
        ]

        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        with open(args.out_csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["rank", "feature", "importance"])
            for idx, (feature_idx, importance) in enumerate(ranked, start=1):
                writer.writerow([idx, feature_names[feature_idx], float(importance)])

        print(f"Wrote feature importances to: {args.out_csv}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
