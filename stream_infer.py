from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.sql import functions as F
from pyspark.sql import SparkSession

from preprocess import clean_df, get_csv_schema


def ensure_kafka_connector(spark: SparkSession) -> None:
    try:
        _ = spark._jvm.org.apache.spark.sql.kafka010.KafkaSourceProvider
    except Exception as exc:
        raise RuntimeError(
            "Spark Kafka connector not found. Run with --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spark Structured Streaming inference")
    parser.add_argument("--bootstrap", default="localhost:9092", help="Kafka bootstrap servers")
    parser.add_argument("--input-topic", default="bank_raw", help="Input Kafka topic")
    parser.add_argument("--output-topic", default="bank_pred", help="Output Kafka topic")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Artifacts directory")
    parser.add_argument("--model-path", default=None, help="PipelineModel path")
    parser.add_argument(
        "--checkpoint-dir", default="artifacts/checkpoints/bank_stream", help="Checkpoint directory"
    )
    parser.add_argument("--log-interval", type=int, default=10, help="Progress log interval (sec)")

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
        default=None,
        help="Override keep_duration if metadata not found",
    )
    return parser.parse_args()


def load_metadata(artifacts_dir: str) -> dict | None:
    meta_path = Path(artifacts_dir) / "metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    model_path = Path(args.model_path) if args.model_path else artifacts_dir / "pipeline_model"
    if not model_path.exists():
        raise FileNotFoundError(f"PipelineModel not found: {model_path}")

    metadata = load_metadata(str(artifacts_dir))
    keep_duration = None
    if metadata is not None:
        keep_duration = bool(metadata.get("keep_duration", False))
    elif args.keep_duration is not None:
        keep_duration = bool(args.keep_duration)
    else:
        keep_duration = False

    spark = SparkSession.builder.appName("bank-stream-infer").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    ensure_kafka_connector(spark)

    schema = get_csv_schema()

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", args.bootstrap)
        .option("subscribe", args.input_topic)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed = raw.select(F.from_json(F.col("value").cast("string"), schema).alias("data")).select("data.*")
    parsed = parsed.withColumn("ingest_ts", F.current_timestamp())

    features_df = clean_df(parsed, keep_duration=keep_duration, include_label=False)

    model = PipelineModel.load(str(model_path))
    preds = model.transform(features_df)

    preds = preds.withColumn("predicted_label", F.col("prediction").cast("int"))
    preds = preds.withColumn("p1", F.col("probability").getItem(1))

    output_cols = [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "balance",
        "housing",
        "loan",
        "contact",
        "day",
        "month",
        "duration",
        "campaign",
        "pdays_clean",
        "previous",
        "poutcome",
        "y",
        "prev_contacted",
        "ingest_ts",
        "predicted_label",
        "p1",
    ]
    output_cols = [c for c in output_cols if c in preds.columns]

    out_df = preds.select(F.to_json(F.struct(*[F.col(c) for c in output_cols])).alias("value"))

    query = (
        out_df.writeStream.format("kafka")
        .option("kafka.bootstrap.servers", args.bootstrap)
        .option("topic", args.output_topic)
        .option("checkpointLocation", args.checkpoint_dir)
        .outputMode("append")
        .start()
    )

    print(
        f"Streaming from {args.input_topic} to {args.output_topic} | keep_duration={keep_duration} | model={model_path}"
    )

    try:
        while query.isActive:
            progress = query.lastProgress
            if progress:
                duration = progress.get("durationMs", {}).get("addBatch")
                num_in = progress.get("numInputRows")
                if duration is not None:
                    print(f"batch_ms={duration} input_rows={num_in}")
            time.sleep(args.log_interval)
    except KeyboardInterrupt:
        print("Stopping stream...")
    finally:
        query.stop()
        spark.stop()


if __name__ == "__main__":
    main()
