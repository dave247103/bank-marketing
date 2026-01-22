import argparse
import json
import logging
import os
import re
import pyspark
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession, functions as F, types as T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spark Structured Streaming scorer for bank marketing events."
    )
    parser.add_argument(
        "--model",
        default="models/pipeline_lr",
        help="Path to saved PipelineModel.",
    )
    parser.add_argument(
        "--broker",
        default="localhost:9092",
        help="Kafka bootstrap servers.",
    )
    parser.add_argument(
        "--in_topic",
        default="bank_raw",
        help="Input Kafka topic.",
    )
    parser.add_argument(
        "--out_topic",
        default="bank_scored",
        help="Output Kafka topic.",
    )
    parser.add_argument(
        "--deadletter_topic",
        default="bank_deadletter",
        help="Kafka topic for dead-letter records.",
    )
    parser.add_argument(
        "--checkpoint",
        default="data/checkpoints",
        help="Checkpoint root path for streaming state.",
    )
    parser.add_argument(
        "--starting_offsets",
        default="latest",
        help="Kafka starting offsets (earliest/latest).",
    )
    return parser.parse_args()


def build_schema() -> T.StructType:
    return T.StructType(
        [
            T.StructField("age", T.IntegerType(), nullable=True),
            T.StructField("job", T.StringType(), nullable=True),
            T.StructField("marital", T.StringType(), nullable=True),
            T.StructField("education", T.StringType(), nullable=True),
            T.StructField("default", T.StringType(), nullable=True),
            T.StructField("balance", T.IntegerType(), nullable=True),
            T.StructField("housing", T.StringType(), nullable=True),
            T.StructField("loan", T.StringType(), nullable=True),
            T.StructField("contact", T.StringType(), nullable=True),
            T.StructField("day", T.IntegerType(), nullable=True),
            T.StructField("month", T.StringType(), nullable=True),
            T.StructField("duration", T.IntegerType(), nullable=True),
            T.StructField("campaign", T.IntegerType(), nullable=True),
            T.StructField("pdays", T.IntegerType(), nullable=True),
            T.StructField("previous", T.IntegerType(), nullable=True),
            T.StructField("poutcome", T.StringType(), nullable=True),
            T.StructField("y", T.StringType(), nullable=True),
            T.StructField("event_time", T.StringType(), nullable=True),
        ]
    )


def resolve_spark_version() -> str:
    match = re.search(r"\d+\.\d+\.\d+", pyspark.__version__)
    return match.group(0) if match else pyspark.__version__


def kafka_packages(spark_version: str) -> str:
    return (
        f"org.apache.spark:spark-sql-kafka-0-10_2.12:{spark_version},"
        f"org.apache.spark:spark-token-provider-kafka-0-10_2.12:{spark_version}"
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger("bank-scorer")

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model path not found: {args.model}")

    spark_version = resolve_spark_version()
    packages = kafka_packages(spark_version)
    logger.info("Using Spark version %s with packages: %s", spark_version, packages)

    query = None
    deadletter_query = None
    spark = (
        SparkSession.builder.appName("bank-stream-scorer")
        .master("local[*]")
        .config("spark.jars.packages", packages)
        .getOrCreate()
    )
    try:
        schema = build_schema()
        model = PipelineModel.load(args.model)

        label_model = model.stages[0]
        if not hasattr(label_model, "labels"):
            raise ValueError("First pipeline stage is missing label metadata.")
        labels = list(label_model.labels)
        if "yes" not in labels or "no" not in labels:
            raise ValueError(f"Unexpected label mapping: {labels}")
        pos_index = labels.index("yes")
        label_array = F.array([F.lit(label) for label in labels])

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
        required_feature_cols = numeric_cols + categorical_cols

        raw_df = (
            spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", args.broker)
            .option("subscribe", args.in_topic)
            .option("startingOffsets", args.starting_offsets)
            .load()
        )

        parsed_df = (
            raw_df.select(
                F.from_json(F.col("value").cast("string"), schema).alias("data")
            )
            .where(F.col("data").isNotNull())
            .select("data.*")
        )

        parsed_df = parsed_df.withColumn("y_original", F.col("y"))
        parsed_df = parsed_df.withColumn(
            "y", F.when(F.col("y").isNull(), F.lit("no")).otherwise(F.col("y"))
        )

        scored_at_col = F.date_format(
            F.to_utc_timestamp(F.current_timestamp(), "UTC"),
            "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'",
        )
        parsed_df = parsed_df.withColumn("scored_at", scored_at_col)
        parsed_df = parsed_df.withColumn(
            "event_time", F.coalesce(F.col("event_time"), F.col("scored_at"))
        )

        null_checks = [F.col(col_name).isNull() for col_name in required_feature_cols]
        has_null = null_checks[0]
        for expr in null_checks[1:]:
            has_null = has_null | expr
        parsed_df = parsed_df.withColumn("has_null", has_null)
        parsed_df = parsed_df.withColumn(
            "null_columns",
            F.array(
                *[
                    F.when(F.col(col_name).isNull(), F.lit(col_name))
                    for col_name in required_feature_cols
                ]
            ),
        )
        parsed_df = parsed_df.withColumn(
            "null_columns", F.expr("filter(null_columns, x -> x is not null)")
        )

        good_df = parsed_df.where(~F.col("has_null"))
        bad_df = parsed_df.where(F.col("has_null"))

        preds = model.transform(good_df)

        probability_yes = vector_to_array("probability").getItem(pos_index)
        prediction_label = label_array.getItem(F.col("prediction").cast("int"))
        model_name = F.lit(os.path.basename(os.path.normpath(args.model)))

        output_df = preds.select(
            F.col("event_time").alias("event_time"),
            F.col("y_original").alias("y"),
            prediction_label.alias("prediction"),
            probability_yes.alias("probability_yes"),
            model_name.alias("model_name"),
            F.col("scored_at").alias("scored_at"),
        )

        output_kafka = output_df.select(
            F.to_json(F.struct(*[F.col(col_name) for col_name in output_df.columns]))
            .alias("value")
        )

        query = (
            output_kafka.writeStream.format("kafka")
            .option("kafka.bootstrap.servers", args.broker)
            .option("topic", args.out_topic)
            .option(
                "checkpointLocation",
                os.path.join(args.checkpoint, "bank_scored"),
            )
            .outputMode("append")
            .start()
        )

        deadletter_df = bad_df.select(
            F.col("event_time").alias("event_time"),
            F.col("y_original").alias("y"),
            F.col("null_columns").alias("null_columns"),
            F.col("scored_at").alias("scored_at"),
        )
        deadletter_kafka = deadletter_df.select(
            F.to_json(
                F.struct(*[F.col(col_name) for col_name in deadletter_df.columns])
            ).alias("value")
        )
        deadletter_query = (
            deadletter_kafka.writeStream.format("kafka")
            .option("kafka.bootstrap.servers", args.broker)
            .option("topic", args.deadletter_topic)
            .option(
                "checkpointLocation",
                os.path.join(args.checkpoint, "bank_deadletter"),
            )
            .outputMode("append")
            .start()
        )

        logger.info(
            "Streaming from '%s' to '%s' (dead-letter '%s').",
            args.in_topic,
            args.out_topic,
            args.deadletter_topic,
        )
        queries = [query, deadletter_query]
        while all(active_query.isActive for active_query in queries):
            for active_query in queries:
                active_query.awaitTermination(timeout=5)
            progress = query.lastProgress
            if progress:
                logger.info("Streaming progress: %s", json.dumps(progress))
    except KeyboardInterrupt:
        logger.info("Stopping streaming query.")
    finally:
        for active_query in [query, deadletter_query]:
            if active_query is None:
                continue
            try:
                active_query.stop()
            except Exception:
                pass
        spark.stop()


if __name__ == "__main__":
    main()
