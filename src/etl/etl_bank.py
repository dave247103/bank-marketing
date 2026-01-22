import argparse

from pyspark.sql import SparkSession, functions as F, types as T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETL for UCI Bank Marketing dataset.")
    parser.add_argument(
        "--input",
        default="data/raw/bank-full.csv",
        help="Input CSV path (semicolon-delimited).",
    )
    parser.add_argument(
        "--output",
        default="data/processed/bank.parquet",
        help="Output parquet path.",
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
        ]
    )


def validate(df) -> None:
    row_count = df.count()
    print(f"Row count: {row_count}")
    if row_count == 0:
        raise ValueError("No rows found in input dataset.")

    null_exprs = [
        F.sum(F.when(F.col(col_name).isNull(), 1).otherwise(0)).alias(col_name)
        for col_name in df.columns
    ]
    null_counts = df.select(null_exprs).collect()[0].asDict()
    print("Null counts per column:")
    for col_name in df.columns:
        print(f"  {col_name}: {null_counts[col_name]}")
    bad_nulls = {k: v for k, v in null_counts.items() if v > 0}
    if bad_nulls:
        raise ValueError(f"Found nulls in columns: {bad_nulls}")

    y_values = {row["y"] for row in df.select("y").distinct().collect()}
    print(f"Distinct y values: {sorted(y_values)}")
    if y_values != {"yes", "no"}:
        raise ValueError(f"Unexpected label values in y: {sorted(y_values)}")


def main() -> None:
    args = parse_args()
    spark = SparkSession.builder.appName("bank-etl").master("local[*]").getOrCreate()
    try:
        schema = build_schema()
        df = (
            spark.read.option("sep", ";")
            .option("quote", '"')
            .option("header", True)
            .schema(schema)
            .csv(args.input)
        )
        validate(df)
        df.write.mode("overwrite").parquet(args.output)
        print(f"Wrote parquet to: {args.output}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
