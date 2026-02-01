from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

from config import CAP_VALUE, CATEGORICAL_COLS, NUMERIC_COLS, TARGET_COL

CSV_COLUMNS = [
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
    "pdays",
    "previous",
    "poutcome",
    "y",
]


def get_spark(app_name: str) -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()


def get_csv_schema() -> StructType:
    fields: List[StructField] = []
    for col in CSV_COLUMNS:
        if col in NUMERIC_COLS:
            fields.append(StructField(col, IntegerType(), True))
        else:
            fields.append(StructField(col, StringType(), True))
    return StructType(fields)


def read_csv(spark: SparkSession, path: str) -> DataFrame:
    schema = get_csv_schema()
    return (
        spark.read.option("header", True)
        .option("sep", ";")
        .option("quote", '"')
        .schema(schema)
        .csv(path)
    )


def clean_df(df: DataFrame, keep_duration: bool, include_label: bool = True) -> DataFrame:
    for col in NUMERIC_COLS:
        df = df.withColumn(col, F.col(col).cast("int"))

    df = df.withColumn("campaign", F.least(F.col("campaign"), F.lit(CAP_VALUE)))
    df = df.withColumn("previous", F.least(F.col("previous"), F.lit(CAP_VALUE)))
    df = df.withColumn(
        "prev_contacted",
        F.when(F.col("pdays") == -1, F.lit(0)).otherwise(F.lit(1)).cast("int"),
    )
    df = df.withColumn(
        "pdays_clean",
        F.when(F.col("pdays") == -1, F.lit(0)).otherwise(F.col("pdays")).cast("int"),
    )
    df = df.drop("pdays")

    if not keep_duration:
        df = df.drop("duration")

    if include_label:
        df = df.withColumn(
            "label",
            F.when(F.col(TARGET_COL) == "yes", F.lit(1)).otherwise(F.lit(0)).cast("int"),
        )

    return df


def get_feature_columns(keep_duration: bool, pdays_features: str = "both") -> Tuple[List[str], List[str]]:
    num_cols = [
        "age",
        "balance",
        "day",
        "campaign",
        "previous",
        "pdays_clean",
        "prev_contacted",
    ]
    if pdays_features == "drop_pdays_clean":
        num_cols = [c for c in num_cols if c != "pdays_clean"]
    elif pdays_features == "drop_prev_contacted":
        num_cols = [c for c in num_cols if c != "prev_contacted"]
    if keep_duration:
        num_cols.append("duration")
    return CATEGORICAL_COLS, num_cols


def _histogram_counts(df: DataFrame, col: str, bins: int = 20):
    stats = df.agg(F.min(col).alias("min"), F.max(col).alias("max")).collect()[0]
    min_val, max_val = stats["min"], stats["max"]
    if min_val is None or max_val is None:
        return None

    if min_val == max_val:
        return {
            "min": min_val,
            "max": max_val,
            "bins": 1,
            "width": 1.0,
            "counts": {0: df.count()},
        }

    width = float(max_val - min_val) / float(bins)
    if width == 0:
        width = 1.0

    bin_idx = F.when(F.col(col) == max_val, F.lit(bins - 1)).otherwise(
        F.floor((F.col(col) - F.lit(min_val)) / F.lit(width))
    )
    counts_df = (
        df.groupBy(bin_idx.cast("int").alias("bin"))
        .count()
        .orderBy("bin")
        .toPandas()
    )
    counts = {int(row["bin"]): int(row["count"]) for _, row in counts_df.iterrows()}
    return {"min": min_val, "max": max_val, "bins": bins, "width": width, "counts": counts}


def save_dataset_summary(df: DataFrame, report_dir: str, keep_duration: bool) -> None:
    import pandas as pd
    from utils_report import save_table

    rows = df.count()
    cols = len(df.columns)
    summary = pd.DataFrame(
        [
            {
                "rows": rows,
                "cols": cols,
                "keep_duration": bool(keep_duration),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        ]
    )
    save_table(summary, Path(report_dir) / "dataset_summary.csv")


def plot_class_balance(df: DataFrame, report_dir: str) -> None:
    import matplotlib.pyplot as plt
    from utils_report import save_fig

    counts = df.groupBy("label").count().orderBy("label").toPandas()
    counts["label_name"] = counts["label"].map({0: "no", 1: "yes"})

    plt.figure(figsize=(4, 3))
    plt.bar(counts["label_name"], counts["count"], color="#4C78A8")
    plt.xlabel("y")
    plt.ylabel("count")
    save_fig(Path(report_dir) / "class_balance.png")


def plot_histograms(df: DataFrame, report_dir: str, columns: Iterable[str]) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt
    from utils_report import save_fig

    for col in columns:
        if col not in df.columns:
            continue
        hist = _histogram_counts(df, col, bins=20)
        if not hist:
            continue

        bins = hist["bins"]
        width = hist["width"]
        min_val = hist["min"]
        counts = hist["counts"]

        pdf = pd.DataFrame({"bin": list(range(bins))})
        pdf["count"] = pdf["bin"].map(counts).fillna(0).astype(int)
        pdf["center"] = min_val + (pdf["bin"] + 0.5) * width

        plt.figure(figsize=(5, 3.5))
        plt.bar(pdf["center"], pdf["count"], width=width * 0.9, color="#F58518")
        plt.xlabel(col)
        plt.ylabel("count")
        save_fig(Path(report_dir) / f"hist_{col}.png")


def plot_corr_heatmap(df: DataFrame, report_dir: str, columns: List[str]) -> None:
    import matplotlib.pyplot as plt
    from utils_report import save_fig

    cols = [c for c in columns if c in df.columns]
    if len(cols) < 2:
        return

    vec_df = VectorAssembler(inputCols=cols, outputCol="num_vec").transform(df.select(cols))
    corr = Correlation.corr(vec_df, "num_vec").head()[0].toArray()

    plt.figure(figsize=(8, 6))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            plt.text(j, i, format(val, ".2g"), ha="center", va="center", color=color)
    save_fig(Path(report_dir) / "corr_heatmap.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bank marketing EDA + preprocessing")
    parser.add_argument("--input", default="data/bank-full.csv", help="Input CSV path")
    parser.add_argument("--report-dir", default="report", help="Report output directory")
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
    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    spark = get_spark("bank-preprocess")
    spark.sparkContext.setLogLevel("WARN")

    df_raw = read_csv(spark, args.input)
    df = clean_df(df_raw, keep_duration=args.keep_duration, include_label=True)

    Path(args.report_dir).mkdir(parents=True, exist_ok=True)
    save_dataset_summary(df, args.report_dir, args.keep_duration)
    plot_class_balance(df, args.report_dir)

    hist_cols = ["age", "balance", "campaign"]
    if args.keep_duration:
        hist_cols.append("duration")
    plot_histograms(df, args.report_dir, hist_cols)

    _, num_cols = get_feature_columns(args.keep_duration, args.pdays_features)
    corr_cols = num_cols + ["label"]
    plot_corr_heatmap(df, args.report_dir, corr_cols)

    rows = df.count()
    print(
        f"Saved report artifacts to {args.report_dir} | rows={rows} cols={len(df.columns)} keep_duration={args.keep_duration} pdays_features={args.pdays_features}"
    )

    spark.stop()


if __name__ == "__main__":
    main()
