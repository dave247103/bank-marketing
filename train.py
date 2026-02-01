from __future__ import annotations

import argparse
import math
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from pyspark.ml import Pipeline
AttributeGroup = None
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

    num_scaled_col = None
    if num_cols:
        stages.append(VectorAssembler(inputCols=num_cols, outputCol="num_vec"))
        stages.append(
            StandardScaler(inputCol="num_vec", outputCol="num_scaled", withMean=False, withStd=True)
        )
        num_scaled_col = "num_scaled"

    feature_inputs = ohe_cols + ([num_scaled_col] if num_scaled_col else [])
    if not feature_inputs:
        raise RuntimeError("No features selected for pipeline.")
    stages.append(VectorAssembler(inputCols=feature_inputs, outputCol="features"))
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
    plt.xticks([0, 1], ["no", "yes"])
    plt.yticks([0, 1], ["no", "yes"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i][j], ha="center", va="center", color="black")
    save_fig(Path(report_dir) / "confusion_matrix.png")


def _get_vector_size(schema, col: str) -> int | None:
    field = schema[col]
    if AttributeGroup is not None:
        group = AttributeGroup.fromStructField(field)
        if group.size is not None:
            return int(group.size)
        if group.attributes is not None:
            return len(group.attributes)
        return None

    meta = field.metadata.get("ml_attr", {})
    if "num_attrs" in meta:
        return int(meta["num_attrs"])
    attrs = meta.get("attrs")
    if not attrs:
        return None
    max_idx = -1
    for group in attrs.values():
        for attr in group:
            idx = attr.get("idx")
            if idx is not None:
                max_idx = max(max_idx, idx)
    return max_idx + 1 if max_idx >= 0 else None


def _get_attr_names_from_metadata(field) -> List[str] | None:
    meta = field.metadata.get("ml_attr", {})
    attrs = meta.get("attrs")
    if not attrs:
        return None
    idx_to_name: Dict[int, str | None] = {}
    max_idx = -1
    for group in attrs.values():
        for attr in group:
            idx = attr.get("idx")
            if idx is None:
                continue
            idx_to_name[idx] = attr.get("name")
            max_idx = max(max_idx, idx)
    if max_idx < 0:
        return None
    return [idx_to_name.get(i) for i in range(max_idx + 1)]


def _get_feature_names_from_metadata(
    transformed_df: DataFrame, features_col: str, num_features: int
) -> List[str]:
    field = transformed_df.schema[features_col]
    names: List[str] = []
    if AttributeGroup is not None:
        group = AttributeGroup.fromStructField(field)
        attrs = group.attributes
        if attrs:
            for i, attr in enumerate(attrs):
                name = attr.name if attr and attr.name else f"f{i}"
                names.append(name)
    else:
        attrs = _get_attr_names_from_metadata(field)
        if attrs:
            for i, name in enumerate(attrs):
                names.append(name or f"f{i}")
    if not names:
        names = [f"f{i}" for i in range(num_features)]
    if len(names) < num_features:
        names.extend([f"f{i}" for i in range(len(names), num_features)])
    return names[:num_features]


def _build_base_feature_names(
    transformed_df: DataFrame,
    pipeline_model,
    cat_cols: List[str],
    num_cols: List[str],
    num_features: int,
) -> List[str]:
    assembler = next(
        (stage for stage in pipeline_model.stages if isinstance(stage, VectorAssembler) and stage.getOutputCol() == "features"),
        None,
    )
    if assembler is None:
        return [f"f{i}" for i in range(num_features)]

    base_names: List[str] = []
    for col in assembler.getInputCols():
        if col == "num_scaled":
            if num_cols:
                base_names.extend(num_cols)
            else:
                size = _get_vector_size(transformed_df.schema, col) or 0
                base_names.extend(["num_scaled"] * size)
            continue

        base = col[:-4] if col.endswith("_ohe") else col
        size = _get_vector_size(transformed_df.schema, col) or 0
        base_names.extend([base] * size)

    if len(base_names) < num_features:
        base_names.extend([f"f{i}" for i in range(len(base_names), num_features)])
    return base_names[:num_features]


def _map_dim_to_base(dim_name: str, cat_cols: List[str], num_cols: List[str]) -> str | None:
    if dim_name in num_cols:
        return dim_name
    if dim_name in cat_cols:
        return dim_name
    for col in cat_cols:
        if dim_name.startswith(f"{col}_") or dim_name.startswith(f"{col}=") or dim_name.startswith(f"{col}:"):
            return col
    if "=" in dim_name:
        prefix = dim_name.split("=", 1)[0]
        if prefix in cat_cols:
            return prefix
    return None


def _aggregate_base_importances(
    pipeline_model, df: DataFrame, cat_cols: List[str], num_cols: List[str]
) -> Dict[str, float]:
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
    elif isinstance(model, LogisticRegressionModel):
        importances = [abs(float(v)) for v in model.coefficients.toArray()]
    else:
        return {}

    transformed = pipeline_model.transform(df)
    dim_names = _get_feature_names_from_metadata(transformed, "features", len(importances))
    fallback_bases = _build_base_feature_names(
        transformed, pipeline_model, cat_cols, num_cols, len(importances)
    )

    grouped: Dict[str, float] = {}
    for i, importance in enumerate(importances):
        dim_name = dim_names[i]
        base = _map_dim_to_base(dim_name, cat_cols, num_cols)
        if base is None and i < len(fallback_bases):
            base = fallback_bases[i]
        if base is None:
            base = dim_name
        grouped[base] = grouped.get(base, 0.0) + float(importance)

    for col in cat_cols + num_cols:
        grouped.setdefault(col, 0.0)
    return grouped


def _rank_base_features(
    grouped_importances: Dict[str, float], cat_cols: List[str], num_cols: List[str]
) -> List[str]:
    base_features = cat_cols + num_cols
    index = {name: i for i, name in enumerate(base_features)}
    return sorted(
        base_features,
        key=lambda name: (-grouped_importances.get(name, 0.0), index[name]),
    )


def plot_feature_importance(
    pipeline_model, df: DataFrame, cat_cols: List[str], num_cols: List[str], report_dir: str
) -> bool:
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
        transformed = pipeline_model.transform(df)
        importances = model.featureImportances.toArray()
        dim_names = _get_feature_names_from_metadata(transformed, "features", len(importances))
        fallback_bases = _build_base_feature_names(
            transformed, pipeline_model, cat_cols, num_cols, len(importances)
        )

        grouped: Dict[str, float] = {}
        for i, importance in enumerate(importances):
            dim_name = dim_names[i]
            base = _map_dim_to_base(dim_name, cat_cols, num_cols)
            if base is None and i < len(fallback_bases):
                base = fallback_bases[i]
            if base is None:
                base = dim_name
            grouped[base] = grouped.get(base, 0.0) + float(importance)

        feature_df = pd.DataFrame(
            {"feature": list(grouped.keys()), "importance": list(grouped.values())}
        ).sort_values("importance", ascending=False)
        save_table(feature_df, Path(report_dir) / "feature_importance.csv")
        pdf = feature_df.head(20)

        plt.figure(figsize=(7, 5))
        plt.barh(pdf["feature"], pdf["importance"], color="#54A24B")
        plt.gca().invert_yaxis()
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

    counts = train.groupBy("label").count().collect()
    count_map = {int(row["label"]): int(row["count"]) for row in counts}
    n0 = count_map.get(0, 0)
    n1 = count_map.get(1, 0)
    total = n0 + n1
    if n0 == 0 or n1 == 0:
        raise RuntimeError("Train split missing one of the classes")

    w0 = total / (2 * n0)
    w1 = total / (2 * n1)
    train = train.withColumn(
        "weight", F.when(F.col("label") == 0, F.lit(w0)).otherwise(F.lit(w1))
    )
    test = test.withColumn(
        "weight", F.when(F.col("label") == 0, F.lit(w0)).otherwise(F.lit(w1))
    )

    def with_weight_col(model):
        if model.hasParam("weightCol"):
            model = model.setParams(weightCol="weight")
        return model

    models = [
        (
            "LogisticRegression",
            with_weight_col(LogisticRegression(labelCol="label", featuresCol="features", maxIter=50)),
        ),
        (
            "RandomForest",
            with_weight_col(
                RandomForestClassifier(
                    labelCol="label", featuresCol="features", numTrees=100, seed=args.seed
                )
            ),
        ),
        (
            "GBT",
            with_weight_col(
                GBTClassifier(labelCol="label", featuresCol="features", maxIter=50, seed=args.seed)
            ),
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

    def rank_key(res):
        return (res["metrics"]["f1"], res["metrics"]["auc"])

    baseline_ranked = sorted(
        [res for res in results if not res["tuned"]], key=rank_key, reverse=True
    )
    top2_names = [res["name"] for res in baseline_ranked[:2]]

    lr = with_weight_col(LogisticRegression(labelCol="label", featuresCol="features", maxIter=50))
    lr_grid = ParamGridBuilder().addGrid(lr.regParam, [0.0, 0.1]).addGrid(
        lr.elasticNetParam, [0.0, 0.5]
    ).build()

    rf = with_weight_col(
        RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, seed=args.seed)
    )
    rf_grid = ParamGridBuilder().addGrid(rf.maxDepth, [5, 10]).addGrid(
        rf.numTrees, [50, 100]
    ).build()

    gbt = with_weight_col(
        GBTClassifier(labelCol="label", featuresCol="features", maxIter=50, seed=args.seed)
    )
    gbt_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxDepth, [3, 5])
        .addGrid(gbt.maxIter, [30, 50])
        .addGrid(gbt.stepSize, [0.05, 0.1])
        .build()
    )

    tuned_specs = {
        "LogisticRegression": (lr, lr_grid),
        "RandomForest": (rf, rf_grid),
        "GBT": (gbt, gbt_grid),
    }

    for name in top2_names:
        model, grid = tuned_specs[name]
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

    best_result = max(results, key=rank_key)
    best_model = best_result["model"]

    preds_best = best_model.transform(test)
    plot_confusion_matrix(preds_best, args.report_dir)

    if not plot_feature_importance(best_model, train, cat_cols, num_cols, args.report_dir):
        print("Feature importance skipped (final model is not tree-based).")

    def build_ablation_estimator():
        fitted_model = best_model.stages[-1]
        if best_result["name"] == "LogisticRegression":
            lr = LogisticRegression(labelCol="label", featuresCol="features")
            lr = lr.setParams(
                maxIter=fitted_model.getOrDefault(fitted_model.maxIter),
                regParam=fitted_model.getOrDefault(fitted_model.regParam),
                elasticNetParam=fitted_model.getOrDefault(fitted_model.elasticNetParam),
            )
            return with_weight_col(lr)
        if best_result["name"] == "RandomForest":
            rf = RandomForestClassifier(
                labelCol="label", featuresCol="features", seed=args.seed
            ).setParams(
                numTrees=fitted_model.getOrDefault(fitted_model.numTrees),
                maxDepth=fitted_model.getOrDefault(fitted_model.maxDepth),
            )
            return with_weight_col(rf)
        if best_result["name"] == "GBT":
            gbt = GBTClassifier(
                labelCol="label", featuresCol="features", seed=args.seed
            ).setParams(
                maxDepth=fitted_model.getOrDefault(fitted_model.maxDepth),
                maxIter=fitted_model.getOrDefault(fitted_model.maxIter),
                stepSize=fitted_model.getOrDefault(fitted_model.stepSize),
            )
            return with_weight_col(gbt)
        raise RuntimeError(f"Unsupported model for ablation: {best_result['name']}")

    base_importances = _aggregate_base_importances(best_model, train, cat_cols, num_cols)
    ordered_features = _rank_base_features(base_importances, cat_cols, num_cols)
    total_features = len(ordered_features)
    ablation_rows = []
    for frac in [1.0, 0.8, 0.6]:
        k = int(math.ceil(frac * total_features))
        selected = set(ordered_features[:k])
        sel_cat = [c for c in cat_cols if c in selected]
        sel_num = [c for c in num_cols if c in selected]
        ablation_model = build_ablation_estimator()
        pipeline = build_pipeline(ablation_model, sel_cat, sel_num)
        fitted = pipeline.fit(train)
        preds = fitted.transform(test)
        metrics = evaluate_predictions(preds)
        ablation_rows.append(
            {
                "fraction": frac,
                "num_features": len(selected),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "auc": metrics["auc"],
                "model_name": best_result["name"],
                "tuned": best_result["tuned"],
            }
        )
        print(f"Ablation {frac:.1f}: f1={metrics['f1']:.4f} auc={metrics['auc']:.4f}")

    import pandas as pd

    ablation_df = pd.DataFrame(ablation_rows)
    save_table(ablation_df, Path(args.report_dir) / "feature_ablation.csv")

    metrics_rows = []
    for res in results:
        row = {
            "model": res["name"],
            "tuned": res["tuned"],
            **res["metrics"],
        }
        metrics_rows.append(row)

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
