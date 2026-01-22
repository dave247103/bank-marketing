"""Shared feature engineering utilities for bank marketing models."""

from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

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


def build_preprocessing_stages(label_col: str = "y"):
    """Return preprocessing stages for label indexing and feature assembly."""
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
    return [label_indexer] + cat_indexers + [encoder, assembler]


def get_label_mapping(pipeline_model):
    """Return ordered label values and the positive index for 'yes'."""
    label_stage = pipeline_model.stages[0]
    if not hasattr(label_stage, "labels"):
        raise ValueError("First pipeline stage is missing label metadata.")
    labels = list(label_stage.labels)
    if "yes" not in labels or "no" not in labels:
        raise ValueError(f"Unexpected label mapping: {labels}")
    return labels, labels.index("yes")
