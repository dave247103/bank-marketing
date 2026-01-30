from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    input_csv: str = "data/bank-full.csv"
    report_dir: str = "report"
    artifacts_dir: str = "artifacts"

    kafka_bootstrap: str = "localhost:9092"
    topic_raw: str = "bank_raw"
    topic_pred: str = "bank_pred"

    spark_kafka_pkg: str = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8"

    def ensure_dirs(self) -> None:
        Path(self.report_dir).mkdir(parents=True, exist_ok=True)
        Path(self.artifacts_dir).mkdir(parents=True, exist_ok=True)
