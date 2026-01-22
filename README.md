# UCI Bank Marketing - Milestones 1-2

## Run ETL (CSV -> Parquet)
```bash
source .venv/bin/activate
python src/etl/etl_bank.py --input data/raw/bank-full.csv --output data/processed/bank.parquet
```

## Train baseline Logistic Regression
```bash
source .venv/bin/activate
python src/ml/train_baseline_lr.py --input data/processed/bank.parquet --model_out models/pipeline_lr --seed 42
```

Expected outputs:
- Parquet dataset: `data/processed/bank.parquet`
- Saved model: `models/pipeline_lr`

## Feature selection experiment (top-k)
```bash
source .venv/bin/activate
python src/ml/feature_selection_experiment.py --input data/processed/bank.parquet --seed 42 --model gbt --top_k 30 --out_json report/feature_selection_experiment.json --out_csv report/feature_selection_topk.csv
```

Expected outputs:
- Report: `report/feature_selection_experiment.json`
- Selected features: `report/feature_selection_topk.csv`

## Streaming scoring (Kafka + Spark Structured Streaming)

Start Kafka (Docker Compose):
```bash
docker compose up -d
```

Optional clean reset (clears old topic messages):
```bash
docker compose down -v
docker compose up -d
```

Optional: create topics explicitly (useful if auto-create is disabled):
```bash
docker exec -it kafka /opt/bitnami/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --if-not-exists --topic bank_raw --partitions 3 --replication-factor 1
docker exec -it kafka /opt/bitnami/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --if-not-exists --topic bank_scored --partitions 3 --replication-factor 1
docker exec -it kafka /opt/bitnami/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --if-not-exists --topic bank_deadletter --partitions 3 --replication-factor 1
```

Then run each command in its own terminal:
```bash
source .venv/bin/activate
python src/streaming/producer.py --input data/processed/bank.parquet --broker localhost:9092 --topic bank_raw --rate 20 --repeat
```
```bash
source .venv/bin/activate
python src/streaming/scorer.py --model models/pipeline_lr --broker localhost:9092 --in_topic bank_raw --out_topic bank_scored --deadletter_topic bank_deadletter --checkpoint data/checkpoints --progress_log report/stream_progress.jsonl
```
```bash
source .venv/bin/activate
python src/streaming/consumer.py --broker localhost:9092 --topic bank_scored
```

Streaming details:
- Input topic: `bank_raw`
- Output topic: `bank_scored`
- Dead-letter topic: `bank_deadletter`
- Checkpoint root: `data/checkpoints`

## Streaming progress summary
```bash
source .venv/bin/activate
python src/streaming/summarize_progress.py --input report/stream_progress.jsonl --out_json report/stream_summary.json
```

Expected output:
- Summary: `report/stream_summary.json`
