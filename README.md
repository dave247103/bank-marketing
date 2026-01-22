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
python src/streaming/producer.py --input data/processed/bank.parquet --broker localhost:9092 --topic bank_raw --rate 50 --repeat
```
```bash
source .venv/bin/activate
python src/streaming/scorer.py --model models/pipeline_lr --broker localhost:9092 --in_topic bank_raw --out_topic bank_scored --deadletter_topic bank_deadletter --checkpoint data/checkpoints
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
