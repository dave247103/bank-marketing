# Bank Marketing — Spark MLlib + Kafka (Binary Classification)

Predict term deposit subscription (y=yes/no) using batch ML (Spark MLlib) and real-time scoring (Spark Structured Streaming + Kafka).

See `AGENTS.md` for the canonical, reproducible command list.

## Requirements
- Python + venv
- Docker + Docker Compose

## Setup
```bash
source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d
```

## Batch pipeline
```bash
python src/etl/etl_bank.py --input data/raw/bank-full.csv --output data/processed/bank.parquet

python src/ml/train_compare_models.py --input data/processed/bank.parquet --seed 42 --report_out report/metrics_models.json --model_dir models
python src/ml/tune_model.py --input data/processed/bank.parquet --model gbt --out_model models/pipeline_tuned_gbt --report_out report/metrics_tuning_gbt.json
python src/ml/tune_model.py --input data/processed/bank.parquet --model rf  --out_model models/pipeline_tuned_rf  --report_out report/metrics_tuning_rf.json
```

## Streaming scoring demo (Kafka + Spark Structured Streaming)
Topics:
- input: `bank_raw`
- output: `bank_scored`
- dead-letter: `bank_deadletter`

Run in three terminals:
```bash
rm -rf data/checkpoints
python src/streaming/producer.py --input data/processed/bank.parquet --broker localhost:9092 --topic bank_raw --rate 20 --repeat
```
```bash
python src/streaming/scorer.py --model models/pipeline_lr --broker localhost:9092 --in_topic bank_raw --out_topic bank_scored --deadletter_topic bank_deadletter --checkpoint data/checkpoints --starting_offsets earliest --progress_log report/stream_progress.jsonl
```
```bash
python src/streaming/consumer.py --broker localhost:9092 --topic bank_scored --from_beginning --latency_out report/stream_latency.json
```

## Streaming performance artifacts
```bash
python src/streaming/summarize_progress.py --input report/stream_progress.jsonl --out_json report/stream_summary.json
```

## Figures for report/presentation
```bash
python src/report/make_figures.py --out_dir report/figures
```

## Kafka UI
If enabled in `docker-compose.yml`, open Kafka UI on:
- http://localhost:8080
