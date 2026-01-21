# UCI Bank Marketing - Milestone 1

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
