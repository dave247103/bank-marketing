RUN STEPS

1) Start Kafka (Docker)
   docker compose up -d

2) Generate reports (optional but recommended)
   python preprocess.py --input data/bank-full.csv --report-dir report --keep-duration false --pdays-features both

3) Train + save model
   python train.py --input data/bank-full.csv --report-dir report --artifacts-dir artifacts --keep-duration false --pdays-features both

4) Terminal 1: consume predictions
   python stream_consumer.py --bootstrap localhost:9092 --topic bank_pred

5) Terminal 2: start streaming inference
   spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8 stream_infer.py --bootstrap localhost:9092 --input-topic bank_raw --output-topic bank_pred --artifacts-dir artifacts

6) Terminal 3: produce streaming data
   python stream_producer.py --input data/bank-full.csv --bootstrap localhost:9092 --topic bank_raw --rate 50

Notes:
- Add --loop to stream_producer.py to keep the stream running.
- stream_infer.py reads artifacts/metadata.json to align keep_duration and pdays_features with training.
