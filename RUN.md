RUN STEPS

1) Start Kafka (Docker)
   docker compose up -d

2) Preprocess reports
   python preprocess.py --input data/bank-full.csv --report-dir report --keep-duration false --pdays-features drop_pdays_clean

3) Train + save model
   python train.py --input data/bank-full.csv --report-dir report --artifacts-dir artifacts --keep-duration false --pdays-features drop_pdays_clean

4) Terminal A: consume predictions
   python stream_consumer.py --bootstrap localhost:9092 --topic bank_pred --group-id bank_consumer_latest

5) Terminal B: start streaming inference
   spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8 stream_infer.py \
     --starting-offsets latest \
     --bootstrap localhost:9092 \
     --input-topic bank_raw \
     --output-topic bank_pred \
     --artifacts-dir artifacts

6) Terminal C: produce streaming data
   python stream_producer.py --input data/bank-full.csv --bootstrap localhost:9092 --topic bank_raw \
     --rate 50 --log-every 1000 --flush-every 5000

Notes:
- Add --loop to stream_producer.py to keep the stream running.
- stream_infer.py reads artifacts/metadata.json to align keep_duration and pdays_features with training.
