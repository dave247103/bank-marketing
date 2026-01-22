#!/usr/bin/env bash
set -euo pipefail

RATE=50
WARMUP=15
MAX_MESSAGES=5000
MODEL="models/pipeline_lr"
TAG=""

usage() {
  cat <<'EOF'
Usage: bash scripts/run_stream_latency.sh [--rate N] [--warmup SECONDS] [--max_messages N] [--model PATH] [--tag TAG]

Defaults:
  --rate 50
  --warmup 15
  --max_messages 5000
  --model models/pipeline_lr
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rate)
      RATE="$2"
      shift 2
      ;;
    --warmup)
      WARMUP="$2"
      shift 2
      ;;
    --max_messages)
      MAX_MESSAGES="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

producer_pid=""
scorer_pid=""

sanitize_tag() {
  local raw="$1"
  local cleaned
  cleaned="$(echo "$raw" | tr -cs 'A-Za-z0-9._-' '_' | sed 's/^_//; s/_$//')"
  if [[ -z "$cleaned" ]]; then
    cleaned="run"
  fi
  echo "$cleaned"
}

if [[ -n "$TAG" ]]; then
  TAG="$(sanitize_tag "$TAG")"
else
  TAG="$(sanitize_tag "$(basename "$MODEL")")"
fi

progress_path="report/stream_progress_${TAG}.jsonl"
latency_path="report/stream_latency_${TAG}.json"
summary_path="report/stream_summary_${TAG}.json"
progress_latest="report/stream_progress.jsonl"
latency_latest="report/stream_latency.json"
summary_latest="report/stream_summary.json"

stop_process() {
  local pid="$1"
  local name="$2"
  local wait_seconds="${3:-10}"

  if [[ -z "$pid" ]]; then
    return 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi

  echo "Stopping ${name} (pid ${pid})..."
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 "$wait_seconds"); do
    if ! kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done

  echo "Force killing ${name} (pid ${pid})..."
  kill -KILL "$pid" 2>/dev/null || true
}

cleanup() {
  stop_process "$producer_pid" "producer" 10
  stop_process "$scorer_pid" "scorer" 30
}

trap cleanup EXIT

echo "Resetting Kafka..."
docker compose down -v
docker compose up -d

echo "Waiting for Kafka broker..."
ready="false"
for _ in $(seq 1 30); do
  if docker compose exec -T kafka /opt/kafka/bin/kafka-topics.sh \
    --bootstrap-server localhost:9092 --list >/dev/null 2>&1; then
    ready="true"
    break
  fi
  sleep 1
done
if [[ "$ready" != "true" ]]; then
  echo "Kafka broker did not become ready in time." >&2
  exit 1
fi

create_topic() {
  local topic="$1"
  docker compose exec -T kafka /opt/kafka/bin/kafka-topics.sh \
    --bootstrap-server localhost:9092 --create --if-not-exists --topic "$topic" >/dev/null
}

create_topic "bank_raw"
create_topic "bank_scored"
create_topic "bank_deadletter"

rm -rf data/checkpoints
rm -f "$progress_path" "$latency_path" "$summary_path"
rm -f "$progress_latest" "$latency_latest" "$summary_latest"

echo "Starting scorer..."
python src/streaming/scorer.py --starting_offsets latest --progress_log "$progress_path" --model "$MODEL" &
scorer_pid=$!

echo "Starting producer..."
python src/streaming/producer.py --rate "$RATE" --repeat --seed 42 &
producer_pid=$!

if [[ "$WARMUP" != "0" ]]; then
  echo "Warming up for ${WARMUP}s..."
  sleep "$WARMUP"
fi

run_id="$(date +%Y%m%d%H%M%S)"
group_id="bank-scored-consumer-${run_id}"

python src/streaming/consumer.py \
  --max_messages "$MAX_MESSAGES" \
  --latency_out "$latency_path" \
  --group "$group_id"

cleanup
producer_pid=""
scorer_pid=""

python src/streaming/summarize_progress.py \
  --input "$progress_path" \
  --out_json "$summary_path"

cp "$progress_path" "$progress_latest"
cp "$latency_path" "$latency_latest"
cp "$summary_path" "$summary_latest"

python src/report/make_figures.py --out_dir report/figures

python - <<'PY'
import json

path = "report/stream_latency.json"
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)

def fmt(value):
    return "n/a" if value is None else f"{value:.2f}"

count = data.get("count")
avg = fmt(data.get("avg"))
p95 = fmt(data.get("p95"))

print(f"Latency summary ({path}): count={count} avg={avg} ms p95={p95} ms")
print("Outputs:")
print("  report/stream_progress.jsonl")
print("  report/stream_latency.json")
print("  report/stream_summary.json")
print("  report/figures")
PY
