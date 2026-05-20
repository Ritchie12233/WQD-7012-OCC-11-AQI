#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/opt/anaconda3/bin/python3"
PID_FILE="$SCRIPT_DIR/uvicorn.pid"
LOG_FILE="$SCRIPT_DIR/uvicorn.log"

if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE")"
  if kill -0 "$PID" 2>/dev/null; then
    echo "API is already running at PID $PID"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

cd "$SCRIPT_DIR"
nohup "$PYTHON_BIN" -m uvicorn app:app --host 127.0.0.1 --port 8000 > "$LOG_FILE" 2>&1 < /dev/null &
echo $! > "$PID_FILE"
disown "$(cat "$PID_FILE")" 2>/dev/null || true

for _ in {1..15}; do
  if kill -0 "$(cat "$PID_FILE")" 2>/dev/null && curl -fsS "http://127.0.0.1:8000/" >/dev/null; then
    break
  fi
  sleep 1
done

if kill -0 "$(cat "$PID_FILE")" 2>/dev/null && curl -fsS "http://127.0.0.1:8000/" >/dev/null; then
  echo "API started on http://127.0.0.1:8000"
  echo "PID: $(cat "$PID_FILE")"
  echo "Log: $LOG_FILE"
else
  echo "API failed to start. Check $LOG_FILE"
  exit 1
fi
