#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/uvicorn.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file found. API may already be stopped."
  exit 0
fi

PID="$(cat "$PID_FILE")"

if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "Stopped API process $PID"
else
  echo "Process $PID is not running."
fi

rm -f "$PID_FILE"
