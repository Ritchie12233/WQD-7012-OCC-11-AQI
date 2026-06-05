#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python3}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" -m streamlit run streamlit_app.py --server.port "${PORT:-8501}" --server.address 127.0.0.1
