#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/opt/anaconda3/bin/python3"

cd "$SCRIPT_DIR"
"$PYTHON_BIN" -m streamlit run streamlit_app.py --server.port 8501 --server.address 127.0.0.1
