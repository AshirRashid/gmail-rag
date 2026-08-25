#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

GREEN='\033[0;32m'; NC='\033[0m'
info() { echo -e "${GREEN}[demo]${NC} $*"; }

if [[ ! -d ".venv" ]]; then
    info "First run - setting up venv and dependencies..."
    PYTHON=$(command -v python3.11 || command -v python3)
    "$PYTHON" -m venv .venv
    source .venv/bin/activate
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
    pip install --quiet -r eval/requirements.txt
else
    info "venv ready, dependencies cached"
    source .venv/bin/activate
fi

CHROMA_PORT="${CHROMA_PORT:-8000}"
CHROMA_DATA_DIR="${CHROMA_DATA_DIR:-./chroma-data}"
if nc -z localhost "${CHROMA_PORT}" 2>/dev/null; then
    info "ChromaDB already running on port ${CHROMA_PORT}"
else
    info "Starting ChromaDB..."
    mkdir -p "$CHROMA_DATA_DIR"
    nohup chroma run --path "$CHROMA_DATA_DIR" --host 127.0.0.1 --port "$CHROMA_PORT" > chroma.log 2>&1 &
    echo $! > chroma.pid
    for i in $(seq 1 60); do
        nc -z localhost "${CHROMA_PORT}" 2>/dev/null && break
        sleep 1
    done
fi

info "Loading synthetic inbox (no Gmail account needed)..."
python -m eval.ingest_synthetic

info "Launching UI → http://127.0.0.1:7860"
python app.py
