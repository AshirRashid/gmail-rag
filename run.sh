#!/usr/bin/env bash
set -euo pipefail

GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
info() { echo -e "${GREEN}[run]${NC} $*"; }
die()  { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

cd "$(dirname "$0")"

# ── 1. Virtual environment ────────────────────────────────────────────────────
[[ -d ".venv" ]] || die ".venv not found — run setup.sh first"
source .venv/bin/activate

# ── 2. ChromaDB ───────────────────────────────────────────────────────────────
CHROMA_PORT="${CHROMA_PORT:-8000}"
CHROMA_DATA_DIR="${CHROMA_DATA_DIR:-./chroma-data}"

if nc -z localhost "${CHROMA_PORT}" 2>/dev/null; then
    info "ChromaDB already running on port ${CHROMA_PORT}"
else
    info "Starting ChromaDB..."
    mkdir -p "$CHROMA_DATA_DIR"
    nohup chroma run \
        --path "$CHROMA_DATA_DIR" \
        --host 0.0.0.0 \
        --port "$CHROMA_PORT" \
        > chroma.log 2>&1 &
    echo $! > chroma.pid

    info "Waiting for ChromaDB to be ready..."
    for i in $(seq 1 60); do
        nc -z localhost "${CHROMA_PORT}" 2>/dev/null && break
        sleep 1
        [[ $i -eq 60 ]] && die "ChromaDB did not start. Check chroma.log."
    done
    info "ChromaDB started"
fi

# ── 3. Launch UI ──────────────────────────────────────────────────────────────
info "Launching UI → http://127.0.0.1:7860"
python app.py
