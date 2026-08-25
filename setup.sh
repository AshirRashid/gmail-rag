#!/usr/bin/env bash
set -euo pipefail

# ── colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
die()   { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

# ── 0. working directory ─────────────────────────────────────────────────────
cd "$(dirname "$0")"

# ── 1. Python version ────────────────────────────────────────────────────────
info "Checking Python version..."
PYTHON=$(command -v python3.11 || command -v python3 || die "Python 3 not found")
PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
[[ "${PY_VER}" < "3.11" ]] && die "Python 3.11+ required (found $PY_VER)"
info "Using Python $PY_VER at $PYTHON"

# ── 2. Virtual environment ───────────────────────────────────────────────────
if [[ ! -d ".venv" ]]; then
    info "Creating virtual environment..."
    "$PYTHON" -m venv .venv
fi
source .venv/bin/activate
info "Virtual environment active"

# ── 3. Install dependencies ──────────────────────────────────────────────────
info "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
info "Dependencies installed"

# ── 4. ChromaDB ──────────────────────────────────────────────────────────────
CHROMA_PORT="${CHROMA_PORT:-8000}"
CHROMA_DATA_DIR="${CHROMA_DATA_DIR:-./chroma-data}"

if nc -z localhost "${CHROMA_PORT}" 2>/dev/null; then
    info "ChromaDB is already running on port ${CHROMA_PORT}"
else
    info "Starting ChromaDB on port ${CHROMA_PORT}..."
    mkdir -p "$CHROMA_DATA_DIR"
    nohup chroma run \
        --path "$CHROMA_DATA_DIR" \
        --host 0.0.0.0 \
        --port "$CHROMA_PORT" \
        > chroma.log 2>&1 &
    CHROMA_PID=$!
    echo "$CHROMA_PID" > chroma.pid

    # Wait up to 60 s for Chroma to be ready (first run is slow - imports + model load)
    info "Waiting for ChromaDB to be ready..."
    for i in $(seq 1 60); do
        if nc -z localhost "${CHROMA_PORT}" 2>/dev/null; then
            info "ChromaDB started (pid $CHROMA_PID)"
            break
        fi
        sleep 1
        [[ $i -eq 60 ]] && die "ChromaDB did not start after 60s. Check chroma.log for details."
    done
fi

# ── 5. Gmail credentials ─────────────────────────────────────────────────────
CREDS_FILE=$(python3 -c "from config import CREDENTIALS_FILE; print(CREDENTIALS_FILE)")

if [[ -f "$CREDS_FILE" ]]; then
    info "Gmail credentials found: $CREDS_FILE"
else
    warn "Gmail credentials file not found: $CREDS_FILE"
    echo ""
    echo "  To get credentials:"
    echo "  1. Go to https://console.cloud.google.com/"
    echo "  2. Create a project and enable the Gmail API"
    echo "  3. Create an OAuth 2.0 Client ID (Desktop app)"
    echo "  4. Download the JSON file and place it here as: $CREDS_FILE"
    echo "  5. Re-run this script, or proceed directly with:"
    echo "       source .venv/bin/activate && python pipeline.py"
    echo ""
fi

# ── 6. Done ──────────────────────────────────────────────────────────────────
echo ""
info "Setup complete. Next steps:"
echo ""
echo "  source .venv/bin/activate"
echo ""
echo "  # Fetch and index emails:"
echo "  python pipeline.py"
echo ""
echo "  # Query for event-related emails:"
echo "  python query.py"
echo ""
