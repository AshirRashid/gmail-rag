# eval/ingest_synthetic.py
"""Ingest the synthetic corpus into the real (persistent) ChromaDB collection -
used by demo.sh so a reviewer can try the actual app.py/query.py UI with no
Gmail account, using the same ingest() code path as the real pipeline."""
import json
from pathlib import Path

from gmail_client import Email
from pipeline import ingest

CORPUS_PATH = Path(__file__).parent / "synthetic_corpus" / "emails.json"

if __name__ == "__main__":
    emails = [Email(**e) for e in json.loads(CORPUS_PATH.read_text())]
    result = ingest(emails)
    print(f"Demo corpus ingested: {result}")
