"""
Ingestion pipeline: Gmail → clean → chunk → embed → ChromaDB upsert.

Run directly:
    python pipeline.py            # uses N_EMAILS env var (default 50)
    N_EMAILS=200 python pipeline.py
"""

import os
from dataclasses import dataclass

import chromadb
from email_reply_parser import EmailReplyParser

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_COLLECTION,
    CHROMA_HOST,
    CHROMA_PORT,
    EMBED_MODEL,
    N_EMAILS,
)
from embeddings import BGEEmbeddings
from gmail_client import Email, fetch_emails


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

_STRIP_SENTINELS = ("unsubscribe", "view this email in your browser")


def _clean(body: str) -> str:
    """Remove quoted replies and common footer noise."""
    text = EmailReplyParser.parse_reply(body)
    lower = text.lower()
    for sentinel in _STRIP_SENTINELS:
        idx = lower.find(sentinel)
        if idx != -1:
            text = text[:idx]
            break
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


# ---------------------------------------------------------------------------
# Chunk  (no external dependency — simple sliding window)
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


def _chunk(email: Email, body: str) -> list[Chunk]:
    header = (
        f"From: {email.sender}\n"
        f"Subject: {email.subject}\n"
        f"Date: {email.date}\n\n"
    )
    full_text = header + body
    pieces: list[str] = []
    start = 0
    while start < len(full_text):
        pieces.append(full_text[start : start + CHUNK_SIZE])
        if start + CHUNK_SIZE >= len(full_text):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP

    base_meta = {
        "email_id": email.id,
        "thread_id": email.thread_id,
        "subject": email.subject,
        "sender": email.sender,
        "date": email.date,
    }
    return [
        Chunk(
            id=f"{email.id}-{i}",
            text=piece,
            metadata={**base_meta, "chunk_index": i},
        )
        for i, piece in enumerate(pieces)
    ]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(n: int = N_EMAILS) -> dict:
    emails = fetch_emails(n)
    if not emails:
        print("No emails to process.")
        return {"status": "done", "chunks_saved": 0}

    all_chunks: list[Chunk] = []
    for email in emails:
        body = _clean(email.body)
        if body:
            all_chunks.extend(_chunk(email, body))

    print(f"Created {len(all_chunks)} chunks from {len(emails)} emails")

    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=BGEEmbeddings(EMBED_MODEL),
        metadata={"hnsw:space": "cosine"},
    )

    # Upsert is idempotent — safe to re-run without creating duplicates
    collection.upsert(
        ids=[c.id for c in all_chunks],
        documents=[c.text for c in all_chunks],
        metadatas=[c.metadata for c in all_chunks],
    )
    print(f"Upserted {len(all_chunks)} chunks → '{CHROMA_COLLECTION}'")
    return {
        "status": "done",
        "emails_processed": len(emails),
        "chunks_saved": len(all_chunks),
        "collection": CHROMA_COLLECTION,
    }


if __name__ == "__main__":
    result = run()
    print("Pipeline complete:", result)
