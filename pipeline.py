"""
Ingestion pipeline: Gmail → clean → embed whole email → ChromaDB upsert.

Only original emails are indexed; replies (identified by the In-Reply-To header)
are skipped so each thread is represented once by its opening message.

Run directly:
    python pipeline.py            # uses N_EMAILS env var (default 50)
    N_EMAILS=200 python pipeline.py
"""

import chromadb
from email_reply_parser import EmailReplyParser

from config import CHROMA_COLLECTION, CHROMA_HOST, CHROMA_PORT, EMBED_MODEL, N_EMAILS
from embeddings import BGEEmbeddings
from gmail_client import Email, fetch_emails


_STRIP_SENTINELS = ("unsubscribe", "view this email in your browser")


def _clean(body: str) -> str:
    """Strip quoted reply blocks and common footer noise."""
    text = EmailReplyParser.parse_reply(body)
    lower = text.lower()
    for sentinel in _STRIP_SENTINELS:
        idx = lower.find(sentinel)
        if idx != -1:
            text = text[:idx]
            break
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def _to_document(email: Email, body: str) -> str:
    """Format the email as a single string for embedding."""
    return (
        f"From: {email.sender}\n"
        f"Subject: {email.subject}\n"
        f"Date: {email.date}\n\n"
        f"{body}"
    )


def ingest(emails: list[Email], client=None) -> dict:
    """Clean, embed, and upsert a list of Email objects. Skips replies."""
    originals = [e for e in emails if not e.is_reply]
    skipped = len(emails) - len(originals)
    print(f"Skipped {skipped} replies, processing {len(originals)} original emails")

    ids, documents, metadatas = [], [], []
    for email in originals:
        body = _clean(email.body)
        if not body:
            continue
        ids.append(email.id)
        documents.append(_to_document(email, body))
        metadatas.append({
            "email_id": email.id,
            "thread_id": email.thread_id,
            "subject": email.subject,
            "sender": email.sender,
            "date": email.date,
        })

    if not ids:
        print("No emails to process.")
        return {"status": "done", "emails_saved": 0}

    if client is None:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=BGEEmbeddings(EMBED_MODEL),
        metadata={"hnsw:space": "cosine"},
    )

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Upserted {len(ids)} emails → '{CHROMA_COLLECTION}'")
    return {
        "status": "done",
        "emails_processed": len(originals),
        "replies_skipped": skipped,
        "emails_saved": len(ids),
        "collection": CHROMA_COLLECTION,
    }


def run(n: int = N_EMAILS) -> dict:
    emails = fetch_emails(n)
    if not emails:
        print("No emails to process.")
        return {"status": "done", "emails_saved": 0}
    return ingest(emails)


if __name__ == "__main__":
    result = run()
    print("Pipeline complete:", result)
