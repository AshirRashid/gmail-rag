"""
Query ChromaDB for event-related emails.

Run directly:
    python query.py
    python query.py -q "team lunch next week"
    python query.py -q "dentist appointment" -n 10
"""

import argparse
import os
import sys

import chromadb

from config import CHROMA_COLLECTION, CHROMA_HOST, CHROMA_PORT, EMBED_MODEL
from embeddings import BGEEmbeddings

DEFAULT_QUERY = (
    "emails about events, meetings, appointments, invitations, "
    "deadlines, schedules, or calendar items"
)
N_RESULTS = int(os.getenv("N_RESULTS", "5"))


def query(text: str = DEFAULT_QUERY, n_results: int = N_RESULTS) -> list[dict]:
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    try:
        collection = client.get_collection(
            name=CHROMA_COLLECTION,
            embedding_function=BGEEmbeddings(EMBED_MODEL),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Collection '{CHROMA_COLLECTION}' not found — run pipeline.py first."
        ) from exc

    results = collection.query(
        query_texts=[text],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "subject": meta.get("subject", "N/A"),
            "sender": meta.get("sender", "N/A"),
            "date": meta.get("date", "N/A"),
            "score": round(1.0 - dist, 3),
            "snippet": doc[:300].replace("\n", " "),
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query indexed emails for event-related content.")
    parser.add_argument("-q", "--query", default=DEFAULT_QUERY, help="Search query (default: event/meeting/schedule query)")
    parser.add_argument("-n", "--n-results", type=int, default=N_RESULTS, help=f"Number of results to return (default: {N_RESULTS})")
    args = parser.parse_args()

    for i, r in enumerate(query(args.query, args.n_results), 1):
        print(f"\n[{i}]  score={r['score']}")
        print(f"     Subject : {r['subject']}")
        print(f"     From    : {r['sender']}")
        print(f"     Date    : {r['date']}")
        print(f"     Snippet : {r['snippet']}...")
