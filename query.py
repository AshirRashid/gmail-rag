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


def query(text: str = DEFAULT_QUERY, n_results: int = N_RESULTS) -> None:
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    try:
        collection = client.get_collection(
            name=CHROMA_COLLECTION,
            embedding_function=BGEEmbeddings(EMBED_MODEL),
        )
    except Exception as exc:
        print(
            f"Collection '{CHROMA_COLLECTION}' not found — run pipeline.py first.\n{exc}"
        )
        return

    results = collection.query(
        query_texts=[text],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    print(f"\nQuery : {text}")
    print("=" * 60)
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        score = 1.0 - dist  # cosine similarity (collection uses cosine space)
        snippet = doc[:300].replace("\n", " ")
        print(f"\n[{i}]  score={score:.3f}")
        print(f"     Subject : {meta.get('subject', 'N/A')}")
        print(f"     From    : {meta.get('sender', 'N/A')}")
        print(f"     Date    : {meta.get('date', 'N/A')}")
        print(f"     Snippet : {snippet}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query indexed emails for event-related content.")
    parser.add_argument("-q", "--query", default=DEFAULT_QUERY, help="Search query (default: event/meeting/schedule query)")
    parser.add_argument("-n", "--n-results", type=int, default=N_RESULTS, help=f"Number of results to return (default: {N_RESULTS})")
    args = parser.parse_args()
    query(args.query, args.n_results)
