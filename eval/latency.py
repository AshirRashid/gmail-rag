"""Ingestion throughput and query latency measurement."""
import json
import statistics
import time
from pathlib import Path

import chromadb

from config import CHROMA_COLLECTION, EMBED_MODEL
from embeddings import BGEEmbeddings
from gmail_client import Email
from pipeline import ingest
from query import search_collection

CORPUS_DIR = Path(__file__).parent / "synthetic_corpus"
RESULTS_DIR = Path(__file__).parent / "results"


def _load_emails() -> list[dict]:
    return json.loads((CORPUS_DIR / "emails.json").read_text())


def measure_ingestion(n_values: list[int]) -> list[dict]:
    emails = _load_emails()
    results = []
    for n in n_values:
        subset = [Email(**e) for e in emails[:n]]
        client = chromadb.EphemeralClient()
        start = time.perf_counter()
        ingest(subset, client=client)
        elapsed = time.perf_counter() - start
        results.append({"n": n, "seconds": round(elapsed, 3), "emails_per_sec": round(n / elapsed, 2)})
    return results


def measure_query_latency(collection, queries: list[str], repeats: int = 3) -> dict:
    timings_ms = []
    for _ in range(repeats):
        for q in queries:
            start = time.perf_counter()
            search_collection(collection, q, n_results=5)
            timings_ms.append((time.perf_counter() - start) * 1000)
    timings_ms.sort()
    return {
        "p50_ms": round(statistics.median(timings_ms), 1),
        "p95_ms": round(timings_ms[int(len(timings_ms) * 0.95) - 1], 1),
        "n_samples": len(timings_ms),
    }


def run_latency_report(n_values: list[int] = None, repeats: int = 3) -> dict:
    n_values = n_values or [50, 200]
    emails = _load_emails()
    queries = json.loads((CORPUS_DIR / "queries.json").read_text())
    query_texts = [q["query"] for q in queries]

    client = chromadb.EphemeralClient()
    ingest([Email(**e) for e in emails], client=client)
    collection = client.get_collection(name=CHROMA_COLLECTION, embedding_function=BGEEmbeddings(EMBED_MODEL))

    report = {
        "cost_usd_marginal": 0.0,
        "cost_note": "Local embedding model (BGE-base-en-v1.5), no external API calls — marginal cost per query or ingest run is $0.",
        "ingestion": measure_ingestion(n_values),
        "query_latency": measure_query_latency(collection, query_texts, repeats=repeats),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "latency.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(run_latency_report(), indent=2))
