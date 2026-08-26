"""Ingestion throughput and query latency measurement."""
import json
import os
import platform
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


def _machine_info() -> dict:
    """Capture the hardware and OS the latency numbers were measured on.

    A latency figure is meaningless without the machine behind it, so this
    is recorded alongside the numbers. The pipeline has no GPU path, so the
    embedding model runs on CPU regardless of the host.
    """
    return {
        "os": f"{platform.system()} {platform.release()}",
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "device": "cpu",
    }


def _load_emails() -> list[dict]:
    return json.loads((CORPUS_DIR / "emails.json").read_text())


def measure_ingestion(n_values: list[int]) -> list[dict]:
    emails = _load_emails()
    results = []
    for n in n_values:
        subset = [Email(**e) for e in emails[:n]]
        client = chromadb.EphemeralClient()
        start = time.perf_counter()
        result = ingest(subset, client=client)
        elapsed = time.perf_counter() - start
        saved = result.get("emails_saved", 0)
        results.append({"n": n, "seconds": round(elapsed, 3), "emails_per_sec": round(saved / elapsed, 2) if elapsed > 0 else 0.0})
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


def run_latency_report(n_values: list[int] = None, repeats: int = 3, output_path: Path = None) -> dict:
    n_values = n_values or [30, 60]
    emails = _load_emails()
    queries = json.loads((CORPUS_DIR / "queries.json").read_text())
    query_texts = [q["query"] for q in queries]

    client = chromadb.EphemeralClient()
    ingest([Email(**e) for e in emails], client=client)
    collection = client.get_collection(name=CHROMA_COLLECTION, embedding_function=BGEEmbeddings(EMBED_MODEL))

    report = {
        "cost_usd_marginal": 0.0,
        "cost_note": "Local embedding model (BGE-base-en-v1.5), no external API calls - marginal cost per query or ingest run is $0.",
        "machine": _machine_info(),
        "latency_note": "Latency is hardware-dependent. Measured on the machine in 'machine', CPU only (no GPU).",
        "ingestion": measure_ingestion(n_values),
        "query_latency": measure_query_latency(collection, query_texts, repeats=repeats),
    }
    if output_path is None:
        output_path = RESULTS_DIR / "latency.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(run_latency_report(), indent=2))
