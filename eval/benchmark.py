"""Retrieval benchmark: semantic pipeline vs. BM25, over the synthetic corpus."""
import json
from pathlib import Path

import chromadb

from config import CHROMA_COLLECTION, CHROMA_HOST, CHROMA_PORT, EMBED_MODEL
from embeddings import BGEEmbeddings
from eval.baseline_bm25 import BM25Retriever
from eval.metrics import mrr, precision_at_k, recall_at_k
from gmail_client import Email
from pipeline import ingest
from query import search_collection

CORPUS_DIR = Path(__file__).parent / "synthetic_corpus"
RESULTS_DIR = Path(__file__).parent / "results"


def _load_corpus() -> tuple[list[dict], list[dict]]:
    emails = json.loads((CORPUS_DIR / "emails.json").read_text())
    queries = json.loads((CORPUS_DIR / "queries.json").read_text())
    return emails, queries


def _build_semantic_collection(emails: list[dict]):
    client = chromadb.EphemeralClient()
    email_objs = [Email(**e) for e in emails]
    ingest(email_objs, client=client)
    return client.get_collection(name=CHROMA_COLLECTION, embedding_function=BGEEmbeddings(EMBED_MODEL))

def _score_method(rank_fn, queries: list[dict], k: int) -> dict:
    per_query = []
    for q in queries:
        relevant = set(q["relevant_ids"])
        ranked = rank_fn(q["query"], k)
        per_query.append({
            "query": q["query"],
            "relevant_ids": q["relevant_ids"],
            "ranked_ids": ranked,
            "precision_at_k": precision_at_k(relevant, ranked, k),
            "recall_at_k": recall_at_k(relevant, ranked, k),
            "reciprocal_rank": mrr(relevant, ranked),
        })
    n = len(per_query)
    return {
        "precision": sum(r["precision_at_k"] for r in per_query) / n,
        "recall": sum(r["recall_at_k"] for r in per_query) / n,
        "mrr": sum(r["reciprocal_rank"] for r in per_query) / n,
        "per_query": per_query,
    }


def _write_markdown_summary(result: dict, path: Path) -> None:
    lines = ["# Synthetic retrieval benchmark", "", f"k = {result['k']}", "",
             "| Method | Precision@k | Recall@k | MRR |", "|---|---|---|---|"]
    for method in ("semantic", "bm25"):
        m = result[method]
        lines.append(f"| {method} | {m['precision']:.3f} | {m['recall']:.3f} | {m['mrr']:.3f} |")
    path.write_text("\n".join(lines) + "\n")


def run_synthetic_benchmark(k: int = 5) -> dict:
    emails, queries = _load_corpus()

    collection = _build_semantic_collection(emails)
    semantic_rank = lambda text, k: [r["id"] for r in search_collection(collection, text, n_results=k)]
    semantic = _score_method(semantic_rank, queries, k)

    bm25 = BM25Retriever([{"id": e["id"], "subject": e["subject"], "body": e["body"]} for e in emails if not e["is_reply"]])
    bm25_scores = _score_method(bm25.search, queries, k)

    result = {"k": k, "semantic": semantic, "bm25": bm25_scores}

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "synthetic_benchmark.json").write_text(json.dumps(result, indent=2))
    _write_markdown_summary(result, RESULTS_DIR / "synthetic_benchmark.md")
    return result


def run_real_benchmark(queries_path: str, k: int = 5) -> dict:
    """
    Run against the real, already-ingested Gmail collection.
    `queries_path` points at a local, gitignored file: a JSON list of
    {"query": str, "relevant_ids": list[str]} that you hand-label yourself
    by running query.py against your own inbox first.
    Only aggregate numbers are written — no email content, subjects, or senders.
    """
    queries = json.loads(Path(queries_path).read_text())

    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = client.get_collection(name=CHROMA_COLLECTION, embedding_function=BGEEmbeddings(EMBED_MODEL))
    semantic_rank = lambda text, k: [r["id"] for r in search_collection(collection, text, n_results=k)]
    semantic = _score_method(semantic_rank, queries, k)

    raw = collection.get(include=["documents", "metadatas"])
    docs = [{"id": rid, "subject": meta.get("subject", ""), "body": doc}
            for rid, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"])]
    bm25 = BM25Retriever(docs)
    bm25_scores = _score_method(bm25.search, queries, k)

    summary = {
        "k": k,
        "n_queries": len(queries),
        "semantic": {"precision": semantic["precision"], "recall": semantic["recall"], "mrr": semantic["mrr"]},
        "bm25": {"precision": bm25_scores["precision"], "recall": bm25_scores["recall"], "mrr": bm25_scores["mrr"]},
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "real_benchmark_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    result = run_synthetic_benchmark()
    print(json.dumps({k: {m: v for m, v in val.items() if m != "per_query"} if k in ("semantic", "bm25") else val
                       for k, val in result.items()}, indent=2))
