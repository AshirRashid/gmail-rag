"""BM25 baseline retriever — the standard IR comparison point for the semantic pipeline."""
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, documents: list[dict]):
        self._ids = [d["id"] for d in documents]
        corpus = [f"{d['subject']} {d['body']}".lower().split() for d in documents]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int) -> list[str]:
        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(zip(self._ids, scores), key=lambda pair: pair[1], reverse=True)
        return [doc_id for doc_id, _ in ranked[:k]]
