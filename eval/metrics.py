"""Retrieval evaluation metrics: precision@k, recall@k, MRR."""


def precision_at_k(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float:
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(top_k)


def recall_at_k(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = ranked_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def mrr(relevant_ids: set[str], ranked_ids: list[str]) -> float:
    for rank, rid in enumerate(ranked_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def lowest_scoring(per_query_results: list[dict], metric: str, k: int) -> list[dict]:
    """Return the k entries with the lowest value for `metric`, ascending."""
    return sorted(per_query_results, key=lambda r: r[metric])[:k]
