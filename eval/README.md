# Evidence pack

This directory holds the evidence that the semantic-search pipeline in this repo actually works, and where it does not. Every number below traces to a committed file produced by a script in this directory, so it is reproducible rather than asserted.

The four pieces:

| Evidence | File | Headline |
|---|---|---|
| Retrieval quality | [`results/synthetic_benchmark.md`](results/synthetic_benchmark.md) | Semantic beats a BM25 baseline on precision, recall, and MRR (see below). |
| Latency & cost | [`results/latency.json`](results/latency.json) | Query p50 60.7ms / p95 67.9ms; ingestion 7.73-13.45 emails/sec; $0 marginal cost. |
| Failure analysis | [`failures.md`](failures.md) | The lowest-recall queries per method, each traced to a concrete cause. |
| Security review | [`security_review.md`](security_review.md) | OAuth read-only scope, unencrypted at-rest storage, one unresolved output-injection gap. |

## Retrieval quality

Measured on a 60-email synthetic corpus (`synthetic_corpus/`) with hand-verified ground truth, over 16 natural-language queries, at k = 5. The semantic pipeline (BGE-base-en-v1.5 whole-email embeddings) is compared against a BM25 bag-of-words baseline.

| Method | Precision@k | Recall@k | MRR |
|---|---|---|---|
| semantic | 0.463 | 0.849 | 0.781 |
| bm25 | 0.350 | 0.688 | 0.682 |

Reproduce: `python -m eval.benchmark`. The gap is largest exactly where you would expect semantic search to help: queries whose wording does not match the correct email's wording (see `failures.md` for the BM25 vocabulary-gap cases where semantic scored a perfect 1.0 recall and BM25 scored 0).

## Latency & cost

From `results/latency.json`, measured on this machine:

- Ingestion throughput: 7.73 emails/sec at n=30, 13.45 emails/sec at n=60.
- Query latency: p50 60.7ms, p95 67.9ms, over 48 samples.
- Marginal cost: $0. The embedding model (BGE-base-en-v1.5) runs locally with no external API calls, so there is no per-query or per-ingest cost.

Reproduce: `python -m eval.latency`.

## Failure analysis

`failures.md` takes the five lowest-recall queries per method and explains each one. The short version: the semantic pipeline's low scores are mostly a benchmark ceiling (12 near-duplicate relevant emails per category evaluated at k=5 caps recall at 5/12 = 0.417) plus one narrow category-boundary miss, not real retrieval failures. BM25's low scores are genuine, caused by literal keyword mismatch and unfiltered stopwords letting short off-topic newsletters outrank correct emails.

## Security review

`security_review.md` documents three concrete findings against the real code: the OAuth scope is read-only (confirmed by test), ChromaDB stores email text unencrypted at rest (accepted tradeoff for a local single-user tool), and retrieved snippets reach Gradio's Markdown renderer unescaped (a real, unresolved output-injection gap, blast radius limited to the operator's own machine).

## Try it

No Gmail account, zero setup:

```bash
./demo.sh
```

This ingests the synthetic corpus and launches the UI so you can query it immediately.

Against your own inbox (needs Gmail OAuth credentials, see the top-level [`README.md`](../README.md)):

```bash
./setup.sh && ./run.sh
```
