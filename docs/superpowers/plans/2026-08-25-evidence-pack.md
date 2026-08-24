# Gmail RAG Evidence Pack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a retrieval benchmark, latency/cost measurement, failure analysis, security check, and one-command onboarding for the Gmail RAG pipeline, then rewrite the public portfolio page around the real results.

**Architecture:** Two small refactors (`pipeline.py`, `query.py`) separate "fetch from Gmail" from "ingest" and "resolve a collection" from "search it," so an `eval/` subsystem can drive the exact same embedding/formatting/search code against an in-memory ChromaDB collection loaded from a synthetic corpus, instead of reimplementing it. Everything in `eval/` is either pure-function-testable (metrics, BM25) or runs against `chromadb.EphemeralClient()` (no network Chroma server required for tests or the benchmark). The real-inbox benchmark and the portfolio rewrite are the only steps that touch live, private data or another repo.

**Tech Stack:** Python 3.11+, chromadb 1.0.0, sentence-transformers (BGE-base-en-v1.5), rank-bm25, pytest.

**Spec:** `docs/superpowers/specs/2026-08-25-evidence-pack-design.md`

## Global Constraints

- Base commit for all work: `d041aa1`. Do not reintroduce chunking, LangChain, or an LLM generation step — those belong to the later APD project, not this one.
- Eval-only dependencies (`pytest`, `rank-bm25`) go in `eval/requirements.txt`, not the root `requirements.txt` — the core tool's dependency footprint stays as-is.
- Nothing that touches the real inbox (Task 7's real-benchmark run, any hand-labeled queries) may be committed with content — only aggregate metrics are ever committed.
- No network calls beyond the existing Gmail API and Hugging Face model download (already present in the codebase). No new external services.
- `demo.sh` must work with zero Gmail credentials and zero manual steps beyond running it.
- Every plan JSON/markdown output under `eval/results/` must be safe to publish as-is — if a step can't guarantee that, it doesn't go in `eval/results/`.

---

## File Structure

```
gmail-rag/
  pipeline.py                    # modified: adds ingest()
  query.py                       # modified: adds search_collection()
  eval/
    requirements.txt             # new: pytest, rank-bm25
    metrics.py                   # new: precision_at_k, recall_at_k, mrr, lowest_scoring
    baseline_bm25.py             # new: BM25Retriever
    benchmark.py                 # new: orchestrates synthetic + real runs
    latency.py                   # new: ingestion + query timing
    security_review.md           # new: writeup
    failures.md                  # new: writeup
    README.md                    # new: entry point
    demo.sh                      # new: zero-setup try-it path
    synthetic_corpus/
      generate.py                # new: deterministic corpus + query generator
      emails.json                # generated
      queries.json                # generated
    results/
      synthetic_benchmark.json    # generated
      synthetic_benchmark.md      # generated
      latency.json                 # generated
      real_benchmark_summary.json  # generated manually by user, aggregate-only
  tests/
    test_pipeline.py             # new
    test_query.py                # new
    test_metrics.py              # new
    test_bm25.py                 # new
    test_benchmark.py            # new
    test_corpus_schema.py         # new
    test_security_injection.py    # new
  setup.sh                       # fixed: uncomment venv/install steps
  run.sh                         # modified: auto-ingest if collection empty
  README.md                      # fixed: N_EMAILS default
```

---

### Task 1: Refactor pipeline.py — extract `ingest()`

**Files:**
- Modify: `pipeline.py:45-90` (the `run()` function)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `gmail_client.Email` (existing dataclass: `id, thread_id, subject, sender, date, body, is_reply`), `embeddings.BGEEmbeddings` (existing).
- Produces: `pipeline.ingest(emails: list[Email], client=None) -> dict` — the return dict has keys `status`, and on success `emails_processed`, `replies_skipped`, `emails_saved`, `collection`. `client` defaults to `None`, in which case a `chromadb.HttpClient` is created exactly as `run()` did before; passing an explicit client (e.g. `chromadb.EphemeralClient()`) is what Task 6 relies on.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py
import chromadb
from gmail_client import Email
from pipeline import ingest


def _email(id_, is_reply=False, body="Some content about a meeting next week."):
    return Email(
        id=id_, thread_id=f"t-{id_}", subject="Test subject",
        sender="a@example.com", date="Mon, 1 Jan 2026", body=body, is_reply=is_reply,
    )


def test_ingest_saves_original_emails_and_skips_replies():
    client = chromadb.EphemeralClient()
    emails = [_email("e1"), _email("e2", is_reply=True)]

    result = ingest(emails, client=client)

    assert result["status"] == "done"
    assert result["emails_processed"] == 1
    assert result["replies_skipped"] == 1
    assert result["emails_saved"] == 1

    collection = client.get_collection(name="emails")
    assert collection.count() == 1


def test_ingest_with_no_emails_returns_zero_saved():
    client = chromadb.EphemeralClient()
    result = ingest([], client=client)
    assert result == {"status": "done", "emails_saved": 0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd <repo> && source .venv/bin/activate && pytest tests/test_pipeline.py -v`
Expected: FAIL — `ImportError: cannot import name 'ingest' from 'pipeline'`

- [ ] **Step 3: Extract `ingest()` from `run()`**

Replace `pipeline.py`'s existing `run()` function (currently lines 45-90) with:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS (2 tests). Note: first run downloads BGE-base-en-v1.5 (~1-3 min); cached after.

- [ ] **Step 5: Commit**

```bash
git add pipeline.py tests/test_pipeline.py
git commit -m "Extract ingest() from pipeline.run() for reuse by eval harness"
```

---

### Task 2: Refactor query.py — extract `search_collection()`

**Files:**
- Modify: `query.py:26-58` (the `query()` function)
- Test: `tests/test_query.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `query.search_collection(collection, text: str, n_results: int = N_RESULTS) -> list[dict]`. Each result dict now has an additional `"id"` key (the email's `email_id` metadata) alongside the existing `subject`, `sender`, `date`, `score`, `snippet` — Task 6's benchmark needs ranked ids to score against ground truth, and `app.py` doesn't read a fixed key set from these dicts, so this is purely additive.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_query.py
import chromadb
from config import CHROMA_COLLECTION, EMBED_MODEL
from embeddings import BGEEmbeddings
from query import search_collection


def test_search_collection_ranks_relevant_doc_first():
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=BGEEmbeddings(EMBED_MODEL),
        metadata={"hnsw:space": "cosine"},
    )
    collection.upsert(
        ids=["e1", "e2"],
        documents=[
            "From: a@x.com\nSubject: Dentist appointment\n\nYour dentist appointment is confirmed for Tuesday at 3pm.",
            "From: b@x.com\nSubject: Weekly newsletter\n\nHere's what happened in tech this week.",
        ],
        metadatas=[
            {"email_id": "e1", "thread_id": "t1", "subject": "Dentist appointment", "sender": "a@x.com", "date": "Mon"},
            {"email_id": "e2", "thread_id": "t2", "subject": "Weekly newsletter", "sender": "b@x.com", "date": "Tue"},
        ],
    )

    results = search_collection(collection, "a confirmed dentist or doctor appointment", n_results=2)

    assert results[0]["id"] == "e1"
    assert results[0]["subject"] == "Dentist appointment"
    assert "id" in results[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_query.py -v`
Expected: FAIL — `ImportError: cannot import name 'search_collection' from 'query'`

- [ ] **Step 3: Extract `search_collection()`**

Replace `query.py`'s existing `query()` function with:

```python
def search_collection(collection, text: str, n_results: int = N_RESULTS) -> list[dict]:
    results = collection.query(
        query_texts=[text],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "id": meta.get("email_id", "N/A"),
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
    return search_collection(collection, text, n_results)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_query.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add query.py tests/test_query.py
git commit -m "Extract search_collection() from query.query(), add id to results"
```

---

### Task 3: eval/metrics.py — retrieval metrics

**Files:**
- Create: `eval/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `precision_at_k(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float`, `recall_at_k(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float`, `mrr(relevant_ids: set[str], ranked_ids: list[str]) -> float`, `lowest_scoring(per_query_results: list[dict], metric: str, k: int) -> list[dict]`. Consumed by Task 6 (benchmark) and Task 10 (failure analysis).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
from eval.metrics import precision_at_k, recall_at_k, mrr, lowest_scoring


def test_precision_and_recall_at_k():
    relevant = {"a", "b"}
    ranked = ["a", "c", "b", "d"]
    assert precision_at_k(relevant, ranked, k=2) == 0.5
    assert recall_at_k(relevant, ranked, k=2) == 0.5
    assert recall_at_k(relevant, ranked, k=4) == 1.0


def test_precision_and_recall_with_no_hits():
    assert precision_at_k({"z"}, ["a", "b"], k=2) == 0.0
    assert recall_at_k({"z"}, ["a", "b"], k=2) == 0.0


def test_precision_with_empty_ranked_list():
    assert precision_at_k({"a"}, [], k=5) == 0.0


def test_mrr_rewards_earlier_hits():
    assert mrr({"a"}, ["a", "b", "c"]) == 1.0
    assert mrr({"c"}, ["a", "b", "c"]) == 1 / 3
    assert mrr({"z"}, ["a", "b", "c"]) == 0.0


def test_lowest_scoring_returns_ascending():
    results = [
        {"query": "q1", "recall_at_5": 0.8},
        {"query": "q2", "recall_at_5": 0.1},
        {"query": "q3", "recall_at_5": 0.5},
    ]
    worst = lowest_scoring(results, metric="recall_at_5", k=2)
    assert [r["query"] for r in worst] == ["q2", "q3"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval'` (create `eval/__init__.py` as an empty file first if pytest can't find the package — see step 3).

- [ ] **Step 3: Implement**

Create `eval/__init__.py` (empty file) and `eval/metrics.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add eval/__init__.py eval/metrics.py tests/test_metrics.py
git commit -m "Add retrieval metrics: precision@k, recall@k, MRR, lowest_scoring"
```

---

### Task 4: Synthetic corpus + query generator

**Files:**
- Create: `eval/synthetic_corpus/generate.py`
- Test: `tests/test_corpus_schema.py`

**Interfaces:**
- Produces: `eval/synthetic_corpus/emails.json` (list of `{id, thread_id, subject, sender, date, body, is_reply}`) and `eval/synthetic_corpus/queries.json` (list of `{query, relevant_ids}`). Consumed by Task 6 (benchmark) and Task 12 (`demo.sh`).
- Corpus size: 60 emails — 12 each of events, deadlines, newsletters, receipts, plus 6 original/reply pairs (12 raw, 6 indexed after reply-skip) — deliberately smaller than the spec's rough "~150" estimate so every email and every query's ground truth stays hand-verifiable rather than combinatorially sprawling.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_corpus_schema.py
import json
from pathlib import Path

CORPUS_DIR = Path("eval/synthetic_corpus")


def test_generated_corpus_has_expected_size_and_categories():
    emails = json.loads((CORPUS_DIR / "emails.json").read_text())
    assert len(emails) == 60
    ids = {e["id"] for e in emails}
    assert len(ids) == 60  # all unique
    replies = [e for e in emails if e["is_reply"]]
    assert len(replies) == 6


def test_every_query_relevant_id_exists_in_corpus():
    emails = json.loads((CORPUS_DIR / "emails.json").read_text())
    email_ids = {e["id"] for e in emails}
    queries = json.loads((CORPUS_DIR / "queries.json").read_text())
    assert len(queries) >= 15
    for q in queries:
        assert q["relevant_ids"], f"query has no relevant_ids: {q['query']}"
        for rid in q["relevant_ids"]:
            assert rid in email_ids, f"{rid} referenced by query but not in corpus"


def test_no_query_relevant_id_is_a_reply():
    emails = json.loads((CORPUS_DIR / "emails.json").read_text())
    reply_ids = {e["id"] for e in emails if e["is_reply"]}
    queries = json.loads((CORPUS_DIR / "queries.json").read_text())
    for q in queries:
        assert not (set(q["relevant_ids"]) & reply_ids), (
            f"query '{q['query']}' points at a reply, which the pipeline skips"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_corpus_schema.py -v`
Expected: FAIL — `FileNotFoundError` (no `emails.json` yet)

- [ ] **Step 3: Write the generator**

```python
# eval/synthetic_corpus/generate.py
"""Deterministic synthetic email corpus + query ground truth for the retrieval benchmark."""
import itertools
import json
from pathlib import Path

OUT_DIR = Path(__file__).parent

SENDERS = {
    "events": ["parties@evite-clone.com", "alumni@nyuad-network.org", "team-social@acme-labs.io"],
    "deadlines": ["research-office@university.edu", "no-reply@conference-system.org", "hr@acme-labs.io"],
    "newsletters": ["digest@techweekly.com", "updates@fintech-brief.io", "news@opensource-monthly.org"],
    "receipts": ["receipts@cloudhost.com", "billing@streamplus.io", "orders@gearshop.com"],
}

EVENT_TEMPLATES = [
    ("You're invited: {kind} on {date}", "RSVP by {date} — {kind} kicks off at {time} at {venue}. Bring a friend."),
    ("Save the date — {kind} at {venue}", "We're hosting {kind} on {date} at {venue}. Details and RSVP link inside."),
    ("Reminder: {kind} starts {time}", "Just a reminder that {kind} starts at {time} on {date}. See you there."),
    ("Join us for {kind}", "{kind} is happening {date} at {venue}, starting {time}. Hope to see you."),
]
EVENT_FILL = [
    {"kind": "the team dinner", "date": "Friday", "time": "7pm", "venue": "The Local Bistro"},
    {"kind": "a dentist appointment", "date": "Tuesday", "time": "3pm", "venue": "Downtown Dental"},
    {"kind": "the alumni mixer", "date": "next Thursday", "time": "6:30pm", "venue": "The Rooftop"},
    {"kind": "a birthday party", "date": "Saturday", "time": "noon", "venue": "Riverside Park"},
]

DEADLINE_TEMPLATES = [
    ("Re: Grant renewal — documents due {date}", "Reminder that the renewal packet, including the budget justification, is due before end of day {date}."),
    ("Your submission window closes {date}", "This is a reminder that the camera-ready deadline for accepted papers is {date}. Please upload your final PDF."),
    ("Action required: onboarding paperwork due {date}", "Please complete and return the onboarding forms by {date} so payroll can be set up on time."),
    ("Lease renewal — sign by {date}", "Please return the signed renewal by {date} so we can lock in the current rate before it expires."),
]
DEADLINE_FILL = [
    {"date": "this Friday"}, {"date": "next Monday"}, {"date": "the 30th"}, {"date": "end of week"},
]

NEWSLETTER_TEMPLATES = [
    ("This week in {topic}", "Top stories in {topic} this week: a roundup of what mattered and why."),
    ("{topic} digest — issue #{n}", "Your {topic} digest for this week. Curated links and short takes below."),
    ("The {topic} roundup", "Five things worth knowing in {topic} this week, summarized."),
    ("{topic} weekly", "Here's what happened in {topic} this week, in five minutes or less."),
]
NEWSLETTER_FILL = [
    {"topic": "AI research", "n": 42}, {"topic": "fintech", "n": 17},
    {"topic": "open source", "n": 88}, {"topic": "climate tech", "n": 9},
]

RECEIPT_TEMPLATES = [
    ("Your receipt for {item}", "Thanks for your purchase. {item} — total {amount}, charged to your card ending in 4412."),
    ("Payment confirmation — {item}", "We've received your payment of {amount} for {item}. This receipt is for your records."),
    ("Order confirmed: {item}", "Your order for {item} ({amount}) has been confirmed and will ship shortly."),
    ("Subscription renewed — {item}", "Your subscription to {item} renewed for {amount}. Manage your plan anytime."),
]
RECEIPT_FILL = [
    {"item": "cloud storage plan", "amount": "$9.99"}, {"item": "wireless keyboard", "amount": "$64.00"},
    {"item": "streaming subscription", "amount": "$12.99"}, {"item": "monitor stand", "amount": "$38.50"},
]


def _build_category(category: str, templates: list[tuple[str, str]], fills: list[dict]) -> list[dict]:
    senders = SENDERS[category]
    emails = []
    combos = list(itertools.product(templates, fills, senders))  # 4*4*3 = 48, take 12
    for i, ((subject_t, body_t), fill, sender) in enumerate(combos[:12], start=1):
        eid = f"{category}-{i:03d}"
        emails.append({
            "id": eid,
            "thread_id": f"t-{eid}",
            "subject": subject_t.format(**fill),
            "sender": sender,
            "date": "Mon, 1 Jan 2026",
            "body": body_t.format(**fill),
            "is_reply": False,
        })
    return emails


def _build_reply_pairs() -> list[dict]:
    emails = []
    for i in range(1, 7):
        orig_id, reply_id = f"thread-orig-{i:03d}", f"thread-reply-{i:03d}"
        emails.append({
            "id": orig_id, "thread_id": f"thread-{i:03d}",
            "subject": f"Project update #{i}",
            "sender": "manager@acme-labs.io", "date": "Mon, 1 Jan 2026",
            "body": f"Here's the status on project {i}: on track, no blockers, next check-in Friday.",
            "is_reply": False,
        })
        emails.append({
            "id": reply_id, "thread_id": f"thread-{i:03d}",
            "subject": f"Re: Project update #{i}",
            "sender": "you@acme-labs.io", "date": "Tue, 2 Jan 2026",
            "body": f"> Here's the status on project {i}: on track...\n\nThanks, sounds good.",
            "is_reply": True,
        })
    return emails


def _find_id(emails: list[dict], subject_contains: str) -> str:
    for e in emails:
        if subject_contains in e["subject"]:
            return e["id"]
    raise ValueError(f"no email found with subject containing {subject_contains!r}")


def build_corpus() -> list[dict]:
    emails = []
    emails += _build_category("events", EVENT_TEMPLATES, EVENT_FILL)
    emails += _build_category("deadlines", DEADLINE_TEMPLATES, DEADLINE_FILL)
    emails += _build_category("newsletters", NEWSLETTER_TEMPLATES, NEWSLETTER_FILL)
    emails += _build_category("receipts", RECEIPT_TEMPLATES, RECEIPT_FILL)
    emails += _build_reply_pairs()
    return emails


def build_queries(emails: list[dict]) -> list[dict]:
    by_category = {
        cat: [e["id"] for e in emails if e["id"].startswith(cat) and not e["is_reply"]]
        for cat in ("events", "deadlines", "newsletters", "receipts")
    }
    queries = [
        {"query": "an email inviting me to an event, party, or gathering with a specific date", "relevant_ids": by_category["events"]},
        {"query": "an email about an upcoming deadline, due date, or submission cutoff", "relevant_ids": by_category["deadlines"]},
        {"query": "a newsletter or weekly digest roundup", "relevant_ids": by_category["newsletters"]},
        {"query": "a receipt or payment confirmation for something I bought", "relevant_ids": by_category["receipts"]},
        {"query": "an email confirming a dentist or doctor appointment",
         "relevant_ids": [_find_id(emails, "dentist appointment")]},
        {"query": "an email about a lease or rental agreement renewal",
         "relevant_ids": [_find_id(emails, "Lease renewal")]},
        {"query": "an email about a grant or research funding deadline",
         "relevant_ids": [_find_id(emails, "Grant renewal")]},
        {"query": "a reminder about onboarding paperwork for a new job",
         "relevant_ids": [_find_id(emails, "onboarding paperwork")]},
        {"query": "an email about AI or machine learning research news",
         "relevant_ids": [e["id"] for e in emails if "AI research" in e["body"]]},
        {"query": "an email about a subscription renewal charge",
         "relevant_ids": [e["id"] for e in emails if "Subscription renewed" in e["subject"]]},
        {"query": "a birthday party invitation",
         "relevant_ids": [_find_id(emails, "birthday party")]},
        {"query": "a conference paper camera-ready submission reminder",
         "relevant_ids": [_find_id(emails, "submission window")]},
        {"query": "an email about a team dinner or social event",
         "relevant_ids": [_find_id(emails, "team dinner")]},
        {"query": "a purchase confirmation for office equipment",
         "relevant_ids": [_find_id(emails, "wireless keyboard")]},
        {"query": "an alumni network event invitation",
         "relevant_ids": [_find_id(emails, "alumni mixer")]},
        {"query": "a fintech industry news roundup",
         "relevant_ids": [e["id"] for e in emails if "fintech" in e["body"]]},
    ]
    return queries


if __name__ == "__main__":
    emails = build_corpus()
    queries = build_queries(emails)
    (OUT_DIR / "emails.json").write_text(json.dumps(emails, indent=2))
    (OUT_DIR / "queries.json").write_text(json.dumps(queries, indent=2))
    print(f"Wrote {len(emails)} emails and {len(queries)} queries to {OUT_DIR}")
```

Run it: `cd <repo> && python eval/synthetic_corpus/generate.py`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_corpus_schema.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add eval/synthetic_corpus/generate.py eval/synthetic_corpus/emails.json eval/synthetic_corpus/queries.json tests/test_corpus_schema.py
git commit -m "Add deterministic synthetic corpus + query ground truth generator"
```

---

### Task 5: eval/baseline_bm25.py — BM25 baseline retriever

**Files:**
- Create: `eval/baseline_bm25.py`
- Create: `eval/requirements.txt`
- Test: `tests/test_bm25.py`

**Interfaces:**
- Consumes: corpus format from Task 4 (`{id, subject, body, ...}`).
- Produces: `class BM25Retriever: __init__(self, documents: list[dict])`, `.search(query: str, k: int) -> list[str]` (ranked ids, best first). Consumed by Task 6.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bm25.py
from eval.baseline_bm25 import BM25Retriever


def test_bm25_ranks_keyword_matching_doc_first():
    docs = [
        {"id": "e1", "subject": "Dentist appointment", "body": "Your dentist appointment is confirmed for Tuesday."},
        {"id": "e2", "subject": "Weekly newsletter", "body": "Top stories in tech this week."},
    ]
    retriever = BM25Retriever(docs)
    ranked = retriever.search("dentist appointment confirmed", k=2)
    assert ranked[0] == "e1"
    assert set(ranked) == {"e1", "e2"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bm25.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rank_bm25'`

- [ ] **Step 3: Implement**

Create `eval/requirements.txt`:

```
pytest
rank-bm25
```

Install it: `pip install -r eval/requirements.txt`

Create `eval/baseline_bm25.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bm25.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add eval/baseline_bm25.py eval/requirements.txt tests/test_bm25.py
git commit -m "Add BM25 baseline retriever for the retrieval benchmark"
```

---

### Task 6: eval/benchmark.py — synthetic benchmark run

**Files:**
- Create: `eval/benchmark.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `pipeline.ingest` (Task 1), `query.search_collection` (Task 2), `eval.metrics.*` (Task 3), `eval.baseline_bm25.BM25Retriever` (Task 5), corpus files (Task 4).
- Produces: `run_synthetic_benchmark(k: int = 5) -> dict` — writes `eval/results/synthetic_benchmark.json` and `eval/results/synthetic_benchmark.md`, returns the same dict it wrote. Dict shape: `{"k": int, "semantic": {"precision": float, "recall": float, "mrr": float, "per_query": [...]}, "bm25": {...same shape...}}`. Each `per_query` entry: `{"query": str, "relevant_ids": list[str], "ranked_ids": list[str], "precision_at_k": float, "recall_at_k": float, "reciprocal_rank": float}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmark.py
import json
from pathlib import Path
from eval.benchmark import run_synthetic_benchmark


def test_synthetic_benchmark_runs_and_writes_results():
    result = run_synthetic_benchmark(k=5)

    assert "semantic" in result and "bm25" in result
    for method in ("semantic", "bm25"):
        assert 0.0 <= result[method]["precision"] <= 1.0
        assert 0.0 <= result[method]["recall"] <= 1.0
        assert 0.0 <= result[method]["mrr"] <= 1.0
        assert len(result[method]["per_query"]) == len(result["bm25"]["per_query"])

    written = json.loads(Path("eval/results/synthetic_benchmark.json").read_text())
    assert written == result
    assert Path("eval/results/synthetic_benchmark.md").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.benchmark'`

- [ ] **Step 3: Implement**

Create `eval/benchmark.py`:

```python
"""Retrieval benchmark: semantic pipeline vs. BM25, over the synthetic corpus."""
import json
from pathlib import Path

import chromadb

from config import CHROMA_COLLECTION, EMBED_MODEL
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


if __name__ == "__main__":
    result = run_synthetic_benchmark()
    print(json.dumps({k: {m: v for m, v in val.items() if m != "per_query"} if k in ("semantic", "bm25") else val
                       for k, val in result.items()}, indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_benchmark.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add eval/benchmark.py eval/results/synthetic_benchmark.json eval/results/synthetic_benchmark.md tests/test_benchmark.py
git commit -m "Add synthetic retrieval benchmark: semantic pipeline vs. BM25"
```

---

### Task 7: Real-inbox benchmark path (manual-run, private)

**Files:**
- Modify: `eval/benchmark.py` (add `run_real_benchmark`)
- Modify: `.gitignore` (add `eval/real_queries.local.json`)

**Interfaces:**
- Produces: `run_real_benchmark(queries_path: str, k: int = 5) -> dict` — writes only `eval/results/real_benchmark_summary.json` (aggregate numbers, no content). This task adds capability; the user runs it themselves against their live inbox, since no automated test can access real Gmail credentials or fabricate real relevance labels.

- [ ] **Step 1: Add the real-benchmark function**

Append to `eval/benchmark.py`:

```python
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
```

Add the missing import at the top of `eval/benchmark.py`: `from config import CHROMA_HOST, CHROMA_PORT` (extend the existing `from config import ...` line rather than adding a second one).

- [ ] **Step 2: Gitignore the private queries file**

Add to `.gitignore`:

```
eval/real_queries.local.json
```

- [ ] **Step 3: Manual run (you, not the implementer, do this step)**

1. Run `pipeline.py` against your real inbox if you haven't already.
2. Run `query.py` with 20-30 varied queries, note which returned results are actually relevant, and hand-write `eval/real_queries.local.json`: `[{"query": "...", "relevant_ids": ["<email_id>", ...]}, ...]` (get ids from `search_collection`'s new `"id"` field).
3. Run: `python -c "from eval.benchmark import run_real_benchmark; print(run_real_benchmark('eval/real_queries.local.json'))"`
4. Confirm `eval/results/real_benchmark_summary.json` contains only numbers — no subjects, senders, or snippets — before committing it.

- [ ] **Step 4: Commit the capability (not the private queries file)**

```bash
git add eval/benchmark.py .gitignore
git commit -m "Add real-inbox benchmark path (aggregate-only output)"
```

(Commit `eval/results/real_benchmark_summary.json` separately, yourself, once you've generated it and confirmed it's content-free.)

---

### Task 8: eval/latency.py — ingestion + query latency

**Files:**
- Create: `eval/latency.py`
- Test: `tests/test_latency.py`

**Interfaces:**
- Consumes: `pipeline.ingest`, corpus files (Task 4).
- Produces: `measure_ingestion(n_values: list[int]) -> dict`, `measure_query_latency(collection, queries: list[str], repeats: int = 3) -> dict`, `run_latency_report() -> dict` — writes `eval/results/latency.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_latency.py
from eval.latency import run_latency_report


def test_latency_report_has_expected_shape():
    report = run_latency_report(n_values=[5, 10], repeats=2)

    assert report["cost_usd_marginal"] == 0.0
    for entry in report["ingestion"]:
        assert entry["n"] in (5, 10)
        assert entry["emails_per_sec"] > 0
    assert "p50_ms" in report["query_latency"]
    assert "p95_ms" in report["query_latency"]
    assert report["query_latency"]["p50_ms"] <= report["query_latency"]["p95_ms"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_latency.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.latency'`

- [ ] **Step 3: Implement**

```python
# eval/latency.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_latency.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add eval/latency.py eval/results/latency.json tests/test_latency.py
git commit -m "Add ingestion and query latency measurement"
```

---

### Task 9: Security review — injection test + writeup

**Files:**
- Create: `tests/test_security_injection.py`
- Create: `eval/security_review.md`

**Interfaces:**
- Consumes: `pipeline._clean`, `pipeline._to_document`, `query.search_collection` (via a real ingested collection).
- Produces: `eval/security_review.md`, documenting three findings.

- [ ] **Step 1: Write the injection test**

```python
# tests/test_security_injection.py
import chromadb
from config import CHROMA_COLLECTION, EMBED_MODEL, SCOPES
from embeddings import BGEEmbeddings
from gmail_client import Email
from pipeline import ingest
from query import search_collection


def test_oauth_scope_is_read_only():
    assert SCOPES == ["https://www.googleapis.com/auth/gmail.readonly"]


def test_html_payload_in_email_body_survives_unescaped_into_snippet():
    """
    Demonstrates (does not exploit) that a malicious email body reaches the
    UI-facing snippet with no sanitization step in the pipeline. app.py
    renders this snippet directly via gr.Markdown.
    """
    payload = '<img src=x onerror="alert(1)"> click <a href="javascript:alert(2)">here</a>'
    malicious = Email(
        id="mal-1", thread_id="t-mal-1", subject="Important: verify your account",
        sender="attacker@example.com", date="Mon, 1 Jan 2026",
        body=f"Please verify your account. {payload}", is_reply=False,
    )
    client = chromadb.EphemeralClient()
    ingest([malicious], client=client)
    collection = client.get_collection(name=CHROMA_COLLECTION, embedding_function=BGEEmbeddings(EMBED_MODEL))

    results = search_collection(collection, "verify your account", n_results=1)

    assert payload in results[0]["snippet"], (
        "payload was altered/stripped — re-check this finding, the pipeline may sanitize after all"
    )
```

- [ ] **Step 2: Run test to verify it passes as-is (this test documents current behavior, it isn't TDD-red/green)**

Run: `pytest tests/test_security_injection.py -v`
Expected: PASS — confirms the payload survives unsanitized. If it fails, the finding in Step 3 is wrong; re-inspect `pipeline._clean` and `query.search_collection` before writing it up.

- [ ] **Step 3: Write the security review**

Create `eval/security_review.md`:

```markdown
# Security review

Scope: the Gmail RAG pipeline at commit `d041aa1`. Three concrete checks, not a generic checklist.

## 1. OAuth scope

`config.py:8` sets `SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]`.
Read-only, confirmed by `tests/test_security_injection.py::test_oauth_scope_is_read_only`.
The pipeline cannot modify, send, or delete anything in the connected account.

## 2. At-rest storage

`pipeline.py`'s `ingest()` upserts full cleaned email text into ChromaDB, which persists to
`./chroma-data` on local disk with no encryption layer.
This is an accepted tradeoff for a local-only, single-user tool — the alternative (encrypting
at rest) would need a key-management story this project has no user-facing need for yet.
Worth stating plainly rather than leaving implicit: anyone with filesystem access to the
machine running this tool can read the indexed email content directly from `chroma-data`.

## 3. Output injection via unsanitized snippets

`app.py` renders retrieved email snippets directly into `gr.Markdown`.
`query.search_collection()` truncates and returns the raw document text as `"snippet"` with no
HTML/Markdown escaping at any stage of the pipeline.
`tests/test_security_injection.py::test_html_payload_in_email_body_survives_unescaped_into_snippet`
confirms an `<img onerror=...>` / `javascript:` payload embedded in an email body survives intact
into the value that reaches `gr.Markdown`.

This is the applicable security surface at this commit — there is no LLM generation step, so
classic prompt-injection-into-generation doesn't apply here. The risk is UI-level: if Gradio's
Markdown renderer executes injected HTML/JS (not verified here — this test stops at confirming
the payload reaches the render boundary unsanitized, not at confirming exploitation in a live
browser), a crafted email could affect the local UI session. Given this is a local, single-user
tool, blast radius is limited to the operator's own machine — but it's a real, documented gap,
not a theoretical one. Recommended follow-up (out of scope for this evidence pack): escape
snippet content before rendering, or switch the result display from `gr.Markdown` to a plain
`gr.Textbox`.
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_security_injection.py eval/security_review.md
git commit -m "Add security review: OAuth scope, at-rest storage, output-injection check"
```

---

### Task 10: Failure analysis writeup

**Files:**
- Create: `eval/failures.md`

**Interfaces:**
- Consumes: `eval/results/synthetic_benchmark.json` (Task 6), `eval.metrics.lowest_scoring` (Task 3).

- [ ] **Step 1: Pull the worst-performing queries**

Run:

```bash
python3 -c "
import json
from eval.metrics import lowest_scoring
data = json.loads(open('eval/results/synthetic_benchmark.json').read())
for method in ('semantic', 'bm25'):
    print(f'--- {method} ---')
    for r in lowest_scoring(data[method]['per_query'], 'recall_at_k', 5):
        print(r['query'], '->', r['recall_at_k'], r['ranked_ids'], 'expected', r['relevant_ids'])
"
```

- [ ] **Step 2: Write eval/failures.md from the actual output**

Create `eval/failures.md` with this structure, filled in with the real query text, real ranked ids, and real recall/precision values printed by Step 1 (do not invent example queries — use the ones the benchmark actually produced):

```markdown
# Failure analysis

Source: `eval/results/synthetic_benchmark.json`, the 5 lowest-recall queries per method.

## Semantic pipeline

For each of the worst-performing queries below: the query text, what came back, what should
have come back, and why.

<!-- One entry per query from Step 1's semantic output, e.g.: -->
### "<real query text>"

- Recall@k: <real value>
- Returned: <real ranked_ids>
- Expected: <real relevant_ids>
- Root cause: <your actual read of why — e.g. whole-email embedding diluting a query aimed at
  one sub-topic inside a longer email, category overlap between two templates, a query phrased
  more abstractly than any email's actual language>

<!-- repeat for the remaining worst semantic queries -->

## BM25 baseline

<!-- same structure, for BM25's worst queries -->

## Pattern

Summarize in 2-3 sentences whether the semantic pipeline's failures cluster around a specific
cause (most likely: the whole-email-embedding tradeoff noted in the design spec — a long email
compresses to one vector, so a query aimed at a sub-topic inside it underperforms). State
plainly if the data doesn't support that theory instead — don't force the expected finding onto
the actual results.
```

- [ ] **Step 3: Commit**

```bash
git add eval/failures.md
git commit -m "Add failure analysis from synthetic benchmark's lowest-recall queries"
```

---

### Task 11: Onboarding fix — setup.sh + README.md mismatch

**Files:**
- Modify: `setup.sh:13-32`
- Modify: `README.md` (the `N_EMAILS` row in the configuration table)

**Interfaces:** none (shell script + doc fix).

- [ ] **Step 1: Uncomment and fix the venv/install steps in setup.sh**

Replace the commented block at `setup.sh:13-32`:

```bash
# ── 1. Python version ────────────────────────────────────────────────────────
info "Checking Python version..."
PYTHON=$(command -v python3.11 || command -v python3 || die "Python 3 not found")
PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
[[ "${PY_VER}" < "3.11" ]] && die "Python 3.11+ required (found $PY_VER)"
info "Using Python $PY_VER at $PYTHON"

# ── 2. Virtual environment ───────────────────────────────────────────────────
if [[ ! -d ".venv" ]]; then
    info "Creating virtual environment..."
    "$PYTHON" -m venv .venv
fi
source .venv/bin/activate
info "Virtual environment active"

# ── 3. Install dependencies ──────────────────────────────────────────────────
info "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
info "Dependencies installed"
```

- [ ] **Step 2: Verify the script runs clean**

Run: `cd <repo> && bash -n setup.sh` (syntax check)
Expected: no output (clean parse)

Then run the real thing in a scratch copy to confirm it doesn't clobber the working `.venv`:

```bash
cp -r <repo> /tmp/gmail-rag-setup-check
cd /tmp/gmail-rag-setup-check && rm -rf .venv chroma-data chroma.pid chroma.log
bash setup.sh
test -d .venv && echo "venv created: OK"
source .venv/bin/activate && python -c "import chromadb, sentence_transformers, gradio" && echo "deps importable: OK"
```

Expected: both `OK` lines print.

- [ ] **Step 3: Fix the README.md N_EMAILS mismatch**

In `README.md`'s configuration table, change the `N_EMAILS` row's default from `50` to `500` to match `config.py:19` — the code's default is the source of truth here since changing runtime behavior silently would be a bigger, unrelated change than fixing a doc.

- [ ] **Step 4: Commit**

```bash
git add setup.sh README.md
git commit -m "Fix setup.sh: actually create venv and install deps; fix N_EMAILS doc mismatch"
```

---

### Task 12: demo.sh + run.sh auto-ingest

**Files:**
- Create: `eval/ingest_synthetic.py`
- Create: `demo.sh`
- Modify: `run.sh`

**Interfaces:**
- Consumes: `pipeline.ingest` (Task 1), corpus files (Task 4).
- Produces: a one-command, zero-Gmail-credential try-it path.

- [ ] **Step 1: Write the synthetic-ingest script**

```python
# eval/ingest_synthetic.py
"""Ingest the synthetic corpus into the real (persistent) ChromaDB collection —
used by demo.sh so a reviewer can try the actual app.py/query.py UI with no
Gmail account, using the same ingest() code path as the real pipeline."""
import json
from pathlib import Path

from gmail_client import Email
from pipeline import ingest

CORPUS_PATH = Path(__file__).parent / "synthetic_corpus" / "emails.json"

if __name__ == "__main__":
    emails = [Email(**e) for e in json.loads(CORPUS_PATH.read_text())]
    result = ingest(emails)
    print(f"Demo corpus ingested: {result}")
```

- [ ] **Step 2: Write demo.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

GREEN='\033[0;32m'; NC='\033[0m'
info() { echo -e "${GREEN}[demo]${NC} $*"; }

if [[ ! -d ".venv" ]]; then
    info "First run — setting up venv and dependencies..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
    pip install --quiet -r eval/requirements.txt
else
    info "venv ready, dependencies cached"
    source .venv/bin/activate
fi

CHROMA_PORT="${CHROMA_PORT:-8000}"
CHROMA_DATA_DIR="${CHROMA_DATA_DIR:-./chroma-data}"
if nc -z localhost "${CHROMA_PORT}" 2>/dev/null; then
    info "ChromaDB already running on port ${CHROMA_PORT}"
else
    info "Starting ChromaDB..."
    mkdir -p "$CHROMA_DATA_DIR"
    nohup chroma run --path "$CHROMA_DATA_DIR" --host 0.0.0.0 --port "$CHROMA_PORT" > chroma.log 2>&1 &
    echo $! > chroma.pid
    for i in $(seq 1 60); do
        nc -z localhost "${CHROMA_PORT}" 2>/dev/null && break
        sleep 1
    done
fi

info "Loading synthetic inbox (no Gmail account needed)..."
python eval/ingest_synthetic.py

info "Launching UI → http://127.0.0.1:7860"
python app.py
```

Make it executable: `chmod +x demo.sh`

- [ ] **Step 3: Add auto-ingest-if-empty to run.sh**

In `run.sh`, insert this block between the "ChromaDB" section and the "Launch UI" section (after ChromaDB is confirmed running, before `python app.py`):

```bash
# ── 2.5. Auto-ingest if the collection is empty ─────────────────────────────
COUNT=$(python3 -c "
import chromadb
from config import CHROMA_COLLECTION, CHROMA_HOST, CHROMA_PORT, EMBED_MODEL
from embeddings import BGEEmbeddings
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
try:
    print(client.get_collection(name=CHROMA_COLLECTION, embedding_function=BGEEmbeddings(EMBED_MODEL)).count())
except Exception:
    print(0)
")
if [[ "$COUNT" -eq 0 ]]; then
    info "No existing collection — ingesting inbox (this can take a while on first run)..."
    python pipeline.py
else
    info "Existing collection found (${COUNT} emails) — skipping ingest"
fi
```

- [ ] **Step 4: Verify demo.sh works end-to-end**

Run: `cd <repo> && ./demo.sh &` then, once "Launching UI" prints, `curl -s http://127.0.0.1:7860 | grep -qi "gradio" && echo "UI reachable: OK"`, then stop it: `kill %1`.
Expected: `UI reachable: OK` prints, no errors in `chroma.log`.

- [ ] **Step 5: Commit**

```bash
git add eval/ingest_synthetic.py demo.sh run.sh
git commit -m "Add demo.sh (zero-setup try-it path) and auto-ingest to run.sh"
```

---

### Task 13: eval/README.md + portfolio rewrite

**Files:**
- Create: `eval/README.md`
- Modify (separate repo): `<personal-site>/_portfolio/2-emailai.md`

**Interfaces:** none new — this task only reads and writes prose from prior tasks' committed outputs.

- [ ] **Step 1: Write eval/README.md**

Create `eval/README.md` tying the four evidence items together: a short intro, then a table of contents linking `results/synthetic_benchmark.md`, `results/latency.json`, `failures.md`, `security_review.md`, and a "try it" section pointing at `../demo.sh` (no Gmail needed) and `../setup.sh && ../run.sh` (your own inbox). Pull the actual headline numbers (precision/recall/mrr, p50/p95 latency) from the committed JSON files into this README rather than restating vague claims.

- [ ] **Step 2: Commit eval/README.md**

```bash
git add eval/README.md
git commit -m "Add eval/README.md tying the evidence pack together"
```

- [ ] **Step 3: Rewrite the portfolio page**

In the `personal-site` repo, rewrite `_portfolio/2-emailai.md` following the same structure as `_portfolio/1-mslesion.md` (MSEval) and `_portfolio/10-mia-quantized-llms.md` (MIA): problem → what was built → what it shows → key design decisions.

Required content, sourced only from committed files (no invented numbers):
- **Architecture**, corrected: whole-email embedding via `pipeline.ingest()`, no chunking, no LangChain `Runnable` pipeline, no LLM generation step — reading directly from `pipeline.py`, `gmail_client.py`, `embeddings.py` at the current commit, not the old chunking/LangChain description.
- **What it shows**: the precision@k/recall@k/MRR table from `eval/results/synthetic_benchmark.md`, semantic vs. BM25.
- **Latency & cost**: the ingestion throughput and p50/p95 query latency from `eval/results/latency.json`, and the explicit $0 marginal cost.
- **Failures**: 2-3 concrete cases from `eval/failures.md`, stated honestly, not softened.
- **Security**: the three findings from `eval/security_review.md`, stated plainly (including the unresolved output-injection gap).
- **Try it**: the `demo.sh` zero-setup path, one line.
- Keep the existing "What this became" section (the link to Agentic Process Discovery) — that lineage claim is accurate and doesn't need correcting.

- [ ] **Step 4: Build and review before touching git**

Run: `cd <personal-site> && bundle exec jekyll build`
Expected: build succeeds with no errors referencing `2-emailai.md`.

Per this repo's own CLAUDE.md: do not commit or push. Show the user the diff and wait for their review before either happens.

---

## Self-Review Notes

- **Spec coverage:** A (Tasks 4-7), B (Task 8), C (Task 10), D (Task 9), E (Tasks 11-12), F (Task 13) — all six spec items map to at least one task.
- **Type consistency checked:** `ingest(emails, client=None)` (Task 1) is the exact signature used in Tasks 6, 7, 8, 9, 12. `search_collection(collection, text, n_results)` (Task 2) is the exact signature used in Tasks 6, 7, 8. The `"id"` key added to `search_collection`'s results in Task 2 is what Task 6's `semantic_rank` lambda reads.
- **No placeholders:** every code step above is complete, runnable code; Task 13's portfolio rewrite is prose sourced from data that doesn't exist until Tasks 6-9 run, which is unavoidable for a data-dependent writing task — it's given a concrete required structure and an explicit no-invention constraint instead.
