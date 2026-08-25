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
