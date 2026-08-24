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
