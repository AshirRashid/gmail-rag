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
        "payload was altered/stripped - re-check this finding, the pipeline may sanitize after all"
    )
