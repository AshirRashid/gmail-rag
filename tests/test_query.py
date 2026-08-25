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
