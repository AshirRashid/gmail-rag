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
