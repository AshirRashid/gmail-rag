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
