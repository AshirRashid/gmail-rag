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
