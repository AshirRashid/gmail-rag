from eval.latency import run_latency_report


def test_latency_report_has_expected_shape(tmp_path):
    report = run_latency_report(
        n_values=[5, 10], repeats=2, output_path=tmp_path / "latency.json"
    )

    assert report["cost_usd_marginal"] == 0.0
    for entry in report["ingestion"]:
        assert entry["n"] in (5, 10)
        assert entry["emails_per_sec"] > 0
    assert "p50_ms" in report["query_latency"]
    assert "p95_ms" in report["query_latency"]
    assert report["query_latency"]["p50_ms"] <= report["query_latency"]["p95_ms"]
