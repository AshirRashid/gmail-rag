"""Shared pytest fixtures.

ChromaDB 1.0.0's in-memory clients (EphemeralClient) share a single
process-wide system cache keyed by settings. Two tests that each call
chromadb.EphemeralClient() therefore end up talking to the same underlying
system, so a collection created by one test (e.g. the 54-email synthetic
corpus ingested by test_benchmark / test_latency) leaks into the next test's
supposedly-fresh client. That makes assertions like collection.count() == 1
fail with the leaked count instead.

Clearing the shared system cache around every test gives each test a genuinely
isolated in-memory ChromaDB, independent of test execution order.
"""
import pytest
from chromadb.api.shared_system_client import SharedSystemClient


@pytest.fixture(autouse=True)
def _isolate_chromadb_state():
    SharedSystemClient.clear_system_cache()
    yield
    SharedSystemClient.clear_system_cache()
