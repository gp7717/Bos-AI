from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Agents.orchestrator.context_index import AgentContextRetriever


@pytest.fixture(scope="module")
def retriever() -> AgentContextRetriever:
    return AgentContextRetriever.from_default_sources()


def test_retriever_has_expected_agents(retriever: AgentContextRetriever) -> None:
    agents = set(retriever.available_agents())
    for expected in {"docs", "sql", "computation"}:
        assert expected in agents


def test_docs_retrieval_matches_net_profit(retriever: AgentContextRetriever) -> None:
    results = retriever.retrieve_for_agent("docs", "net profit endpoint", top_k=3)
    assert results, "Expected at least one API docs snippet"
    titles = {result.document.title.lower() for result in results}
    assert any("net profit" in title for title in titles)


def test_sql_retrieval_mentions_orders(retriever: AgentContextRetriever) -> None:
    results = retriever.retrieve_for_agent("sql", "shopify order revenue", top_k=5)
    assert results, "Expected SQL context for Shopify order queries"
    ids = {result.document.document_id for result in results}
    assert any("shopify" in identifier for identifier in ids)


def test_computation_retrieval_for_ratios(retriever: AgentContextRetriever) -> None:
    results = retriever.retrieve_for_agent("computation", "calculate growth ratio", top_k=2)
    assert results, "Computation agent should provide capability context"
    assert all(result.score > 0 for result in results)

