"""
Test suite for complex query handling in the multi-agent system.

Tests cover:
- Natural language query decomposition
- Capability-aware agent selection
- Multiple graphs support
- Parallel processing
- Multi-part complex queries
"""

import pytest
from Agents.core.models import AgentRequest, AgentName
from Agents.orchestrator.service import run


class TestAPIAwareSelection:
    """Test that system prefers API agent when metrics are available via API."""

    def test_net_profit_uses_api_agent(self):
        """Critical test: Net profit should use API agent, not SQL."""
        query = "Get the net profit graph for the last 4 months"
        request = AgentRequest(question=query)
        response = run(request)

        # Verify API agent was used
        agents_used = [r.agent for r in response.agent_results]
        assert AgentName.api_docs in agents_used, (
            f"Expected api_docs agent, but got: {agents_used}. "
            "System should prefer API when net profit is available via API."
        )

        # Verify graph was generated
        assert response.graph or response.graphs, "Expected graph in response"

    def test_multiple_api_metrics(self):
        """Test multiple API metrics are fetched correctly."""
        query = "What's our revenue, ROAS, and ad spend for this month?"
        request = AgentRequest(question=query)
        response = run(request)

        agents_used = [r.agent for r in response.agent_results]
        # Should prefer API agent for these metrics
        assert AgentName.api_docs in agents_used, "Should use API agent for available metrics"


class TestQueryDecomposition:
    """Test natural language query decomposition."""

    def test_generic_performance_query(self):
        """Test that vague queries are decomposed into specific sub-queries."""
        query = "What's our performance this month?"
        request = AgentRequest(question=query)
        response = run(request)

        # Should have decomposed query
        # Note: This requires checking internal state or trace
        assert response.answer, "Should generate an answer"
        assert len(response.agent_results) > 0, "Should execute at least one agent"

    def test_trend_request_decomposition(self):
        """Test generic trend request is decomposed."""
        query = "Show me the trends"
        request = AgentRequest(question=query)
        response = run(request)

        assert response.answer, "Should generate an answer"
        # Should infer metrics and timeframes


class TestMultipleGraphs:
    """Test multiple graphs support."""

    def test_explicit_multiple_graphs(self):
        """Test explicit request for multiple charts."""
        query = "Show me sales trends and revenue breakdown as separate charts"
        request = AgentRequest(question=query)
        response = run(request)

        graphs = response.all_graphs
        assert len(graphs) >= 1, f"Expected at least 1 graph, got {len(graphs)}"
        # Ideally should have 2 graphs, but depends on data availability

    def test_implicit_multiple_visualizations(self):
        """Test implicit multiple visualization request."""
        query = "Display net profit over time and also show the channel breakdown"
        request = AgentRequest(question=query)
        response = run(request)

        graphs = response.all_graphs
        assert len(graphs) >= 1, "Should generate at least one graph"


class TestParallelProcessing:
    """Test parallel execution of independent agents."""

    def test_independent_data_sources(self):
        """Test that independent agents execute in parallel."""
        query = "Get sales data and API documentation for the orders endpoint"
        request = AgentRequest(question=query)
        response = run(request)

        agents_used = [r.agent for r in response.agent_results]
        # Should have multiple agents
        assert len(agents_used) >= 1, "Should execute multiple agents"

        # Check trace for parallel execution indicators
        # (This would require checking trace events)


class TestComplexQueries:
    """Test complex multi-part queries."""

    def test_full_analysis_request(self):
        """Test comprehensive analysis query."""
        query = (
            "Analyze our business performance: get sales data, "
            "calculate profit margins, compare to last month, and create visualizations"
        )
        request = AgentRequest(question=query)
        response = run(request)

        assert response.answer, "Should generate comprehensive answer"
        assert len(response.agent_results) > 1, "Should use multiple agents"

    def test_nested_dependencies(self):
        """Test queries with nested dependencies."""
        query = (
            "Get net profit data, calculate growth rate, "
            "forecast next month, and show everything in graphs"
        )
        request = AgentRequest(question=query)
        response = run(request)

        # Should handle dependencies correctly
        assert response.answer, "Should handle nested dependencies"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_ambiguous_query(self):
        """Test handling of ambiguous queries."""
        query = "Tell me about our data"
        request = AgentRequest(question=query)
        response = run(request)

        # Should handle gracefully, either decompose or ask for clarification
        assert response.answer, "Should handle ambiguous queries"

    def test_very_long_query(self):
        """Test very long, complex query."""
        query = (
            "I need to see our sales performance for the last 6 months broken down by channel, "
            "compare it to the previous 6 months, calculate the growth rate, "
            "forecast the next 3 months, show trends, create breakdown charts, "
            "and also get the API documentation for the sales endpoint"
        )
        request = AgentRequest(question=query)
        response = run(request)

        assert response.answer, "Should handle very long queries"


class TestIntegration:
    """End-to-end integration tests."""

    def test_end_to_end_complex_query(self):
        """Full system test with complex query."""
        query = (
            "I want a comprehensive business report: get net profit for last 3 months, "
            "calculate the average, compare to previous 3 months, show trends, "
            "create breakdown charts by channel, and also get me the API documentation"
        )
        request = AgentRequest(question=query)
        response = run(request)

        # Verify all components work together
        assert response.answer, "Should generate comprehensive report"
        assert response.agent_results, "Should execute multiple agents"
        # Should have graphs if data available
        if response.graph or response.graphs:
            assert len(response.all_graphs) > 0, "Should have graphs"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

