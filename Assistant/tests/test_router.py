"""Tests for Router Agent."""
import pytest
from app.agents.router import RouterAgent


@pytest.fixture
def router():
    """Create router agent instance."""
    return RouterAgent()


def test_parse_simple_query(router):
    """Test parsing a simple query."""
    # Skip if OpenAI API key not configured
    pytest.importorskip("openai")
    
    query = "What was ROAS last week for SB campaigns?"
    task_spec = router.parse(query)
    
    assert task_spec.intent.value == "analytics.query"
    assert "roas" in [m.lower() for m in task_spec.metrics]
    assert task_spec.time.tz == "Asia/Kolkata"


def test_extract_time_range(router):
    """Test time range extraction."""
    range_str = "last_week"
    start, end = router._resolve_relative_range(range_str)
    
    assert start is not None
    assert end is not None
    assert end >= start

