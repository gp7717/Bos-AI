from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from langchain_core.messages import AIMessage

httpx = pytest.importorskip("httpx")

from Agents.ApiDocsAgent.agent import ApiDocsAgent
from Agents.core.models import AgentExecutionStatus, TraceEventType


class _QueuedLLM:
    """LLM stub that returns queued responses for successive invocations."""

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.invocations = 0
        self.history = []

    def invoke(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        if not self._responses:
            raise AssertionError("LLM stub exhausted responses")
        self.invocations += 1
        self.history.append(messages)
        return AIMessage(content=self._responses.pop(0))

    __call__ = invoke


def _write_context(tmp_path: Path) -> Path:
    payload = {
        "sections": [
            {
                "id": "health",
                "title": "Server Health",
                "source": "Docs/API Docs/API_DOCUMENTATION_WITH_LOGIC.md#health",
                "content": "- GET `/health`: Health check endpoint.",
            }
        ]
    }
    path = tmp_path / "api_context.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_api_agent_executes_http_call(tmp_path: Path) -> None:
    context_path = _write_context(tmp_path)

    plan_payload = json.dumps(
        {
            "make_request": True,
            "method": "GET",
            "path": "/health",
            "path_params": {},
            "query_params": {},
            "json_body": None,
            "headers": {},
            "reason": "Verify the service health endpoint.",
        }
    )
    summary_response = "Health endpoint returned a healthy status."
    llm = _QueuedLLM(plan_payload, summary_response)

    def _mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "healthy", "database": "connected"})

    transport = httpx.MockTransport(_mock_handler)
    client = httpx.Client(transport=transport)
    try:
        agent = ApiDocsAgent(llm=llm, context_path=context_path, http_client=client, top_k=1)
        result = agent.invoke("Call the GET /health endpoint")
    finally:
        client.close()

    assert result.status == AgentExecutionStatus.succeeded
    assert "healthy" in (result.answer or "").lower()
    assert llm.invocations == 2
    assert llm.history and "Server Health" in llm.history[0][-1].content
    assert any(event.event_type is TraceEventType.TOOL for event in result.trace)


def test_api_agent_retries_with_inferred_time_range(tmp_path: Path, monkeypatch) -> None:
    context_path = _write_context(tmp_path)

    plan_payload = json.dumps(
        {
            "make_request": True,
            "method": "GET",
            "path": "/api/product_spend_and_sales_by_title",
            "path_params": {},
            "query_params": {},
            "json_body": None,
            "headers": {},
            "reason": "Fetch product spend and sales summary.",
        }
    )
    summary_response = "Product spend was retrieved successfully."
    llm = _QueuedLLM(plan_payload, summary_response)

    monkeypatch.setenv("API_AGENT_TIMEZONE", "UTC")
    monkeypatch.setattr(
        "Agents.ApiDocsAgent.agent.ApiDocsAgent._derive_default_hour_range",
        lambda self, context: ("2025-11-09 00", "2025-11-09 23"),
    )

    calls = {"count": 0}

    def _mock_handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            assert "startDateTime" not in request.url.params
            assert "endDateTime" not in request.url.params
            return httpx.Response(
                400,
                json={
                    "error": "startDateTime and endDateTime query params are required, format: YYYY-MM-DD HH"
                },
            )

        assert request.url.params.get("startDateTime") == "2025-11-09 00"
        assert request.url.params.get("endDateTime") == "2025-11-09 23"
        return httpx.Response(200, json={"success": True, "data": []})

    transport = httpx.MockTransport(_mock_handler)
    client = httpx.Client(transport=transport)
    try:
        agent = ApiDocsAgent(llm=llm, context_path=context_path, http_client=client, top_k=1)
        result = agent.invoke("Fetch product spend summary for today using the API")
    finally:
        client.close()

    assert calls["count"] == 2
    assert result.status == AgentExecutionStatus.succeeded
    assert llm.invocations == 2
    assert any(
        event.message
        and "Auto-filled startDateTime and endDateTime" in event.message
        for event in result.trace
    )


def test_api_agent_reports_planning_decline(tmp_path: Path) -> None:
    context_path = _write_context(tmp_path)
    plan_payload = json.dumps(
        {
            "make_request": False,
            "method": "GET",
            "path": "/health",
            "path_params": {},
            "query_params": {},
            "json_body": None,
            "headers": {},
            "reason": "Health check requires authentication token.",
            "failure_reason": "Missing authentication token in context.",
        }
    )
    llm = _QueuedLLM(plan_payload)

    agent = ApiDocsAgent(llm=llm, context_path=context_path, top_k=1)
    result = agent.invoke("Call GET /health with proper auth")

    assert result.status == AgentExecutionStatus.failed
    assert result.error is not None
    assert "authentication" in result.error.message.lower()
    assert llm.invocations == 1
    assert llm.history and "Server Health" in llm.history[0][-1].content
    assert any(event.event_type is TraceEventType.ERROR for event in result.trace)
