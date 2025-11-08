from __future__ import annotations

import yaml
from langchain_core.messages import AIMessage

from Agents.ApiDocsAgent.agent import ApiDocsAgent
from Agents.core.models import AgentExecutionStatus


class _StubLLM:
    """Minimal stub that mimics the required LangChain LLM interface."""

    def __init__(self, response: str) -> None:
        self._response = response

    def invoke(self, messages, config=None):  # noqa: D401 - signature matches LangChain
        return AIMessage(content=self._response)


def test_api_docs_agent_returns_answer(tmp_path):
    context_payload = {
        "sections": [
            {
                "id": "health",
                "title": "Server Health",
                "source": "Docs/API Docs/API_DOCUMENTATION_WITH_LOGIC.md#health",
                "content": "GET `/health` checks database connectivity.",
            },
            {
                "id": "metrics",
                "title": "Server Metrics",
                "source": "Docs/API Docs/API_DOCUMENTATION_WITH_LOGIC.md#metrics",
                "content": "GET `/metrics` returns uptime and memory usage.",
            },
        ]
    }
    context_path = tmp_path / "context.yaml"
    context_path.write_text(yaml.safe_dump(context_payload), encoding="utf-8")

    agent = ApiDocsAgent(
        llm=_StubLLM("The `/health` endpoint verifies DB connectivity."),
        context_path=context_path,
        top_k=2,
    )

    result = agent.invoke("What does the GET /health endpoint do?")

    assert result.status == AgentExecutionStatus.succeeded
    assert "DB connectivity" in (result.answer or "")
    assert any(event.data.get("matches") for event in result.trace if event.event_type == "message")
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Agents.ApiDocsAgent import ApiDocsAgent
from Agents.core.models import AgentExecutionStatus


class _FakeLLM:
    """Minimal stub that mimics ChatModel.invoke behaviour for testing."""

    def __init__(self) -> None:
        self.invocations = 0
        self.messages = None

    def invoke(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.invocations += 1
        self.messages = messages

        # Extract user message content for deterministic answer
        try:
            user_message = messages[-1].content
        except AttributeError:  # pragma: no cover - defensive
            user_message = messages[-1]["content"]

        return type("LLMMessage", (), {"content": f"Stubbed answer based on: {user_message[:60]}"})()

    __call__ = invoke


def test_api_docs_agent_returns_answer_without_real_llm(tmp_path: Path) -> None:
    context_path = Path(__file__).resolve().parents[1] / "Docs" / "api_docs_context.yaml"
    assert context_path.exists(), "API docs context YAML is required for the agent test."

    llm = _FakeLLM()
    agent = ApiDocsAgent(context_path=context_path, llm=llm)

    question = "What does the GET /health endpoint report?"
    result = agent.invoke(question)

    assert result.status is AgentExecutionStatus.succeeded
    assert result.answer is not None and "Stubbed answer" in result.answer
    assert result.trace, "Trace events should include retrieval metadata."
    assert llm.invocations == 1

