from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Agents.orchestrator.planner import Planner


class _StubLLM:
    def __init__(self, response: str) -> None:
        self._response = response
        self.invocations = 0
        self.messages = None

    def invoke(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.invocations += 1
        self.messages = messages
        return type("LLMMessage", (), {"content": self._response})()


class _FailingLLM:
    def invoke(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("LLM unavailable")


def test_planner_prefers_docs_for_net_profit_question() -> None:
    stub = _StubLLM(
        '{"agents": ["docs", "sql"], "rationale": "Docs explain the net profit endpoint.", "confidence": 0.82}'
    )
    planner = Planner(llm=stub)

    decision = planner.plan(question="What is the net profit this year?")

    assert decision.chosen_agents[0] == "docs"
    assert decision.confidence == pytest.approx(0.82, rel=0.01)
    assert stub.invocations == 1


def test_planner_fallback_when_llm_fails() -> None:
    planner = Planner(llm=_FailingLLM())

    decision = planner.plan(question="List Shopify orders revenue by month.")

    assert decision.chosen_agents[0] == "sql"
    assert decision.confidence == pytest.approx(0.4)

