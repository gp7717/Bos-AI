from __future__ import annotations

from types import SimpleNamespace

from Agents.core.models import AgentExecutionStatus, AgentResult, TabularResult
from Agents.orchestrator.composer import Composer


class _CapturingLLM:
    """Stub LLM that records incoming messages and returns a canned response."""

    def __init__(self) -> None:
        self.messages: list[list[str]] = []

    def invoke(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        captured = []
        for message in messages:
            content = getattr(message, "content", str(message))
            captured.append(content)
        self.messages.append(captured)
        return SimpleNamespace(content="Stub final answer including revenue insights.")


def test_composer_prioritises_structured_metrics() -> None:
    llm = _CapturingLLM()
    composer = Composer(llm=llm)

    docs_result = AgentResult(
        agent="docs",  # type: ignore[assignment]
        status=AgentExecutionStatus.succeeded,
        answer="Structured data not available; revenue cannot be determined.",
    )
    sql_result = AgentResult(
        agent="sql",  # type: ignore[assignment]
        status=AgentExecutionStatus.succeeded,
        tabular=TabularResult(
            columns=[
                "campaign_id",
                "campaign_name",
                "total_spend",
                "orders_count",
                "revenue",
                "roas",
            ],
            rows=[
                {
                    "campaign_id": 123,
                    "campaign_name": "Example Campaign",
                    "total_spend": "18286.18",
                    "orders_count": 14,
                    "revenue": "52684.00",
                    "roas": "2.88",
                },
                {
                    "campaign_id": 456,
                    "campaign_name": "Another Campaign",
                    "total_spend": "22182.57",
                    "orders_count": 16,
                    "revenue": "39079.00",
                    "roas": "1.76",
                },
            ],
            row_count=2,
        ),
        answer="Query executed successfully.",
    )

    response = composer.compose(
        question="Top performing campaigns for this week in meta",
        planner_rationale="Review SQL data before summarising.",
        agent_results=[docs_result, sql_result],
    )

    assert llm.messages, "Expected the stub LLM to receive a prompt."
    user_message = llm.messages[0][1]
    assert "structured rows=2" in user_message
    assert "revenue=52684.00" in user_message
    assert user_message.index("structured") < user_message.index("text_answer=")
    assert "Structured data not available" in user_message

    assert response.answer == "Stub final answer including revenue insights."
    assert response.data == sql_result.tabular

