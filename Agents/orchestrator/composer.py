"""Composer agent that fuses individual agent outputs into a final answer."""

from __future__ import annotations

from typing import Iterable, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI

from Agents.core.models import AgentResult, OrchestratorResponse, TabularResult
from Agents.QueryAgent.config import get_resources


class Composer:
    """LLM-backed composer that consolidates agent answers."""

    def __init__(self, llm: AzureChatOpenAI | None = None) -> None:
        resources = get_resources()
        self.llm = llm or resources.llm
        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You synthesise answers from multiple specialised agents. "
                    "Provide a concise, truthful final answer. Mention any limitations. "
                    "If tabular data is available, describe key takeaways but do not repeat the entire table.",
                ),
                (
                    "user",
                    "User question: {question}\n"
                    "Planner rationale: {planner_rationale}\n"
                    "Agent summaries:\n{agent_summaries}",
                ),
            ]
        )

    def compose(
        self,
        *,
        question: str,
        planner_rationale: str,
        agent_results: Iterable[AgentResult],
        metadata: Optional[dict] = None,
    ) -> OrchestratorResponse:
        results = list(agent_results)
        summaries = []
        tabular = self._select_tabular(results)
        for result in results:
            status = result.status
            base = f"Agent: {result.agent} | status: {status}."
            if result.answer:
                base += f" Answer: {result.answer}"
            if result.error:
                base += f" Error: {result.error.message}"
            summaries.append(base)

        llm_response = self._prompt | self.llm | RunnableLambda(lambda message: message.content)
        answer = llm_response.invoke(
            {
                "question": question,
                "planner_rationale": planner_rationale,
                "agent_summaries": "\n".join(summaries) or "No agents produced outputs.",
            }
        )

        return OrchestratorResponse(
            answer=str(answer),
            data=tabular,
            agent_results=results,
            metadata=metadata or {},
        )

    @staticmethod
    def _select_tabular(results: Iterable[AgentResult]) -> Optional[TabularResult]:
        for result in results:
            if result.tabular:
                return result.tabular
        return None


__all__ = ["Composer"]


