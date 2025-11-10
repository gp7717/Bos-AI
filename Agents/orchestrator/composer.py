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

    def __init__(self, llm: Optional[AzureChatOpenAI] = None) -> None:
        resources = get_resources()
        self.llm = llm or resources.llm
        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a Senior Business Intelligence Agent. "
                        "Provide a concise and direct answer to the user query. Do not provide any explanation or context. "
                        "If any metrics is 0 or not available , dont mention that in the answer. "
                        "Structured metrics appear in the agent summaries as entries beginning with 'structured'. "
                        "Treat structured metrics as authoritative, especially for financial figures such as revenue, spend, ROAS, CPA, conversion rate, and orders. "
                        "Only claim a metric is missing when it is absent from all structured summaries. "
                        "Prefer structured data from quantitative agents over narrative text, and never repeat statements that contradict available structured metrics. "
                        "Do not explain your process or how conclusions were reached. "
                        "If no data is available or no rows are returned, clearly state that no data was returned."
                        "Use Indian Rupees (Rs.) with commas and two decimals for all monetary amounts (e.g., Rs.1,75,206.00), never ₹ or INR. "
                        "Be professional. Do not use emojis, hashtags, or unnecessary formatting—only use '\\n' for new lines. "
                        "Do not include boilerplate, incomplete placeholders, or repeat explanations. Only summarize the most important findings and information relevant to the user question."
                    ),
                ),
                (
                    "user",
                    (
                        "User question: {question}\n"
                        "Planner rationale: {planner_rationale}\n"
                        "Agent summaries:\n{agent_summaries}"
                    ),
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


