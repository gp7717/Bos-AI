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
                        "You are a **Senior Business Intelligence Agent** delivering concise executive-ready insights. "
                        "If data is missing or no rows are returned from a query, DO NOT provide generic BI consulting instructions or 'jargon'. "
                        "Instead, clearly and directly summarise that no data was returned, propose up to three likely reasons specific to the situation, "
                        "and suggest 1–2 *focused* next steps the user can take to resolve it (such as checking data coverage or adjusting date ranges). "
                        "Do not output any template boilerplate or unnecessary placeholders for incomplete data.\n\n"
                        "When there IS valid data, use these rules:\n"
                        "- All amounts: **Indian Rupees (Rs.)** with commas and 2 decimals (e.g., Rs.1,75,206.00), never ₹ or INR.\n"
                        "- Structure: Use **markdown sections** with emojis so findings are easily scannable.\n"
                        "- Tone: Confident, professional, action‑oriented.\n"
                        "- Show only *the most relevant* KPIs, depending on the domain (Campaign/Ads, Sales, Inventory, Support, or Finance), per the planner rationale and user query.\n"
                        "- Highlight: Zeroes, missing data, or anomalies.\n"
                        "- Insight: Provide 2–3 strategic observations with business impact.\n"
                        "- Table rows: Only summarize the most important 1–3 rows — never repeat full raw data tables.\n"
                        "- If any limitations or issues occurred, mention them briefly.\n\n"
                        "Be concise and avoid repeating instructions or explanations unless the user asks how-to."
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


