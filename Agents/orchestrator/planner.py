"""Planner for selecting which agents to execute for a query."""

from __future__ import annotations

from typing import Iterable, List, Sequence

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

from Agents.core.models import AgentName, PlannerDecision
from Agents.QueryAgent.config import get_resources


_SQL_KEYWORDS = {
    "table",
    "tables",
    "list",
    "show",
    "top",
    "per",
    "group",
    "average",
    "sum",
    "count",
    "trend",
    "breakdown",
    "report",
}

_COMPUTE_KEYWORDS = {
    "calculate",
    "difference",
    "ratio",
    "project",
    "simulate",
    "compute",
    "forecast",
    "estimate",
    "compare",
    "what is",
    "percent",
}


class _PlannerResponse(BaseModel):
    agents: List[AgentName]
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class Planner:
    """Combines heuristics with LLM reasoning to pick agent ordering."""

    def __init__(self, llm: AzureChatOpenAI | None = None) -> None:
        resources = get_resources()
        self.llm = llm or resources.llm
        self._parser = PydanticOutputParser(pydantic_object=_PlannerResponse)
        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an orchestration planner. Use the proposed agent list and optionally drop "
                    "agents that add no value. Keep the order unless there is a compelling reason. "
                    "Return JSON that matches the format instructions.",
                ),
                (
                    "user",
                    "Question: {question}\n"
                    "Context keys: {context_keys}\n"
                    "Preferred agents: {preferred}\n"
                    "Candidates: {candidates}\n"
                    "Disabled: {disabled}\n"
                    "Format instructions: {format_instructions}",
                ),
            ]
        )

    def plan(
        self,
        *,
        question: str,
        prefer: Sequence[AgentName] = (),
        disable: Sequence[AgentName] = (),
        context: dict | None = None,
    ) -> PlannerDecision:
        candidates = self._heuristic_candidates(question)
        candidates = self._apply_preferences(candidates, prefer, disable)

        if not candidates:
            raise RuntimeError("No eligible agents remain after applying preferences")

        try:
            response = (
                self._prompt.partial(
                    format_instructions=self._parser.get_format_instructions()
                )
                | self.llm
                | self._parser
            ).invoke(
                {
                    "question": question,
                    "context_keys": list((context or {}).keys()),
                    "preferred": list(prefer),
                    "candidates": list(candidates),
                    "disabled": list(disable),
                }
            )
        except Exception:  # pragma: no cover - fallback resilience
            response = _PlannerResponse(
                agents=list(candidates),
                rationale="Using heuristic ordering due to planner error",
                confidence=0.4,
            )

        filtered_agents = [agent for agent in response.agents if agent in candidates]
        if not filtered_agents:
            filtered_agents = list(candidates)

        return PlannerDecision(
            rationale=response.rationale,
            chosen_agents=tuple(filtered_agents),
            confidence=float(min(1.0, max(0.0, response.confidence))),
            guardrails={"disabled": list(disable)},
        )

    @staticmethod
    def _heuristic_candidates(question: str) -> List[AgentName]:
        lowered = question.lower()
        sql_score = sum(keyword in lowered for keyword in _SQL_KEYWORDS)
        comp_score = sum(keyword in lowered for keyword in _COMPUTE_KEYWORDS)

        order: List[AgentName] = []
        if sql_score >= comp_score and sql_score > 0:
            order.append("sql")
        if comp_score >= sql_score and comp_score > 0:
            order.append("computation")

        if not order:
            order = ["sql"]

        return order

    @staticmethod
    def _apply_preferences(
        candidates: Iterable[AgentName],
        prefer: Sequence[AgentName],
        disable: Sequence[AgentName],
    ) -> List[AgentName]:
        prefer_set = list(dict.fromkeys(agent for agent in prefer if agent not in disable))
        filtered = [agent for agent in candidates if agent not in disable]

        ordered: List[AgentName] = []
        for agent in prefer_set:
            if agent not in ordered and agent in filtered:
                ordered.append(agent)
        for agent in filtered:
            if agent not in ordered:
                ordered.append(agent)
        return ordered


__all__ = ["Planner"]


