"""Planner for selecting which agents to execute for a query."""

from __future__ import annotations

import logging
from typing import Iterable, List, Sequence

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

from Agents.core.models import AgentName, PlannerDecision
from Agents.QueryAgent.config import get_resources

logger = logging.getLogger(__name__)


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

_API_DOCS_KEYWORDS = {
    "api",
    "endpoint",
    "route",
    "http",
    "/api/",
    "get /",
    "post /",
    "put /",
    "delete /",
    "patch /",
    "status code",
    "request body",
}

_GRAPH_KEYWORDS = {
    "graph",
    "chart",
    "plot",
    "visualize",
    "visualization",
    "trend",
    "trends",
    "over time",
    "show me",
    "display",
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
                    "CRITICAL RULE: If the user query contains visualization keywords (graph, chart, plot, visualize, visualization, trends, 'show me', display), "
                    "you MUST include the 'graph' agent in your response, even if you think it's not needed. "
                    "The graph agent will automatically process tabular data from sql/computation agents. "
                    "Do NOT remove the graph agent when visualization is explicitly requested. "
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
        # FIRST: Check for graph keywords BEFORE any processing
        lowered_question = question.lower()
        has_graph_keywords = any(keyword in lowered_question for keyword in _GRAPH_KEYWORDS)
        
        if has_graph_keywords:
            logger.info(f"Graph keywords detected in query: {question[:100]}...")
        
        candidates = self._heuristic_candidates(question)
        
        # If graph keywords detected, ensure graph is in candidates BEFORE applying preferences
        if has_graph_keywords and "graph" not in candidates and "graph" not in disable:
            candidates.append("graph")
            logger.info("Graph agent added to candidates via keyword detection")
        
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
            # Fallback: create response object with agents field
            response = _PlannerResponse(
                agents=list(candidates),
                rationale="Using heuristic ordering due to planner error",
                confidence=0.4,
            )

        # Extract agents from LLM response (handle both list and missing field)
        try:
            llm_agents = getattr(response, 'agents', []) or []
        except (AttributeError, TypeError):
            llm_agents = []
        
        logger.info(f"LLM returned agents: {llm_agents}, candidates: {candidates}")
        
        # Filter LLM agents to only include valid candidates
        filtered_agents = [agent for agent in llm_agents if agent in candidates]
        if not filtered_agents:
            filtered_agents = list(candidates)
        
        logger.info(f"Filtered agents before safeguard: {filtered_agents}")
        
        # CRITICAL: Ensure graph agent is ALWAYS added if explicitly requested via keywords
        # This runs AFTER filtering to override LLM decisions
        # This MUST happen BEFORE final agent selection to ensure graph is included
        if has_graph_keywords:
            logger.info(f"Graph keywords detected - enforcing graph agent inclusion")
            
            # Ensure we have a data source agent first (SQL or computation)
            has_data_source = any(agent in filtered_agents for agent in ["sql", "computation"])
            if not has_data_source:
                # Add SQL as prerequisite if no data source exists
                if "sql" not in filtered_agents and "sql" not in disable:
                    filtered_agents.insert(0, "sql")
                    logger.info("SQL agent added as prerequisite for graph")
            
            # ALWAYS add graph agent if requested, regardless of what LLM said
            # Only skip if explicitly disabled
            if "graph" not in filtered_agents:
                if "graph" not in disable:
                    # Insert graph agent after data source agents but before other agents
                    # Find the position after the last data source agent
                    insert_pos = len(filtered_agents)
                    for i, agent in enumerate(filtered_agents):
                        if agent in ["sql", "computation"]:
                            insert_pos = i + 1
                    filtered_agents.insert(insert_pos, "graph")
                    logger.warning(f"Graph agent FORCED via safeguard at position {insert_pos} for query: {question[:50]}...")
                else:
                    logger.warning(f"Graph agent requested but explicitly disabled")
            else:
                logger.info("Graph agent already in filtered agents")
        
        logger.info(f"Final agents after safeguard: {filtered_agents}")

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
        api_score = sum(keyword in lowered for keyword in _API_DOCS_KEYWORDS)
        graph_score = sum(keyword in lowered for keyword in _GRAPH_KEYWORDS)

        order: List[AgentName] = []
        
        # Check for graph keywords FIRST - this is critical for visualization requests
        has_graph_request = graph_score > 0
        
        if api_score >= max(sql_score, comp_score, graph_score) and api_score > 0:
            order.append("api_docs")
        if sql_score >= comp_score and sql_score > 0:
            order.append("sql")
        if comp_score >= sql_score and comp_score > 0:
            order.append("computation")
        
        # Graph agent MUST be added if graph keywords are detected
        # It will consume tabular data from SQL/computation agents
        if has_graph_request:
            # If graph is requested, ensure we have a data source agent first
            if "sql" not in order and "computation" not in order:
                order.append("sql")  # Add SQL as prerequisite for graph
            # Always add graph agent if keywords detected - this is non-negotiable
            if "graph" not in order:
                order.append("graph")
                logger.info(f"Heuristic: Added graph agent due to graph keywords (score: {graph_score})")

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


