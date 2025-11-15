"""Planner for selecting which agents to execute for a query."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

from Agents.core.models import AgentName, DecomposedQuery, PlannerDecision
from Agents.QueryAgent.config import get_resources
from Agents.orchestrator.capability_registry import CapabilityRegistry

logger = logging.getLogger(__name__)

_REGISTRY = CapabilityRegistry()  # Initialize capability registry


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
        decomposed_query: Optional[DecomposedQuery] = None,
        prefer: Sequence[AgentName] = (),
        disable: Sequence[AgentName] = (),
        context: dict | None = None,
    ) -> PlannerDecision:
        # TEMPORARILY DISABLE API_DOCS AGENT
        # Add api_docs to disable list if not already there
        disable_list = list(disable)
        if "api_docs" not in disable_list:
            disable_list.append("api_docs")
            logger.info("API_DOCS agent is temporarily disabled")
        
        # Use decomposed query if available
        if decomposed_query and len(decomposed_query.sub_queries) > 0:
            return self._plan_from_decomposition(
                decomposed_query=decomposed_query,
                prefer=prefer,
                disable=tuple(disable_list),  # Pass updated disable list
                context=context,
            )
        
        # FIRST: Check for graph keywords BEFORE any processing
        lowered_question = question.lower()
        has_graph_keywords = any(keyword in lowered_question for keyword in _GRAPH_KEYWORDS)
        
        if has_graph_keywords:
            logger.info(f"Graph keywords detected in query: {question[:100]}...")
        
        candidates = self._heuristic_candidates(question)
        
        # If graph keywords detected, ensure graph is in candidates BEFORE applying preferences
        if has_graph_keywords and "graph" not in candidates and "graph" not in disable_list:
            candidates.append("graph")
            logger.info("Graph agent added to candidates via keyword detection")
        
        candidates = self._apply_preferences(candidates, prefer, tuple(disable_list))

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
                    "disabled": list(disable_list),
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
                if "sql" not in filtered_agents and "sql" not in disable_list:
                    filtered_agents.insert(0, "sql")
                    logger.info("SQL agent added as prerequisite for graph")
            
            # ALWAYS add graph agent if requested, regardless of what LLM said
            # Only skip if explicitly disabled
            if "graph" not in filtered_agents:
                if "graph" not in disable_list:
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
            guardrails={"disabled": list(disable_list)},
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
        
        # Check if metrics are available via API using capability registry
        detected_metrics = _REGISTRY.extract_metrics_from_query(question)
        has_api_metrics = False
        if detected_metrics:
            for metric in detected_metrics:
                tools = _REGISTRY.find_tools_for_metric(metric)
                api_tools = [t for t in tools if t.type == "api_endpoint" and t.agent == "api_docs"]
                if api_tools:
                    has_api_metrics = True
                    logger.info(f"Metric '{metric}' available via API, preferring api_docs agent")
                    break
        
        # Prefer API agent when metrics are available via API
        if has_api_metrics:
            order.append("api_docs")
        elif api_score >= max(sql_score, comp_score, graph_score) and api_score > 0:
            order.append("api_docs")
        
        if sql_score >= comp_score and sql_score > 0:
            # Only add SQL if API doesn't have the metrics
            if not has_api_metrics:
                order.append("sql")
        if comp_score >= sql_score and comp_score > 0:
            order.append("computation")
        
        # Graph agent MUST be added if graph keywords are detected
        # It will consume tabular data from SQL/computation/API agents
        if has_graph_request:
            # If graph is requested, ensure we have a data source agent first
            if "api_docs" not in order and "sql" not in order and "computation" not in order:
                # Prefer API if metrics available, otherwise SQL
                if has_api_metrics:
                    order.append("api_docs")
                else:
                    order.append("sql")  # Add SQL as prerequisite for graph
            # Always add graph agent if keywords detected - this is non-negotiable
            if "graph" not in order:
                order.append("graph")
                logger.info(f"Heuristic: Added graph agent due to graph keywords (score: {graph_score})")

        if not order:
            # Default: prefer API if metrics detected, otherwise SQL
            if has_api_metrics:
                order = ["api_docs"]
            else:
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

    def _plan_from_decomposition(
        self,
        *,
        decomposed_query: DecomposedQuery,
        prefer: Sequence[AgentName] = (),
        disable: Sequence[AgentName] = (),
        context: dict | None = None,
    ) -> PlannerDecision:
        """
        Plan agent execution based on decomposed sub-queries.
        
        Builds execution order considering:
        - Sub-query priorities
        - Dependencies between sub-queries
        - Agent requirements
        """
        # TEMPORARILY DISABLE API_DOCS AGENT
        disable_list = list(disable)
        if "api_docs" not in disable_list:
            disable_list.append("api_docs")
            logger.info("API_DOCS agent is temporarily disabled in decomposition planning")
        
        execution_order: List[AgentName] = []
        rationale_parts = []
        
        # Sort sub-queries by priority and resolve dependencies
        sorted_sub_queries = sorted(
            decomposed_query.sub_queries,
            key=lambda sq: (sq.priority, len(sq.dependencies))
        )
        
        # Build agent execution plan
        agent_to_subqueries: Dict[AgentName, List] = {}
        
        for sub_query in sorted_sub_queries:
            # Get agents needed for this sub-query
            agents = sub_query.required_agents or self._infer_agents_from_intent(sub_query.intent)
            
            # Filter disabled agents
            agents = [a for a in agents if a not in disable_list]
            
            # Apply preferences (prefer agents come first)
            if prefer:
                agents = [a for a in prefer if a in agents] + [a for a in agents if a not in prefer]
            
            # Track which sub-queries map to which agents
            for agent in agents:
                if agent not in agent_to_subqueries:
                    agent_to_subqueries[agent] = []
                agent_to_subqueries[agent].append(sub_query)
            
            # Add to execution order (avoid duplicates, respect dependencies)
            for agent in agents:
                if agent not in execution_order:
                    # Check if this agent depends on others
                    if sub_query.dependencies:
                        # Find agents for dependent sub-queries
                        for dep_id in sub_query.dependencies:
                            dep_sq = next(
                                (sq for sq in decomposed_query.sub_queries if sq.id == dep_id),
                                None
                            )
                            if dep_sq:
                                dep_agents = dep_sq.required_agents or self._infer_agents_from_intent(dep_sq.intent)
                                for dep_agent in dep_agents:
                                    if dep_agent not in execution_order:
                                        execution_order.append(dep_agent)
                    
                    execution_order.append(agent)
            
            rationale_parts.append(
                f"Sub-query '{sub_query.detailed_query[:60]}...' → {', '.join(agents)}"
            )
        
        # Ensure at least one agent
        if not execution_order:
            execution_order = ["sql"]  # Default fallback
        
        rationale = f"Decomposed into {len(decomposed_query.sub_queries)} sub-queries. " + " | ".join(rationale_parts)
        
        return PlannerDecision(
            rationale=rationale,
            chosen_agents=tuple(execution_order),
            confidence=decomposed_query.confidence,
            guardrails={
                "disabled": list(disable_list),
                "decomposed": True,
                "sub_query_count": len(decomposed_query.sub_queries)
            }
        )
    
    @staticmethod
    def _infer_agents_from_intent(intent: str) -> List[AgentName]:
        """Infer agents needed based on sub-query intent."""
        intent_to_agents = {
            "data_retrieval": ["sql"],
            "computation": ["computation"],
            "api_call": ["api_docs"],
            "visualization": ["graph"],
            "analysis": ["sql", "computation"],
        }
        return intent_to_agents.get(intent, ["sql"])


__all__ = ["Planner"]


