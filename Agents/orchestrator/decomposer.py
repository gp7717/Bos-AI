"""Natural language query decomposition system."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

from Agents.core.models import (
    AgentName,
    DecomposedQuery,
    QueryIntent,
    SubQuery,
)
from Agents.QueryAgent.config import get_resources
from Agents.orchestrator.capability_registry import CapabilityRegistry

logger = logging.getLogger(__name__)


class _DecompositionResponse(BaseModel):
    """LLM response for query decomposition."""

    interpreted_query: str = Field(
        ..., description="Detailed interpretation of what the user wants"
    )
    primary_intent: str
    secondary_intents: List[str] = Field(default_factory=list)
    inferred_metrics: List[str] = Field(default_factory=list)
    inferred_timeframes: Dict[str, str] = Field(default_factory=dict)
    inferred_filters: Dict[str, Any] = Field(default_factory=dict)
    sub_queries: List[Dict[str, Any]] = Field(default_factory=list)
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class QueryDecomposer:
    """
    Natural language query decomposer that converts generic user queries
    into specific, actionable sub-queries.

    Handles queries like:
    - "What's our performance this month?" → Specific queries for revenue, orders, trends
    - "Show me the trends" → Time-series data retrieval and visualization
    - "Compare last quarter to this quarter" → Two data retrieval queries + comparison computation
    """

    def __init__(
        self, llm: Optional[AzureChatOpenAI] = None, registry: Optional[CapabilityRegistry] = None
    ) -> None:
        resources = get_resources()
        self.llm = llm or resources.llm
        self.registry = registry or CapabilityRegistry()
        self._parser = PydanticOutputParser(pydantic_object=_DecompositionResponse)
        self._prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert Business Intelligence query decomposition system. Your task is to understand generic, natural language user queries and convert them into specific, actionable sub-queries.

CRITICAL INSTRUCTIONS:
1. Understand the user's INTENT even if the query is vague or generic
2. Infer missing details (timeframes, metrics, filters) from context
3. Break down complex queries into specific, executable sub-queries
4. Identify dependencies between sub-queries
5. Map each sub-query to appropriate agents and tools using the capability registry

AVAILABLE AGENTS:
- api_docs: HTTP API integration - PREFERRED when metrics are available via API (net_profit, revenue, sales, ROAS, COGS)
- sql: Database queries - Use when API doesn't have the data or custom queries needed
- computation: Calculations, forecasts, statistical analysis - Use for complex calculations
- graph: Data visualization, charts, trends - Use when visualization is requested

CAPABILITY REGISTRY INSIGHTS:
- Net profit: Available via API endpoint /api/net_profit (PREFER api_docs agent)
- Revenue/Sales: Available via API endpoint /api/sales (PREFER api_docs agent)
- ROAS: Available via API endpoint /api/roas (PREFER api_docs agent)
- COGS: Available via API endpoint /api/cogs (PREFER api_docs agent)
- Ad Spend: Available via API endpoint /api/ad_spend (PREFER api_docs agent)

IMPORTANT: Always check if metrics are available via API first. Only use SQL/computation when API doesn't have the data.

COMMON BUSINESS METRICS:
- Revenue, Sales, Orders, Conversion Rate, ROAS, CPA, AOV, Customer Count, Net Profit, COGS, Ad Spend

TIME FRAME INFERENCE:
- "this month" → current month (use current_date from context)
- "last quarter" → previous quarter
- "this year" → current year
- "last week" → previous 7 days
- "recent" → last 30 days (default)
- "last 4 months" → last 4 months from current date

EXAMPLES:

User Query: "Get the net profit graph for the last 4 months"
Interpreted: "Retrieve net profit data for the last 4 months and generate a visualization"
Sub-queries:
1. "Fetch net profit data via API for last 4 months" → api_docs (tool: api_net_profit), priority: 0
2. "Generate trend visualization for net profit over last 4 months" → graph (tool: graph_generator), priority: 1, depends on: [1]

User Query: "What's our performance this month?"
Interpreted: "Retrieve and analyze key performance metrics for the current month including revenue, orders, conversion rate, and trends"
Sub-queries:
1. "Retrieve sales/revenue data for current month via API" → api_docs (tool: api_sales), priority: 0
2. "Retrieve net profit data for current month via API" → api_docs (tool: api_net_profit), priority: 0
3. "Calculate performance metrics and compare to previous month" → computation, priority: 1, depends on: [1, 2]
4. "Generate trend visualization for monthly performance" → graph, priority: 2, depends on: [1, 2]

User Query: "Show me the trends"
Interpreted: "Display trend analysis and visualization of key business metrics over time"
Sub-queries:
1. "Retrieve time-series data for key metrics (revenue, orders) over the last 3 months via API" → api_docs (tool: api_sales), priority: 0
2. "Generate trend visualization chart showing metrics over time" → graph (tool: graph_generator), priority: 1, depends on: [1]

User Query: "Compare last quarter to this quarter"
Interpreted: "Retrieve and compare performance metrics between previous quarter and current quarter"
Sub-queries:
1. "Retrieve sales data for previous quarter (Q3) via API" → api_docs (tool: api_sales), priority: 0
2. "Retrieve sales data for current quarter (Q4) via API" → api_docs (tool: api_sales), priority: 0
3. "Calculate comparison metrics (growth rate, difference, percentage change)" → computation, priority: 1, depends on: [1, 2]
4. "Generate comparison visualization" → graph, priority: 2, depends on: [1, 2]

OUTPUT FORMAT:
- interpreted_query: Your detailed understanding of what user wants
- primary_intent: Main intent (data_retrieval, analysis, comparison, trend_analysis, forecast, visualization, computation, api_inquiry)
- secondary_intents: Additional intents if query is multi-faceted
- inferred_metrics: List of metrics user likely wants (e.g., ["net_profit", "revenue", "orders"])
- inferred_timeframes: Dict with "start" and "end" dates if time-bound
- inferred_filters: Any filters inferred (region, product category, etc.)
- sub_queries: List of specific sub-queries, each with:
  - original_phrase: Part of user query that triggered this
  - detailed_query: Specific, actionable query text
  - intent: data_retrieval, computation, api_call, visualization, analysis
  - required_agents: List of agent names needed (PREFER api_docs when metric available via API)
  - selected_tools: List of tool IDs from capability registry (e.g., ["api_net_profit"])
  - dependencies: IDs of sub-queries this depends on (empty if independent)
  - priority: Execution order (0 = first, higher = later)
  - context: Additional context (metrics, timeframes, etc.)
- rationale: Explanation of decomposition
- confidence: How confident you are (0.0 to 1.0)

Return JSON matching the format instructions.""",
            ),
            (
                "user",
                """User Query: {query}

Available Context:
- Current Date: {current_date}
- Current DateTime (UTC): {current_datetime}
- Context Keys: {context_keys}

Available Capabilities:
{capability_summary}

Format instructions: {format_instructions}""",
            ),
        ])

    def decompose(
        self, *, question: str, context: Optional[Dict[str, Any]] = None
    ) -> DecomposedQuery:
        """
        Decompose a natural language query into specific sub-queries.

        Args:
            question: User's natural language query (can be generic)
            context: Request context (includes temporal context)

        Returns:
            DecomposedQuery with specific sub-queries ready for execution
        """
        context = context or {}
        current_date = context.get("current_date", "unknown")
        current_datetime = context.get("current_datetime_utc", "unknown")

        # Extract metrics from query to help with capability matching
        detected_metrics = self.registry.extract_metrics_from_query(question)

        # Build capability summary for LLM
        capability_summary = self._build_capability_summary(detected_metrics)

        try:
            response = (
                self._prompt.partial(
                    format_instructions=self._parser.get_format_instructions()
                )
                | self.llm
                | self._parser
            ).invoke({
                "query": question,
                "current_date": current_date,
                "current_datetime": current_datetime,
                "context_keys": list(context.keys()),
                "capability_summary": capability_summary,
            })

            # Convert to SubQuery objects
            sub_queries = []
            for idx, sq_dict in enumerate(response.sub_queries):
                # Ensure selected_tools are populated from capability registry if not provided
                selected_tools = sq_dict.get("selected_tools", [])
                if not selected_tools:
                    # Try to infer tools from metrics and agents
                    metrics = sq_dict.get("context", {}).get("metrics", [])
                    agents = sq_dict.get("required_agents", [])
                    selected_tools = self._infer_tools_from_metrics_and_agents(metrics, agents)

                # Normalize dependencies: convert integers to strings, ensure all are strings
                dependencies = sq_dict.get("dependencies", [])
                normalized_dependencies = []
                for dep in dependencies:
                    if isinstance(dep, int):
                        # Convert integer to string (LLM sometimes returns indices)
                        normalized_dependencies.append(str(dep))
                    elif isinstance(dep, str):
                        normalized_dependencies.append(dep)
                    else:
                        # Skip invalid types
                        logger.warning(f"Invalid dependency type: {type(dep)}, value: {dep}")

                sub_query = SubQuery(
                    id=f"sq_{uuid.uuid4().hex[:8]}",
                    original_phrase=sq_dict.get("original_phrase", ""),
                    detailed_query=sq_dict.get("detailed_query", ""),
                    intent=sq_dict.get("intent", "data_retrieval"),
                    required_agents=sq_dict.get("required_agents", []),
                    selected_tools=selected_tools,
                    dependencies=normalized_dependencies,
                    priority=sq_dict.get("priority", idx),
                    context=sq_dict.get("context", {}),
                    metadata=sq_dict.get("metadata", {}),
                )
                sub_queries.append(sub_query)

            # Build QueryIntent
            intent = QueryIntent(
                primary_intent=response.primary_intent,  # type: ignore[assignment]
                secondary_intents=response.secondary_intents,
                confidence=response.confidence,
            )

            # Build capability matches - collect unique tool IDs from all sub-queries
            matched_tools_list = []
            try:
                for sq in sub_queries:
                    # Safely get selected_tools, ensuring it's a list
                    if hasattr(sq, 'selected_tools'):
                        selected_tools = sq.selected_tools
                    else:
                        selected_tools = []
                    
                    # Ensure selected_tools is a list
                    if not isinstance(selected_tools, list):
                        logger.warning(f"selected_tools is not a list for sub_query {sq.id}: {type(selected_tools)}")
                        selected_tools = []
                    
                    # Collect valid tool IDs
                    for tool_id in selected_tools:
                        if isinstance(tool_id, str) and tool_id and tool_id in self.registry.tools_by_id:
                            if tool_id not in matched_tools_list:  # Avoid duplicates
                                matched_tools_list.append(tool_id)
            except Exception as e:
                logger.error(f"Error building capability matches: {e}", exc_info=True)
                # Continue with empty list rather than failing completely
            
            capability_matches = {
                "detected_metrics": detected_metrics,
                "matched_tools": matched_tools_list,
            }

            return DecomposedQuery(
                original_query=question,
                interpreted_query=response.interpreted_query,
                intent=intent,
                sub_queries=sub_queries,
                inferred_metrics=response.inferred_metrics,
                inferred_timeframes=response.inferred_timeframes,
                inferred_filters=response.inferred_filters,
                capability_matches=capability_matches,
                decomposition_rationale=response.rationale,
                confidence=response.confidence,
            )

        except Exception as exc:
            logger.warning(f"Query decomposition failed: {exc}. Using fallback.", exc_info=True)
            return self._create_fallback_decomposition(question, context, detected_metrics)

    def _build_capability_summary(self, metrics: List[str]) -> str:
        """Build a summary of available capabilities for the LLM."""
        if not metrics:
            return "No specific metrics detected. All agents available."

        summary_parts = []
        for metric in metrics:
            tools = self.registry.find_tools_for_metric(metric)
            if tools:
                api_tools = [t for t in tools if t.type == "api_endpoint"]
                if api_tools:
                    summary_parts.append(
                        f"Metric '{metric}': Available via API ({', '.join([t.name for t in api_tools[:3]])}) - PREFER api_docs agent"
                    )
                else:
                    summary_parts.append(
                        f"Metric '{metric}': Available via {tools[0].type} - Use {tools[0].agent} agent"
                    )
            else:
                summary_parts.append(
                    f"Metric '{metric}': Not directly available - May need SQL/computation"
                )

        return "\n".join(summary_parts) if summary_parts else "No specific capabilities matched."

    def _infer_tools_from_metrics_and_agents(
        self, metrics: List[str], agents: List[AgentName]
    ) -> List[str]:
        """Infer tool IDs from metrics and agents."""
        tool_ids = []
        for metric in metrics:
            tools = self.registry.find_tools_for_metric(metric)
            for tool in tools:
                if tool.agent in agents and tool.id not in tool_ids:
                    tool_ids.append(tool.id)

        return tool_ids

    def _create_fallback_decomposition(
        self, question: str, context: Optional[Dict[str, Any]], detected_metrics: List[str]
    ) -> DecomposedQuery:
        """Create a fallback decomposition when LLM fails."""
        # Try to use capability registry to find best agent
        best_agent = self.registry.find_best_agent(question, [], detected_metrics)
        selected_tools = []
        required_agents = []

        if best_agent:
            required_agents = [best_agent.agent_name]
            # Find tools for detected metrics
            for metric in detected_metrics:
                tools = self.registry.find_tools_for_metric(metric)
                if tools:
                    selected_tools.append(tools[0].id)

        if not required_agents:
            required_agents = ["sql"]  # Default fallback

        fallback_sub_query = SubQuery(
            id="sq_fallback",
            original_phrase=question,
            detailed_query=question,  # Use original as-is
            intent="data_retrieval",
            required_agents=required_agents,  # type: ignore[assignment]
            selected_tools=selected_tools,
            priority=0,
        )

        return DecomposedQuery(
            original_query=question,
            interpreted_query=f"Process query: {question}",
            intent=QueryIntent(primary_intent="data_retrieval", confidence=0.5),
            sub_queries=[fallback_sub_query],
            inferred_metrics=detected_metrics,
            capability_matches={"detected_metrics": detected_metrics},
            decomposition_rationale="Fallback decomposition due to error",
            confidence=0.5,
        )


__all__ = ["QueryDecomposer"]

