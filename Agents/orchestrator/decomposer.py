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
- sql: Database queries - PREFERRED for data retrieval (more flexible, direct database access)
- api_docs: HTTP API integration - Use when SQL is not suitable or API provides pre-computed aggregations
- computation: Calculations, forecasts, statistical analysis - Use for complex calculations
- graph: Data visualization, charts, trends - Use when visualization is requested

CAPABILITY REGISTRY INSIGHTS:
- Net profit: Available via SQL (shopify_orders table) or API endpoint /api/net_profit (PREFER sql agent)
- Revenue/Sales: Available via SQL (shopify_orders table) or API endpoint /api/sales (PREFER sql agent)
- ROAS: Available via SQL (calculated from orders and ad spend) or API endpoint /api/roas (PREFER sql agent)
- COGS: Available via SQL (shopify_orders table) or API endpoint /api/cogs (PREFER sql agent)
- Ad Spend: Available via SQL (ad_spend tables) or API endpoint /api/ad_spend (PREFER sql agent)

CRITICAL RULES FOR AGENT SELECTION:
1. HOURLY QUERIES: ALWAYS use SQL agent. API endpoints typically return daily/monthly data, not hourly. Use attribution table or shopify_orders with hour-level grouping.
2. CHANNEL/ATTRIBUTION QUERIES: ALWAYS use SQL agent. Channel breakdowns (Meta, Google, Organic) require attribution table queries.
3. GRANULARITY KEYWORDS: "hourly", "by hour", "per hour" → SQL agent (attribution table)
4. CHANNEL KEYWORDS: "channel", "meta", "google", "organic", "attribution", "utm" → SQL agent (attribution table)
5. DEFAULT: PREFER SQL agent for data retrieval. SQL provides more flexibility, direct database access, and custom query capabilities.
6. API FALLBACK: Use API only when SQL is not suitable or when pre-computed daily/monthly aggregations are explicitly needed.

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

User Query: "plot hourly sales and spend for meta and google channel for today hourly"
Interpreted: "Retrieve hourly sales and ad spend data for Meta and Google channels for today and generate a visualization"
Sub-queries:
1. "Query attribution table for hourly sales and ad spend by channel (Meta, Google) for today" → sql (tool: sql_query), priority: 0
   Context: {granularity: "hourly", channels: ["meta", "google"], date: "today"}
2. "Generate hourly trend visualization for sales and spend by channel" → graph (tool: graph_generator), priority: 1, depends on: [1]

User Query: "Get the net profit graph for the last 4 months"
Interpreted: "Retrieve net profit data for the last 4 months and generate a visualization"
Sub-queries:
1. "Query shopify_orders table for net profit data for last 4 months" → sql (tool: sql_query), priority: 0
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
  - required_agents: List of agent names needed (PREFER sql for hourly/channel queries, otherwise prefer sql over api_docs)
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

        # Build capability summary for LLM (pass query for hourly/channel detection)
        capability_summary = self._build_capability_summary(detected_metrics, question)

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
            
            # Post-process: Use dynamic query pattern detection from capability registry
            pattern_matches = self.registry.detect_query_patterns(question)
            
            # Find agents that have strong pattern matches (high boost scores)
            preferred_agents = {}
            for agent_name, match_info in pattern_matches.items():
                if match_info["total_boost"] >= 5:  # Threshold for strong pattern match
                    preferred_agents[agent_name] = match_info
                    logger.info(
                        f"Query pattern detected for {agent_name}: boost={match_info['total_boost']}, "
                        f"patterns={[m['pattern'] for m in match_info['matches']]}"
                    )
            
            # Apply pattern-based agent selection to sub-queries
            if preferred_agents:
                for sq_dict in response.sub_queries:
                    intent = sq_dict.get("intent", "")
                    if intent in ["data_retrieval", "api_call"]:
                        # Find the best matching agent for this sub-query
                        # Check if sub-query text also matches patterns
                        sq_text = sq_dict.get("detailed_query", "") or sq_dict.get("original_phrase", "")
                        sq_patterns = self.registry.detect_query_patterns(sq_text)
                        
                        # Merge query and sub-query pattern matches
                        all_matches = {**pattern_matches}
                        for agent_name, match_info in sq_patterns.items():
                            if agent_name in all_matches:
                                all_matches[agent_name]["total_boost"] += match_info["total_boost"]
                                all_matches[agent_name]["matches"].extend(match_info["matches"])
                            else:
                                all_matches[agent_name] = match_info
                        
                        # Select agent with highest boost
                        best_agent = None
                        best_score = 0
                        for agent_name, match_info in all_matches.items():
                            if match_info["total_boost"] > best_score:
                                best_score = match_info["total_boost"]
                                best_agent = agent_name
                        
                        if best_agent and best_score >= 5:
                            # Force the preferred agent
                            current_agents = sq_dict.get("required_agents", [])
                            if best_agent not in current_agents:
                                sq_dict["required_agents"].insert(0, best_agent)
                                logger.info(
                                    f"Post-processed sub-query to use {best_agent} agent "
                                    f"(pattern boost: {best_score})"
                                )
                            
                            # Remove conflicting agents if pattern strongly indicates specific agent
                            if best_score >= 10:  # Very strong pattern match
                                # Remove api_docs if SQL is preferred (or vice versa)
                                conflicting_agents = []
                                if best_agent == "sql" and "api_docs" in current_agents:
                                    conflicting_agents.append("api_docs")
                                elif best_agent == "api_docs" and "sql" in current_agents:
                                    conflicting_agents.append("sql")
                                
                                for conflicting in conflicting_agents:
                                    if conflicting in sq_dict["required_agents"]:
                                        sq_dict["required_agents"].remove(conflicting)
                                        logger.info(
                                            f"Removed conflicting agent {conflicting} due to strong pattern match for {best_agent}"
                                        )
                                
                                # Update selected_tools based on preferred agent
                                if best_agent == "sql":
                                    sq_dict["selected_tools"] = ["sql_query"]
                                elif best_agent == "api_docs" and "api_" in str(sq_dict.get("selected_tools", [])):
                                    # Keep API tools if they exist
                                    pass

            # Convert to SubQuery objects
            sub_queries = []
            for idx, sq_dict in enumerate(response.sub_queries):
                # Ensure selected_tools are populated from capability registry if not provided
                selected_tools = sq_dict.get("selected_tools", [])
                if not selected_tools:
                    # Try to infer tools from metrics and agents
                    metrics = sq_dict.get("context", {}).get("metrics", [])
                    agents = sq_dict.get("required_agents", [])
                    selected_tools = self._infer_tools_from_metrics_and_agents(metrics, agents, question)

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

    def _build_capability_summary(self, metrics: List[str], query: str = "") -> str:
        """Build a summary of available capabilities for the LLM."""
        # Use dynamic query pattern detection
        pattern_matches = self.registry.detect_query_patterns(query) if query else {}
        
        # Check for strong pattern matches
        strong_patterns = {}
        for agent_name, match_info in pattern_matches.items():
            if match_info["total_boost"] >= 5:
                strong_patterns[agent_name] = match_info
        
        if not metrics:
            if strong_patterns:
                pattern_info = []
                for agent_name, match_info in strong_patterns.items():
                    reasons = [m["reason"] for m in match_info["matches"]]
                    pattern_info.append(f"{agent_name.upper()}: {', '.join(set(reasons))}")
                return f"⚠️ QUERY PATTERN DETECTED: {' | '.join(pattern_info)}"
            return "No specific metrics detected. PREFER SQL agent for data retrieval."

        summary_parts = []
        if strong_patterns:
            pattern_warnings = []
            for agent_name, match_info in strong_patterns.items():
                reasons = [m["reason"] for m in match_info["matches"]]
                pattern_warnings.append(f"{agent_name.upper()}: {', '.join(set(reasons))}")
            summary_parts.append(f"⚠️ QUERY PATTERN DETECTED: {' | '.join(pattern_warnings)}")
        
        for metric in metrics:
            tools = self.registry.find_tools_for_metric(metric)
            if tools:
                api_tools = [t for t in tools if t.type == "api_endpoint"]
                # Check if SQL pattern is strongly preferred
                sql_preferred = "sql" in strong_patterns and strong_patterns["sql"]["total_boost"] >= 5
                
                if api_tools and not sql_preferred:
                    summary_parts.append(
                        f"Metric '{metric}': Available via API ({', '.join([t.name for t in api_tools[:3]])}) - But PREFER sql agent for flexibility"
                    )
                else:
                    summary_parts.append(
                        f"Metric '{metric}': Available via {tools[0].type} - Use {tools[0].agent} agent (PREFER sql)"
                    )
            else:
                summary_parts.append(
                    f"Metric '{metric}': Not directly available - Use SQL agent (attribution/shopify_orders tables)"
                )

        return "\n".join(summary_parts) if summary_parts else "No specific capabilities matched. Default to SQL agent."

    def _infer_tools_from_metrics_and_agents(
        self, metrics: List[str], agents: List[AgentName], query: str = ""
    ) -> List[str]:
        """Infer tool IDs from metrics and agents, with dynamic pattern-based bias."""
        # Use dynamic query pattern detection
        pattern_matches = self.registry.detect_query_patterns(query) if query else {}
        
        # Check for strong pattern matches that indicate specific agent preference
        for agent_name, match_info in pattern_matches.items():
            if match_info["total_boost"] >= 5:
                # Strong pattern match - prioritize this agent
                if agent_name not in agents:
                    agents.insert(0, agent_name)
                    logger.info(
                        f"Query pattern detected for {agent_name} (boost: {match_info['total_boost']}) - "
                        f"prioritizing {agent_name} agent"
                    )
        
        tool_ids = []
        
        # Check if SQL has strong pattern match (should be prioritized)
        sql_has_strong_pattern = (
            "sql" in pattern_matches and 
            pattern_matches["sql"]["total_boost"] >= 5
        )
        
        # Process SQL agent first (highest priority)
        if "sql" in agents:
            if "sql_query" not in tool_ids:
                tool_ids.append("sql_query")
        
        # Then process other agents
        for agent in agents:
            if agent == "sql":
                continue  # Already handled
            elif agent == "api_docs":
                # Only use API tools if SQL doesn't have strong pattern match
                if not sql_has_strong_pattern:
                    for metric in metrics:
                        tools = self.registry.find_tools_for_metric(metric)
                        api_tools = [t for t in tools if t.type == "api_endpoint" and t.agent == "api_docs"]
                        for tool in api_tools:
                            if tool.id not in tool_ids:
                                tool_ids.append(tool.id)
            elif agent == "computation":
                if "computation" not in tool_ids:
                    tool_ids.append("computation")
            elif agent == "graph":
                if "graph_generator" not in tool_ids:
                    tool_ids.append("graph_generator")
            else:
                # Fallback: find tools for metrics matching this agent
                for metric in metrics:
                    tools = self.registry.find_tools_for_metric(metric)
                    for tool in tools:
                        if tool.agent == agent and tool.id not in tool_ids:
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

