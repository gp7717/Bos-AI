"""Composer agent that fuses individual agent outputs into a final answer."""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from typing import Any, Iterable, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI

from Agents.core.models import AgentResult, GraphResult, OrchestratorResponse, Scratchpad, TabularResult
from Agents.QueryAgent.config import get_resources

logger = logging.getLogger(__name__)


def _normalize_tabular_value(value: Any) -> Any:
    """
    Normalize a tabular cell value to ensure it's JSON-serializable and table-renderable.
    
    Converts complex objects (dicts, lists) to comma-separated strings for better readability,
    handles Decimal types, and preserves simple types (str, int, float, bool, None).
    
    Args:
        value: The value to normalize
        
    Returns:
        Normalized value that can be safely serialized to JSON and rendered in tables
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    
    # Handle lists/arrays - convert to comma-separated string
    if isinstance(value, list):
        if not value:
            return ""
        # Convert each element to a readable string
        normalized_items = []
        for item in value:
            if isinstance(item, dict):
                # Try to extract a meaningful field from the object
                readable = _extract_readable_value(item)
                normalized_items.append(readable)
            elif isinstance(item, (str, int, float, bool)):
                normalized_items.append(str(item))
            else:
                normalized_items.append(str(item))
        return ", ".join(normalized_items)
    
    # Handle single dict/object - convert to readable format
    if isinstance(value, dict):
        return _extract_readable_value(value)
    
    # For any other type, convert to string
    return str(value)


def _extract_readable_value(obj: dict) -> str:
    """
    Extract a readable string representation from a dictionary/object.
    
    Uses intelligent field selection based on:
    - Field name patterns (name-like, title-like, label-like fields)
    - Value characteristics (type, length, readability)
    - Field priority scoring
    
    Args:
        obj: Dictionary to extract readable value from
        
    Returns:
        Readable string representation
    """
    if not isinstance(obj, dict) or not obj:
        return str(obj)
    
    def _score_field(key: str, value: Any) -> float:
        """
        Score a field based on how likely it is to be a readable identifier.
        Higher score = more likely to be the best readable value.
        
        Returns:
            Score from 0.0 to 1.0
        """
        score = 0.0
        key_lower = key.lower()
        
        # Pattern matching for field names (dynamic, not hardcoded list)
        # Name-like patterns
        if any(pattern in key_lower for pattern in ["name", "title", "label"]):
            score += 0.4
        # Identifier patterns
        if key_lower in ["id", "identifier", "key", "code", "sku"]:
            score += 0.3
        # Description/display patterns
        if any(pattern in key_lower for pattern in ["description", "text", "value", "display"]):
            score += 0.2
        # Product/item specific patterns
        if any(pattern in key_lower for pattern in ["product", "item", "variant"]):
            score += 0.15
        
        # Value type scoring
        if isinstance(value, str):
            # Prefer short, meaningful strings (not too long, not empty)
            str_len = len(value.strip())
            if 1 <= str_len <= 100:  # Reasonable length
                score += 0.2
            elif str_len > 200:  # Too long, less preferred
                score -= 0.1
        elif isinstance(value, (int, float)):
            # Numeric IDs are okay but less readable than strings
            score += 0.1
        elif isinstance(value, (dict, list)):
            # Complex objects are not directly readable
            score -= 0.2
        
        # Avoid technical/internal fields
        if key_lower.startswith("_") or key_lower in ["metadata", "config", "settings"]:
            score -= 0.3
        
        # Prefer fields that are not None/empty
        if value is None or value == "":
            score -= 0.5
        
        return max(0.0, min(1.0, score))  # Clamp between 0 and 1
    
    # Score all fields and find the best one
    scored_fields = []
    for key, value in obj.items():
        if value is not None:
            score = _score_field(key, value)
            scored_fields.append((score, key, value))
    
    # Sort by score (highest first)
    scored_fields.sort(key=lambda x: x[0], reverse=True)
    
    # Return the highest scoring field if it has a good score
    if scored_fields and scored_fields[0][0] > 0.3:
        best_score, best_key, best_value = scored_fields[0]
        # Convert to string if it's a simple type
        if isinstance(best_value, (str, int, float, bool)):
            return str(best_value)
        elif isinstance(best_value, (dict, list)):
            # For complex values, try to extract from them recursively
            if isinstance(best_value, dict):
                nested = _extract_readable_value(best_value)
                if nested != str(best_value):  # If we got something meaningful
                    return nested
    
    # If no good field found, try to find first simple readable value
    for key, val in obj.items():
        if val is not None and val != "":
            if isinstance(val, (str, int, float, bool)):
                # Check if it's not too long
                str_val = str(val)
                if len(str_val) <= 150:  # Reasonable length for display
                    return str_val
    
    # Fallback: create comma-separated key-value pairs for readability
    try:
        # Limit to first few key-value pairs to avoid overly long strings
        items = list(obj.items())[:3]
        kv_pairs = [f"{k}: {v}" for k, v in items if v is not None and not isinstance(v, (dict, list))]
        if kv_pairs:
            return ", ".join(kv_pairs)
    except Exception:
        pass
    
    # Final fallback: JSON string (truncated if too long)
    try:
        json_str = json.dumps(obj, default=str, ensure_ascii=False)
        if len(json_str) > 200:
            return json_str[:197] + "..."
        return json_str
    except (TypeError, ValueError):
        return str(obj)


def _normalize_tabular_data(tabular: Optional[TabularResult]) -> Optional[TabularResult]:
    """
    Normalize all values in tabular data to ensure proper serialization.
    
    Args:
        tabular: The TabularResult to normalize
        
    Returns:
        A new TabularResult with normalized values, or None if input is None
    """
    if not tabular or not tabular.rows:
        return tabular
    
    normalized_rows = []
    for row in tabular.rows:
        if isinstance(row, dict):
            normalized_row = {
                str(key): _normalize_tabular_value(value)
                for key, value in row.items()
            }
            normalized_rows.append(normalized_row)
        else:
            # If row is not a dict, try to normalize it as-is
            normalized_rows.append(_normalize_tabular_value(row))
    
    return TabularResult(
        columns=tabular.columns,
        rows=normalized_rows,
        row_count=tabular.row_count,
    )


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
                        "Do not include boilerplate, incomplete placeholders, or repeat explanations. Only summarize the most important findings and information relevant to the user question. "
                        "CRITICAL: NEVER generate SVG, HTML, or any visualization code. NEVER ask for clarification about graph formats, granularity, or output types. "
                        "If the user requested a graph/chart and graph data is available in the response, simply mention that the visualization has been generated and provide a summary of the data insights. "
                        "If graph data is not yet available but the user requested visualization, the graph agent will handle it automatically - just summarize the available data. "
                        "The graph data is handled separately by the graph agent - you should only provide a text summary of the data. "
                        "IMPORTANT: Use the shared memory/scratchpad context to provide meaningful data summaries. When tabular data is available, "
                        "provide key insights, trends, and notable findings from the data rather than just stating row counts. "
                        "For example, if sales data is provided, mention key metrics like total revenue, peak periods, trends, or notable patterns."
                    ),
                ),
                (
                    "user",
                    (
                        "User question: {question}\n"
                        "Planner rationale: {planner_rationale}\n"
                        "Shared Memory/Context:\n{scratchpad_summary}\n"
                        "Agent summaries:\n{agent_summaries}"
                    ),
                ),
            ]
        )
        self._simple_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a helpful Business Intelligence assistant. "
                        "Respond naturally and conversationally to the user's query. "
                        "Be friendly, professional, and concise. "
                        "For greetings, respond warmly and offer to help with data queries or analysis. "
                        "Do not use emojis, hashtags, or unnecessary formatting—only use '\\n' for new lines. "
                        "Keep responses brief and to the point."
                    ),
                ),
                (
                    "user",
                    "User query: {question}",
                ),
            ]
        )

    def compose(
        self,
        *,
        question: str,
        planner_rationale: str,
        agent_results: Iterable[AgentResult],
        context: Optional[dict] = None,
        metadata: Optional[dict] = None,
        scratchpad: Optional[Scratchpad] = None,
    ) -> OrchestratorResponse:
        results = list(agent_results)
        summaries = []
        tabular = self._select_tabular(results)
        graphs = self._collect_all_graphs(results)
        graph = graphs[0] if graphs else None  # Backward compatibility
        
        # Normalize tabular data to ensure proper serialization for tables
        normalized_tabular = _normalize_tabular_data(tabular)
        
        for result in results:
            status = result.status
            base = f"Agent: {result.agent} | status: {status}."
            if result.answer:
                base += f" Answer: {result.answer}"
            if result.error:
                base += f" Error: {result.error.message}"
            summaries.append(base)

        # Get scratchpad summary for context
        scratchpad_summary = ""
        if scratchpad:
            scratchpad_summary = scratchpad.get_summary()
        else:
            scratchpad_summary = "No shared memory context available."

        llm_response = self._prompt | self.llm | RunnableLambda(lambda message: message.content)
        answer = llm_response.invoke(
            {
                "question": question,
                "planner_rationale": planner_rationale,
                "scratchpad_summary": scratchpad_summary,
                "agent_summaries": "\n".join(summaries) or "No agents produced outputs.",
            }
        )

        return OrchestratorResponse(
            answer=str(answer),
            data=normalized_tabular,
            graph=graph,  # Backward compatibility
            graphs=graphs,  # All graphs
            agent_results=results,
            metadata=metadata or {},
        )

    def compose_simple_response(
        self,
        *,
        question: str,
        router_rationale: str = "",
        context: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> OrchestratorResponse:
        """
        Compose a response for simple queries that don't require agents.
        Uses a conversational prompt instead of the BI-focused prompt.
        """
        llm_response = self._simple_prompt | self.llm | RunnableLambda(lambda message: message.content)
        answer = llm_response.invoke(
            {
                "question": question,
            }
        )

        response_metadata = metadata or {}
        if router_rationale:
            response_metadata["router_rationale"] = router_rationale

        return OrchestratorResponse(
            answer=str(answer),
            data=None,
            agent_results=[],
            metadata=response_metadata,
        )

    @staticmethod
    def _select_tabular(results: Iterable[AgentResult]) -> Optional[TabularResult]:
        for result in results:
            if result.tabular:
                return result.tabular
        return None

    @staticmethod
    def _select_graph(results: Iterable[AgentResult]) -> Optional[GraphResult]:
        """Select the first graph result from agent results (backward compatibility)."""
        for result in results:
            if result.graph:
                return result.graph
        return None

    @staticmethod
    def _collect_all_graphs(results: Iterable[AgentResult]) -> List[GraphResult]:
        """Collect all graph results from agent results."""
        graphs = []
        for result in results:
            if result.graph:
                graphs.append(result.graph)
        return graphs


__all__ = ["Composer"]


